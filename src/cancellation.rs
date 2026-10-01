//! Abandoning a query nobody is waiting for any more.
//!
//! The application searches as the user types — a query per keystroke — so every query
//! but the last is obsolete by the time the next one arrives. A semantic query does not
//! finish quickly on its own: it embeds the text and then scans every stored vector, with
//! no index to narrow the scan (`O(N·D)`; see
//! [`VectorSearchBackend`](crate::semantic::store_backend::VectorSearchBackend)). Over a
//! library-sized artifact that is on the order of a second: `benches/vector_search.rs`
//! extrapolates 1.1 s for the 6,058,210-line library at the production model's 256
//! dimensions, single-threaded, on an Apple M4. Left to run, the queries a fast typist
//! abandons queue up, and the one that matters waits behind all of them.
//!
//! A [`CancellationToken`] is how a caller says "stop". It hands the token in with the
//! search, keeps a clone, and calls [`CancellationToken::cancel`] when a newer query
//! supersedes this one. The search looks at the token at fixed points and, once it is
//! set, returns [`SemanticSearchError::Cancelled`] instead of a result.
//!
//! # Where a query looks
//!
//! * before anything else — before the result cache is consulted and before the query is
//!   embedded — so a query cancelled before it starts costs nothing;
//! * after the query is embedded, before the scan;
//! * inside every vector scan, once per [`SCAN_CHECK_INTERVAL`] records;
//! * after the scan, before fusion;
//! * after fusion, before the result is cached, counted, or handed back to be hydrated.
//!
//! Embedding a query is one inference call, which nothing outside it can interrupt, so the
//! stretch between the first two checkpoints is the one place a cancelled query keeps
//! running: for the length of one query embedding.
//!
//! # Not a failure
//!
//! A cancelled search is one its caller no longer wants, and nothing went wrong in it.
//! It is therefore not logged, and it is not degraded to the lexical results the way a
//! failed semantic path is — those would answer a question nobody is asking any more.
//! It leaves nothing behind: no query-cache entry, no embedding-cache entry, no telemetry
//! record. The locks it holds are read locks, released on the way out, and returning an
//! error poisons none of them.
//!
//! # Not the indexing cancel/resume the product contract rules out
//!
//! `docs/PRODUCT_CONTRACT.md` §4 rules out a cancellable, resumable *indexing* run,
//! because the application never indexes. This is the other end: a query, which the
//! application runs constantly.

use crate::errors::{SemanticSearchError, VectorStoreError};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// How many records a vector scan visits between two looks at its token.
///
/// A look is one relaxed load of a flag that is written at most once, so it is an L1 hit:
/// about a nanosecond. A record is a dot product over every dimension plus the walk to it
/// and the heap bookkeeping around it: ~180 ns at the production model's 256 dimensions and
/// ~370 ns at 1,024 (`benches/vector_search.rs` on an Apple M4: 300,000 vectors in 52–57 ms,
/// 100,000 in 36–38 ms). Looking at every record would already cost under 1%; looking every
/// 1,024 costs one load per 0.2–0.4 ms of scanning, and the benchmark's scans measured the
/// same with the looks as without them, within run-to-run noise. That is also how late a
/// cancel can be noticed — 0.2–0.4 ms, far below anything a keystroke can tell apart. A
/// power of two, so the test is a mask and not a division.
pub const SCAN_CHECK_INTERVAL: usize = 1024;

/// A caller's way to abandon a search it no longer wants.
///
/// Cheap to clone — every clone shares one flag — and `Send + Sync`, so the thread that
/// starts a search and the one that supersedes it can each hold one. Cancelling is one-way
/// and idempotent: a token cannot be reset, so a new search takes a new token.
///
/// A token nobody cancels changes nothing. That is what every entry point without one
/// passes, which makes `search` and `search_cancellable` with a fresh token the same call.
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask every search holding this token, or a clone of it, to stop.
    ///
    /// Returns at once. A search notices at its next checkpoint — see the module
    /// documentation — not here.
    pub fn cancel(&self) {
        // Relaxed: the flag publishes no data, only the instruction to stop. A scan that
        // sees it one checkpoint late has visited one more interval of records, nothing
        // worse.
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Whether [`Self::cancel`] has been called on this token or any of its clones.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// A checkpoint between two stages of a query.
    pub(crate) fn checkpoint(&self) -> Result<(), SemanticSearchError> {
        if self.is_cancelled() {
            return Err(SemanticSearchError::Cancelled);
        }
        Ok(())
    }

    /// A checkpoint inside a vector scan, `scanned` records in. The scan calls it before
    /// its first record and then every [`SCAN_CHECK_INTERVAL`] records.
    #[inline]
    pub(crate) fn scan_checkpoint(&self, scanned: usize) -> Result<(), VectorStoreError> {
        probe::observe(scanned);
        if self.is_cancelled() {
            return Err(VectorStoreError::Cancelled);
        }
        Ok(())
    }
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

/// Nothing outside a test build: the call compiles to nothing.
#[cfg(not(test))]
mod probe {
    #[inline(always)]
    pub(super) fn observe(_scanned: usize) {}
}

/// What a scan's checkpoints report to a test running on the same thread.
///
/// Thread-local, so tests running side by side cannot see each other's scans. It is what
/// lets a test cancel a scan *during* the scan, deterministically — from another thread,
/// at a checkpoint of its choosing — and see how far the scan got, rather than racing a
/// timer against it.
#[cfg(test)]
pub(crate) mod probe {
    use super::CancellationToken;
    use std::cell::RefCell;
    use std::rc::Rc;

    type Observer = Box<dyn FnMut(usize)>;

    thread_local! {
        static OBSERVER: RefCell<Option<Observer>> = const { RefCell::new(None) };
    }

    /// Run `body` and return, with its result, every checkpoint its scans reached on this
    /// thread, in order — each as the number of records visited before it.
    ///
    /// An empty list means no scan ever started.
    pub(crate) fn checkpoints_of<R>(body: impl FnOnce() -> R) -> (R, Vec<usize>) {
        observing(|_| {}, body)
    }

    /// As [`checkpoints_of`], with `token` cancelled from another thread when a scan
    /// reaches the checkpoint `at` records in.
    ///
    /// The scan waits at that checkpoint until the other thread has cancelled, so where it
    /// stops is decided by the test, not raced: the checkpoint at `at` is the last one it
    /// reports, and it has visited exactly `at` records.
    pub(crate) fn cancelling_at<R>(
        token: &CancellationToken,
        at: usize,
        body: impl FnOnce() -> R,
    ) -> (R, Vec<usize>) {
        let token = token.clone();
        observing(
            move |scanned| {
                if scanned == at {
                    let remote = token.clone();
                    std::thread::spawn(move || remote.cancel())
                        .join()
                        .expect("the cancelling thread must not panic");
                }
            },
            body,
        )
    }

    fn observing<R>(
        mut react: impl FnMut(usize) + 'static,
        body: impl FnOnce() -> R,
    ) -> (R, Vec<usize>) {
        struct Uninstall;
        impl Drop for Uninstall {
            fn drop(&mut self) {
                OBSERVER.with(|slot| slot.borrow_mut().take());
            }
        }

        let seen = Rc::new(RefCell::new(Vec::new()));
        let record = Rc::clone(&seen);
        OBSERVER.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move |scanned| {
                record.borrow_mut().push(scanned);
                react(scanned);
            }))
        });
        let uninstall = Uninstall;
        let result = body();
        drop(uninstall);
        let seen = seen.borrow().clone();
        (result, seen)
    }

    pub(super) fn observe(scanned: usize) {
        OBSERVER.with(|slot| {
            if let Some(observer) = slot.borrow_mut().as_mut() {
                observer(scanned);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_token_is_not_cancelled_and_every_clone_sees_a_cancel() {
        let token = CancellationToken::new();
        let held_elsewhere = token.clone();
        assert!(!token.is_cancelled());
        assert!(token.checkpoint().is_ok());

        held_elsewhere.cancel();
        assert!(token.is_cancelled());
        assert!(matches!(
            token.checkpoint(),
            Err(SemanticSearchError::Cancelled)
        ));
        assert!(matches!(
            token.scan_checkpoint(0),
            Err(VectorStoreError::Cancelled)
        ));

        // One-way and idempotent.
        token.cancel();
        assert!(held_elsewhere.is_cancelled());
        assert_eq!(
            format!("{token:?}"),
            "CancellationToken { cancelled: true }"
        );
    }

    #[test]
    fn two_tokens_do_not_share_a_flag() {
        let first = CancellationToken::new();
        let second = CancellationToken::default();
        first.cancel();
        assert!(!second.is_cancelled());
    }

    /// The host cancels from a thread other than the one searching.
    #[test]
    fn a_token_can_be_cancelled_from_another_thread() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CancellationToken>();

        let token = CancellationToken::new();
        let remote = token.clone();
        std::thread::spawn(move || remote.cancel())
            .join()
            .expect("the cancelling thread must not panic");
        assert!(token.is_cancelled());
    }

    #[test]
    fn the_probe_sees_only_its_own_threads_checkpoints() {
        let token = CancellationToken::new();
        let ((), seen) = probe::checkpoints_of(|| {
            token.scan_checkpoint(0).unwrap();
            let other = token.clone();
            std::thread::spawn(move || other.scan_checkpoint(7).unwrap())
                .join()
                .unwrap();
            token.scan_checkpoint(SCAN_CHECK_INTERVAL).unwrap();
        });
        assert_eq!(seen, vec![0, SCAN_CHECK_INTERVAL]);

        // And it is gone once the body returns.
        let ((), after) = probe::checkpoints_of(|| {});
        assert!(after.is_empty());
    }

    #[test]
    fn the_probe_cancels_at_the_checkpoint_it_was_asked_to() {
        let token = CancellationToken::new();
        let (results, seen) = probe::cancelling_at(&token, SCAN_CHECK_INTERVAL, || {
            [0, SCAN_CHECK_INTERVAL, 2 * SCAN_CHECK_INTERVAL]
                .map(|scanned| token.scan_checkpoint(scanned).is_ok())
        });
        assert_eq!(results, [true, false, false]);
        assert_eq!(seen, vec![0, SCAN_CHECK_INTERVAL, 2 * SCAN_CHECK_INTERVAL]);
        assert!(token.is_cancelled());
    }
}
