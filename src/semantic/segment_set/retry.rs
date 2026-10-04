//! The set's renames and removals. On Windows, one refused because another process holds the
//! file without delete sharing (a scanner, the indexer, the host) is retried for up to ~2 s;
//! elsewhere each is the `std::fs` call it names, once.

use std::fs;
use std::io;
use std::path::Path;
#[cfg(any(windows, test))]
use std::time::Duration;

/// 1,888 ms in all, the policy of `otzaria_search_engine`'s index directory: outlasts a
/// scanner, and bounds a refusal that is not transient.
#[cfg(any(windows, test))]
pub(super) const DELAYS: [Duration; 14] = [
    Duration::from_millis(1),
    Duration::from_millis(2),
    Duration::from_millis(5),
    Duration::from_millis(10),
    Duration::from_millis(20),
    Duration::from_millis(50),
    Duration::from_millis(100),
    Duration::from_millis(200),
    Duration::from_millis(250),
    Duration::from_millis(250),
    Duration::from_millis(250),
    Duration::from_millis(250),
    Duration::from_millis(250),
    Duration::from_millis(250),
];

#[cfg(all(windows, test))]
thread_local! {
    /// The paths of every refused attempt on this thread, for the tests that hold a file open.
    pub(crate) static REFUSED: std::cell::RefCell<Vec<std::path::PathBuf>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `fs::rename`.
pub(crate) fn rename(from: &Path, to: &Path) -> io::Result<()> {
    retried("moving", &[from, to], || fs::rename(from, to))
}

/// `fs::remove_file`.
pub(crate) fn remove_file(path: &Path) -> io::Result<()> {
    retried("removing", &[path], || fs::remove_file(path))
}

/// `fs::remove_dir`.
pub(crate) fn remove_dir(path: &Path) -> io::Result<()> {
    retried("removing", &[path], || fs::remove_dir(path))
}

/// `fs::remove_dir_all`.
pub(crate) fn remove_dir_all(path: &Path) -> io::Result<()> {
    retried("removing", &[path], || fs::remove_dir_all(path))
}

#[cfg(windows)]
fn retried(
    what: &str,
    paths: &[&Path],
    mut operation: impl FnMut() -> io::Result<()>,
) -> io::Result<()> {
    let mut refused = 0usize;
    let result = retry(
        || {
            let attempt = operation();
            if attempt.as_ref().is_err_and(is_transient) {
                refused += 1;
            }
            attempt
        },
        is_transient,
        &DELAYS,
        std::thread::sleep,
    );
    #[cfg(test)]
    REFUSED.with(|log| {
        for _ in 0..refused {
            log.borrow_mut()
                .extend(paths.iter().map(|path| path.to_path_buf()));
        }
    });
    let describe = || {
        let paths: Vec<String> = paths
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        format!("{what} {}", paths.join(" to "))
    };
    match &result {
        Ok(()) if refused > 0 => log::info!(
            "Retried {}: it succeeded after {refused} refused attempt(s)",
            describe()
        ),
        Err(error) if refused > DELAYS.len() => log::warn!(
            "Gave up {} after {refused} refused attempts: {error}",
            describe()
        ),
        _ => {}
    }
    result
}

#[cfg(not(windows))]
#[inline]
fn retried(
    _what: &str,
    _paths: &[&Path],
    mut operation: impl FnMut() -> io::Result<()>,
) -> io::Result<()> {
    operation()
}

/// Windows' refusals while another handle holds a file without delete sharing: ACCESS_DENIED
/// (5) replacing it, SHARING_VIOLATION (32) moving or removing it.
#[cfg(windows)]
fn is_transient(error: &io::Error) -> bool {
    const ERROR_ACCESS_DENIED: i32 = 5;
    const ERROR_SHARING_VIOLATION: i32 = 32;
    matches!(
        error.raw_os_error(),
        Some(ERROR_ACCESS_DENIED | ERROR_SHARING_VIOLATION)
    )
}

/// Run `operation` until it succeeds, fails with an error `transient` rejects, or is refused
/// once more than `delays` has entries; the last error is returned as is.
#[cfg(any(windows, test))]
fn retry<T>(
    mut operation: impl FnMut() -> io::Result<T>,
    transient: impl Fn(&io::Error) -> bool,
    delays: &[Duration],
    mut sleep: impl FnMut(Duration),
) -> io::Result<T> {
    let mut delays = delays.iter();
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) if transient(&error) => match delays.next() {
                Some(&delay) => sleep(delay),
                None => return Err(error),
            },
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const SHORT: [Duration; 4] = [
        Duration::from_millis(1),
        Duration::from_millis(2),
        Duration::from_millis(4),
        Duration::from_millis(8),
    ];

    fn refused(attempt: usize) -> io::Error {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("refusal {attempt}"),
        )
    }

    fn is_refused(error: &io::Error) -> bool {
        error.kind() == io::ErrorKind::PermissionDenied
    }

    #[test]
    fn retry_succeeds_once_the_refusals_stop() {
        for refusals in 0..=SHORT.len() {
            let attempts = Cell::new(0);
            let mut slept = Vec::new();
            let result = retry(
                || {
                    attempts.set(attempts.get() + 1);
                    match attempts.get() <= refusals {
                        true => Err(refused(attempts.get())),
                        false => Ok(attempts.get()),
                    }
                },
                is_refused,
                &SHORT,
                |delay| slept.push(delay),
            );
            assert_eq!(result.unwrap(), refusals + 1);
            assert_eq!(slept, SHORT[..refusals]);
        }
    }

    #[test]
    fn retry_returns_the_last_refusal_once_the_delays_run_out() {
        let attempts = Cell::new(0);
        let mut slept = Vec::new();
        let result: io::Result<()> = retry(
            || {
                attempts.set(attempts.get() + 1);
                Err(refused(attempts.get()))
            },
            is_refused,
            &SHORT,
            |delay| slept.push(delay),
        );
        let error = result.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(error.to_string(), format!("refusal {}", SHORT.len() + 1));
        assert_eq!(attempts.get(), SHORT.len() + 1);
        assert_eq!(slept, SHORT);
    }

    #[test]
    fn retry_returns_an_error_it_does_not_accept_at_once() {
        let attempts = Cell::new(0);
        let mut slept = Vec::new();
        let result: io::Result<()> = retry(
            || {
                attempts.set(attempts.get() + 1);
                match attempts.get() {
                    1 => Err(refused(1)),
                    _ => Err(io::Error::new(io::ErrorKind::NotFound, "gone")),
                }
            },
            is_refused,
            &SHORT,
            |delay| slept.push(delay),
        );
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
        assert_eq!(attempts.get(), 2);
        assert_eq!(slept, SHORT[..1]);

        // Not even once, when the first error is not a refusal.
        let attempts = Cell::new(0);
        let result: io::Result<()> = retry(
            || {
                attempts.set(attempts.get() + 1);
                Err(io::Error::new(io::ErrorKind::NotFound, "gone"))
            },
            is_refused,
            &SHORT,
            |_| panic!("nothing to wait for"),
        );
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
        assert_eq!(attempts.get(), 1);
    }

    #[test]
    fn the_backoff_totals_what_its_comment_says() {
        let total: Duration = DELAYS.iter().sum();
        assert_eq!(total, Duration::from_millis(1_888));
    }

    /// An error that is no refusal is returned from the first attempt.
    #[test]
    fn an_operation_that_is_not_refused_fails_at_once() {
        let missing = std::env::temp_dir().join(format!(
            "otzaria_retry_missing_{}_{}",
            std::process::id(),
            line!()
        ));
        let started = std::time::Instant::now();
        let error = rename(&missing, &missing.with_extension("to")).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert_eq!(
            remove_file(&missing).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(
            remove_dir(&missing).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(
            remove_dir_all(&missing).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert!(started.elapsed() < Duration::from_millis(500));
    }

    #[cfg(windows)]
    #[test]
    fn only_a_windows_refusal_is_transient() {
        assert!(is_transient(&io::Error::from_raw_os_error(5)));
        assert!(is_transient(&io::Error::from_raw_os_error(32)));
        // ERROR_FILE_NOT_FOUND, ERROR_DISK_FULL, ERROR_DIR_NOT_EMPTY.
        for code in [2, 112, 145] {
            assert!(!is_transient(&io::Error::from_raw_os_error(code)), "{code}");
        }
        // Not from the OS, so not the operation's.
        assert!(!is_transient(&refused(1)));
    }
}
