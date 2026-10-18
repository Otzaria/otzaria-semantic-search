//! Compaction: a set's segments merged into one, on the device, by file operations alone.
//!
//! Every delta adds about 6% to a set and leaves the vectors it superseded in place, so a
//! set grows; compaction writes the live part of it again as one segment and drops the
//! rest. Nothing is embedded: vectors are copied byte for byte, which is why every segment
//! of a set shares one codec epoch — and every block is verified before it is copied, since
//! the copy is checksummed afresh.
//!
//! Records are gathered book by book from the newest complete snapshot of each key;
//! distinct parent anchors within a book survive. On a live index of the same version,
//! the host can verify and refresh each occurrence, including inner windows. Legacy
//! sources expose only the anchor column and conservatively keep unverified windows.
//! Vectors remain deduplicated: the first appearance takes a slot, later books an extra.

use super::files::{self, io_error, Durable, SetLock, SetSegment, STAGING_DIR};
use super::install::{full_or_io, NewGeneration};
use super::{recover, retry, space, SegmentSet, SetInfo};
use crate::cancellation::CancellationToken;
use crate::distribution::package::PackageKind;
use crate::errors::{ArtifactError, SemanticSearchError, VectorStoreError};
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::occurrence::ALIAS_WINDOW;
use crate::semantic::oxv::reader::Segment;
use crate::semantic::oxv::writer::{SegmentBuilder, SegmentSpec};
use crate::semantic::resolve::LiveKeySource;
use crate::semantic::versioning::hex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// When to compact, and how.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPolicy {
    /// Compact once the deltas together are this fraction of the base or more.
    pub max_delta_ratio: f64,
    /// Compact once the set has more segments than this.
    pub max_segments: u32,
    /// Compact once this fraction of the slots is dead.
    pub max_dead_ratio: f64,
    /// Refuse to start without this many times the output's size free.
    pub min_free_space_factor: f64,
    /// Re-anchor every record on the live index, when it holds the set's library version.
    pub refresh_hints: bool,
    /// Compact whatever the thresholds say.
    pub force: bool,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            max_delta_ratio: 0.20,
            max_segments: 4,
            max_dead_ratio: 0.05,
            min_free_space_factor: 1.15,
            refresh_hints: true,
            force: false,
        }
    }
}

impl CompactionPolicy {
    /// Why this policy would compact a set described by `info`, or `None`.
    pub fn wants(&self, info: &SetInfo) -> Option<String> {
        if self.force {
            return Some("forced".to_string());
        }
        if info.segments.len() > self.max_segments as usize {
            return Some(format!(
                "{} segments, more than {}",
                info.segments.len(),
                self.max_segments
            ));
        }
        let base = info.segments.first().map_or(0, |segment| segment.size);
        let deltas: u64 = info
            .segments
            .iter()
            .skip(1)
            .map(|segment| segment.size)
            .sum();
        if base > 0 && deltas as f64 > self.max_delta_ratio * base as f64 {
            return Some(format!(
                "the deltas hold {deltas} bytes, more than {} of the base's {base}",
                self.max_delta_ratio
            ));
        }
        let slots = info.slots_live + info.slots_dead;
        if slots > 0 && info.slots_dead as f64 > self.max_dead_ratio * slots as f64 {
            return Some(format!(
                "{} of {slots} slots are dead, more than {}",
                info.slots_dead, self.max_dead_ratio
            ));
        }
        None
    }
}

/// What a compaction did.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionReport {
    pub compacted: bool,
    /// Why it compacted, or why not.
    pub reason: String,
    /// The set's generation afterwards.
    pub generation: u64,
    pub bytes_before: u64,
    pub bytes_after: u64,
    pub slots_before: u64,
    pub slots_after: u64,
    /// Records dropped because their book no longer holds their key.
    pub records_pruned: u64,
    /// Records re-anchored on another live line.
    pub hints_refreshed: u64,
    pub elapsed_ms: u64,
}

/// The slot assigned to a live key, while slots are being assigned.
const UNASSIGNED: u32 = u32::MAX;

/// Merge the set at `dir` into one segment when `policy` asks for it.
///
/// `live`, when given and when it holds the set's library version, re-anchors the records
/// on the lines that hold their keys today. The set is locked throughout; the new
/// generation replaces the old in one flip, and the old segments go with the garbage.
///
/// Every block of every segment is read and checked against its CRC first: the bytes are
/// copied, and their CRCs computed again, so a damaged source would otherwise pass its
/// damage on where no scrub can find it. A source that fails is condemned, as
/// [`scrub`](super::scrub) condemns it, and the compaction returns
/// [`VectorStoreError::Corrupted`].
///
/// Cancellable at every block read, between books and at every 1 MiB written; a cancelled
/// or failed compaction leaves the set as it was and its partial output is removed.
pub fn compact(
    dir: &Path,
    policy: &CompactionPolicy,
    live: Option<&dyn LiveKeySource>,
    cancel: &CancellationToken,
) -> Result<CompactionReport, SemanticSearchError> {
    let started = Instant::now();
    let _lock = SetLock::take(dir)?;
    recover(dir)?;
    let set = SegmentSet::open_unlocked(dir)?;
    let info = set.info().clone();
    let slots_before = info.slots_live + info.slots_dead;
    let Some(reason) = policy.wants(&info) else {
        return Ok(CompactionReport {
            compacted: false,
            reason: "no threshold of the policy is reached".to_string(),
            generation: info.generation,
            bytes_before: info.bytes_on_disk,
            bytes_after: info.bytes_on_disk,
            slots_before,
            slots_after: slots_before,
            records_pruned: 0,
            hints_refreshed: 0,
            elapsed_ms: started.elapsed().as_millis() as u64,
        });
    };

    let codec = set.codec().clone();
    let width = codec.bytes_per_vector() as u64 + if codec.has_vector_scales() { 4 } else { 0 };
    let estimate = info.slots_live * (width + 20) + (64 << 10);
    let needed = (estimate as f64 * policy.min_free_space_factor) as u64;
    if let Some(available) = space::available(dir) {
        if available < needed {
            return Err(ArtifactError::InsufficientSpace { needed, available }.into());
        }
    }

    // 0. Every block of every source. What is copied lands under CRCs computed afresh, so
    // damage copied from a block no scrub has read would be past every scrub after it; a
    // source that fails is condemned as a scrub condemns it, and nothing is written.
    for (segment, entry) in set.segments().iter().zip(&set.document().segments) {
        if let Err(error) = segment.verify_blocks(cancel, |_| {}) {
            if let VectorStoreError::Corrupted { reason } = &error {
                files::damage_found();
                let failed = files::sha256_cancellable(segment.file_bytes(), cancel)?;
                files::condemn(dir, &entry.id, &failed, reason, cancel)?;
                log::error!("Compaction of {}: {reason}", dir.display());
            }
            return Err(error.into());
        }
    }

    // 1. Every live slot, by key, the newest segment's copy first.
    let segments = set.segments();
    let deleted = set.deleted();
    let mut live_slots: Vec<(ChunkKey, u16, u32)> = Vec::with_capacity(info.slots_live as usize);
    for (seg, segment) in segments.iter().enumerate() {
        for (slot, key) in segment.keys().enumerate() {
            if !deleted[seg].is_set(slot as u32) {
                live_slots.push((key, seg as u16, slot as u32));
            }
        }
    }
    // Total: a key a segment holds live twice — the format allows it — keeps its first slot.
    live_slots.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    live_slots.dedup_by(|later, earlier| later.0 == earlier.0);
    let mut new_slot = vec![UNASSIGNED; live_slots.len()];

    // 2. and 3. Records book by book, slots assigned as a build assigns them.
    let mut books: Vec<(&str, u16, usize)> = segments
        .iter()
        .enumerate()
        .flat_map(|(seg, segment)| {
            segment
                .books()
                .iter()
                .enumerate()
                .map(move |(index, book)| (&*book.name, seg as u16, index))
        })
        .collect();
    books.sort_unstable_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(&b.1)));

    let document = set.document();
    let refresh = policy.refresh_hints
        && live.is_some_and(|live| live.library_version() == document.library_version);
    let mut builder = SegmentBuilder::new(
        SegmentSpec {
            kind: PackageKind::Compacted,
            identity_digest: segments[0].identity_digest(),
            from_library_version: 0,
            to_library_version: document.library_version,
            library_release_tag: document.library_release_tag.clone(),
        },
        codec.clone(),
    );
    // The set's own key width: a compaction never changes what its keys are.
    builder
        .set_format_version(segments[0].format_version())
        .map_err(io_error("the segment format".to_string()))?;
    let mut order: Vec<(u16, u32)> = Vec::with_capacity(live_slots.len());
    let (mut pruned, mut refreshed) = (0u64, 0u64);
    let mut live_lines: Vec<(u32, u64)> = Vec::new();
    let mut start = 0;
    while start < books.len() {
        if cancel.is_cancelled() {
            return Err(SemanticSearchError::Cancelled);
        }
        let name = books[start].0;
        let end = start
            + books[start..]
                .iter()
                .take_while(|book| book.0 == name)
                .count();
        // Resolve effective snapshots before merging: a newer key's locations replace
        // all older ones, and distinct parents in the same book remain distinct.
        let mut records = Vec::new();
        set.book_occurrences(name, &mut records);
        start = end;
        if refresh {
            let source = live.expect("refresh implies a source");
            let mut resolved = Vec::new();
            live_lines.clear();
            let has_keys = source.book_keys(name, &mut live_lines)?;
            let mut by_key: HashMap<u64, Vec<u32>> = HashMap::new();
            if has_keys {
                for &(ordinal, value) in &live_lines {
                    if value != 0 {
                        by_key.entry(value).or_default().push(ordinal);
                    }
                }
            }
            for (key, mut record) in records {
                if let Some(verified) = source.resolve_record(key, &record, cancel)? {
                    if verified.is_empty() {
                        pruned += 1;
                    }
                    for mut fresh in verified {
                        // A legacy key may also occur in a book the live index cannot
                        // inspect. Keep the entire key legacy instead of publishing a
                        // partial snapshot that would hide those unknown occurrences.
                        if record.anchor.is_none() {
                            fresh.anchor = None;
                        }
                        if fresh.hint != record.hint {
                            refreshed += 1;
                        }
                        resolved.push((key, fresh));
                    }
                    continue;
                }
                // Legacy hosts only expose the parent's column. It can refresh a
                // recorded parent, but cannot prove an inner window was removed.
                let lookup = record.anchor.unwrap_or(key).column_value();
                match by_key.get(&lookup) {
                    Some(ordinals) => {
                        let closest = *ordinals
                            .iter()
                            .min_by_key(|ordinal| (ordinal.abs_diff(record.hint), **ordinal))
                            .expect("nonempty");
                        if closest != record.hint {
                            refreshed += 1;
                            record.hint = closest;
                        }
                        resolved.push((key, record));
                    }
                    None if has_keys && (record.window == 0 || record.window == ALIAS_WINDOW) => {
                        pruned += 1;
                    }
                    None => resolved.push((key, record)),
                }
            }
            records = resolved;
        }
        records.sort_unstable_by(|(ka, a), (kb, b)| {
            a.hint
                .cmp(&b.hint)
                .then(ka.cmp(kb))
                .then(a.window.cmp(&b.window))
                .then(a.anchor.cmp(&b.anchor))
        });
        records.dedup();
        let mut primary = Vec::new();
        let mut windows = Vec::new();
        let mut extras = Vec::new();
        let mut occurrences = Vec::new();
        let mut least = HashMap::<ChunkKey, u8>::new();
        // An alias still needs its backing vector when the neighbour's own occurrence
        // has gone stale. Keep a slot/extra for every surviving key; the complete
        // occurrence snapshot supplies its actual roles, without inventing a normal
        // occurrence at the alias's hint. An alias-only slot keeps the penalized role.
        for (key, record) in &records {
            least
                .entry(*key)
                .and_modify(|window| *window = (*window).min(record.window))
                .or_insert(record.window);
        }
        let mut recorded = std::collections::HashSet::new();
        let mut placed = std::collections::HashSet::new();
        for (key, record) in records {
            if let Some(anchor) = record.anchor {
                occurrences.push(crate::semantic::oxv::occurrence::Occurrence {
                    key,
                    book: 0,
                    hint: record.hint,
                    anchor,
                    window: record.window,
                });
            }
            if !recorded.insert(key) {
                continue;
            }
            let Ok(position) = live_slots.binary_search_by(|probe| probe.0.cmp(&key)) else {
                continue;
            };
            if new_slot[position] == UNASSIGNED {
                new_slot[position] = order.len() as u32;
                let (_, seg, slot) = live_slots[position];
                order.push((seg, slot));
                primary.push((key, record.hint));
                windows.push(least[&key]);
            } else {
                builder.minimize_slot_window(new_slot[position], least[&key]);
                extras.push((record.hint, new_slot[position]));
            }
            placed.insert(key);
        }
        occurrences.retain(|row| !row.is_alias() || placed.contains(&row.key));
        builder
            .add_book_windowed_hashed(
                name,
                &primary,
                &windows,
                &extras,
                &[],
                set.book_source_hash(name),
            )
            .map_err(io_error(format!("assembling book {name:?}")))?;
        builder
            .add_occurrences(name, &occurrences)
            .map_err(io_error(format!("assembling occurrences of {name:?}")))?;
    }
    drop(new_slot);
    drop(live_slots);
    if order.is_empty() {
        return Err(VectorStoreError::Corrupted {
            reason: format!(
                "{}: compacting would leave no record, which no set describes",
                dir.display()
            ),
        }
        .into());
    }

    // 4. The segment, vectors copied in the new slot order.
    let staging = dir.join(STAGING_DIR);
    fs::create_dir_all(&staging).map_err(io_error(format!("creating {}", staging.display())))?;
    let partial = staging.join("compacted.oxv.partial");
    let written = (|| {
        let mut sink = builder
            .write(&partial)
            .map_err(|error| full_or_io(error, &partial, needed, 0))?;
        for (index, (seg, slot)) in order.iter().enumerate() {
            if index % 4096 == 0 && cancel.is_cancelled() {
                return Err(SemanticSearchError::Cancelled);
            }
            let source = &segments[*seg as usize];
            sink.push_encoded(source.vector(*slot), source.vector_scale(*slot))
                .map_err(|error| full_or_io(error, &partial, needed, index as u64 * width))?;
        }
        sink.finish()
            .map_err(|error| full_or_io(error, &partial, needed, order.len() as u64 * width))
    })();
    let written = match written {
        Ok(written) => written,
        Err(error) => {
            let _ = retry::remove_file(&partial);
            return Err(error);
        }
    };
    // `VectorSink::finish` flushed it.
    files::note(|| Durable::File(partial.clone()));

    // 5. Re-open it and compare a sample with its sources.
    {
        let output = Segment::open(&partial)?;
        if output.slot_count() as usize != order.len() {
            return Err(VectorStoreError::Corrupted {
                reason: "the compacted segment does not hold the slots it was given".into(),
            }
            .into());
        }
        let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ document.generation;
        for _ in 0..1000.min(order.len()) {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let slot = (state >> 33) as usize % order.len();
            let (seg, source) = order[slot];
            let original = &segments[seg as usize];
            if output.vector(slot as u32) != original.vector(source)
                || output.vector_scale(slot as u32) != original.vector_scale(source)
                || output.key(slot as u32) != original.key(source)
            {
                return Err(VectorStoreError::Corrupted {
                    reason: format!(
                        "slot {slot} of the compacted segment is not segment {seg}'s slot {source}"
                    ),
                }
                .into());
            }
        }
    }

    // 6. Publish it as a generation of its own. Its id names its content: a file of that name
    // already there holds these bytes — the same compaction, done before — or is damage.
    let id = hex(&written.segment_id);
    let file = files::segment_file(&id);
    let target = dir.join(&file);
    let placed = files::place_segment(dir, &id, &partial, &hex(&written.sha256), cancel, |held| {
        VectorStoreError::Corrupted {
            reason: format!(
                "segment {id} is served with SHA-256 {held}, and a compaction wrote other bytes \
                 under its name"
            ),
        }
        .into()
    })?;
    if placed == files::Placed::Kept {
        retry::remove_file(&partial)
            .map_err(io_error(format!("removing {}", partial.display())))?;
    }
    let segment = Segment::open(&target)?;
    let mut generation = NewGeneration::empty(&document.identity, &document.codec_params_sha256);
    generation.library_release_tag = document.library_release_tag.clone();
    generation.push(
        SetSegment {
            id,
            file,
            kind: PackageKind::Compacted,
            from: 0,
            to: document.library_version,
            sha256: hex(&written.sha256),
            size: written.size,
            slots: written.counts.slots,
            package_digest: None,
            // A merge of segments that share a family and, in practice, a passage package:
            // the base's stands for them all.
            provenance: document.segments[0].provenance.clone(),
            del: Default::default(),
            links: None,
        },
        &segment,
        &[],
        cancel,
    )?;
    drop(segment);
    let base = set.pointer().clone();
    drop(set);
    let committed = generation.commit(dir, Some(&base))?;
    log::info!(
        "Compacted {} ({reason}): {} segment(s) and {slots_before} slot(s) into one of {}, \
         {pruned} record(s) pruned, {refreshed} re-anchored",
        dir.display(),
        info.segments.len(),
        written.counts.slots
    );
    Ok(CompactionReport {
        compacted: true,
        reason,
        generation: committed.generation,
        bytes_before: info.bytes_on_disk,
        bytes_after: written.size,
        slots_before,
        slots_after: written.counts.slots,
        records_pruned: pruned,
        hints_refreshed: refreshed,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}
