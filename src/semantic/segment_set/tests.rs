//! Segment sets end to end: install, apply, crash, recover, collect, compact, scrub.

use super::install::{Step, CRASH_AT};
use super::*;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::codec::Codec;
use crate::semantic::oxv::scan::ScanRequest;
use crate::semantic::oxv::testing::{key, spec, write_segment, Random, TempDir, TestBook};
use crate::semantic::resolve::{BookSet, LiveKeySource, ResolveError};
use crate::semantic::versioning::{test_identity, test_provenance, StoreIdentity};
use std::collections::{BTreeMap, BTreeSet};

const DIM: usize = 32;

/// A library version, as a set's records describe it: book → (hint → key).
#[derive(Clone, Default)]
struct Library {
    version: u32,
    books: BTreeMap<String, BTreeMap<u32, u64>>,
}

impl Library {
    fn with(version: u32, books: &[(&str, &[(u32, u64)])]) -> Self {
        Self {
            version,
            books: books
                .iter()
                .map(|(name, lines)| (name.to_string(), lines.iter().copied().collect()))
                .collect(),
        }
    }

    fn keys(&self) -> BTreeSet<u64> {
        self.books
            .values()
            .flat_map(|lines| lines.values().copied())
            .collect()
    }

    /// One record per book and key, at the lowest hint the key has in the book.
    fn pairs(&self) -> BTreeMap<(String, u64), u32> {
        let mut pairs = BTreeMap::new();
        for (book, lines) in &self.books {
            for (hint, key) in lines {
                pairs.entry((book.clone(), *key)).or_insert(*hint);
            }
        }
        pairs
    }
}

/// The vector a key embeds as: the same in every segment, as content addressing promises.
fn vector_of(key: u64) -> Vec<f32> {
    Random(key.wrapping_mul(0x9E37_79B9) ^ 0xABCD).unit(DIM)
}

fn codec() -> Codec {
    Codec::i8_sym_vec(DIM).unwrap()
}

fn identity() -> IndexVersion {
    let mut identity = test_identity();
    identity.model.embedding_dim = DIM as u32;
    identity.store = StoreIdentity {
        backend_id: STORE_BACKEND_ID.to_string(),
        store_format_version: 2,
        vector_precision: "i8-sym-vec".to_string(),
    };
    identity
}

fn expectation() -> InstallExpectation {
    InstallExpectation {
        identity: identity(),
        published_manifest_sha256: None,
    }
}

/// Books for a segment of `kind`, assigning slots as §1.4 does: books in name order,
/// records in hint order, a key's first appearance a slot and every later one an extra.
/// `shipped` decides which keys this segment ships; `foreign` which records it carries for
/// keys an older segment ships.
fn books_for(records: &BTreeMap<(String, u64), u32>, shipped: &BTreeSet<u64>) -> Vec<TestBook> {
    let mut by_book: BTreeMap<&str, Vec<(u32, u64)>> = BTreeMap::new();
    for ((book, key), hint) in records {
        by_book.entry(book).or_default().push((*hint, *key));
    }
    let mut slot_of: BTreeMap<u64, u32> = BTreeMap::new();
    let mut books = Vec::new();
    for (name, mut lines) in by_book {
        lines.sort();
        let mut book = TestBook::named(name);
        for (hint, key) in lines {
            if shipped.contains(&key) {
                match slot_of.get(&key) {
                    Some(slot) => book.extras.push((hint, *slot)),
                    None => {
                        let slot = slot_of.len() as u32;
                        slot_of.insert(key, slot);
                        book.primary.push((self::key(key), hint, vector_of(key)));
                    }
                }
            } else {
                book.foreign.push((self::key(key), hint));
            }
        }
        books.push(book);
    }
    books
}

/// Write a release of `library` — a base, or a delta from `previous` — and its manifest.
fn release(dir: &TempDir, library: &Library, previous: Option<&Library>) -> (PathBuf, String) {
    let pairs = library.pairs();
    let (kind, from, books, tombstones) = match previous {
        None => (
            PackageKind::Base,
            0,
            books_for(&pairs, &library.keys()),
            Vec::new(),
        ),
        Some(previous) => {
            let old_keys = previous.keys();
            let old_pairs = previous.pairs();
            let new_keys: BTreeSet<u64> = library.keys().difference(&old_keys).copied().collect();
            // Every record of a new key, and the new records of old keys.
            let records: BTreeMap<(String, u64), u32> = pairs
                .iter()
                .filter(|((book, key), _)| {
                    new_keys.contains(key) || !old_pairs.contains_key(&(book.clone(), *key))
                })
                .map(|(pair, hint)| (pair.clone(), *hint))
                .collect();
            let gone: Vec<ChunkKey> = old_keys
                .difference(&library.keys())
                .map(|k| key(*k))
                .collect();
            (
                PackageKind::Delta,
                previous.version,
                books_for(&records, &new_keys),
                gone,
            )
        }
    };
    let mut spec = spec(kind, from, library.version);
    spec.identity_digest = identity().identity_digest();
    let path = dir.join(&format!("{kind}-{}.oxv", library.version));
    let _ = std::fs::remove_file(&path);
    let written = write_segment(&path, spec, codec(), &books, &tombstones);
    let manifest = ReleaseManifest::for_segment(
        &written,
        &identity(),
        codec().params_sha256(),
        test_provenance(),
        "2026-10-02T00:00:00Z".to_string(),
    );
    (path, manifest.to_json())
}

fn install(dir: &Path, release: &(PathBuf, String)) -> Result<ApplyReport, SemanticSearchError> {
    install_package(
        dir,
        &InstallSource {
            segment: &release.0,
            manifest_json: &release.1,
        },
        &expectation(),
        &CancellationToken::new(),
    )
}

/// Every hit of a full scan, as `key → (score, {(book, hint)})`.
fn everything(
    set: &SegmentSet,
    query_seed: u64,
) -> BTreeMap<ChunkKey, (f32, BTreeSet<(String, u32)>)> {
    let query = Random(query_seed).unit(DIM);
    set.scan(
        &query,
        &ScanRequest {
            top_k: 100_000,
            books: None,
            threads: 2,
        },
        &CancellationToken::new(),
    )
    .unwrap()
    .into_iter()
    .map(|hit| {
        let records = hit
            .records
            .iter()
            .map(|record| (record.book.to_string(), record.hint))
            .collect();
        (hit.key, (hit.score, records))
    })
    .collect()
}

fn v29() -> Library {
    Library::with(
        29,
        &[
            ("id:1", &[(0, 1), (1, 2), (2, 3), (5, 1)]),
            ("id:2", &[(0, 4), (1, 2), (3, 5)]),
            ("id:3", &[(0, 6), (1, 7)]),
        ],
    )
}

/// v29 → v30: key 3 is gone (a tombstone); key 8 is new, in an old book and a new one; key
/// 6 appears in a new book (a foreign record); key 4 moves within its book.
fn v30() -> Library {
    Library::with(
        30,
        &[
            ("id:1", &[(0, 1), (1, 2), (2, 8)]),
            ("id:2", &[(0, 5), (1, 2), (4, 4)]),
            ("id:3", &[(0, 6), (1, 7)]),
            ("id:4", &[(0, 6), (1, 8)]),
        ],
    )
}

#[test]
fn a_base_installs_opens_and_scans() {
    let work = TempDir::new("set_base");
    let dir = work.join("vectors");
    let base = release(&work, &v29(), None);
    let report = install(&dir, &base).unwrap();
    assert_eq!(report.kind, PackageKind::Base);
    assert_eq!(
        (report.library_version, report.generation, report.segments),
        (29, 1, 1)
    );
    assert_eq!(report.slots_added, 7);
    assert!(!report.already_applied);
    assert!(
        base.0.exists(),
        "a segment outside incoming/ is copied, not taken"
    );

    let set = SegmentSet::open(&dir).unwrap();
    let info = set.info();
    assert_eq!((info.generation, info.library_version), (1, 29));
    assert_eq!((info.slots_live, info.slots_dead), (7, 0));
    assert!(!info.recovered_from_previous);
    assert_eq!(info.identity_digest, identity().identity_digest_hex());
    assert!(dir.join("CURRENT").exists() && !dir.join("PREVIOUS").exists());
    let id = &info.segments[0].id;
    assert!(dir.join(format!("segments/{id}.package.json")).exists());

    let hits = everything(&set, 1);
    assert_eq!(hits.len(), 7);
    let records = &hits[&key(2)].1;
    assert_eq!(
        records,
        &BTreeSet::from([("id:1".to_string(), 1), ("id:2".to_string(), 1)])
    );
    assert_eq!(info, &self::info(&dir).unwrap().unwrap());
}

#[test]
fn a_delta_tombstones_ships_and_links_against_the_base() {
    let work = TempDir::new("set_delta");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let report = install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    assert_eq!(report.kind, PackageKind::Delta);
    assert_eq!(
        (report.library_version, report.generation, report.segments),
        (30, 2, 2)
    );
    assert_eq!(report.slots_added, 1, "key 8 alone is new");
    assert_eq!(report.tombstones_applied, 1, "key 3 is gone");
    assert_eq!(report.foreign_unresolved, 0);

    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!(set.info().slots_dead, 1);
    let hits = everything(&set, 2);
    assert!(
        !hits.contains_key(&key(3)),
        "a tombstoned key is never returned"
    );
    // Key 6: its base record, and the delta's record in the new book.
    assert_eq!(
        hits[&key(6)].1,
        BTreeSet::from([("id:3".to_string(), 0), ("id:4".to_string(), 0)])
    );
    // Key 8, shipped by the delta, in both books.
    assert_eq!(
        hits[&key(8)].1,
        BTreeSet::from([("id:1".to_string(), 2), ("id:4".to_string(), 1)])
    );
    // Key 4 moved within id:2. The pair is not new, so the delta carries nothing for it:
    // the base's record keeps its old hint, which resolution re-anchors and compaction
    // refreshes.
    assert_eq!(hits[&key(4)].1, BTreeSet::from([("id:2".to_string(), 0)]));
    let previous = files::read_pointer(&dir, PREVIOUS).unwrap().unwrap();
    assert_eq!(previous.generation, 1);
}

#[test]
fn a_delta_applied_twice_or_out_of_date_changes_nothing() {
    let work = TempDir::new("set_twice");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let delta = release(&work, &v30(), Some(&v29()));
    install(&dir, &delta).unwrap();
    let again = install(&dir, &delta).unwrap();
    assert!(again.already_applied);
    assert_eq!((again.generation, again.library_version), (2, 30));

    let mut v28 = v29();
    v28.version = 28;
    let stale = release(&work, &v29(), Some(&v28));
    assert!(install(&dir, &stale).unwrap().already_applied);
    assert_eq!(SegmentSet::open(&dir).unwrap().generation(), 2);
}

#[test]
fn a_delta_with_a_gap_another_epoch_or_no_base_is_refused() {
    let work = TempDir::new("set_gap");
    let dir = work.join("vectors");
    let delta = release(&work, &v30(), Some(&v29()));
    match install(&dir, &delta) {
        Err(SemanticSearchError::Artifact(ArtifactError::DeltaDoesNotApply { field, .. })) => {
            assert_eq!(field, "delta.from_library_version")
        }
        other => panic!("a delta with no base must be refused, got {other:?}"),
    }

    install(&dir, &release(&work, &v29(), None)).unwrap();
    let mut v31 = v30();
    v31.version = 31;
    let mut v32 = v30();
    v32.version = 32;
    match install(&dir, &release(&work, &v32, Some(&v31))) {
        Err(SemanticSearchError::Artifact(ArtifactError::DeltaDoesNotApply { field, .. })) => {
            assert_eq!(field, "delta.from_library_version")
        }
        other => panic!("a gap must be refused, got {other:?}"),
    }

    // The same delta quantized in another epoch.
    let (path, json) = delta;
    let mut manifest: ReleaseManifest = serde_json::from_str(&json).unwrap();
    manifest.codec_params_sha256 = "e".repeat(64);
    manifest.package_digest = manifest.package().digest();
    match install(&dir, &(path, manifest.to_json())) {
        Err(SemanticSearchError::Artifact(ArtifactError::DeltaDoesNotApply { field, .. })) => {
            assert_eq!(field, "delta.codec_params")
        }
        other => panic!("another codec epoch must be refused, got {other:?}"),
    }
    assert_eq!(SegmentSet::open(&dir).unwrap().generation(), 1);
}

#[test]
fn a_release_that_is_not_the_published_one_or_not_its_segment_is_refused() {
    let work = TempDir::new("set_refusals");
    let dir = work.join("vectors");
    let (path, json) = release(&work, &v29(), None);

    let mut published = expectation();
    published.published_manifest_sha256 = Some("0".repeat(64));
    let refused = install_package(
        &dir,
        &InstallSource {
            segment: &path,
            manifest_json: &json,
        },
        &published,
        &CancellationToken::new(),
    );
    assert!(matches!(
        refused,
        Err(SemanticSearchError::Artifact(
            ArtifactError::UnexpectedArtifactDigest { .. }
        ))
    ));
    published.published_manifest_sha256 = Some(files::sha256_hex(json.as_bytes()));
    let mut wrong = published.clone();
    wrong.identity.text.line_text_version = 2;
    assert!(matches!(
        install_package(
            &dir,
            &InstallSource {
                segment: &path,
                manifest_json: &json
            },
            &wrong,
            &CancellationToken::new()
        ),
        Err(SemanticSearchError::Artifact(
            ArtifactError::IdentityMismatch { .. }
        ))
    ));

    // A manifest whose counts were edited no longer has its own package digest.
    let mut edited: ReleaseManifest = serde_json::from_str(&json).unwrap();
    edited.counts.extras += 1;
    assert!(matches!(
        install(&dir, &(path.clone(), edited.to_json())),
        Err(SemanticSearchError::Artifact(
            ArtifactError::ManifestDisagreesWithPayload { .. }
        ))
    ));

    // A segment damaged after its manifest was written.
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let damaged = work.join("damaged.oxv");
    std::fs::write(&damaged, bytes).unwrap();
    assert!(matches!(
        install(&dir, &(damaged, json.clone())),
        Err(SemanticSearchError::Artifact(
            ArtifactError::PayloadChecksumFailed { .. }
        ))
    ));

    // Nothing was installed by any of it, and the real one still installs.
    assert!(!dir.join("CURRENT").exists());
    install(&dir, &(path, json)).unwrap();
}

#[test]
fn a_crash_at_any_step_leaves_the_old_generation_or_the_new() {
    for step in [
        Step::Staged,
        Step::Moved,
        Step::GenerationWritten,
        Step::PreviousWritten,
        Step::CurrentFlipped,
    ] {
        let work = TempDir::new("set_crash");
        let dir = work.join("vectors");
        install(&dir, &release(&work, &v29(), None)).unwrap();
        let delta = release(&work, &v30(), Some(&v29()));

        CRASH_AT.with(|crash| crash.set(Some(step)));
        let crashed = install(&dir, &delta);
        CRASH_AT.with(|crash| crash.set(None));
        assert!(
            matches!(
                crashed,
                Err(SemanticSearchError::Artifact(
                    ArtifactError::InterruptedInstall { .. }
                ))
            ),
            "{step:?}: {crashed:?}"
        );

        // The next open sees one of the two generations, never a broken one.
        let set = SegmentSet::open(&dir).unwrap();
        let expected = if step == Step::CurrentFlipped { 2 } else { 1 };
        assert_eq!(set.generation(), expected, "{step:?}");
        assert!(!set.info().recovered_from_previous, "{step:?}");
        drop(set);
        assert!(
            !dir.join("staging").exists(),
            "{step:?}: recovery removes staging/"
        );

        // And the install, repeated, lands.
        let report = install(&dir, &delta).unwrap();
        assert_eq!(
            report.already_applied,
            step == Step::CurrentFlipped,
            "{step:?}"
        );
        let set = SegmentSet::open(&dir).unwrap();
        assert_eq!(
            (set.generation(), set.info().library_version),
            (2, 30),
            "{step:?}"
        );
        drop(set);
        let generations: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("gen-"))
            .collect();
        assert_eq!(generations.len(), 2, "{step:?}: {generations:?}");
    }
}

#[test]
fn a_generation_that_does_not_open_falls_back_to_the_previous_one() {
    let work = TempDir::new("set_fallback");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();

    // A derived file of the live generation, damaged.
    let current = files::read_pointer(&dir, CURRENT).unwrap().unwrap();
    let generation = dir.join(files::generation_dir(current.generation));
    let del = std::fs::read_dir(&generation)
        .unwrap()
        .flatten()
        .find(|entry| entry.file_name().to_string_lossy().ends_with(".del"))
        .unwrap()
        .path();
    let mut bytes = std::fs::read(&del).unwrap();
    bytes[20] ^= 1;
    std::fs::write(&del, bytes).unwrap();

    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!(set.generation(), 1);
    assert!(set.info().recovered_from_previous);
    assert_eq!(
        info(&dir).unwrap().unwrap().generation,
        2,
        "info reads, it does not check bitmaps"
    );
    drop(set);

    // Both broken: corrupt, not missing.
    std::fs::write(dir.join(files::generation_dir(1)).join("set.json"), b"{}").unwrap();
    assert!(matches!(
        SegmentSet::open(&dir),
        Err(SemanticSearchError::VectorStore(
            VectorStoreError::Corrupted { .. }
        ))
    ));
}

#[test]
fn garbage_goes_after_a_flip_and_an_open_reader_keeps_its_generation() {
    let work = TempDir::new("set_garbage");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let reader = SegmentSet::open(&dir).unwrap();
    let before = everything(&reader, 3);
    let base_id = reader.info().segments[0].id.clone();

    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    // A new base replaces the set; the old base stays while PREVIOUS names it.
    let mut v40 = v30();
    v40.version = 40;
    install(&dir, &release(&work, &v40, None)).unwrap();
    assert!(dir.join(format!("segments/{base_id}.oxv")).exists() || cfg!(windows));
    let mut v41 = v40.clone();
    v41.version = 41;
    v41.books.get_mut("id:3").unwrap().insert(9, 99);
    install(&dir, &release(&work, &v41, Some(&v40))).unwrap();

    let generations: BTreeSet<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("gen-"))
        .collect();
    assert_eq!(
        generations,
        BTreeSet::from(["gen-000003".to_string(), "gen-000004".to_string()])
    );
    if !cfg!(windows) {
        // On Windows the reader's mapping keeps the file until it is closed.
        assert!(!dir.join(format!("segments/{base_id}.oxv")).exists());
    }
    // The reader opened before all of it still answers from its own generation.
    assert_eq!(everything(&reader, 3), before);
    drop(reader);
    // And the next open, with nothing mapped, collects what was left.
    SegmentSet::open(&dir).unwrap();
    assert!(!dir.join(format!("segments/{base_id}.oxv")).exists());
}

#[test]
fn a_segment_left_in_incoming_is_taken_and_one_elsewhere_copied() {
    let work = TempDir::new("set_incoming");
    let dir = work.join("vectors");
    let (path, json) = release(&work, &v29(), None);
    std::fs::create_dir_all(incoming_dir(&dir)).unwrap();
    let incoming = incoming_dir(&dir).join("download.oxv");
    std::fs::copy(&path, &incoming).unwrap();
    install(&dir, &(incoming.clone(), json)).unwrap();
    assert!(!incoming.exists(), "taken into the set");
    assert!(path.exists());
}

#[test]
fn an_install_never_waits_behind_another_and_a_reader_never_waits_at_all() {
    let work = TempDir::new("set_lock");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let held = files::SetLock::take(&dir).unwrap();
    match install(&dir, &release(&work, &v30(), Some(&v29()))) {
        Err(SemanticSearchError::Artifact(ArtifactError::Io { source, .. })) => {
            assert_eq!(source.kind(), std::io::ErrorKind::WouldBlock)
        }
        other => panic!("a locked set must refuse an install, got {other:?}"),
    }
    assert_eq!(SegmentSet::open(&dir).unwrap().generation(), 1);
    drop(held);
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
}

#[test]
fn a_cancelled_install_leaves_the_set_as_it_was() {
    let work = TempDir::new("set_cancel");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let (path, json) = release(&work, &v30(), Some(&v29()));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let cancelled = install_package(
        &dir,
        &InstallSource {
            segment: &path,
            manifest_json: &json,
        },
        &expectation(),
        &cancel,
    );
    assert!(matches!(cancelled, Err(SemanticSearchError::Cancelled)));
    assert_eq!(SegmentSet::open(&dir).unwrap().generation(), 1);
}

#[test]
fn a_scrub_marks_a_damaged_segment_and_every_open_after_refuses_it() {
    let work = TempDir::new("set_scrub");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let report = scrub(&dir, &CancellationToken::new()).unwrap();
    assert_eq!((report.generation, report.segments), (1, 1));
    assert!(report.bytes_checked > 0);

    let id = info(&dir).unwrap().unwrap().segments[0].id.clone();
    let segment = dir.join(format!("segments/{id}.oxv"));
    let mut bytes = std::fs::read(&segment).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&segment, bytes).unwrap();

    assert!(matches!(
        scrub(&dir, &CancellationToken::new()),
        Err(SemanticSearchError::VectorStore(
            VectorStoreError::Corrupted { .. }
        ))
    ));
    assert!(dir.join(format!("segments/{id}.corrupt")).exists());
    match SegmentSet::open(&dir) {
        Err(SemanticSearchError::VectorStore(VectorStoreError::Corrupted { reason })) => {
            assert!(reason.contains("scrub"), "{reason}")
        }
        other => panic!("a scrubbed-out segment must not open, got {other:?}"),
    }
}

#[test]
fn nothing_installed_is_missing_and_a_v1_artifact_is_another_store() {
    let work = TempDir::new("set_missing");
    let dir = work.join("vectors");
    std::fs::create_dir_all(&dir).unwrap();
    assert_eq!(info(&dir).unwrap(), None);
    assert!(matches!(
        SegmentSet::open(&dir),
        Err(SemanticSearchError::Artifact(
            ArtifactError::MetadataUnusable { .. }
        ))
    ));

    std::fs::write(
        dir.join("manifest.json"),
        r#"{"metadata_version":2,"identity":{"store":{"backend_id":"zevc-persistent-v1"}}}"#,
    )
    .unwrap();
    for result in [SegmentSet::open(&dir).map(|_| ()), info(&dir).map(|_| ())] {
        match result {
            Err(SemanticSearchError::Artifact(ArtifactError::IdentityMismatch { mismatches })) => {
                assert_eq!(mismatches[0].field, IdentityField::StoreBackendId);
                assert_eq!(mismatches[0].artifact, "zevc-persistent-v1");
            }
            other => panic!("a v1 artifact must be refused by its store, got {other:?}"),
        }
    }
}

/// What compaction must preserve: every live key with its score, and every live record.
#[test]
fn compaction_merges_a_set_into_one_segment_that_answers_alike() {
    let work = TempDir::new("set_compact");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let before = {
        let set = SegmentSet::open(&dir).unwrap();
        (everything(&set, 4), everything(&set, 5), set.info().clone())
    };

    let not_needed = compact(
        &dir,
        &CompactionPolicy {
            max_dead_ratio: 0.5,
            max_delta_ratio: 10.0,
            ..CompactionPolicy::default()
        },
        None,
        &CancellationToken::new(),
    )
    .unwrap();
    assert!(!not_needed.compacted);

    let report = compact(
        &dir,
        &CompactionPolicy {
            force: true,
            ..CompactionPolicy::default()
        },
        None,
        &CancellationToken::new(),
    )
    .unwrap();
    assert!(report.compacted);
    assert_eq!(
        report.slots_before,
        before.2.slots_live + before.2.slots_dead
    );
    assert_eq!(report.slots_after, before.2.slots_live);
    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!(set.info().segments.len(), 1);
    assert_eq!(set.info().segments[0].kind, PackageKind::Compacted);
    assert_eq!((set.info().library_version, set.info().slots_dead), (30, 0));
    assert_eq!(everything(&set, 4), before.0);
    assert_eq!(everything(&set, 5), before.1);
}

/// Compacting a base and its delta writes the segment a base of the newer library would
/// be: the same slots, keys, hints, records and vector bytes.
#[test]
fn compacting_base_and_delta_gives_the_base_of_the_newer_library() {
    let work = TempDir::new("set_equivalence");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    // The delta keeps key 4's base record at hint 3 as well as its new one at 4: one record
    // per book and key keeps the newest segment's, which is the newer library's own.
    let live = Fixed {
        version: 30,
        books: v30()
            .books
            .iter()
            .map(|(book, lines)| {
                (
                    book.clone(),
                    lines
                        .iter()
                        .map(|(hint, k)| (*hint, key(*k).column_value()))
                        .collect(),
                )
            })
            .collect(),
    };
    compact(
        &dir,
        &CompactionPolicy {
            force: true,
            ..CompactionPolicy::default()
        },
        Some(&live),
        &CancellationToken::new(),
    )
    .unwrap();
    let compacted = SegmentSet::open(&dir).unwrap();

    let direct_dir = work.join("direct");
    install(&direct_dir, &release(&work, &v30(), None)).unwrap();
    let direct = SegmentSet::open(&direct_dir).unwrap();

    let (a, b) = (&compacted.segments()[0], &direct.segments()[0]);
    assert_eq!(a.slot_count(), b.slot_count());
    assert_eq!(a.counts(), b.counts());
    for slot in 0..a.slot_count() {
        assert_eq!(a.key(slot), b.key(slot), "slot {slot}");
        assert_eq!(a.hint(slot), b.hint(slot), "slot {slot}");
        assert_eq!(a.vector(slot), b.vector(slot), "slot {slot}");
    }
    for extra in 0..a.extra_count() {
        assert_eq!(a.extra(extra), b.extra(extra));
    }
    let names = |set: &SegmentSet| -> Vec<String> {
        set.segments()[0]
            .books()
            .iter()
            .map(|book| book.name.to_string())
            .collect()
    };
    assert_eq!(names(&compacted), names(&direct));
}

/// A key index of a fixed library, for compaction to re-anchor against.
struct Fixed {
    version: u32,
    books: BTreeMap<String, Vec<(u32, u64)>>,
}

impl LiveKeySource for Fixed {
    fn library_version(&self) -> u32 {
        self.version
    }

    fn book_keys(&self, book: &str, out: &mut Vec<(u32, u64)>) -> Result<bool, ResolveError> {
        match self.books.get(book) {
            Some(lines) => {
                out.extend_from_slice(lines);
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

#[test]
fn compaction_re_anchors_records_on_the_live_index() {
    let work = TempDir::new("set_refresh");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    // The live index at the set's version: in id:1 key 2 moved from line 1 to line 6 and key
    // 3 is gone; id:2 is not in the index at all; id:3 is as it was.
    let live = Fixed {
        version: 29,
        books: BTreeMap::from([
            (
                "id:1".to_string(),
                vec![
                    (0, key(1).column_value()),
                    (6, key(2).column_value()),
                    (7, 0),
                ],
            ),
            (
                "id:3".to_string(),
                vec![(0, key(6).column_value()), (1, key(7).column_value())],
            ),
        ]),
    };
    let report = compact(
        &dir,
        &CompactionPolicy {
            force: true,
            ..CompactionPolicy::default()
        },
        Some(&live),
        &CancellationToken::new(),
    )
    .unwrap();
    assert_eq!((report.records_pruned, report.hints_refreshed), (1, 1));
    let set = SegmentSet::open(&dir).unwrap();
    let hits = everything(&set, 6);
    assert!(
        !hits.contains_key(&key(3)),
        "its only record was pruned, so its vector went"
    );
    assert_eq!(
        hits[&key(2)].1,
        BTreeSet::from([("id:1".to_string(), 6), ("id:2".to_string(), 1)]),
        "re-anchored in id:1, kept as it was in id:2, which the index does not hold"
    );
    drop(set);

    // Against an index of another version, nothing is re-anchored.
    let other = Fixed {
        version: 30,
        books: BTreeMap::new(),
    };
    let report = compact(
        &dir,
        &CompactionPolicy {
            force: true,
            ..CompactionPolicy::default()
        },
        Some(&other),
        &CancellationToken::new(),
    )
    .unwrap();
    assert_eq!((report.records_pruned, report.hints_refreshed), (0, 0));
}

#[test]
fn the_policy_names_its_reason() {
    let work = TempDir::new("set_policy");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let info = info(&dir).unwrap().unwrap();
    let policy = CompactionPolicy::default();
    assert!(
        policy.wants(&info).unwrap().contains("deltas"),
        "a large delta against a tiny base"
    );
    let lenient = CompactionPolicy {
        max_delta_ratio: 100.0,
        max_dead_ratio: 0.5,
        ..policy.clone()
    };
    assert_eq!(lenient.wants(&info), None);
    let few = CompactionPolicy {
        max_segments: 1,
        ..lenient.clone()
    };
    assert!(few.wants(&info).unwrap().contains("segments"));
    let dead = CompactionPolicy {
        max_dead_ratio: 0.01,
        ..lenient
    };
    assert!(dead.wants(&info).unwrap().contains("dead"));
}

#[test]
fn a_filtered_scan_of_a_set_reaches_foreign_records() {
    let work = TempDir::new("set_filtered");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let set = SegmentSet::open(&dir).unwrap();
    let books: BookSet = ["id:4"].into_iter().collect();
    let hits = set
        .scan(
            &Random(7).unit(DIM),
            &ScanRequest {
                top_k: 100,
                books: Some(&books),
                threads: 1,
            },
            &CancellationToken::new(),
        )
        .unwrap();
    let keys: BTreeSet<ChunkKey> = hits.iter().map(|hit| hit.key).collect();
    // id:4 holds key 6 through a foreign record into the base, and key 8 as a slot.
    assert_eq!(keys, BTreeSet::from([key(6), key(8)]));
    assert!(hits.iter().all(|hit| &*hit.records[0].book == "id:4"));
}

/// v30 and one more line, a new key: what installs after v30 in the tests below.
fn v31() -> Library {
    let mut v31 = v30();
    v31.version = 31;
    v31.books.get_mut("id:3").unwrap().insert(5, 31);
    v31
}

/// How a pointer file can stop reading: bytes that are no pointer, and — as the review
/// broke it — a directory where the file should be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Broken {
    Garbage,
    Directory,
}

fn break_current(dir: &Path, how: Broken) {
    let current = dir.join(CURRENT);
    match std::fs::symlink_metadata(&current) {
        Ok(metadata) if metadata.is_dir() => std::fs::remove_dir_all(&current).unwrap(),
        Ok(_) => std::fs::remove_file(&current).unwrap(),
        Err(_) => {}
    }
    match how {
        Broken::Garbage => std::fs::write(&current, b"damaged").unwrap(),
        Broken::Directory => std::fs::create_dir(&current).unwrap(),
    }
}

/// `PREVIOUS` and every file of every generation, by path: what "as it was" holds a set to.
fn on_disk(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name == PREVIOUS {
            files.insert(name, std::fs::read(entry.path()).unwrap());
        } else if name.starts_with("gen-") {
            for file in std::fs::read_dir(entry.path()).unwrap().flatten() {
                let path = format!("{name}/{}", file.file_name().to_string_lossy());
                files.insert(path, std::fs::read(file.path()).unwrap());
            }
        }
    }
    files
}

fn previous_generation(dir: &Path) -> u64 {
    files::read_pointer(dir, PREVIOUS)
        .unwrap()
        .expect("a PREVIOUS")
        .generation
}

/// The review's sequence: two generations, `CURRENT` broken, the set answering from
/// `PREVIOUS` — and then an install. Numbered from an unreadable `CURRENT` as from nothing,
/// it removed the generation `PREVIOUS` names to write its own there, and copied the broken
/// `CURRENT` over `PREVIOUS` (or failed on the directory); the fallback was gone either way.
#[test]
fn an_install_over_a_broken_current_keeps_the_generation_previous_names() {
    for how in [Broken::Garbage, Broken::Directory] {
        let work = TempDir::new("set_broken_install");
        let dir = work.join("vectors");
        install(&dir, &release(&work, &v29(), None)).unwrap();
        install(&dir, &release(&work, &v30(), None)).unwrap();
        break_current(&dir, how);
        let fallback = SegmentSet::open(&dir).unwrap();
        assert_eq!(
            (fallback.generation(), fallback.info().library_version),
            (1, 29),
            "{how:?}"
        );
        assert!(fallback.info().recovered_from_previous, "{how:?}");
        let answers = everything(&fallback, 8);
        drop(fallback);

        let report = install(&dir, &release(&work, &v31(), None)).unwrap();
        assert_eq!(
            report.generation, 3,
            "{how:?}: past both generations on disk"
        );
        let set = SegmentSet::open(&dir).unwrap();
        assert_eq!(
            (set.generation(), set.info().library_version),
            (3, 31),
            "{how:?}"
        );
        assert!(!set.info().recovered_from_previous, "{how:?}");
        drop(set);
        assert_eq!(
            previous_generation(&dir),
            1,
            "{how:?}: PREVIOUS names the generation that opened"
        );

        // And that generation is the fallback it was.
        break_current(&dir, how);
        let fallback = SegmentSet::open(&dir).unwrap();
        assert_eq!(fallback.generation(), 1, "{how:?}");
        assert_eq!(everything(&fallback, 8), answers, "{how:?}");
    }
}

/// Every way an install can fail — refused, cancelled, cut off at any step — over a
/// `CURRENT` that does not read leaves the set as it found it: the same fallback with the
/// same answers, and `PREVIOUS` and every generation's files byte for byte.
#[test]
fn a_failed_install_over_a_broken_current_leaves_the_set_as_it_was() {
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Failure {
        Refused,
        Cancelled,
        Crash(Step),
    }
    let failures = [
        Failure::Refused,
        Failure::Cancelled,
        Failure::Crash(Step::Staged),
        Failure::Crash(Step::Moved),
        Failure::Crash(Step::GenerationWritten),
        Failure::Crash(Step::PreviousWritten),
        Failure::Crash(Step::CurrentFlipped),
    ];
    for how in [Broken::Garbage, Broken::Directory] {
        for failure in failures {
            let work = TempDir::new("set_broken_failure");
            let dir = work.join("vectors");
            install(&dir, &release(&work, &v29(), None)).unwrap();
            install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
            break_current(&dir, how);
            let answers = everything(&SegmentSet::open(&dir).unwrap(), 9);
            let before = on_disk(&dir);

            let (path, json) = release(&work, &v31(), None);
            let mut expect = expectation();
            let cancel = CancellationToken::new();
            match failure {
                Failure::Refused => expect.published_manifest_sha256 = Some("0".repeat(64)),
                Failure::Cancelled => cancel.cancel(),
                Failure::Crash(step) => CRASH_AT.with(|crash| crash.set(Some(step))),
            }
            let result = install_package(
                &dir,
                &InstallSource {
                    segment: &path,
                    manifest_json: &json,
                },
                &expect,
                &cancel,
            );
            CRASH_AT.with(|crash| crash.set(None));
            assert!(result.is_err(), "{how:?} {failure:?}: {result:?}");

            let set = SegmentSet::open(&dir)
                .unwrap_or_else(|error| panic!("{how:?} {failure:?}: the set must open: {error}"));
            if failure == Failure::Crash(Step::CurrentFlipped) {
                // The flip is the install: the new generation, with the old fallback behind.
                assert_eq!((set.generation(), set.info().library_version), (3, 31));
                drop(set);
                assert_eq!(previous_generation(&dir), 1, "{how:?}");
                break_current(&dir, how);
                let set = SegmentSet::open(&dir).unwrap();
                assert_eq!(set.generation(), 1, "{how:?}");
                assert_eq!(everything(&set, 9), answers, "{how:?}");
                continue;
            }
            assert_eq!(set.generation(), 1, "{how:?} {failure:?}");
            assert!(set.info().recovered_from_previous, "{how:?} {failure:?}");
            assert_eq!(everything(&set, 9), answers, "{how:?} {failure:?}");
            drop(set);
            let after = on_disk(&dir);
            for (file, bytes) in &before {
                assert_eq!(
                    after.get(file),
                    Some(bytes),
                    "{how:?} {failure:?}: {file} changed"
                );
            }
        }
    }
}

/// A generation only an unreadable pointer could name — or one a crash left where no
/// collection could prove it dead — is never written into: the next is numbered past them
/// all.
#[test]
fn a_new_generation_is_numbered_past_every_generation_on_disk() {
    let work = TempDir::new("set_numbering");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let stray = dir.join(files::generation_dir(9));
    std::fs::create_dir(&stray).unwrap();
    std::fs::write(stray.join("set.json"), b"what a crash left").unwrap();
    break_current(&dir, Broken::Garbage);

    let report = install(&dir, &release(&work, &v31(), None)).unwrap();
    assert_eq!(report.generation, 10);
    assert_eq!(previous_generation(&dir), 1);
    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!((set.generation(), set.info().library_version), (10, 31));
}

/// `CURRENT` names a generation that does not open, and the set answers from `PREVIOUS`'s.
/// An install or a compaction builds on that one; it must neither take the number of the one
/// `CURRENT` names — removing it while it is named — nor move `PREVIOUS` off the one that
/// opens.
#[test]
fn work_on_a_fallback_never_reuses_the_generation_current_names() {
    for compaction in [false, true] {
        let work = TempDir::new("set_fallback_work");
        let dir = work.join("vectors");
        install(&dir, &release(&work, &v29(), None)).unwrap();
        let delta = release(&work, &v30(), Some(&v29()));
        install(&dir, &delta).unwrap();
        let named = dir.join(files::generation_dir(2));
        let del = std::fs::read_dir(&named)
            .unwrap()
            .flatten()
            .find(|entry| entry.file_name().to_string_lossy().ends_with(".del"))
            .unwrap()
            .path();
        let mut bytes = std::fs::read(&del).unwrap();
        bytes[20] ^= 1;
        std::fs::write(&del, bytes).unwrap();
        let fallback = SegmentSet::open(&dir).unwrap();
        assert!(fallback.info().recovered_from_previous);
        assert_eq!(fallback.generation(), 1);
        drop(fallback);
        let current_names: BTreeMap<String, Vec<u8>> = on_disk(&dir)
            .into_iter()
            .filter(|(file, _)| file.starts_with("gen-000002/"))
            .collect();

        let work_once = || {
            if compaction {
                compact(
                    &dir,
                    &CompactionPolicy {
                        force: true,
                        ..CompactionPolicy::default()
                    },
                    None,
                    &CancellationToken::new(),
                )
                .map(|report| report.generation)
            } else {
                install(&dir, &delta).map(|report| report.generation)
            }
        };

        // Cut off once its generation is written: what CURRENT names is untouched.
        CRASH_AT.with(|crash| crash.set(Some(Step::GenerationWritten)));
        let crashed = work_once();
        CRASH_AT.with(|crash| crash.set(None));
        assert!(crashed.is_err(), "compaction {compaction}: {crashed:?}");
        let after = on_disk(&dir);
        for (file, bytes) in &current_names {
            assert_eq!(
                after.get(file),
                Some(bytes),
                "compaction {compaction}: {file}"
            );
        }
        assert_eq!(SegmentSet::open(&dir).unwrap().generation(), 1);

        // And done: past it, with PREVIOUS still on the generation it was built from — v29,
        // which a compaction merges as it is and the delta takes to v30.
        assert_eq!(work_once().unwrap(), 3, "compaction {compaction}");
        assert_eq!(previous_generation(&dir), 1, "compaction {compaction}");
        let set = SegmentSet::open(&dir).unwrap();
        let version = if compaction { 29 } else { 30 };
        assert_eq!((set.generation(), set.info().library_version), (3, version));
        assert!(!set.info().recovered_from_previous);
    }
}

/// With `CURRENT` unreadable the set is `PREVIOUS`'s generation, to `open` and to `info`
/// alike; a delta from that generation's version applies to it.
#[test]
fn a_delta_over_an_unreadable_current_applies_to_the_generation_that_opens() {
    let work = TempDir::new("set_broken_delta");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let delta = release(&work, &v30(), Some(&v29()));
    install(&dir, &delta).unwrap();
    break_current(&dir, Broken::Garbage);
    assert_eq!(info(&dir).unwrap().unwrap().library_version, 29);

    let report = install(&dir, &delta).unwrap();
    assert!(!report.already_applied);
    assert_eq!((report.generation, report.library_version), (3, 30));
    assert_eq!(previous_generation(&dir), 1);
    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!((set.generation(), set.info().library_version), (3, 30));
}

#[test]
fn a_full_disk_is_insufficient_space() {
    let full = std::io::Error::from(std::io::ErrorKind::StorageFull);
    match install::full_or_io(full, Path::new("x"), 100, 40) {
        SemanticSearchError::Artifact(ArtifactError::InsufficientSpace { needed, available }) => {
            assert_eq!((needed, available), (100, 40))
        }
        other => panic!("{other:?}"),
    }
    let other = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
    assert!(matches!(
        install::full_or_io(other, Path::new("x"), 1, 0),
        SemanticSearchError::Artifact(ArtifactError::Io { .. })
    ));
}
