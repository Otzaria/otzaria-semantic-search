//! Segment sets end to end: install, apply, crash, recover, collect, compact, scrub.

use super::install::{Step, CRASH_AT};
use super::*;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::codec::Codec;
use crate::semantic::oxv::scan::ScanRequest;
use crate::semantic::oxv::testing::{key, spec, write_segment, Random, TempDir, TestBook};
use crate::semantic::resolve::{BookSet, LiveKeySource, ResolveError, SlotRef};
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
    vector_in(key, 0)
}

/// The vector a key embeds as under `embedding`: 0 is [`vector_of`]'s; another stands for a
/// worker or a passage package that embeds the same texts a little differently.
fn vector_in(key: u64, embedding: u64) -> Vec<f32> {
    Random(key.wrapping_mul(0x9E37_79B9) ^ 0xABCD ^ embedding.wrapping_mul(0x5851_F42D)).unit(DIM)
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
fn books_for(
    records: &BTreeMap<(String, u64), u32>,
    shipped: &BTreeSet<u64>,
    embedding: u64,
) -> Vec<TestBook> {
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
                        book.primary
                            .push((self::key(key), hint, vector_in(key, embedding)));
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
    release_embedded(dir, library, previous, 0)
}

/// [`release`], its vectors embedded under `embedding` ([`vector_in`]): another embedding of
/// the same release is the same keys in the same slots — the same segment id — and other
/// bytes, as a version published again would be.
fn release_embedded(
    dir: &TempDir,
    library: &Library,
    previous: Option<&Library>,
    embedding: u64,
) -> (PathBuf, String) {
    let pairs = library.pairs();
    let (kind, from, books, tombstones) = match previous {
        None => (
            PackageKind::Base,
            0,
            books_for(&pairs, &library.keys(), embedding),
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
                books_for(&records, &new_keys, embedding),
                gone,
            )
        }
    };
    let mut spec = spec(kind, from, library.version);
    spec.identity_digest = identity().identity_digest();
    let path = match embedding {
        0 => dir.join(&format!("{kind}-{}.oxv", library.version)),
        _ => dir.join(&format!(
            "{kind}-{}-embedding-{embedding}.oxv",
            library.version
        )),
    };
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

/// Which slots of a generation are live is the set's to say: a tombstoned one is not, nor
/// one past a segment's end or in no segment. A scan weighs, besides the books it admits,
/// the live slots it is handed — at the score a full scan gives them — and passes over the
/// dead ones.
#[test]
fn a_set_weighs_the_live_slots_it_is_handed_besides_the_books_it_admits() {
    let work = TempDir::new("set_scan_with");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!(set.segments().len(), 2);
    let slot_of = |seg: u16, wanted: u64| -> SlotRef {
        let segment = &set.segments()[seg as usize];
        let slot = (0..segment.slot_count())
            .find(|slot| segment.key(*slot) == key(wanted))
            .expect("the segment holds the key");
        SlotRef {
            seg,
            slot,
            key: key(wanted),
        }
    };
    // Key 3 was tombstoned by the delta; key 1 is live in the base, key 8 in the delta.
    let (gone, one, eight) = (slot_of(0, 3), slot_of(0, 1), slot_of(1, 8));
    assert!(!set.is_live(gone.seg, gone.slot));
    assert!(set.is_live(one.seg, one.slot) && set.is_live(eight.seg, eight.slot));
    assert!(!set.is_live(0, set.segments()[0].slot_count()));
    assert!(!set.is_live(2, 0));

    let query = Random(11).unit(DIM);
    let scan = |books: Option<&BookSet>, also: &[SlotRef]| {
        set.scan_with(
            &query,
            &ScanRequest {
                top_k: 100,
                books,
                threads: 1,
            },
            also,
            &CancellationToken::new(),
        )
        .unwrap()
    };
    let everything = scan(None, &[]);
    let books: BookSet = ["id:3"].into_iter().collect();
    let filtered = scan(Some(&books), &[]);
    let merged = scan(Some(&books), &[gone, one]);
    assert_eq!(
        merged
            .iter()
            .filter(|hit| hit.key != key(1))
            .cloned()
            .collect::<Vec<_>>(),
        filtered,
        "the admitted books' hits are as the filtered scan's, and the tombstoned key is not one"
    );
    let weighed = merged.iter().find(|hit| hit.key == key(1)).unwrap();
    let full = everything.iter().find(|hit| hit.key == key(1)).unwrap();
    assert_eq!(weighed.score.to_bits(), full.score.to_bits());
    assert_eq!(weighed.records, full.records);
    assert_eq!(
        scan(Some(&books), &[]),
        set.scan(
            &query,
            &ScanRequest {
                top_k: 100,
                books: Some(&books),
                threads: 1
            },
            &CancellationToken::new()
        )
        .unwrap()
    );
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

/// What `work` did to make its writes durable, in order.
fn journal_of(work: impl FnOnce()) -> Vec<files::Durable> {
    files::JOURNAL.with(|journal| journal.borrow_mut().clear());
    work();
    files::JOURNAL.with(|journal| journal.borrow_mut().drain(..).collect())
}

/// The order one publish into `dir` must show: the segment's bytes flushed where they are,
/// then renamed into `segments/`; the new generation's `set.json` renamed in; `CURRENT`
/// renamed last. And on Unix, where a directory can be flushed, `segments/` flushed after the
/// segment lands, the set's directory — the generation's own entry — after the generation is
/// written, and again after the flip, all before what follows; on Windows, which flushes no
/// directory and keeps a rename's order itself, no directory flush is claimed at all.
fn assert_published_in_order(steps: &[files::Durable], dir: &Path, what: &str) {
    use files::Durable;
    let renamed_to = |to: &Path| {
        steps
            .iter()
            .position(|step| matches!(step, Durable::Renamed(_, path) if path == to))
    };
    let flip =
        renamed_to(&dir.join(CURRENT)).unwrap_or_else(|| panic!("{what}: no flip in {steps:#?}"));
    let segments = dir.join(SEGMENTS_DIR);
    let placed = steps
        .iter()
        .position(|step| {
            matches!(step, Durable::Renamed(_, path)
                if path.parent() == Some(segments.as_path())
                    && path.extension().is_some_and(|extension| extension == "oxv"))
        })
        .unwrap_or_else(|| panic!("{what}: no segment placed in {steps:#?}"));
    let Durable::Renamed(source, _) = &steps[placed] else {
        unreachable!("a placement is a rename")
    };
    assert!(
        steps[..placed].contains(&Durable::File(source.clone())),
        "{what}: the segment's bytes are flushed where they are before they are renamed into \
         place: {steps:#?}"
    );
    let generation = steps[..flip]
        .iter()
        .rposition(|step| {
            matches!(step, Durable::Renamed(_, path)
                if path.file_name().is_some_and(|name| name == files::SET_FILE))
        })
        .unwrap_or_else(|| panic!("{what}: no generation written in {steps:#?}"));
    assert!(placed < generation, "{what}: {steps:#?}");
    if cfg!(unix) {
        assert!(
            steps[placed..flip].contains(&Durable::Dir(segments.clone())),
            "{what}: segments/ is flushed after the segment lands and before the flip: {steps:#?}"
        );
        assert!(
            steps[generation..flip].contains(&Durable::Dir(dir.to_path_buf())),
            "{what}: the set's directory — the new generation's entry — is flushed before the \
             flip: {steps:#?}"
        );
        assert_eq!(
            steps.get(flip + 1),
            Some(&Durable::Dir(dir.to_path_buf())),
            "{what}: and so is the flip"
        );
    } else {
        assert!(
            !steps.iter().any(|step| matches!(step, Durable::Dir(_))),
            "{what}: no directory flush is claimed where none happens: {steps:#?}"
        );
    }
}

/// Every publish writes, flushes the file, renames it into place and flushes the directory,
/// and only then flips the pointer that names it: a base copied in, a delta taken from
/// `incoming/` — a file the set did not write, so not yet flushed — and a compaction.
#[test]
fn every_publish_is_flushed_before_the_flip_that_names_it() {
    let work = TempDir::new("set_durable");
    let dir = work.join("vectors");
    let steps = journal_of(|| {
        install(&dir, &release(&work, &v29(), None)).unwrap();
    });
    assert_published_in_order(&steps, &dir, "a base, copied");
    let flushed = journal_of(|| files::sync_set_dir(&dir).unwrap());
    match cfg!(unix) {
        true => assert_eq!(flushed, [files::Durable::Dir(dir.clone())]),
        false => assert_eq!(flushed, []),
    }

    let (path, json) = release(&work, &v30(), Some(&v29()));
    std::fs::create_dir_all(incoming_dir(&dir)).unwrap();
    let incoming = incoming_dir(&dir).join("download.oxv");
    std::fs::copy(&path, &incoming).unwrap();
    let steps = journal_of(|| {
        install(&dir, &(incoming, json)).unwrap();
    });
    assert_published_in_order(&steps, &dir, "a delta, taken from incoming/");

    let steps = journal_of(|| {
        compact(
            &dir,
            &CompactionPolicy {
                force: true,
                ..CompactionPolicy::default()
            },
            None,
            &CancellationToken::new(),
        )
        .unwrap();
    });
    assert_published_in_order(&steps, &dir, "a compaction");
}

fn damage_last_byte(path: &Path) -> Vec<u8> {
    let mut bytes = std::fs::read(path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(path, &bytes).unwrap();
    bytes
}

fn is_corrupt<T>(result: Result<T, SemanticSearchError>) -> bool {
    matches!(
        result,
        Err(SemanticSearchError::VectorStore(
            VectorStoreError::Corrupted { .. }
        ))
    )
}

/// The review's sequence: a scrub condemns a damaged segment, and installing the release
/// again — the repair the verdict asks for — succeeded while the verdict, written for the old
/// bytes, stayed and refused the new ones. Open, info and scrub agree before the repair and
/// after it.
#[test]
fn a_reinstall_clears_the_verdict_on_the_bytes_it_replaced() {
    let work = TempDir::new("set_verdict");
    let dir = work.join("vectors");
    let base = release(&work, &v29(), None);
    install(&dir, &base).unwrap();
    let id = info(&dir).unwrap().unwrap().segments[0].id.clone();
    let segment = dir.join(format!("segments/{id}.oxv"));
    let damaged = damage_last_byte(&segment);
    assert!(is_corrupt(scrub(&dir, &CancellationToken::new())));

    // The verdict names the bytes it condemned.
    let marker = dir.join(format!("segments/{id}.corrupt"));
    let verdict: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&marker).unwrap()).expect("a verdict is JSON");
    assert_eq!(verdict["sha256"], files::sha256_hex(&damaged));

    // Nothing opens, and open, info and scrub all say so.
    assert!(is_corrupt(SegmentSet::open(&dir)));
    assert!(is_corrupt(info(&dir)));
    assert!(is_corrupt(scrub(&dir, &CancellationToken::new())));

    // The repair: the same release again.
    install(&dir, &base).unwrap();
    assert!(
        !marker.exists(),
        "the verdict was on bytes the file no longer holds"
    );
    let manifest: ReleaseManifest = serde_json::from_str(&base.1).unwrap();
    assert_eq!(
        files::sha256_hex(&std::fs::read(&segment).unwrap()),
        manifest.segment.sha256
    );
    let generation = SegmentSet::open(&dir).unwrap().generation();
    let described = info(&dir).unwrap().unwrap();
    assert_eq!(described.generation, generation);
    assert!(!described.recovered_from_previous);
    assert_eq!(
        scrub(&dir, &CancellationToken::new()).unwrap().generation,
        generation
    );
}

/// A verdict on a delta condemns the generation that holds it, and every reader falls back
/// alike: open and info to `PREVIOUS`'s generation, and a scrub checks that one. The delta
/// installed again clears it.
#[test]
fn a_condemned_generation_falls_back_alike_for_open_info_and_scrub() {
    let work = TempDir::new("set_verdict_fallback");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let delta = release(&work, &v30(), Some(&v29()));
    install(&dir, &delta).unwrap();
    let id = info(&dir).unwrap().unwrap().segments[1].id.clone();
    damage_last_byte(&dir.join(format!("segments/{id}.oxv")));
    assert!(is_corrupt(scrub(&dir, &CancellationToken::new())));
    assert!(dir.join(format!("segments/{id}.corrupt")).exists());

    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!(set.generation(), 1);
    assert!(set.info().recovered_from_previous);
    drop(set);
    let described = info(&dir).unwrap().unwrap();
    assert_eq!(
        (described.generation, described.recovered_from_previous),
        (1, true)
    );
    assert_eq!(
        scrub(&dir, &CancellationToken::new()).unwrap().generation,
        1
    );

    // The set stands at v29 for every reader, and the delta applies there again.
    let report = install(&dir, &delta).unwrap();
    assert_eq!((report.generation, report.library_version), (3, 30));
    assert!(!dir.join(format!("segments/{id}.corrupt")).exists());
    assert_eq!(SegmentSet::open(&dir).unwrap().generation(), 3);
    assert_eq!(info(&dir).unwrap().unwrap().generation, 3);
    assert_eq!(
        scrub(&dir, &CancellationToken::new()).unwrap().generation,
        3
    );
}

/// A scrub that read a damaged file while an install replaced it would leave a verdict on
/// bytes the file no longer holds; it withdraws it, so the install's bytes are never
/// condemned. One on the bytes the file holds stands — damaged bytes, which no generation
/// names — and so does a marker that names no bytes, as every marker before verdicts did. One
/// on exactly the bytes the generation names condemns nothing: those are the bytes it serves.
#[test]
fn a_verdict_stands_only_on_the_bytes_the_file_holds() {
    let work = TempDir::new("set_verdict_race");
    let dir = work.join("vectors");
    let base = release(&work, &v29(), None);
    install(&dir, &base).unwrap();
    let id = info(&dir).unwrap().unwrap().segments[0].id.clone();
    let file = dir.join(format!("segments/{id}.oxv"));
    let marker = dir.join(format!("segments/{id}.corrupt"));

    assert!(!files::condemn(
        &dir,
        &id,
        &"0".repeat(64),
        "a block failed",
        &CancellationToken::new()
    )
    .unwrap());
    assert!(!marker.exists());
    SegmentSet::open(&dir).unwrap();

    let sound = std::fs::read(&file).unwrap();
    let damaged = damage_last_byte(&file);
    assert!(files::condemn(
        &dir,
        &id,
        &files::sha256_hex(&damaged),
        "a block failed",
        &CancellationToken::new()
    )
    .unwrap());
    assert!(is_corrupt(SegmentSet::open(&dir)));
    assert!(is_corrupt(info(&dir)));

    std::fs::write(&file, &sound).unwrap();
    assert!(files::condemn(
        &dir,
        &id,
        &files::sha256_hex(&sound),
        "a block failed",
        &CancellationToken::new()
    )
    .unwrap());
    SegmentSet::open(&dir).unwrap();

    std::fs::write(&marker, b"a block failed").unwrap();
    assert!(is_corrupt(SegmentSet::open(&dir)));
    assert!(is_corrupt(info(&dir)));
    install(&dir, &base).unwrap();
    assert!(!marker.exists());
    SegmentSet::open(&dir).unwrap();
}

/// v29's live index with key 2 of id:1 at `line` instead of line 1.
fn v29_with_key_2_at(line: u32) -> Fixed {
    let mut books: BTreeMap<String, Vec<(u32, u64)>> = v29()
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
        .collect();
    for (ordinal, value) in books.get_mut("id:1").unwrap() {
        if *value == key(2).column_value() {
            *ordinal = line;
        }
    }
    Fixed { version: 29, books }
}

/// The review's sequence: two compactions that re-anchor key 2 differently are two
/// different segments. With an id over the keys alone they shared one file name, the second
/// replaced the first's file, and falling back to `PREVIOUS` served the second's records.
#[test]
fn two_compactions_that_differ_are_two_segments_and_previous_keeps_its_own() {
    let work = TempDir::new("set_compaction_ids");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let force = CompactionPolicy {
        force: true,
        ..CompactionPolicy::default()
    };
    compact(
        &dir,
        &force,
        Some(&v29_with_key_2_at(6)),
        &CancellationToken::new(),
    )
    .unwrap();
    let first = info(&dir).unwrap().unwrap().segments[0].clone();
    compact(
        &dir,
        &force,
        Some(&v29_with_key_2_at(8)),
        &CancellationToken::new(),
    )
    .unwrap();
    let second = info(&dir).unwrap().unwrap().segments[0].clone();
    assert_ne!(first.sha256, second.sha256);
    assert_ne!(first.id, second.id, "a segment id names its whole content");

    // PREVIOUS's segment is still the file it names, byte for byte, and answers as it did.
    let held = std::fs::read(dir.join(format!("segments/{}.oxv", first.id))).unwrap();
    assert_eq!(files::sha256_hex(&held), first.sha256);
    break_current(&dir, Broken::Garbage);
    let fallback = SegmentSet::open(&dir).unwrap();
    assert_eq!(fallback.generation(), 2);
    let records = &everything(&fallback, 4)[&key(2)].1;
    assert!(
        records.contains(&("id:1".to_string(), 6)),
        "the first compaction's records: {records:?}"
    );
}

/// The release `library` is, published again: embedded anew, so the same keys in the same
/// slots — the same segment id — and other bytes.
fn republished(dir: &TempDir, library: &Library, previous: Option<&Library>) -> (PathBuf, String) {
    release_embedded(dir, library, previous, 1)
}

fn manifest_of(release: &(PathBuf, String)) -> ReleaseManifest {
    serde_json::from_str(&release.1).unwrap()
}

/// A version that is installed, published again — embedded anew, or its hints anchored
/// anew — is the installed segment's id with other bytes. While a generation that opens
/// stands on the installed bytes it is refused as what it is, `SegmentIdTaken`, naming both,
/// and nothing changes. Once no generation that opens names them — two other bases later,
/// `PREVIOUS` included — it installs.
#[test]
fn a_republished_version_is_refused_while_its_installed_bytes_are_served() {
    let work = TempDir::new("set_republished");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let installed = info(&dir).unwrap().unwrap().segments[0].clone();
    let file = dir.join(format!("segments/{}.oxv", installed.id));
    let answers = everything(&SegmentSet::open(&dir).unwrap(), 13);
    let again = republished(&work, &v29(), None);
    let offered = manifest_of(&again);
    assert_eq!(offered.segment_id, installed.id, "one release, so one id");
    assert_ne!(offered.segment.sha256, installed.sha256);

    let refused = |result: Result<ApplyReport, SemanticSearchError>| match result {
        Err(SemanticSearchError::Artifact(ArtifactError::SegmentIdTaken {
            id,
            installed_sha256,
            offered_sha256,
        })) => assert_eq!(
            (id, installed_sha256, offered_sha256),
            (
                installed.id.clone(),
                installed.sha256.clone(),
                offered.segment.sha256.clone()
            )
        ),
        other => panic!("a version published again must be refused as one, got {other:?}"),
    };
    refused(install(&dir, &again));
    assert_eq!(
        files::sha256_hex(&std::fs::read(&file).unwrap()),
        installed.sha256
    );
    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!(set.generation(), 1);
    assert_eq!(everything(&set, 13), answers);
    drop(set);

    // Another base: v29 is PREVIOUS's, which opens, so still refused.
    let mut v40 = v30();
    v40.version = 40;
    install(&dir, &release(&work, &v40, None)).unwrap();
    refused(install(&dir, &again));
    // One more, and nothing that opens names v29's bytes.
    let mut v41 = v40.clone();
    v41.version = 41;
    install(&dir, &release(&work, &v41, None)).unwrap();
    assert_eq!(install(&dir, &again).unwrap().library_version, 29);
    assert_eq!(
        files::sha256_hex(&std::fs::read(&file).unwrap()),
        offered.segment.sha256
    );
}

/// The review's scenario: a base damaged under a delta — scrubbed, so both generations are
/// condemned, or not yet, so both open on damaged bytes — and its version published again.
/// No generation that opens stands on the bytes the file holds, so the release installs as a
/// repair and the set opens on it. The generations that name the old bytes stay shut: those
/// bytes are gone, and they never open on the new ones.
#[test]
fn a_damaged_base_is_repaired_by_its_version_published_again() {
    for scrubbed in [true, false] {
        let work = TempDir::new("set_republished_repair");
        let dir = work.join("vectors");
        install(&dir, &release(&work, &v29(), None)).unwrap();
        install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
        let id = info(&dir).unwrap().unwrap().segments[0].id.clone();
        let file = dir.join(format!("segments/{id}.oxv"));
        damage_last_byte(&file);
        if scrubbed {
            assert!(is_corrupt(scrub(&dir, &CancellationToken::new())));
            assert!(is_corrupt(SegmentSet::open(&dir)));
        }

        let again = republished(&work, &v29(), None);
        let report = install(&dir, &again).unwrap_or_else(|error| {
            panic!("scrubbed {scrubbed}: the repair must install: {error}")
        });
        assert_eq!((report.generation, report.library_version), (3, 29));
        assert_eq!(
            files::sha256_hex(&std::fs::read(&file).unwrap()),
            manifest_of(&again).segment.sha256
        );
        let set = SegmentSet::open(&dir).unwrap();
        assert_eq!(set.generation(), 3, "scrubbed {scrubbed}");
        assert!(!set.info().recovered_from_previous);
        drop(set);
        assert_eq!(info(&dir).unwrap().unwrap().generation, 3);
        assert_eq!(
            scrub(&dir, &CancellationToken::new()).unwrap().generation,
            3
        );

        // PREVIOUS names the base's old bytes: it does not open on the new ones.
        assert_eq!(previous_generation(&dir), 2, "scrubbed {scrubbed}");
        break_current(&dir, Broken::Garbage);
        assert!(is_corrupt(SegmentSet::open(&dir)), "scrubbed {scrubbed}");
        assert!(is_corrupt(info(&dir)), "scrubbed {scrubbed}");
    }
}

/// A delta published again, offered to a set on a fallback: `CURRENT` names the generation
/// with the delta, which does not open, and `PREVIOUS` the base's, which does. Nothing that
/// opens stands on the delta's installed bytes, so the release applies to the generation
/// that opens, as any delta from its version would.
#[test]
fn a_republished_delta_applies_to_a_fallback() {
    let work = TempDir::new("set_republished_delta");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let delta = info(&dir).unwrap().unwrap().segments[1].clone();
    let del = std::fs::read_dir(dir.join(files::generation_dir(2)))
        .unwrap()
        .flatten()
        .find(|entry| entry.file_name().to_string_lossy().ends_with(".del"))
        .unwrap()
        .path();
    let mut bytes = std::fs::read(&del).unwrap();
    bytes[20] ^= 1;
    std::fs::write(&del, bytes).unwrap();
    assert_eq!(SegmentSet::open(&dir).unwrap().generation(), 1);

    let again = republished(&work, &v30(), Some(&v29()));
    assert_eq!(manifest_of(&again).segment_id, delta.id);
    let report = install(&dir, &again).unwrap();
    assert_eq!((report.generation, report.library_version), (3, 30));
    assert_eq!(previous_generation(&dir), 1);
    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!((set.generation(), set.info().library_version), (3, 30));
    assert_eq!(
        set.info().segments[1].sha256,
        manifest_of(&again).segment.sha256
    );
    drop(set);
    assert_eq!(
        files::sha256_hex(&std::fs::read(dir.join(format!("segments/{}.oxv", delta.id))).unwrap()),
        manifest_of(&again).segment.sha256
    );
}

/// Compacting a set that is one compacted segment already writes that segment again: the
/// same bytes, so the same id — and the file in place, mapped by a reader, is kept rather
/// than replaced.
#[test]
fn compacting_again_keeps_the_segment_file_in_place() {
    let work = TempDir::new("set_compact_again");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let force = CompactionPolicy {
        force: true,
        ..CompactionPolicy::default()
    };
    compact(&dir, &force, None, &CancellationToken::new()).unwrap();
    let compacted = info(&dir).unwrap().unwrap().segments[0].clone();
    let file = dir.join(format!("segments/{}.oxv", compacted.id));
    let reader = SegmentSet::open(&dir).unwrap();
    let before = everything(&reader, 11);
    #[cfg(unix)]
    let inode = std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&file).unwrap());

    let report = compact(&dir, &force, None, &CancellationToken::new()).unwrap();
    assert_eq!(report.generation, 3);
    let again = info(&dir).unwrap().unwrap().segments[0].clone();
    assert_eq!(
        (again.id.as_str(), again.sha256.as_str()),
        (compacted.id.as_str(), compacted.sha256.as_str())
    );
    assert_eq!(
        files::sha256_hex(&std::fs::read(&file).unwrap()),
        compacted.sha256
    );
    #[cfg(unix)]
    assert_eq!(
        std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&file).unwrap()),
        inode,
        "the file was kept, not replaced"
    );
    assert_eq!(everything(&reader, 11), before);
    drop(reader);
    assert_eq!(everything(&SegmentSet::open(&dir).unwrap(), 11), before);
}

/// A release manifest comes with a download, and its segment id names files — in staging/
/// and segments/ — before anything compares it with the segment's own. One that is not 32
/// lowercase hex digits is refused before it names any: `../../escape` wrote the segment's
/// copy outside the set.
#[test]
fn a_segment_id_that_is_not_one_names_no_file() {
    let work = TempDir::new("set_unsafe_id");
    let dir = work.join("vectors");
    let (path, json) = release(&work, &v29(), None);
    for id in [
        "../../escape",
        "../escape",
        "ABCDEF0123456789ABCDEF0123456789",
        "",
    ] {
        let mut manifest: ReleaseManifest = serde_json::from_str(&json).unwrap();
        manifest.segment_id = id.to_string();
        match install(&dir, &(path.clone(), manifest.to_json())) {
            Err(SemanticSearchError::Artifact(ArtifactError::UnsafePayloadName {
                name, ..
            })) => assert_eq!(name, id),
            other => panic!("{id:?} must be refused as a name, got {other:?}"),
        }
        assert!(!work.join("escape.oxv").exists(), "{id:?}");
        assert!(!dir.join("escape.oxv").exists(), "{id:?}");
    }
    install(&dir, &(path, json)).unwrap();
}

/// A generation's derived files are named by its segments' ids, in one directory: an id
/// twice would be two segments over one `.del`, and a file named for another segment — or
/// for no segment, outside the directory — another segment's bitmap. A generation that says
/// so is refused, and one being assembled takes no segment twice.
#[test]
fn a_generation_names_each_segment_once_and_each_file_by_its_segment() {
    let work = TempDir::new("set_names");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let pointer = files::read_pointer(&dir, CURRENT).unwrap().unwrap();
    let document = files::read_generation(&dir, &pointer).unwrap();
    assert_eq!(document.check(2), Ok(()));

    let mut twice = document.clone();
    twice.segments.push(twice.segments[1].clone());
    assert!(twice.check(2).unwrap_err().contains("twice"));
    let mut misnamed = document.clone();
    misnamed.segments[0].del.file = format!("{}.del", misnamed.segments[1].id);
    assert!(misnamed.check(2).is_err());
    let mut escaping = document.clone();
    escaping.segments[1].links.as_mut().unwrap().file = "../gen-000001/x.links".to_string();
    assert!(escaping.check(2).is_err());
    let mut not_an_id = document;
    not_an_id.segments[1].id = "../x".to_string();
    not_an_id.segments[1].file = files::segment_file("../x");
    assert!(not_an_id.check(2).is_err());

    let set = SegmentSet::open(&dir).unwrap();
    let mut generation = install::NewGeneration::from_set(&set);
    let entry = set.document().segments[1].clone();
    let segment = crate::semantic::oxv::reader::Segment::open(&dir.join(&entry.file)).unwrap();
    assert!(is_corrupt(generation.push(
        entry,
        &segment,
        set.segments(),
        &CancellationToken::new()
    )));
}

/// Compaction copies vectors byte for byte into a segment it checksums afresh: damage in a
/// source it never read whole would be written under CRCs that agree with it, past any scrub.
/// It reads every block of every source first, and one that fails is condemned as a scrub
/// condemns it; nothing is written.
#[test]
fn compaction_verifies_what_it_copies() {
    let work = TempDir::new("set_compact_damage");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let id = info(&dir).unwrap().unwrap().segments[0].id.clone();
    let damaged = damage_last_byte(&dir.join(format!("segments/{id}.oxv")));
    SegmentSet::open(&dir).expect("opening reads no vector block");

    let result = compact(
        &dir,
        &CompactionPolicy {
            force: true,
            ..CompactionPolicy::default()
        },
        None,
        &CancellationToken::new(),
    );
    assert!(is_corrupt(result));
    let verdict: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(format!("segments/{id}.corrupt"))).unwrap())
            .unwrap();
    assert_eq!(verdict["sha256"], files::sha256_hex(&damaged));
    assert!(!dir.join(files::generation_dir(3)).exists());
    assert!(is_corrupt(SegmentSet::open(&dir)));
    assert!(is_corrupt(info(&dir)));
}

/// "A record is a book and a key", and its hint the first line that holds the key: a book
/// that holds one key twice in one segment — the format allows it — keeps the lower hint
/// when it is compacted, whatever order its records came in.
#[test]
fn compaction_keeps_the_first_line_of_a_key_a_book_holds_twice() {
    let work = TempDir::new("set_compact_twice");
    let dir = work.join("vectors");
    let mut book = TestBook::named("id:1");
    book.primary = vec![(key(1), 5, vector_of(1)), (key(2), 6, vector_of(2))];
    // Key 1 again, at a lower line of the same book.
    book.extras = vec![(2, 0)];
    let mut spec = spec(PackageKind::Base, 0, 29);
    spec.identity_digest = identity().identity_digest();
    let path = work.join("twice.oxv");
    let written = write_segment(&path, spec, codec(), &[book], &[]);
    let manifest = ReleaseManifest::for_segment(
        &written,
        &identity(),
        codec().params_sha256(),
        test_provenance(),
        "2026-10-02T00:00:00Z".to_string(),
    );
    install(&dir, &(path, manifest.to_json())).unwrap();

    compact(
        &dir,
        &CompactionPolicy {
            force: true,
            ..CompactionPolicy::default()
        },
        None,
        &CancellationToken::new(),
    )
    .unwrap();
    let hits = everything(&SegmentSet::open(&dir).unwrap(), 12);
    assert_eq!(hits[&key(1)].1, BTreeSet::from([("id:1".to_string(), 2)]));
}

/// `PREVIOUS` without `CURRENT` is no state a flip leaves — a crash between removing a
/// directory where `CURRENT` should be and writing the pointer is one way there — and what
/// `CURRENT` named cannot be known: no collection runs until a `CURRENT` is written again,
/// and then the generations neither pointer names go.
#[test]
fn garbage_waits_while_current_is_missing() {
    let work = TempDir::new("set_no_current");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    let delta = info(&dir).unwrap().unwrap().segments[1].id.clone();
    std::fs::remove_file(dir.join(CURRENT)).unwrap();

    let set = SegmentSet::open(&dir).unwrap();
    assert_eq!(set.generation(), 1);
    assert!(set.info().recovered_from_previous);
    drop(set);
    assert!(dir.join(files::generation_dir(2)).exists());
    assert!(dir.join(format!("segments/{delta}.oxv")).exists());

    install(&dir, &release(&work, &v31(), None)).unwrap();
    assert_eq!(previous_generation(&dir), 1);
    assert!(!dir.join(files::generation_dir(2)).exists());
}

/// `release` left in `incoming/` as `name`, as a host's download is.
fn downloaded(dir: &Path, release: &(PathBuf, String), name: &str) -> (PathBuf, String) {
    std::fs::create_dir_all(incoming_dir(dir)).unwrap();
    let download = incoming_dir(dir).join(name);
    std::fs::copy(&release.0, &download).unwrap();
    (download, release.1.clone())
}

/// A download the host left in `incoming/` read-only installs. The set flushes it where it
/// lies before it takes it: on Unix a read-only handle flushes a file; Windows flushes only
/// through a handle that writes, so there a read-only download is copied instead, and left
/// where it was.
#[test]
fn a_read_only_download_in_incoming_installs() {
    let work = TempDir::new("set_read_only_download");
    let dir = work.join("vectors");
    let (download, json) = downloaded(&dir, &release(&work, &v29(), None), "download.oxv");
    let mut permissions = std::fs::metadata(&download).unwrap().permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(&download, permissions).unwrap();

    let report = install(&dir, &(download.clone(), json))
        .unwrap_or_else(|error| panic!("a read-only download must install: {error}"));
    assert_eq!(report.generation, 1);
    assert_eq!(SegmentSet::open(&dir).unwrap().info().slots_live, 7);
    assert_eq!(
        download.exists(),
        cfg!(windows),
        "taken, unless it had to be copied"
    );
    #[cfg(windows)]
    {
        // So that the test's directory can be removed.
        let mut permissions = std::fs::metadata(&download).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        std::fs::set_permissions(&download, permissions).unwrap();
    }
}

/// A download in `incoming/` is the host's until it is installed: an install that does not
/// happen — refused, cancelled, a segment that does not verify, a failure after the segment
/// was placed — leaves it where it was, byte for byte, for the host to retry or remove.
#[test]
fn a_download_that_does_not_install_stays_in_incoming() {
    let work = TempDir::new("set_download_kept");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let still_there = |download: &Path, bytes: &[u8], what: &str| {
        assert_eq!(
            std::fs::read(download).ok().as_deref(),
            Some(bytes),
            "{what}: the download is lost"
        );
    };

    // Refused: the installed version, published again.
    let (download, json) = downloaded(&dir, &republished(&work, &v29(), None), "again.oxv");
    let bytes = std::fs::read(&download).unwrap();
    assert!(matches!(
        install(&dir, &(download.clone(), json)),
        Err(SemanticSearchError::Artifact(
            ArtifactError::SegmentIdTaken { .. }
        ))
    ));
    still_there(&download, &bytes, "refused");

    // Cancelled.
    let (download, json) = downloaded(&dir, &release(&work, &v30(), Some(&v29())), "delta.oxv");
    let bytes = std::fs::read(&download).unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let cancelled = install_package(
        &dir,
        &InstallSource {
            segment: &download,
            manifest_json: &json,
        },
        &expectation(),
        &cancel,
    );
    assert!(matches!(cancelled, Err(SemanticSearchError::Cancelled)));
    still_there(&download, &bytes, "cancelled");

    // A segment that is not the manifest's.
    let mut damaged = bytes.clone();
    let last = damaged.len() - 1;
    damaged[last] ^= 1;
    std::fs::write(&download, &damaged).unwrap();
    assert!(install(&dir, &(download.clone(), json.clone())).is_err());
    still_there(&download, &damaged, "damaged");
    std::fs::remove_file(&download).unwrap();

    // Placed, and then the generation cannot be published: CURRENT is a directory with a
    // file in it, which no pointer can be written over.
    let current = dir.join(CURRENT);
    std::fs::remove_file(&current).unwrap();
    std::fs::create_dir(&current).unwrap();
    std::fs::write(current.join("left here"), b"by someone").unwrap();
    let (download, json) = downloaded(&dir, &release(&work, &v31(), None), "v31.oxv");
    let bytes = std::fs::read(&download).unwrap();
    assert!(install(&dir, &(download.clone(), json)).is_err());
    still_there(&download, &bytes, "failed after it was placed");
}

/// A pointer that reads and names the last generation number, which no install writes and no
/// directory holds, blocked every install with "no generation number is left" — and, copied
/// into `PREVIOUS`, every collection after. A number only a pointer names, whose generation
/// does not read, is passed over when nothing is left past it, and the next install heals the
/// set: with a fallback to build on, and without one.
#[test]
fn a_pointer_naming_the_last_generation_number_blocks_no_install() {
    let last = files::Pointer {
        generation: u64::MAX,
        set: format!("{}/{}", files::generation_dir(u64::MAX), files::SET_FILE),
        set_sha256: "0".repeat(64),
    };
    let point_current_at_the_last =
        |dir: &Path| std::fs::write(dir.join(CURRENT), serde_json::to_vec(&last).unwrap()).unwrap();

    let work = TempDir::new("set_last_number");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
    point_current_at_the_last(&dir);
    assert_eq!(SegmentSet::open(&dir).unwrap().generation(), 1);
    let report = install(&dir, &release(&work, &v31(), None))
        .unwrap_or_else(|error| panic!("with a fallback: {error}"));
    assert_eq!(report.generation, 3);
    assert_eq!(previous_generation(&dir), 1);
    assert!(
        !dir.join(files::generation_dir(2)).exists(),
        "both pointers read again, and garbage is collected"
    );

    let alone = work.join("alone");
    install(&alone, &release(&work, &v29(), None)).unwrap();
    point_current_at_the_last(&alone);
    assert!(is_corrupt(SegmentSet::open(&alone)));
    let report = install(&alone, &release(&work, &v31(), None))
        .unwrap_or_else(|error| panic!("without a fallback: {error}"));
    assert_eq!(report.generation, 2);
    assert!(
        !alone.join(PREVIOUS).exists(),
        "a pointer whose generation does not read is no fallback"
    );
    assert_eq!(SegmentSet::open(&alone).unwrap().generation(), 2);
}

/// Run `work` with `hook` as what happens the moment a block is found to fail.
fn on_damage<T>(hook: impl FnMut() + 'static, work: impl FnOnce() -> T) -> T {
    files::ON_DAMAGE.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    let result = work();
    files::ON_DAMAGE.with(|slot| *slot.borrow_mut() = None);
    result
}

/// A scrub or a compaction cancelled as it finds a block that fails returns at once and
/// records nothing. Naming the bytes that failed reads the whole segment, twice, and the
/// cancel stops it there — before a verdict exists, so nothing changes, as a cancel promises;
/// the next scrub finds the damage again.
#[test]
fn a_scrub_or_a_compaction_cancelled_on_damage_records_nothing() {
    for compaction in [false, true] {
        let work = TempDir::new("set_cancel_on_damage");
        let dir = work.join("vectors");
        install(&dir, &release(&work, &v29(), None)).unwrap();
        install(&dir, &release(&work, &v30(), Some(&v29()))).unwrap();
        let id = info(&dir).unwrap().unwrap().segments[0].id.clone();
        damage_last_byte(&dir.join(format!("segments/{id}.oxv")));

        let cancel = CancellationToken::new();
        let cancelling = cancel.clone();
        let result = on_damage(
            move || cancelling.cancel(),
            || match compaction {
                true => compact(
                    &dir,
                    &CompactionPolicy {
                        force: true,
                        ..CompactionPolicy::default()
                    },
                    None,
                    &cancel,
                )
                .map(|_| ()),
                false => scrub(&dir, &cancel).map(|_| ()),
            },
        );
        assert!(
            matches!(result, Err(SemanticSearchError::Cancelled)),
            "compaction {compaction}: {result:?}"
        );
        assert!(
            !dir.join(format!("segments/{id}.corrupt")).exists(),
            "compaction {compaction}"
        );
        assert!(is_corrupt(scrub(&dir, &CancellationToken::new())));
    }
}

/// An install replaces a segment while a scrub reads it: the scrub finds a block that fails
/// in bytes the file no longer holds. It condemns nothing, and says so — a report marked
/// `superseded` rather than a `Corrupted` that would send a host downloading for nothing. On
/// Unix alone: Windows replaces no file that is mapped, so there it cannot happen.
#[cfg(unix)]
#[test]
fn a_scrub_of_bytes_an_install_replaced_reports_it_superseded() {
    let work = TempDir::new("set_scrub_superseded");
    let dir = work.join("vectors");
    install(&dir, &release(&work, &v29(), None)).unwrap();
    let id = info(&dir).unwrap().unwrap().segments[0].id.clone();
    let file = dir.join(format!("segments/{id}.oxv"));
    let sound = std::fs::read(&file).unwrap();
    damage_last_byte(&file);
    // Sound bytes, put back as an install puts a segment: renamed over the file.
    let fresh = work.join("fresh.oxv");
    std::fs::write(&fresh, &sound).unwrap();
    let replaced = file.clone();

    let report = on_damage(
        move || std::fs::rename(&fresh, &replaced).unwrap(),
        || scrub(&dir, &CancellationToken::new()),
    )
    .unwrap_or_else(|error| panic!("the scrub read bytes no longer installed: {error}"));
    assert!(report.superseded);
    assert!(!dir.join(format!("segments/{id}.corrupt")).exists());
    let again = scrub(&dir, &CancellationToken::new()).unwrap();
    assert!(!again.superseded);
    assert!(again.bytes_checked > 0);
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
