use super::files::{hex, sha256};
use super::ledger::{classify, write_plan_ledger, BaseRecord, Disposition, Ledger};
use super::plan::*;
use super::testing::{corpus, family, passage_package, stub_model, Holding, TempDir};
use crate::errors::PackError;
use crate::semantic::chunker::ChunkerConfig;
use crate::semantic::oxv::codec::Codec;
use std::path::Path;

const A: &str = "otzaria/a.txt";
const B: &str = "otzaria/b.txt";
const C: &str = "otzaria/c.txt";
/// Each long enough to be embedded as itself under the default recipe.
const L1: &str = "בראשית ברא אלהים את השמים ואת הארץ";
const L2: &str = "והארץ היתה תהו ובהו וחשך על פני תהום רבה";
const L3: &str = "ויאמר אלהים יהי אור ויהי אור וירא אלהים";
const L4: &str = "מאימתי קורין את שמע בערבית משעה שהכהנים";
const L5: &str = "אשרי יושבי ביתך עוד יהללוך סלה אשרי העם";
const L6: &str = "שיר המעלות בשוב יהוה את שיבת ציון היינו";

fn plan(
    dir: &TempDir,
    name: &str,
    version: u32,
    lines: &[(u64, &str, &str)],
    previous: Option<&Ledger>,
    warehouse: Option<&dyn HeldVectors>,
) -> Plan {
    let model_path = stub_model(dir.path());
    let out = dir.join(name);
    plan_from_corpus(
        &corpus(dir.path(), name, version, lines),
        PlanRequest {
            out_dir: out.clone(),
            model: family(&model_path),
            chunking: ChunkerConfig::default(),
            passage_package: passage_package(&model_path),
            previous,
            warehouse,
            created_at: "2026-10-02T00:00:00Z".to_string(),
        },
    )
    .unwrap();
    Plan::open(&out).unwrap()
}

fn codec() -> Codec {
    Codec::i8_sym_dim(vec![0.5; 64], 1.0).unwrap()
}

fn ledger(plan: &Plan, previous: Option<&Ledger>, dir: &Path) -> Ledger {
    std::fs::create_dir_all(dir).unwrap();
    write_plan_ledger(
        plan,
        previous,
        &codec(),
        BaseRecord {
            library_version: 1,
            size: 1_000,
        },
        0,
        dir,
    )
    .unwrap();
    Ledger::open(dir, Some(plan.manifest.library_version)).unwrap()
}

fn key(text: &str) -> [u8; 16] {
    sha256(text.as_bytes())[..16].try_into().unwrap()
}

#[test]
fn records_round_trip_and_refuse_disorder_and_damage() {
    let dir = TempDir::new("plan_records");
    let path = dir.join("records.bin");
    let records = [(0, 0, L1), (0, 3, L2), (2, 1, L3)].map(|(book, ordinal, text)| PlanRecord {
        book,
        ordinal,
        sha256: sha256(text.as_bytes()),
    });
    let mut writer = RecordsWriter::create(&path).unwrap();
    for record in records {
        writer.push(record).unwrap();
    }
    assert!(matches!(
        writer.push(records[0]),
        Err(PackError::MalformedInput { .. })
    ));
    assert_eq!(writer.finish().unwrap(), 3);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 16 + 3 * 40);

    let read = PlanRecords::open(&path).unwrap();
    assert_eq!(read.iter().collect::<Vec<_>>(), records);
    assert_eq!(read.get(2).key().to_hex(), hex(&key(L3)));

    // Windows refuses to write a file while a mapping of it is open.
    drop(read);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() - 1);
    std::fs::write(&path, &bytes).unwrap();
    assert!(
        PlanRecords::open(&path).is_err(),
        "a truncated file is refused"
    );
}

#[test]
fn books_must_ascend_by_bytes() {
    assert!(BookList::new(vec![B.into(), A.into()]).is_err());
    assert!(BookList::new(vec![A.into(), A.into()]).is_err());
    let books = BookList::new(vec![A.into(), B.into(), C.into()]).unwrap();
    assert_eq!(books.index_of(B), Some(1));
    assert_eq!(books.index_of("otzaria/z.txt"), None);
}

/// A plan keys every embedded line at its ordinal, and asks for each text once, in order
/// of first appearance, unless the warehouse has its vector already.
#[test]
fn a_plan_keys_every_line_and_embeds_each_text_once() {
    let dir = TempDir::new("plan_corpus");
    let lines = [
        (4_294_967_297, B, L1),
        (4_294_967_298, B, L3),
        (8_589_934_593, A, L2),
        (8_589_934_594, A, L1),
        (8_589_934_595, A, L2),
    ];
    let first = plan(&dir, "v1", 1, &lines, None, None);
    assert_eq!(first.books.names(), [A, B], "books in byte order");
    let records: Vec<(u32, u32, [u8; 16])> = first
        .records
        .iter()
        .map(|record| (record.book, record.ordinal, record.key_bytes()))
        .collect();
    assert_eq!(
        records,
        [
            (0, 0, key(L2)),
            (0, 1, key(L1)),
            (0, 2, key(L2)),
            (1, 0, key(L1)),
            (1, 1, key(L3))
        ]
    );
    let counts = first.manifest.counts;
    assert_eq!((counts.records, counts.unique, counts.to_embed), (5, 3, 3));
    assert_eq!(counts.to_ship, 3, "a first plan ships every key");

    let manifest = EmbedManifest::read(&first.dir).unwrap();
    let texts: Vec<String> = read_embed_plan(&first.dir, 0, u64::MAX)
        .unwrap()
        .map(|record| record.unwrap().1.embedding_text)
        .collect();
    assert_eq!(texts, [L2, L1, L3]);
    assert_eq!(manifest.records, 3);
    assert_eq!(
        manifest.plan_sha256,
        hex(&sha256(
            &std::fs::read(first.dir.join(EMBED_PLAN_FILE)).unwrap()
        ))
    );
    for (index, record) in read_embed_plan(&first.dir, 1, 1).unwrap().enumerate() {
        let (position, record) = record.unwrap();
        assert_eq!((index, position), (0, 1));
        assert_eq!(record.check(position).unwrap(), sha256(L1.as_bytes()));
        let edited = EmbedRecord {
            embedding_text: format!("{L1}!"),
            ..record
        };
        assert!(matches!(
            edited.check(position),
            Err(PackError::PlanTextChanged { record: 1, .. })
        ));
    }

    // A warehouse that holds a text's vector already takes it out of the embedding plan.
    let held = Holding::of(&[L1]);
    let warm = plan(&dir, "v1-warm", 1, &lines, None, Some(&held));
    assert_eq!(warm.manifest.counts.to_embed, 2);
    assert_eq!(warm.manifest.counts.revived, 1);

    // A records file that is not the one the manifest names is refused.
    let plan_dir = first.dir.clone();
    let path = plan_dir.join(RECORDS_FILE);
    drop(first);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[20] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    assert!(Plan::open(&plan_dir).is_err());
}

/// The split of a release against the one before it: what is held already, what ships as
/// a slot, what ships as a foreign record, what is tombstoned and what the warehouse
/// revives — and the ledger the release then leaves for the next one.
#[test]
fn the_split_against_the_previous_release() {
    let dir = TempDir::new("plan_split");
    let v1 = plan(
        &dir,
        "v1",
        1,
        &[(1, A, L1), (2, A, L2), (3, A, L6), (4, B, L3)],
        None,
        None,
    );
    let ledger_v1 = ledger(&v1, None, &dir.join("ledger-1"));
    assert_eq!(ledger_v1.key_count(), 4);

    // L2 moves up a line; L1 moves to book B; L6 goes; L4 is new, twice in A; L5 is new,
    // in a new book, and the warehouse holds its vector already.
    let lines = [
        (1, A, L2),
        (2, A, L4),
        (3, A, L4),
        (4, B, L3),
        (5, B, L1),
        (6, C, L5),
    ];
    let held = Holding::of(&[L5]);
    let v2 = plan(&dir, "v2", 2, &lines, Some(&ledger_v1), Some(&held));
    let counts = v2.manifest.counts;
    assert_eq!(
        (
            counts.records,
            counts.unique,
            counts.reused,
            counts.to_ship,
            counts.revived,
            counts.tombstones,
            counts.foreign_pairs
        ),
        (6, 5, 3, 2, 1, 1, 1)
    );
    assert_eq!(
        super::ledger::read_keys_file(&v2.dir.join(TOMBSTONES_FILE)).unwrap(),
        [key(L6)]
    );
    assert_eq!(
        v2.manifest.previous.as_ref().unwrap().library_version,
        1,
        "the plan names the release it was split against"
    );

    let mut dispositions = Vec::new();
    classify(
        &v2.records,
        &v2.books,
        Some(&ledger_v1),
        |_, record, disposition| {
            dispositions.push((record.book, record.ordinal, disposition));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        dispositions,
        [
            (0, 0, Disposition::Held),
            (0, 1, Disposition::Slot { slot: 0 }),
            (0, 2, Disposition::Repeat),
            (1, 0, Disposition::Held),
            (1, 1, Disposition::Foreign),
            (2, 0, Disposition::Slot { slot: 1 }),
        ]
    );

    // The next ledger: the keys v2 holds, and its pairs by v2's books.
    let ledger_v2 = ledger(&v2, Some(&ledger_v1), &dir.join("ledger-2"));
    let mut expected = [L1, L2, L3, L4, L5].map(key);
    expected.sort_unstable();
    assert_eq!(
        (0..ledger_v2.key_count())
            .map(|index| ledger_v2.key(index))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(ledger_v2.pair_count(), 5);
    assert!(
        ledger_v2.contains_pair(1, &key(L1)),
        "(B, L1) is held at v2"
    );
    assert!(!ledger_v2.contains_pair(0, &key(L1)), "(A, L1) is not");
    assert!(ledger_v2.contains_pair(2, &key(L5)));
}

#[test]
fn a_ledger_refuses_another_chain_and_damage() {
    let dir = TempDir::new("plan_ledger");
    let v1 = plan(&dir, "v1", 1, &[(1, A, L1), (2, B, L2)], None, None);
    let at = dir.join("ledger");
    let opened = ledger(&v1, None, &at);
    // The ledger's identity is the release's: the plan's, with the codec's precision.
    let digest =
        super::assemble::release_identity(&v1.manifest.identity, &codec()).identity_digest_hex();
    assert_ne!(digest, v1.manifest.identity.identity_digest_hex());
    let epoch = hex(&codec().params_sha256());
    opened
        .manifest
        .ensure_matches(&digest, Some(&epoch))
        .unwrap();
    assert!(opened
        .manifest
        .ensure_matches(&"0".repeat(64), None)
        .is_err());
    assert!(opened
        .manifest
        .ensure_matches(&digest, Some(&"0".repeat(64)))
        .is_err());
    assert_eq!(opened.codec().unwrap().params(), codec().params());

    let keys = at.join(super::ledger::keys_file_name(1));
    drop(opened);
    let mut bytes = std::fs::read(&keys).unwrap();
    bytes[16] ^= 1;
    std::fs::write(&keys, bytes).unwrap();
    assert!(
        Ledger::open(&at, Some(1)).is_err(),
        "a damaged keys file is refused"
    );
}
