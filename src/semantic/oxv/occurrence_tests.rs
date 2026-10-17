//! Occurrence snapshots are correctness metadata, so malformed snapshots must fail closed.
use super::codec::Codec;
use super::format::{
    DirectoryEntry, Header, SectionKind, HEADER_LEN, HINT_MASK, SECTION_BLOCKSUMMED,
    SECTION_CRITICAL,
};
use super::occurrence::Occurrence;
use super::reader::Segment;
use super::testing::{key, spec, TempDir};
use super::writer::SegmentBuilder;
use crate::cancellation::CancellationToken;
use crate::distribution::package::PackageKind;
use crate::errors::VectorStoreError;
use std::path::Path;

fn write_snapshot(path: &Path) -> Vec<Occurrence> {
    let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), Codec::f32(2).unwrap());
    builder
        .add_book_windowed("a", &[(key(1), 7), (key(2), 7)], &[0, 1], &[], &[])
        .unwrap();
    let mut records = vec![
        Occurrence {
            key: key(1),
            book: 0,
            hint: 7,
            anchor: key(1),
            window: 0,
        },
        Occurrence {
            key: key(2),
            book: 0,
            hint: 7,
            anchor: key(1),
            window: 1,
        },
    ];
    builder.add_occurrences("a", &records).unwrap();
    let mut sink = builder.write(path).unwrap();
    sink.push_f32(&[1.0, 0.0]).unwrap();
    sink.push_f32(&[0.0, 1.0]).unwrap();
    sink.finish().unwrap();
    records.sort_unstable();
    records
}

fn entry(header: &Header) -> DirectoryEntry {
    *header
        .sections
        .iter()
        .find(|entry| entry.kind == SectionKind::Occurrences.code())
        .unwrap()
}

fn rejection(path: &Path) -> String {
    match Segment::open(path) {
        Err(VectorStoreError::Corrupted { reason }) => reason,
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("malformed occurrence snapshot was accepted"),
    }
}

#[test]
fn occurrence_snapshot_round_trip_and_payload_crc() {
    let dir = TempDir::new("occurrence_snapshot_round_trip");
    let path = dir.join("base.oxv");
    let records = write_snapshot(&path);
    let segment = Segment::open(&path).unwrap();
    assert_eq!(segment.occurrences().collect::<Vec<_>>(), records);
    for record in &records {
        assert_eq!(
            segment
                .key_occurrences(record.key)
                .unwrap()
                .collect::<Vec<_>>(),
            [*record]
        );
    }
    assert!(segment.key_occurrences(key(100)).is_none());
    segment
        .verify_blocks(&CancellationToken::new(), |_| {})
        .unwrap();
    drop(segment);
    let mut bytes = std::fs::read(&path).unwrap();
    let section = entry(&Header::decode(&bytes).unwrap());
    assert_eq!(section.flags, SECTION_CRITICAL | SECTION_BLOCKSUMMED);
    assert_eq!((section.elem_size, section.elem_count), (44, 2));
    // Corrupt the inner occurrence's anchor while leaving its structure valid.
    let inner = records.iter().position(|record| record.window > 0).unwrap();
    bytes[section.offset as usize + inner * 44 + 24] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    match Segment::open(&path)
        .unwrap()
        .verify_blocks(&CancellationToken::new(), |_| {})
    {
        Err(VectorStoreError::Corrupted { reason }) => {
            assert!(reason.contains("OCCURRENCES"), "{reason}")
        }
        other => panic!("snapshot payload corruption was not detected: {other:?}"),
    }
}

#[test]
fn occurrence_snapshots_must_be_critical_and_block_checked() {
    for removed in [SECTION_CRITICAL, SECTION_BLOCKSUMMED] {
        let dir = TempDir::new("occurrence_snapshot_flags");
        let path = dir.join("base.oxv");
        write_snapshot(&path);
        let mut bytes = std::fs::read(&path).unwrap();
        let mut header = Header::decode(&bytes).unwrap();
        let section = header
            .sections
            .iter_mut()
            .find(|entry| entry.kind == SectionKind::Occurrences.code())
            .unwrap();
        section.flags &= !removed;
        bytes[..HEADER_LEN].copy_from_slice(&header.encode());
        std::fs::write(&path, bytes).unwrap();
        let reason = rejection(&path);
        assert!(reason.contains("OCCURRENCES"), "{reason}");
    }
}

#[test]
fn occurrence_reader_rejects_invalid_location_role_order_and_reserved_bytes() {
    for damage in ["book", "hint", "anchor", "order", "reserved"] {
        let dir = TempDir::new("occurrence_snapshot_bad_record");
        let path = dir.join("base.oxv");
        let records = write_snapshot(&path);
        let mut bytes = std::fs::read(&path).unwrap();
        let section = entry(&Header::decode(&bytes).unwrap());
        let at = section.offset as usize;
        let anchor = records
            .iter()
            .position(|record| record.window == 0)
            .unwrap();
        match damage {
            "book" => bytes[at + 16..at + 20].copy_from_slice(&99u32.to_le_bytes()),
            "hint" => bytes[at + 20..at + 24].copy_from_slice(&(HINT_MASK + 1).to_le_bytes()),
            "anchor" => bytes[at + anchor * 44 + 24] ^= 1,
            "order" => {
                let first = bytes[at..at + 44].to_vec();
                bytes[at + 44..at + 88].copy_from_slice(&first);
            }
            "reserved" => bytes[at + 41] = 1,
            _ => unreachable!(),
        }
        std::fs::write(&path, bytes).unwrap();
        let reason = rejection(&path);
        assert!(
            reason.to_lowercase().contains("occurrence"),
            "{damage}: {reason}"
        );
    }
}

#[test]
fn newest_occurrence_snapshot_replaces_removed_books_and_query_penalty() {
    use super::scan::{scan, Link, ScanRequest, ScanSegment, ScanSet};
    use crate::semantic::resolve::BookSet;
    let dir = TempDir::new("occurrence_snapshot_replacement");
    let base_path = dir.join("base.oxv");
    let mut base = SegmentBuilder::new(spec(PackageKind::Base, 0, 29), Codec::f32(2).unwrap());
    base.add_book("a", &[(key(1), 0)], &[], &[]).unwrap();
    base.add_occurrences(
        "a",
        &[Occurrence {
            key: key(1),
            book: 0,
            hint: 0,
            anchor: key(1),
            window: 0,
        }],
    )
    .unwrap();
    let mut sink = base.write(&base_path).unwrap();
    sink.push_f32(&[1.0, 0.0]).unwrap();
    sink.finish().unwrap();
    let delta_path = dir.join("delta.oxv");
    let mut delta = SegmentBuilder::new(spec(PackageKind::Delta, 29, 30), Codec::f32(2).unwrap());
    delta.add_book("b", &[], &[], &[(key(1), 40)]).unwrap();
    delta
        .add_occurrences(
            "b",
            &[Occurrence {
                key: key(1),
                book: 0,
                hint: 40,
                anchor: key(2),
                window: 1,
            }],
        )
        .unwrap();
    delta.write(&delta_path).unwrap().finish().unwrap();
    let segments = [
        Segment::open(&base_path).unwrap(),
        Segment::open(&delta_path).unwrap(),
    ];
    let links = [Link {
        seg: 0,
        slot: 0,
        flags: 0,
    }];
    let views = [
        ScanSegment {
            segment: &segments[0],
            deleted: &[],
            links: &[],
        },
        ScanSegment {
            segment: &segments[1],
            deleted: &[],
            links: &links,
        },
    ];
    let set = ScanSet {
        segments: &views,
        reverse_links: &[],
    };
    let query = super::kernel::PreparedQuery::new(&[1.0, 0.0], segments[0].codec()).unwrap();
    let a: BookSet = ["a"].into_iter().collect();
    let b: BookSet = ["b"].into_iter().collect();
    for books in [None, Some(&a), Some(&b)] {
        let hits = scan(
            &set,
            &query,
            &ScanRequest {
                top_k: 10,
                books,
                threads: 1,
                window_penalty: 0.04,
            },
            &CancellationToken::new(),
        )
        .unwrap();
        if books == Some(&a) {
            assert!(
                hits.is_empty(),
                "the removed old book must no longer admit the key"
            );
        } else {
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].window, 1);
            assert!((hits[0].score - 0.96).abs() < 1e-6);
            assert_eq!(hits[0].records.len(), 1);
            assert_eq!(hits[0].records[0].book.as_ref(), "b");
            assert_eq!(hits[0].records[0].hint, 40);
            assert_eq!(hits[0].records[0].anchor, Some(key(2)));
        }
    }
    // A live-index arrival can name a new book absent from the published snapshot.
    let c: BookSet = ["c"].into_iter().collect();
    let request = ScanRequest {
        top_k: 1,
        books: Some(&c),
        threads: 1,
        window_penalty: 0.04,
    };
    assert!(scan(&set, &query, &request, &CancellationToken::new())
        .unwrap()
        .is_empty());
    let hits = super::scan::scan_with(
        &set,
        &query,
        &request,
        &[crate::semantic::resolve::SlotRef {
            key: key(1),
            seg: 0,
            slot: 0,
        }],
        &CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].window, 1);
    assert!((hits[0].score - 0.96).abs() < 1e-6);
    assert_eq!(hits[0].records[0].book.as_ref(), "b");
}

#[test]
fn occurrence_record_budget_keeps_other_books_and_filtered_role() {
    use super::scan::{scan_segments, ScanRequest};
    use crate::semantic::resolve::{BookSet, MAX_RECORDS_PER_HIT};
    let dir = TempDir::new("occurrence_snapshot_budget");
    let path = dir.join("base.oxv");
    let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), Codec::f32(2).unwrap());
    builder
        .add_book_windowed("a", &[(key(1), 0)], &[1], &[], &[])
        .unwrap();
    let repeated: Vec<_> = (0..MAX_RECORDS_PER_HIT as u32 + 5)
        .map(|hint| Occurrence {
            key: key(1),
            book: 0,
            hint,
            anchor: key(100 + hint as u64),
            window: 1,
        })
        .collect();
    builder.add_occurrences("a", &repeated).unwrap();
    builder.add_book("z", &[], &[(4, 0)], &[]).unwrap();
    builder
        .add_occurrences(
            "z",
            &[Occurrence {
                key: key(1),
                book: 0,
                hint: 4,
                anchor: key(1),
                window: 0,
            }],
        )
        .unwrap();
    let mut sink = builder.write(&path).unwrap();
    sink.push_f32(&[1.0, 0.0]).unwrap();
    sink.finish().unwrap();
    let segment = Segment::open(&path).unwrap();
    let a: BookSet = ["a"].into_iter().collect();
    let z: BookSet = ["z"].into_iter().collect();
    for books in [None, Some(&a), Some(&z)] {
        let hits = scan_segments(
            &[&segment],
            &[1.0, 0.0],
            &ScanRequest {
                top_k: 1,
                books,
                threads: 1,
                window_penalty: 0.04,
            },
            &CancellationToken::new(),
        )
        .unwrap();
        let hit = &hits[0];
        assert_eq!(hit.records.len(), MAX_RECORDS_PER_HIT);
        if books == Some(&a) {
            assert_eq!(hit.window, 1);
            assert!((hit.score - 0.96).abs() < 1e-6);
            assert!(hit.records.iter().all(|record| record.book.as_ref() == "a"));
        } else {
            assert_eq!(hit.window, 0);
            assert!((hit.score - 1.0).abs() < 1e-6);
            assert_eq!(hit.records[0].book.as_ref(), "z");
            assert!(hit.records.iter().any(|record| record.book.as_ref() == "a"));
        }
    }
}
