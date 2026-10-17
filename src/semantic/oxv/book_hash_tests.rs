use super::{
    codec::Codec,
    format::{Header, SectionKind, SECTION_BLOCKSUMMED, SECTION_CRITICAL},
    reader::Segment,
    testing::{key, spec, TempDir},
    writer::SegmentBuilder,
};
use crate::{cancellation::CancellationToken, distribution::package::PackageKind};

#[test]
fn source_hashes_round_trip_including_metadata_only_and_unknown_books() {
    let dir = TempDir::new("book_hashes");
    let path = dir.join("base.oxv");
    let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), Codec::f32(2).unwrap());
    builder
        .add_book_windowed_hashed("a", &[], &[], &[], &[], Some(11))
        .unwrap();
    builder
        .add_book_windowed_hashed("b", &[(key(1), 0)], &[], &[], &[], Some(22))
        .unwrap();
    builder
        .add_book_windowed_hashed("c", &[], &[], &[], &[], Some(0))
        .unwrap();
    let mut sink = builder.write(&path).unwrap();
    sink.push_f32(&[1.0, 0.0]).unwrap();
    sink.finish().unwrap();
    let segment = Segment::open(&path).unwrap();
    assert_eq!(segment.books().len(), 3);
    assert_eq!(segment.book_source_hash("a"), Some(11));
    assert_eq!(segment.book_source_hash("b"), Some(22));
    assert_eq!(segment.book_source_hash("c"), None);
    assert_eq!(segment.book_source_hash("missing"), None);
    segment
        .verify_blocks(&CancellationToken::new(), |_| {})
        .unwrap();
    // Windows cannot rewrite a file while its read-only mapping is open.
    drop(segment);
    let bytes = std::fs::read(&path).unwrap();
    let mut header = Header::decode(&bytes).unwrap();
    let entry = header
        .sections
        .iter_mut()
        .find(|entry| entry.kind == SectionKind::BookHashes.code())
        .unwrap();
    assert_eq!((entry.elem_size, entry.elem_count), (8, 3));
    assert_eq!(entry.flags, SECTION_CRITICAL | SECTION_BLOCKSUMMED);
    entry.flags &= !SECTION_CRITICAL;
    let mut bytes = bytes;
    bytes[..4096].copy_from_slice(&header.encode());
    std::fs::write(&path, &bytes).unwrap();
    assert!(
        Segment::open(&path).is_err(),
        "hash metadata must fail closed for older readers"
    );
}

#[test]
fn repeated_parent_occurrences_use_one_first_hint_but_distinct_parents_remain() {
    use super::occurrence::Occurrence;
    let dir = TempDir::new("coalesced_parents");
    let path = dir.join("base.oxv");
    let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), Codec::f32(2).unwrap());
    builder
        .add_book_windowed("a", &[(key(1), 0)], &[1], &[], &[])
        .unwrap();
    let mut rows: Vec<_> = (0..100)
        .map(|hint| Occurrence {
            key: key(1),
            book: 0,
            hint,
            anchor: key(2),
            window: 1,
        })
        .collect();
    rows.push(Occurrence {
        key: key(1),
        book: 0,
        hint: 100,
        anchor: key(3),
        window: 1,
    });
    builder.add_occurrences("a", &rows).unwrap();
    let mut sink = builder.write(&path).unwrap();
    sink.push_f32(&[1.0, 0.0]).unwrap();
    sink.finish().unwrap();
    let segment = Segment::open(&path).unwrap();
    let rows: Vec<_> = segment.occurrences().collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].hint, 0);
    assert_eq!(rows[1].hint, 100);
    assert_ne!(rows[0].anchor, rows[1].anchor);
}
