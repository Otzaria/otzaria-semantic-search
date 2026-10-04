//! The segment format end to end: what the writer writes, the reader reads back, and
//! every kind of damage is refused where the format says it is.

use super::codec::Codec;
use super::format::{
    DirectoryEntry, Header, SectionKind, BLOCK_SIZE, HEADER_LEN, SECTION_BLOCKSUMMED,
    SECTION_CRITICAL,
};
use super::reader::Segment;
use super::testing::{key, random_books, spec, write_segment, Random, TempDir, TestBook};
use super::writer::{segment_id, SegmentBuilder, SegmentSpec};
use crate::cancellation::CancellationToken;
use crate::distribution::package::{PackageCounts, PackageKind};
use crate::errors::VectorStoreError;
use crate::semantic::chunk_key::ChunkKey;
use sha2::{Digest, Sha256};
use std::path::Path;

const DIM: usize = 16;

/// A delta with every kind of record: three books, primaries in all of them, extras of a
/// key within its own book and across books, foreign records, tombstones.
fn delta_books(random: &mut Random) -> Vec<TestBook> {
    let mut a = TestBook::named("id:1");
    a.primary = vec![(key(1), 0, random.unit(DIM)), (key(2), 3, random.unit(DIM))];
    // Key 1 again, further down the same book.
    a.extras = vec![(9, 0)];
    let mut b = TestBook::named("id:10");
    b.primary = vec![(key(3), 1, random.unit(DIM))];
    // Keys 2 and 3 in this book too, out of hint order on purpose: the writer sorts.
    b.extras = vec![(7, 1), (4, 2)];
    b.foreign = vec![(key(100), 5), (key(101), 2)];
    let mut c = TestBook::named("id:2");
    c.foreign = vec![(key(102), 0)];
    vec![a, b, c]
}

fn open(path: &Path) -> Segment {
    Segment::open(path).unwrap()
}

fn corrupted(result: Result<Segment, VectorStoreError>) -> String {
    match result {
        Err(VectorStoreError::Corrupted { reason }) => reason,
        Err(other) => panic!("expected a corrupted segment, got {other}"),
        Ok(_) => panic!("expected a corrupted segment, and it opened"),
    }
}

#[test]
fn a_written_segment_reads_back_record_for_record() {
    let dir = TempDir::new("round_trip");
    let mut random = Random(7);
    let books = delta_books(&mut random);
    let codec = Codec::calibrate_i8_sym_dim(
        &books
            .iter()
            .flat_map(|book| book.primary.iter().map(|(_, _, v)| v.as_slice()))
            .collect::<Vec<_>>(),
        1.0,
    )
    .unwrap();
    let path = dir.join("delta.oxv");
    let written = write_segment(
        &path,
        spec(PackageKind::Delta, 29, 30),
        codec.clone(),
        &books,
        &[key(201), key(200), key(201)],
    );

    assert_eq!(
        written.counts,
        PackageCounts {
            books: 3,
            slots: 3,
            extras: 3,
            foreign: 3,
            tombstones: 2,
        }
    );
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(written.size, bytes.len() as u64);
    assert_eq!(written.sha256, <[u8; 32]>::from(Sha256::digest(&bytes)));
    assert_eq!(written.clipped_components, 0);

    let segment = open(&path);
    assert_eq!(segment.kind(), PackageKind::Delta);
    assert_eq!(segment.counts(), written.counts);
    assert_eq!(segment.segment_id(), written.segment_id);
    assert_eq!(segment.codec(), &codec);
    assert_eq!(segment.codec_params_sha256(), codec.params_sha256());
    assert_eq!(
        (segment.from_library_version(), segment.to_library_version()),
        (29, 30)
    );
    assert_eq!(segment.library_release_tag(), "v30-20260930120000");
    segment
        .verify_blocks(&CancellationToken::new(), |_| {})
        .unwrap();

    let names: Vec<&str> = segment.books().iter().map(|book| &*book.name).collect();
    assert_eq!(names, ["id:1", "id:10", "id:2"]);
    assert_eq!(segment.books()[0].slots, 0..2);
    assert_eq!(segment.books()[1].slots, 2..3);
    assert_eq!(segment.books()[2].slots, 3..3);

    // Primary records: key, hint and the vector the codec made of it.
    let mut expected_bytes = vec![0u8; codec.bytes_per_vector()];
    for (slot, (k, hint, vector)) in books.iter().flat_map(|b| b.primary.iter()).enumerate() {
        let slot = slot as u32;
        assert_eq!(segment.key(slot), *k);
        assert_eq!(segment.hint(slot), *hint);
        codec.encode(vector, &mut expected_bytes);
        assert_eq!(segment.vector(slot), expected_bytes.as_slice());
    }
    assert_eq!(segment.book_of_slot(0), 0);
    assert_eq!(segment.book_of_slot(1), 0);
    assert_eq!(segment.book_of_slot(2), 1);

    // Extras: grouped by book, in hint order, and reachable from their slot.
    let extras: Vec<(u32, u32)> = (0..3).map(|i| segment.extra(i)).collect();
    assert_eq!(extras, [(0, 9), (2, 4), (1, 7)]);
    assert!(segment.has_extras(0) && segment.has_extras(1) && segment.has_extras(2));
    let of = |slot| segment.extras_of_slot(slot).collect::<Vec<_>>();
    assert_eq!(of(0), [0]);
    assert_eq!(of(1), [2]);
    assert_eq!(of(2), [1]);
    assert_eq!(segment.book_of_extra(0), 0);
    assert_eq!(segment.book_of_extra(1), 1);
    assert_eq!(segment.book_of_extra(2), 1);

    // Foreign records, in hint order within their book, and the tombstones sorted once.
    let foreign: Vec<(ChunkKey, u32, u32)> = (0..3).map(|i| segment.foreign_record(i)).collect();
    assert_eq!(
        foreign,
        [(key(101), 1, 2), (key(100), 1, 5), (key(102), 2, 0)]
    );
    let mut tombstones = vec![key(200), key(201)];
    tombstones.sort();
    assert_eq!(
        (0..2).map(|i| segment.tombstone(i)).collect::<Vec<_>>(),
        tombstones
    );
}

#[test]
fn the_f32_codec_stores_the_vector_exactly() {
    let dir = TempDir::new("f32");
    let mut random = Random(8);
    let books = random_books(&mut random, 3, 4, DIM, 0);
    let path = dir.join("base.oxv");
    write_segment(
        &path,
        spec(PackageKind::Base, 0, 30),
        Codec::f32(DIM).unwrap(),
        &books,
        &[],
    );
    let segment = open(&path);
    let mut decoded = vec![0f32; DIM];
    for (slot, (_, _, vector)) in books.iter().flat_map(|b| b.primary.iter()).enumerate() {
        segment.codec().decode(
            segment.vector(slot as u32),
            segment.vector_scale(slot as u32),
            &mut decoded,
        );
        assert_eq!(&decoded, vector);
    }
}

/// Sections past 1 MiB are cut into several blocks, each with its own CRC.
/// `i8-sym-vec`: every slot's scale is the one its vector encoded with, in VECTOR_SCALES
/// after the codes; each vector decodes to within half a step of itself; and a flipped
/// byte among the scales is a block CRC that fails.
#[test]
fn a_scale_per_vector_is_stored_apart_and_checked_by_block() {
    let dir = TempDir::new("per_vector");
    let mut random = Random(9);
    let books = random_books(&mut random, 4, 300, DIM, 0);
    let codec = Codec::i8_sym_vec(DIM).unwrap();
    let path = dir.join("base.oxv");
    write_segment(
        &path,
        spec(PackageKind::Base, 0, 30),
        codec.clone(),
        &books,
        &[],
    );
    let segment = open(&path);
    let scales = section(&segment_header(&path), SectionKind::VectorScales);
    let vectors = section(&segment_header(&path), SectionKind::Vectors);
    assert_eq!(scales.elem_count, 1200);
    assert!(
        scales.offset > vectors.offset,
        "the scales follow the codes"
    );
    let (mut codes, mut decoded) = (vec![0u8; DIM], vec![0f32; DIM]);
    for (slot, (_, _, vector)) in books.iter().flat_map(|b| b.primary.iter()).enumerate() {
        let scale = codec.encode(vector, &mut codes).scale;
        assert_eq!(segment.vector_scale(slot as u32), scale);
        assert_eq!(segment.vector(slot as u32), codes.as_slice());
        segment
            .codec()
            .decode(segment.vector(slot as u32), scale, &mut decoded);
        for (value, original) in decoded.iter().zip(vector) {
            assert!((value - original).abs() <= scale.unwrap() / 2.0 + 1e-7);
        }
    }
    segment
        .verify_blocks(&CancellationToken::new(), |_| {})
        .unwrap();
    drop(segment);

    flip(&path, scales.offset + 17);
    let damaged = open(&path);
    assert!(damaged
        .verify_blocks(&CancellationToken::new(), |_| {})
        .is_err());
}

fn segment_header(path: &Path) -> Header {
    Header::decode(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn a_segment_of_several_blocks_verifies_block_by_block() {
    let dir = TempDir::new("blocks");
    let mut random = Random(9);
    // 5,000 slots × 256 bytes is 1.28 MB of vectors: two blocks, the second partial.
    let books = random_books(&mut random, 50, 100, 256, 0);
    let refs: Vec<&[f32]> = books
        .iter()
        .flat_map(|b| b.primary.iter().map(|(_, _, v)| v.as_slice()))
        .collect();
    let codec = Codec::calibrate_i8_sym_dim(&refs, 0.999).unwrap();
    let path = dir.join("base.oxv");
    let written = write_segment(&path, spec(PackageKind::Base, 0, 30), codec, &books, &[]);
    assert!(
        written.clipped_components > 0,
        "a 0.999 quantile clips something"
    );

    let segment = open(&path);
    let mut checked = 0u64;
    segment
        .verify_blocks(&CancellationToken::new(), |bytes| checked += bytes)
        .unwrap();
    assert_eq!(checked, 5000 * (4 + 16 + 256));
    // Unmapped before the file is touched: Windows refuses to write a mapped file.
    drop(segment);

    // Damage in the second block of VECTORS is named as that block.
    let header = Header::decode(&std::fs::read(&path).unwrap()).unwrap();
    let vectors = section(&header, SectionKind::Vectors);
    flip(&path, vectors.offset + BLOCK_SIZE as u64 + 17);
    let damaged = open(&path);
    match damaged.verify_blocks(&CancellationToken::new(), |_| {}) {
        Err(VectorStoreError::Corrupted { reason }) => {
            assert!(reason.contains("block 1 of VECTORS"), "{reason}")
        }
        other => panic!("a damaged block must be refused, got {other:?}"),
    }

    // And a scrub can be abandoned between blocks.
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(matches!(
        damaged.verify_blocks(&cancel, |_| {}),
        Err(VectorStoreError::Cancelled)
    ));
}

#[test]
fn every_section_is_aligned_and_the_file_ends_where_the_last_one_does() {
    let dir = TempDir::new("aligned");
    let mut random = Random(10);
    let path = dir.join("delta.oxv");
    write_segment(
        &path,
        spec(PackageKind::Delta, 29, 30),
        Codec::i8_sym_dim(vec![0.5; DIM], 1.0).unwrap(),
        &delta_books(&mut random),
        &[key(9)],
    );
    let bytes = std::fs::read(&path).unwrap();
    let header = Header::decode(&bytes).unwrap();
    assert_eq!(header.sections.len(), 11);
    for entry in &header.sections {
        let align = if entry.kind == SectionKind::Vectors.code() {
            4096
        } else {
            64
        };
        assert_eq!(entry.offset % align, 0, "{entry:?}");
        assert!(entry.is_critical());
    }
    let last = header
        .sections
        .iter()
        .map(DirectoryEntry::end)
        .max()
        .unwrap();
    assert_eq!(last, bytes.len() as u64);
    // Small metadata first, the vectors last.
    assert_eq!(
        header.sections.last().unwrap().kind,
        SectionKind::Vectors.code()
    );

    // A directory entry moved off its alignment is refused.
    let mut moved = header.clone();
    let books = moved
        .sections
        .iter_mut()
        .find(|entry| entry.kind == SectionKind::Books.code())
        .unwrap();
    books.offset += 8;
    rewrite_header(&path, &moved);
    assert!(corrupted(Segment::open(&path)).contains("not on 64"));
}

#[test]
fn a_truncated_segment_is_refused_at_open() {
    let dir = TempDir::new("truncated");
    let mut random = Random(11);
    let path = dir.join("base.oxv");
    write_segment(
        &path,
        spec(PackageKind::Base, 0, 30),
        Codec::i8_sym_dim(vec![0.5; DIM], 1.0).unwrap(),
        &random_books(&mut random, 4, 3, DIM, 0),
        &[],
    );
    let full = std::fs::read(&path).unwrap();
    for keep in [
        0,
        100,
        HEADER_LEN - 1,
        HEADER_LEN,
        full.len() / 2,
        full.len() - 1,
    ] {
        std::fs::write(&path, &full[..keep]).unwrap();
        corrupted(Segment::open(&path));
    }
    // And bytes appended past the last section are not ignored either.
    let mut longer = full.clone();
    longer.extend_from_slice(&[0; 64]);
    std::fs::write(&path, &longer).unwrap();
    assert!(corrupted(Segment::open(&path)).contains("sections end at"));
}

/// One flipped byte in each section: a small section's CRC refuses it at open; a
/// block-summed one opens — reading it is what open avoids — and the block check names it.
#[test]
fn a_flipped_byte_in_any_section_is_caught_by_its_own_checksum() {
    let dir = TempDir::new("flipped");
    let mut random = Random(12);
    let source = dir.join("source.oxv");
    write_segment(
        &source,
        spec(PackageKind::Delta, 29, 30),
        Codec::i8_sym_dim(vec![0.5; DIM], 1.0).unwrap(),
        &delta_books(&mut random),
        &[key(9), key(8)],
    );
    let header = Header::decode(&std::fs::read(&source).unwrap()).unwrap();
    assert!(header.sections.iter().all(|entry| entry.length > 0));

    for entry in &header.sections {
        let name = SectionKind::from_code(entry.kind).unwrap().name();
        let path = dir.join(&format!("{name}.oxv"));
        std::fs::copy(&source, &path).unwrap();
        flip(&path, entry.offset + entry.length / 2);
        if entry.flags & SECTION_BLOCKSUMMED == 0 {
            let reason = corrupted(Segment::open(&path));
            assert!(
                reason.contains(name) || reason.contains("CRC"),
                "{name}: {reason}"
            );
        } else {
            let segment = Segment::open(&path).unwrap();
            match segment.verify_blocks(&CancellationToken::new(), |_| {}) {
                Err(VectorStoreError::Corrupted { reason }) => {
                    assert!(reason.contains(name), "{name}: {reason}")
                }
                other => panic!("{name}: a flipped byte must be found, got {other:?}"),
            }
        }
    }

    // And in the header, where its own CRC refuses it.
    let path = dir.join("header.oxv");
    std::fs::copy(&source, &path).unwrap();
    flip(&path, 150);
    assert!(corrupted(Segment::open(&path)).contains("header CRC"));
}

#[test]
fn an_unknown_ancillary_section_is_skipped_and_an_unknown_critical_one_refused() {
    let dir = TempDir::new("unknown");
    let mut random = Random(13);
    let source = dir.join("source.oxv");
    write_segment(
        &source,
        spec(PackageKind::Base, 0, 30),
        Codec::i8_sym_dim(vec![0.5; DIM], 1.0).unwrap(),
        &random_books(&mut random, 2, 2, DIM, 0),
        &[],
    );

    for (kind, flags, opens) in [
        (0x0999, 0, true),
        (SectionKind::Tier0Sign.code(), 0, true),
        (0x0999, SECTION_CRITICAL, false),
    ] {
        let path = dir.join(&format!("with-{kind:x}-{flags}.oxv"));
        std::fs::copy(&source, &path).unwrap();
        append_section(&path, kind, flags, &[0xAB; 100]);
        let opened = Segment::open(&path);
        if opens {
            let segment = opened.unwrap();
            assert_eq!(segment.slot_count(), 4);
        } else {
            assert!(corrupted(opened).contains("critical"));
        }
    }
}

#[test]
fn the_writer_refuses_tables_no_reader_could_use() {
    let dir = TempDir::new("writer_refusals");
    let codec = || Codec::i8_sym_dim(vec![0.5; DIM], 1.0).unwrap();
    let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), codec());
    builder.add_book("id:2", &[(key(1), 0)], &[], &[]).unwrap();
    // Out of byte order, a repeated name, a hint that would collide with bit 31.
    assert!(builder.add_book("id:10", &[(key(2), 0)], &[], &[]).is_err());
    assert!(builder.add_book("id:2", &[(key(2), 0)], &[], &[]).is_err());
    assert!(builder
        .add_book("id:3", &[(key(2), 1 << 31)], &[], &[])
        .is_err());
    // A book with nothing in it is simply not stored.
    assert_eq!(builder.add_book("id:4", &[], &[], &[]).unwrap(), 1);
    // An extra past the last slot.
    builder.add_book("id:5", &[], &[(0, 7)], &[]).unwrap();
    assert!(builder.write(&dir.join("a.oxv")).is_err());

    // A base cannot carry tombstones.
    let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), codec());
    builder.add_book("id:1", &[(key(1), 0)], &[], &[]).unwrap();
    builder.set_tombstones(vec![key(2)]);
    assert!(builder.write(&dir.join("b.oxv")).is_err());

    // Vectors: of the codec's width, as many as there are slots, no more.
    let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), codec());
    builder.add_book("id:1", &[(key(1), 0)], &[], &[]).unwrap();
    let mut sink = builder.write(&dir.join("c.oxv")).unwrap();
    assert!(sink.push(&[0; DIM - 1]).is_err());
    assert!(sink.push_f32(&[0.0; DIM + 1]).is_err());
    sink.push(&[0; DIM]).unwrap();
    assert!(sink.push(&[0; DIM]).is_err());
    sink.finish().unwrap();

    let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), codec());
    builder
        .add_book("id:1", &[(key(1), 0), (key(2), 1)], &[], &[])
        .unwrap();
    let sink = builder.write(&dir.join("d.oxv")).unwrap();
    assert!(sink.finish().is_err(), "a slot without its vector");

    // And the file is never overwritten.
    let builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 30), codec());
    assert!(builder.write(&dir.join("c.oxv")).is_err());
}

/// The id `docs/ARTIFACT_CONTRACT.md` §3.6 defines for a compacted segment, from the file's
/// bytes alone: SHA-256 over the domain, the header's CRC-covered bytes with the id zeroed,
/// and every section's SHA-256 in directory order, cut to 16 bytes.
fn compacted_id(path: &Path) -> [u8; 16] {
    let bytes = std::fs::read(path).unwrap();
    let header = Header::decode(&bytes).unwrap();
    let mut head = bytes[..4088].to_vec();
    head[32..48].fill(0);
    let mut hasher = Sha256::new();
    hasher.update(b"oxv-compacted-segment-id\n");
    hasher.update(&head);
    for entry in &header.sections {
        let section = &bytes[entry.offset as usize..(entry.offset + entry.length) as usize];
        hasher.update(Sha256::digest(section));
    }
    hasher.finalize()[..16].try_into().unwrap()
}

/// A compacted segment's id is a digest of everything it holds: the same compaction done
/// twice is one file name, and one that differs anywhere — one hint, one vector, its release
/// tag, its library version, its identity — is another. Over the keys alone, as a release's
/// is, two compactions that re-anchored differently had one name, and the second replaced
/// the first's file under a generation that still named it.
#[test]
fn a_compacted_segments_id_is_a_digest_of_everything_it_holds() {
    let dir = TempDir::new("compacted_id");
    let books = random_books(&mut Random(14), 2, 3, DIM, 0);
    let codec = Codec::i8_sym_dim(vec![0.5; DIM], 1.0).unwrap();
    let write = |name: &str, spec: SegmentSpec, books: &[TestBook]| {
        write_segment(&dir.join(name), spec, codec.clone(), books, &[])
    };
    let compacted = |to| spec(PackageKind::Compacted, 0, to);
    let first = write("first.oxv", compacted(30), &books);
    let again = write("again.oxv", compacted(30), &books);
    assert_eq!(
        (first.segment_id, first.sha256),
        (again.segment_id, again.sha256)
    );

    let mut moved = books.clone();
    moved[0].primary[1].1 += 7;
    let mut nudged = books.clone();
    nudged[1].primary[0].2[0] = -nudged[1].primary[0].2[0];
    let mut tagged = compacted(30);
    tagged.library_release_tag = "v30-20261002000000".to_string();
    let mut other_identity = compacted(30);
    other_identity.identity_digest = [7; 32];
    let others = [
        write("moved.oxv", compacted(30), &moved),
        write("nudged.oxv", compacted(30), &nudged),
        write("tagged.oxv", tagged, &books),
        write("later.oxv", compacted(31), &books),
        write("identity.oxv", other_identity, &books),
    ];
    for other in &others {
        assert_ne!(other.sha256, first.sha256, "{}", other.path.display());
        assert_ne!(
            other.segment_id,
            first.segment_id,
            "{}",
            other.path.display()
        );
    }
    for written in std::iter::once(&first).chain(&others) {
        assert_eq!(
            compacted_id(&written.path),
            written.segment_id,
            "{}",
            written.path.display()
        );
        assert_eq!(open(&written.path).segment_id(), written.segment_id);
    }

    // A release keeps the id releases are published under, over its keys and its place in
    // the chain alone — whatever its hints.
    let release = write("release.oxv", spec(PackageKind::Base, 0, 30), &books);
    let moved_release = write("moved-release.oxv", spec(PackageKind::Base, 0, 30), &moved);
    let keys: Vec<u8> = books
        .iter()
        .flat_map(|book| book.primary.iter().flat_map(|(key, _, _)| key.0))
        .collect();
    let published = segment_id(
        &spec(PackageKind::Base, 0, 30).identity_digest,
        PackageKind::Base,
        0,
        30,
        &Sha256::digest(&keys).into(),
    );
    assert_eq!(release.segment_id, published);
    assert_eq!(moved_release.segment_id, published);
}

#[test]
fn a_segment_id_names_the_content_and_the_place_in_the_chain() {
    let keys = <[u8; 32]>::from(Sha256::digest(b"keys"));
    let id = segment_id(&[1; 32], PackageKind::Delta, 29, 30, &keys);
    assert_eq!(id, segment_id(&[1; 32], PackageKind::Delta, 29, 30, &keys));
    for other in [
        segment_id(&[2; 32], PackageKind::Delta, 29, 30, &keys),
        segment_id(&[1; 32], PackageKind::Compacted, 29, 30, &keys),
        segment_id(&[1; 32], PackageKind::Delta, 28, 30, &keys),
        segment_id(&[1; 32], PackageKind::Delta, 29, 31, &keys),
        segment_id(&[1; 32], PackageKind::Delta, 29, 30, &[0; 32]),
    ] {
        assert_ne!(other, id);
    }

    // Two builds of the same segment are one file name.
    let dir = TempDir::new("deterministic");
    let books = random_books(&mut Random(14), 2, 3, DIM, 0);
    let codec = Codec::i8_sym_dim(vec![0.5; DIM], 1.0).unwrap();
    let first = write_segment(
        &dir.join("1.oxv"),
        spec(PackageKind::Base, 0, 30),
        codec.clone(),
        &books,
        &[],
    );
    let second = write_segment(
        &dir.join("2.oxv"),
        spec(PackageKind::Base, 0, 30),
        codec,
        &books,
        &[],
    );
    assert_eq!(first.segment_id, second.segment_id);
    assert_eq!(first.sha256, second.sha256);
}

fn section(header: &Header, kind: SectionKind) -> DirectoryEntry {
    *header
        .sections
        .iter()
        .find(|entry| entry.kind == kind.code())
        .unwrap()
}

fn flip(path: &Path, at: u64) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[at as usize] ^= 0x01;
    std::fs::write(path, bytes).unwrap();
}

fn rewrite_header(path: &Path, header: &Header) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[..HEADER_LEN].copy_from_slice(&header.encode());
    std::fs::write(path, bytes).unwrap();
}

/// Append a section the writer never writes, as a newer writer might.
fn append_section(path: &Path, kind: u16, flags: u16, content: &[u8]) {
    let mut bytes = std::fs::read(path).unwrap();
    let mut header = Header::decode(&bytes).unwrap();
    let offset = bytes.len().div_ceil(64) * 64;
    bytes.resize(offset, 0);
    bytes.extend_from_slice(content);
    header.sections.push(DirectoryEntry {
        kind,
        flags,
        elem_size: 1,
        offset: offset as u64,
        length: content.len() as u64,
        elem_count: content.len() as u64,
        crc32: crc32fast::hash(content),
    });
    bytes[..HEADER_LEN].copy_from_slice(&header.encode());
    std::fs::write(path, bytes).unwrap();
}
