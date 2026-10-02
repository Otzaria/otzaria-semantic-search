//! Writing a segment: every table first, then the vectors, streamed.
//!
//! [`SegmentBuilder`] collects what a segment says about its records — books in name
//! order, each with its primary records (which take the next slots), its extras and its
//! foreign records, and the tombstones. That is everything but the vectors, and it is small
//! next to them: 16 bytes of key and 4 of hint per slot, 8 per extra, 24 per foreign record —
//! about 130 MB for a full library, against 1.6 GB of vectors.
//!
//! [`SegmentBuilder::write`] lays the file out, writes those tables, and hands back a
//! [`VectorSink`] that takes the vectors in slot order through a 4 MiB buffer. Nothing holds
//! more than one vector at a time, so a build streams a library through it in bounded memory
//! and a compaction copies vectors straight out of the segments it merges.
//! [`VectorSink::finish`] writes the block checksums and the header, flushes the file, and
//! reads it back once for its SHA-256 — the digest a package declares for it.
//!
//! The file is created new: a writer never replaces a segment, which is what lets a reader
//! map one without a lock.

use crate::distribution::package::{validate_release_tag, PackageCounts, PackageKind};
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::codec::Codec;
use crate::semantic::oxv::format::{
    align_up, BookEntry, DirectoryEntry, Header, SectionKind, BLOCK_SIZE, EXTRA_LEN,
    FLAG_HAS_FOREIGN, FLAG_HAS_TOMBSTONES, FOREIGN_LEN, HAS_EXTRAS, HEADER_LEN, HINT_MASK,
    SECTION_ALIGN, SECTION_BLOCKSUMMED, SECTION_CRITICAL, VECTORS_ALIGN,
};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// What a segment is, apart from what it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentSpec {
    pub kind: PackageKind,
    /// [`IndexVersion::identity_digest`](crate::semantic::versioning::IndexVersion::identity_digest)
    /// of the identity the vectors were built under.
    pub identity_digest: [u8; 32],
    /// The library version a delta applies on top of; `0` otherwise.
    pub from_library_version: u32,
    pub to_library_version: u32,
    /// For people; at most 64 bytes, no control characters.
    pub library_release_tag: String,
}

/// A segment's tables, collected before its vectors are written.
pub struct SegmentBuilder {
    spec: SegmentSpec,
    codec: Codec,
    books: Vec<BookEntry>,
    names: Vec<u8>,
    keys: Vec<[u8; 16]>,
    hints: Vec<u32>,
    extras: Vec<(u32, u32)>,
    foreign: Vec<([u8; 16], u32, u32)>,
    tombstones: Vec<[u8; 16]>,
}

/// What [`VectorSink::finish`] wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenSegment {
    pub path: PathBuf,
    pub size: u64,
    /// SHA-256 of the whole file.
    pub sha256: [u8; 32],
    /// The deterministic id the file is installed under — see [`segment_id`].
    pub segment_id: [u8; 16],
    pub spec: SegmentSpec,
    pub counts: PackageCounts,
    /// Components the codec clipped, over every vector pushed as floats.
    pub clipped_components: u64,
}

impl SegmentBuilder {
    pub fn new(spec: SegmentSpec, codec: Codec) -> Self {
        Self {
            spec,
            codec,
            books: Vec::new(),
            names: Vec::new(),
            keys: Vec::new(),
            hints: Vec::new(),
            extras: Vec::new(),
            foreign: Vec::new(),
            tombstones: Vec::new(),
        }
    }

    /// Add the next book, in strictly increasing byte order of names.
    ///
    /// * `primary` — `(key, hint)` for each record whose vector this segment ships, in the
    ///   order their slots are to be numbered: the next `primary.len()` slots are this
    ///   book's, which is what makes a book filter a range.
    /// * `extras` — `(hint, slot)` for each further record of a key this segment ships, at
    ///   a slot of this book or an earlier one.
    /// * `foreign` — `(key, hint)` for each record whose vector an older segment ships.
    ///
    /// Extras and foreign records are stored sorted by hint. A book with no record at all
    /// is not stored.
    ///
    /// Returns the book's first slot.
    pub fn add_book(
        &mut self,
        name: &str,
        primary: &[(ChunkKey, u32)],
        extras: &[(u32, u32)],
        foreign: &[(ChunkKey, u32)],
    ) -> io::Result<u32> {
        let first_slot = self.keys.len() as u32;
        if primary.is_empty() && extras.is_empty() && foreign.is_empty() {
            return Ok(first_slot);
        }
        if name.is_empty() || name.len() > u16::MAX as usize {
            return Err(invalid(format!(
                "a book name is 1 to 65535 bytes, and {name:?} is {}",
                name.len()
            )));
        }
        if let Some(last) = self.books.last() {
            let previous = &self.names[last.name_off as usize..][..last.name_len as usize];
            if name.as_bytes() <= previous {
                return Err(invalid(format!(
                    "books are added in increasing byte order, and {name:?} follows {:?}",
                    String::from_utf8_lossy(previous)
                )));
            }
        }
        let hint_too_large = |hint: u32| hint > HINT_MASK;
        if primary.iter().any(|(_, hint)| hint_too_large(*hint))
            || extras.iter().any(|(hint, _)| hint_too_large(*hint))
            || foreign.iter().any(|(_, hint)| hint_too_large(*hint))
        {
            return Err(invalid(format!(
                "a hint of book {name:?} is past {HINT_MASK}, which bit 31 of a slot's hint \
                 cannot share"
            )));
        }
        let total = |count: usize| u32::try_from(count).map_err(|_| invalid("too many records"));
        let name_off = total(self.names.len())?;
        let entry = BookEntry {
            name_off,
            name_len: name.len() as u16,
            flags: 0,
            slot_start: first_slot,
            slot_count: total(primary.len())?,
            extra_start: total(self.extras.len())?,
            extra_count: total(extras.len())?,
            foreign_start: total(self.foreign.len())?,
            foreign_count: total(foreign.len())?,
        };
        total(self.keys.len() + primary.len())?;
        total(self.extras.len() + extras.len())?;
        total(self.foreign.len() + foreign.len())?;
        total(self.names.len() + name.len())?;

        self.names.extend_from_slice(name.as_bytes());
        for (key, hint) in primary {
            self.keys.push(key.0);
            self.hints.push(*hint);
        }
        let mut sorted: Vec<(u32, u32)> = extras.to_vec();
        sorted.sort_unstable();
        self.extras
            .extend(sorted.into_iter().map(|(hint, slot)| (slot, hint)));
        let mut sorted: Vec<(u32, [u8; 16])> = foreign.iter().map(|(k, h)| (*h, k.0)).collect();
        sorted.sort_unstable();
        let book = self.books.len() as u32;
        self.foreign
            .extend(sorted.into_iter().map(|(hint, key)| (key, book, hint)));
        self.books.push(entry);
        Ok(first_slot)
    }

    /// The keys a delta tells the set to forget. Sorted and deduplicated here.
    pub fn set_tombstones(&mut self, mut keys: Vec<ChunkKey>) {
        keys.sort_unstable();
        keys.dedup();
        self.tombstones = keys.into_iter().map(|key| key.0).collect();
    }

    /// Slots assigned so far.
    pub fn slot_count(&self) -> u32 {
        self.keys.len() as u32
    }

    pub fn codec(&self) -> &Codec {
        &self.codec
    }

    /// Lay the file out at `path`, which must not exist, write every table, and return the
    /// sink the vectors go through.
    pub fn write(self, path: &Path) -> io::Result<VectorSink> {
        validate_release_tag(&self.spec.library_release_tag)
            .map_err(|error| invalid(error.to_string()))?;
        let slot_count = self.keys.len() as u32;
        if let Some((slot, _)) = self.extras.iter().find(|(slot, _)| *slot >= slot_count) {
            return Err(invalid(format!(
                "an extra record names slot {slot}, and the segment has {slot_count}"
            )));
        }
        let has_foreign = !self.foreign.is_empty();
        let has_tombstones = !self.tombstones.is_empty();
        if self.spec.kind != PackageKind::Delta && (has_foreign || has_tombstones) {
            return Err(invalid(format!(
                "a {} carries no foreign records and no tombstones",
                self.spec.kind
            )));
        }

        // The tables, as bytes.
        let mut hints = self.hints;
        for (slot, _) in &self.extras {
            hints[*slot as usize] |= HAS_EXTRAS;
        }
        let mut by_slot: Vec<u32> = (0..self.extras.len() as u32).collect();
        // Within one slot, extras are in book order already, because they were appended
        // book by book; the index breaks the tie the same way.
        by_slot.sort_unstable_by_key(|index| (self.extras[*index as usize].0, *index));

        let mut books = Vec::with_capacity(self.books.len() * 40);
        for book in &self.books {
            book.encode(&mut books);
        }
        let mut extras = Vec::with_capacity(self.extras.len() * EXTRA_LEN);
        for (slot, hint) in &self.extras {
            extras.extend_from_slice(&slot.to_le_bytes());
            extras.extend_from_slice(&hint.to_le_bytes());
        }
        let mut extras_by_slot = Vec::with_capacity(by_slot.len() * 4);
        for index in &by_slot {
            extras_by_slot.extend_from_slice(&index.to_le_bytes());
        }
        let mut foreign = Vec::with_capacity(self.foreign.len() * FOREIGN_LEN);
        for (key, book, hint) in &self.foreign {
            foreign.extend_from_slice(key);
            foreign.extend_from_slice(&book.to_le_bytes());
            foreign.extend_from_slice(&hint.to_le_bytes());
        }
        let tombstones: Vec<u8> = self.tombstones.concat();
        let hint_bytes: Vec<u8> = hints.iter().flat_map(|hint| hint.to_le_bytes()).collect();
        let key_bytes: Vec<u8> = self.keys.concat();

        let vector_bytes = self.codec.bytes_per_vector() as u64;
        let vectors_len = u64::from(slot_count) * vector_bytes;
        let scales_len = if self.codec.has_vector_scales() {
            u64::from(slot_count) * 4
        } else {
            0
        };
        let blocks = |len: u64| len.div_ceil(BLOCK_SIZE as u64);
        let crc_count = blocks(hint_bytes.len() as u64)
            + blocks(key_bytes.len() as u64)
            + blocks(vectors_len)
            + blocks(scales_len);

        // The layout: small metadata first, vectors last.
        let fixed = |kind: SectionKind| kind.fixed_elem_size().expect("a fixed-size section");
        let mut plan: Vec<(SectionKind, u32, u64, u64, bool)> = vec![
            (
                SectionKind::Books,
                fixed(SectionKind::Books),
                books.len() as u64,
                self.books.len() as u64,
                false,
            ),
            (
                SectionKind::BookNames,
                1,
                self.names.len() as u64,
                self.names.len() as u64,
                false,
            ),
            (
                SectionKind::CodecParams,
                1,
                self.codec.params().len() as u64,
                self.codec.params().len() as u64,
                false,
            ),
            (
                SectionKind::Extras,
                fixed(SectionKind::Extras),
                extras.len() as u64,
                self.extras.len() as u64,
                false,
            ),
            (
                SectionKind::ExtrasBySlot,
                4,
                extras_by_slot.len() as u64,
                by_slot.len() as u64,
                false,
            ),
            (
                SectionKind::Foreign,
                fixed(SectionKind::Foreign),
                foreign.len() as u64,
                self.foreign.len() as u64,
                false,
            ),
            (
                SectionKind::Tombstones,
                16,
                tombstones.len() as u64,
                self.tombstones.len() as u64,
                false,
            ),
            (SectionKind::BlockCrcs, 4, crc_count * 4, crc_count, false),
            (
                SectionKind::SlotHints,
                4,
                hint_bytes.len() as u64,
                u64::from(slot_count),
                true,
            ),
            (
                SectionKind::SlotKeys,
                16,
                key_bytes.len() as u64,
                u64::from(slot_count),
                true,
            ),
            (
                SectionKind::Vectors,
                vector_bytes as u32,
                vectors_len,
                u64::from(slot_count),
                true,
            ),
        ];
        // A codec with a scale per vector keeps them apart from the codes, after them, so
        // the rows of VECTORS stay `dim` bytes and aligned.
        if self.codec.has_vector_scales() {
            plan.push((
                SectionKind::VectorScales,
                4,
                scales_len,
                u64::from(slot_count),
                true,
            ));
        }
        let mut offset = HEADER_LEN as u64;
        let mut sections = Vec::with_capacity(plan.len());
        for (kind, elem_size, length, elem_count, blocksummed) in plan {
            let align = if kind == SectionKind::Vectors {
                VECTORS_ALIGN
            } else {
                SECTION_ALIGN
            };
            offset = align_up(offset, align);
            sections.push(DirectoryEntry {
                kind: kind.code(),
                flags: SECTION_CRITICAL | if blocksummed { SECTION_BLOCKSUMMED } else { 0 },
                elem_size,
                offset,
                length,
                elem_count,
                crc32: 0,
            });
            offset += length;
        }

        let segment_id = segment_id(
            &self.spec.identity_digest,
            self.spec.kind,
            self.spec.from_library_version,
            self.spec.to_library_version,
            &Sha256::digest(&key_bytes).into(),
        );
        let mut flags = Header::kind_flag(self.spec.kind);
        if has_foreign {
            flags |= FLAG_HAS_FOREIGN;
        }
        if has_tombstones {
            flags |= FLAG_HAS_TOMBSTONES;
        }
        let header = Header {
            flags,
            dim: self.codec.dim() as u16,
            codec_id: self.codec.id(),
            block_size: BLOCK_SIZE as u32,
            segment_id,
            identity_digest: self.spec.identity_digest,
            codec_params_sha256: self.codec.params_sha256(),
            slot_count: u64::from(slot_count),
            extra_count: self.extras.len() as u64,
            foreign_count: self.foreign.len() as u64,
            tombstone_count: self.tombstones.len() as u64,
            book_count: self.books.len() as u32,
            from_library_version: self.spec.from_library_version,
            to_library_version: self.spec.to_library_version,
            library_release_tag: self.spec.library_release_tag.clone(),
            sections,
        };

        let file = OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .open(path)?;
        let mut out = Positioned {
            file: BufWriter::with_capacity(4 << 20, file),
            at: 0,
        };
        // The header is written last, once every CRC is known; until then, zeros.
        out.write_all(&[0u8; HEADER_LEN])?;
        let contents: [&[u8]; 7] = [
            &books,
            &self.names,
            self.codec.params(),
            &extras,
            &extras_by_slot,
            &foreign,
            &tombstones,
        ];
        let mut header = header;
        for (index, bytes) in contents.into_iter().enumerate() {
            out.pad_to(header.sections[index].offset)?;
            out.write_all(bytes)?;
            header.sections[index].crc32 = crc32fast::hash(bytes);
        }
        // The block checksums of HINTS and KEYS are known now; those of VECTORS when the
        // last vector arrives.
        let mut block_crcs = Vec::with_capacity(crc_count as usize);
        block_crcs.extend(hint_bytes.chunks(BLOCK_SIZE).map(crc32fast::hash));
        block_crcs.extend(key_bytes.chunks(BLOCK_SIZE).map(crc32fast::hash));
        let crc_entry = 7;
        out.pad_to(header.sections[crc_entry].offset)?;
        out.write_all(&vec![0u8; header.sections[crc_entry].length as usize])?;
        out.pad_to(header.sections[8].offset)?;
        out.write_all(&hint_bytes)?;
        out.pad_to(header.sections[9].offset)?;
        out.write_all(&key_bytes)?;
        out.pad_to(header.sections[10].offset)?;

        let has_scales = self.codec.has_vector_scales();
        let counts = PackageCounts {
            books: header.book_count,
            slots: header.slot_count,
            extras: header.extra_count,
            foreign: header.foreign_count,
            tombstones: header.tombstone_count,
        };
        Ok(VectorSink {
            out,
            path: path.to_path_buf(),
            header,
            spec: self.spec,
            codec: self.codec,
            counts,
            expected: u64::from(slot_count),
            written: 0,
            block: crc32fast::Hasher::new(),
            in_block: 0,
            block_crcs,
            scratch: Vec::new(),
            scales: has_scales.then(|| Vec::with_capacity(scales_len as usize)),
            clipped: 0,
        })
    }
}

/// The vectors' end of a segment being written. Take them in slot order, then
/// [`Self::finish`].
pub struct VectorSink {
    out: Positioned,
    path: PathBuf,
    header: Header,
    spec: SegmentSpec,
    codec: Codec,
    counts: PackageCounts,
    expected: u64,
    written: u64,
    block: crc32fast::Hasher,
    in_block: usize,
    block_crcs: Vec<u32>,
    scratch: Vec<u8>,
    /// VECTOR_SCALES, for a codec with a scale per vector: written after the last vector.
    scales: Option<Vec<u8>>,
    clipped: u64,
}

impl VectorSink {
    /// The next slot's vector, already encoded in a codec with no scale per vector.
    pub fn push(&mut self, encoded: &[u8]) -> io::Result<()> {
        self.push_encoded(encoded, None)
    }

    /// The next slot's vector, already encoded, with its own scale for a codec that has
    /// one per vector — `None` for any other.
    pub fn push_encoded(&mut self, encoded: &[u8], scale: Option<f32>) -> io::Result<()> {
        match (&self.scales, scale) {
            (Some(_), Some(scale)) if scale.is_finite() && scale >= 0.0 => {}
            (Some(_), _) => {
                return Err(invalid(format!(
                    "codec {} needs each vector's scale, finite and not negative",
                    self.codec.name()
                )))
            }
            (None, Some(_)) => {
                return Err(invalid(format!(
                    "codec {} has no scale per vector",
                    self.codec.name()
                )))
            }
            (None, None) => {}
        }
        if encoded.len() != self.codec.bytes_per_vector() {
            return Err(invalid(format!(
                "a vector of this codec is {} bytes, and this one is {}",
                self.codec.bytes_per_vector(),
                encoded.len()
            )));
        }
        if self.written == self.expected {
            return Err(invalid(format!(
                "the segment has {} slot(s), and every one has its vector",
                self.expected
            )));
        }
        self.out.write_all(encoded)?;
        if let (Some(scales), Some(scale)) = (&mut self.scales, scale) {
            scales.extend_from_slice(&scale.to_le_bytes());
        }
        let mut rest = encoded;
        while !rest.is_empty() {
            let take = rest.len().min(BLOCK_SIZE - self.in_block);
            self.block.update(&rest[..take]);
            self.in_block += take;
            rest = &rest[take..];
            if self.in_block == BLOCK_SIZE {
                let full = std::mem::replace(&mut self.block, crc32fast::Hasher::new());
                self.block_crcs.push(full.finalize());
                self.in_block = 0;
            }
        }
        self.written += 1;
        Ok(())
    }

    /// The next slot's vector, as floats: encoded here, clipped components counted.
    pub fn push_f32(&mut self, vector: &[f32]) -> io::Result<()> {
        if vector.len() != self.codec.dim() {
            return Err(invalid(format!(
                "a vector of this segment has {} components, and this one has {}",
                self.codec.dim(),
                vector.len()
            )));
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize(self.codec.bytes_per_vector(), 0);
        let encoded = self.codec.encode(vector, &mut scratch);
        self.clipped += encoded.clipped as u64;
        let pushed = self.push_encoded(&scratch, encoded.scale);
        self.scratch = scratch;
        pushed
    }

    /// Vectors pushed so far.
    pub fn written(&self) -> u64 {
        self.written
    }

    pub fn codec(&self) -> &Codec {
        &self.codec
    }

    /// Write the block checksums and the header, flush the file to disk, and read it back
    /// once for its SHA-256.
    pub fn finish(mut self) -> io::Result<WrittenSegment> {
        if self.written != self.expected {
            return Err(invalid(format!(
                "the segment has {} slot(s), and {} vector(s) arrived",
                self.expected, self.written
            )));
        }
        if self.in_block > 0 {
            self.block_crcs.push(self.block.clone().finalize());
        }
        if let Some(scales) = self.scales.take() {
            let entry = self
                .header
                .sections
                .iter()
                .position(|entry| entry.kind == SectionKind::VectorScales.code())
                .expect("a codec with scales lays out their section");
            self.out.pad_to(self.header.sections[entry].offset)?;
            self.out.write_all(&scales)?;
            self.block_crcs
                .extend(scales.chunks(BLOCK_SIZE).map(crc32fast::hash));
        }
        let crc_bytes: Vec<u8> = self
            .block_crcs
            .iter()
            .flat_map(|crc| crc.to_le_bytes())
            .collect();
        let crc_entry = &mut self.header.sections[7];
        if crc_bytes.len() as u64 != crc_entry.length {
            return Err(invalid("the block checksums do not fill their section"));
        }
        crc_entry.crc32 = crc32fast::hash(&crc_bytes);
        let crc_offset = crc_entry.offset;

        let mut file = self
            .out
            .file
            .into_inner()
            .map_err(|error| error.into_error())?;
        file.seek(SeekFrom::Start(crc_offset))?;
        file.write_all(&crc_bytes)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&self.header.encode())?;
        file.sync_all()?;

        file.seek(SeekFrom::Start(0))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        let mut size = 0u64;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            size += read as u64;
        }
        Ok(WrittenSegment {
            path: self.path,
            size,
            sha256: hasher.finalize().into(),
            segment_id: self.header.segment_id,
            spec: self.spec,
            counts: self.counts,
            clipped_components: self.clipped,
        })
    }
}

/// The id a segment is installed under: SHA-256 over `"oxv-segment-id"`, the identity
/// digest, the kind, the two library versions and the SHA-256 of its keys, cut to 16
/// bytes. Deterministic, so the same segment built twice is the same file name, and a
/// segment applied twice is one segment.
pub fn segment_id(
    identity_digest: &[u8; 32],
    kind: PackageKind,
    from_library_version: u32,
    to_library_version: u32,
    keys_sha256: &[u8; 32],
) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"oxv-segment-id");
    hasher.update(identity_digest);
    hasher.update([Header::kind_flag(kind) as u8]);
    hasher.update(from_library_version.to_le_bytes());
    hasher.update(to_library_version.to_le_bytes());
    hasher.update(keys_sha256);
    let digest = hasher.finalize();
    digest[..16].try_into().expect("16 bytes")
}

/// A buffered writer that knows where it is, so padding to a section's offset is
/// arithmetic rather than a seek.
struct Positioned {
    file: BufWriter<File>,
    at: u64,
}

impl Positioned {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.file.write_all(bytes)?;
        self.at += bytes.len() as u64;
        Ok(())
    }

    fn pad_to(&mut self, offset: u64) -> io::Result<()> {
        debug_assert!(offset >= self.at, "sections are laid out in file order");
        let zeros = [0u8; 4096];
        while self.at < offset {
            let take = ((offset - self.at) as usize).min(zeros.len());
            self.write_all(&zeros[..take])?;
        }
        Ok(())
    }
}

fn invalid(reason: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, reason.into())
}
