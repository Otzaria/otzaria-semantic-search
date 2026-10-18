//! Opening a segment: map it, check everything that is cheap to check, and read records
//! in place.
//!
//! [`Segment::open`] is the per-open layer of the store's integrity (the spec's §1.5(2)):
//! the header's CRC; a directory whose sections are sorted, disjoint, aligned, inside a
//! file of exactly the length they add up to, and of kinds this build knows when they are
//! critical; a CRC over every small section; and the invariants of the book table — slot,
//! extra and foreign ranges that tile their sections. About 5 MB of a full base is read to
//! do it. The block-summed sections — keys, hints, vectors — are only mapped;
//! [`Segment::verify_blocks`] reads and checks them, once at install and on demand after.
//!
//! Nothing is copied into the heap but the book table, so the vectors of an open set are
//! file-backed pages the operating system can share and reclaim.

use crate::cancellation::CancellationToken;
use crate::distribution::package::{PackageCounts, PackageKind};
use crate::errors::VectorStoreError;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::codec::Codec;
use crate::semantic::oxv::format::{
    foreign_len, key_len, le_u16, le_u32, occurrence_len, BookEntry, DirectoryEntry, Header,
    SectionKind, BLOCK_SIZE, BOOK_ENTRY_LEN, EXTRA_LEN, HAS_EXTRAS, HEADER_CRC_OFFSET, HEADER_LEN,
    HINT_MASK, MAGIC, READABLE_FORMATS, SECTION_ALIGN, VECTORS_ALIGN,
};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One book of a segment, as the reader holds it.
#[derive(Debug, Clone)]
pub struct SegmentBook {
    pub name: Arc<str>,
    /// Its primary records: slots `slots.start..slots.end`.
    pub slots: Range<u32>,
    /// Its extra records: indexes into the segment's extras.
    pub extras: Range<u32>,
    /// Its foreign records: indexes into the segment's foreign records.
    pub foreign: Range<u32>,
}

/// A mapped, checked segment.
pub struct Segment {
    path: PathBuf,
    map: Mmap,
    header: Header,
    kind: PackageKind,
    codec: Codec,
    /// 16 bytes in format 2, 8 in format 3.
    key_len: usize,
    books: Box<[SegmentBook]>,
    hints: Range<usize>,
    keys: Range<usize>,
    vectors: Range<usize>,
    /// VECTOR_SCALES, for a codec with a scale per vector.
    scales: Option<Range<usize>>,
    /// SLOT_WINDOWS, when the segment has it.
    windows: Option<Range<usize>>,
    occurrences: Option<Range<usize>>,
    book_hashes: Option<Range<usize>>,
    extras: Range<usize>,
    extras_by_slot: Range<usize>,
    foreign: Range<usize>,
    tombstones: Range<usize>,
    block_crcs: Range<usize>,
    /// The block-summed sections, in directory order, for [`Self::verify_blocks`].
    blocksummed: Vec<(SectionKind, Range<usize>)>,
}

impl std::fmt::Debug for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Segment")
            .field("path", &self.path)
            .field("kind", &self.kind)
            .field("slots", &self.header.slot_count)
            .finish_non_exhaustive()
    }
}

impl Segment {
    /// Map `path` and run every check that does not read the block-summed sections.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::OpenFailed`] when the file cannot be opened or mapped, or is a
    /// sound segment of a format or codec this build does not read — newer than it, not
    /// damaged — and [`VectorStoreError::Corrupted`], naming the file and what is wrong, for
    /// anything else about its bytes.
    pub fn open(path: &Path) -> Result<Self, VectorStoreError> {
        if cfg!(target_endian = "big") {
            return Err(VectorStoreError::OpenFailed {
                reason: "segments are little-endian, and this host is not".to_string(),
            });
        }
        let file = File::open(path).map_err(|error| VectorStoreError::OpenFailed {
            reason: format!("{}: {error}", path.display()),
        })?;
        // SAFETY: a mapping is only sound while nothing else writes or truncates the file,
        // and the store never does: a segment is written under staging/, renamed into
        // segments/ once complete, and never opened for writing again — every change to a
        // set is a new file. What remains is another process editing the file in place,
        // which this crate cannot prevent and which would be damage, not a mode of use; the
        // per-block CRCs exist to catch it.
        let map = unsafe { Mmap::map(&file) }.map_err(|error| VectorStoreError::OpenFailed {
            reason: format!("{}: could not be mapped: {error}", path.display()),
        })?;
        if let Some(reason) = unsupported(&map) {
            return Err(VectorStoreError::OpenFailed {
                reason: format!("segment {}: {reason}", path.display()),
            });
        }
        Self::check(path, map).map_err(|reason| VectorStoreError::Corrupted {
            reason: format!("segment {}: {reason}", path.display()),
        })
    }

    fn check(path: &Path, map: Mmap) -> Result<Self, String> {
        let header = Header::decode(&map)?;
        let kind = header
            .kind()
            .expect("decode refuses a header without a kind");
        let file_len = map.len() as u64;
        if header.block_size as usize != BLOCK_SIZE {
            return Err(format!(
                "its block size is {}, not {BLOCK_SIZE}",
                header.block_size
            ));
        }

        // The directory: sorted, disjoint, aligned, inside the file, and the file ends
        // where the last section does.
        let mut sections: HashMap<SectionKind, DirectoryEntry> = HashMap::new();
        let mut previous_end = HEADER_LEN as u64;
        for entry in &header.sections {
            let label = SectionKind::from_code(entry.kind).map_or_else(
                || format!("section {:#06x}", entry.kind),
                |k| k.name().to_string(),
            );
            if entry.offset < previous_end {
                return Err(format!(
                    "{label} starts at {} inside what comes before it, which ends at \
                     {previous_end}",
                    entry.offset
                ));
            }
            let align = if entry.kind == SectionKind::Vectors.code() {
                VECTORS_ALIGN
            } else {
                SECTION_ALIGN
            };
            if entry.offset % align != 0 {
                return Err(format!(
                    "{label} starts at {}, not on {align}",
                    entry.offset
                ));
            }
            if entry.end() > file_len || entry.offset.checked_add(entry.length).is_none() {
                return Err(format!(
                    "{label} runs to {}, past the file's {file_len} bytes — a truncated file",
                    entry.end()
                ));
            }
            previous_end = entry.end();
            match entry.known_kind() {
                Some(kind) => {
                    if sections.insert(kind, *entry).is_some() {
                        return Err(format!("{label} appears twice"));
                    }
                    if let Some(size) = kind.fixed_elem_size(header.format_version) {
                        if entry.elem_size != size
                            || entry.elem_count.checked_mul(u64::from(size)) != Some(entry.length)
                        {
                            return Err(format!(
                                "{label} declares {} element(s) of {} bytes in {} bytes",
                                entry.elem_count, entry.elem_size, entry.length
                            ));
                        }
                    }
                }
                None if entry.is_critical() => {
                    return Err(format!(
                        "{label} is critical and this build does not know it"
                    ));
                }
                // An ancillary section this build does not know: skipped, as the format
                // promises.
                None => {}
            }
        }
        if previous_end != file_len {
            return Err(format!(
                "its sections end at {previous_end} and the file holds {file_len} bytes"
            ));
        }

        // Every small section's CRC.
        for entry in sections.values().filter(|entry| !entry.is_blocksummed()) {
            let bytes = &map[range(entry)];
            let actual = crc32fast::hash(bytes);
            if actual != entry.crc32 {
                return Err(format!(
                    "{} hashes to CRC {actual:08x}, and the directory declares {:08x}",
                    SectionKind::from_code(entry.kind).map_or("a section", SectionKind::name),
                    entry.crc32
                ));
            }
        }

        let required = |kind: SectionKind| {
            sections
                .get(&kind)
                .copied()
                .ok_or_else(|| format!("it has no {} section", kind.name()))
        };
        // Optional only while there is nothing to put in them.
        let optional = |kind: SectionKind, count: u64| match sections.get(&kind) {
            Some(entry) => Ok(*entry),
            None if count == 0 => Ok(DirectoryEntry {
                kind: kind.code(),
                flags: 0,
                elem_size: kind.fixed_elem_size(header.format_version).unwrap_or(1),
                offset: 0,
                length: 0,
                elem_count: 0,
                crc32: 0,
            }),
            None => Err(format!(
                "the header counts {count} for {} and it has no such section",
                kind.name()
            )),
        };
        let books_entry = required(SectionKind::Books)?;
        let names_entry = required(SectionKind::BookNames)?;
        let params_entry = required(SectionKind::CodecParams)?;
        let crcs_entry = required(SectionKind::BlockCrcs)?;
        let hints_entry = required(SectionKind::SlotHints)?;
        let keys_entry = required(SectionKind::SlotKeys)?;
        let vectors_entry = required(SectionKind::Vectors)?;
        let extras_entry = optional(SectionKind::Extras, header.extra_count)?;
        let by_slot_entry = optional(SectionKind::ExtrasBySlot, header.extra_count)?;
        let foreign_entry = optional(SectionKind::Foreign, header.foreign_count)?;
        let tombstones_entry = optional(SectionKind::Tombstones, header.tombstone_count)?;

        for (entry, kind, expected) in [
            (
                &books_entry,
                SectionKind::Books,
                u64::from(header.book_count),
            ),
            (&hints_entry, SectionKind::SlotHints, header.slot_count),
            (&keys_entry, SectionKind::SlotKeys, header.slot_count),
            (&vectors_entry, SectionKind::Vectors, header.slot_count),
            (&extras_entry, SectionKind::Extras, header.extra_count),
            (
                &by_slot_entry,
                SectionKind::ExtrasBySlot,
                header.extra_count,
            ),
            (&foreign_entry, SectionKind::Foreign, header.foreign_count),
            (
                &tombstones_entry,
                SectionKind::Tombstones,
                header.tombstone_count,
            ),
        ] {
            if entry.elem_count != expected {
                return Err(format!(
                    "{} holds {} element(s), and the header counts {expected}",
                    kind.name(),
                    entry.elem_count
                ));
            }
        }
        if header.slot_count > u64::from(u32::MAX)
            || header.extra_count > u64::from(u32::MAX)
            || header.foreign_count > u64::from(u32::MAX)
        {
            return Err("its counts do not fit the 32-bit indexes its records use".into());
        }
        if kind != PackageKind::Delta && (header.foreign_count > 0 || header.tombstone_count > 0) {
            return Err(format!("a {kind} holds foreign records or tombstones"));
        }

        let codec = Codec::from_params(
            header.codec_id,
            usize::from(header.dim),
            &map[range(&params_entry)],
        )?;
        if codec.params_sha256() != header.codec_params_sha256 {
            return Err("its codec parameters are not the ones its header names".into());
        }
        if vectors_entry.elem_size as usize != codec.bytes_per_vector() {
            return Err(format!(
                "its vectors are {} bytes, and codec {} at width {} takes {}",
                vectors_entry.elem_size,
                codec.name(),
                codec.dim(),
                codec.bytes_per_vector()
            ));
        }
        let scales_entry = match (
            codec.has_vector_scales(),
            sections.get(&SectionKind::VectorScales),
        ) {
            (true, Some(entry)) => {
                if entry.elem_count != header.slot_count || !entry.is_blocksummed() {
                    return Err(format!(
                        "VECTOR_SCALES holds {} scale(s) for {} slot(s), or is not covered by \
                         block CRCs",
                        entry.elem_count, header.slot_count
                    ));
                }
                Some(*entry)
            }
            (true, None) => {
                return Err(format!(
                    "codec {} keeps a scale per vector, and it has no VECTOR_SCALES",
                    codec.name()
                ))
            }
            (false, Some(entry)) if entry.length > 0 => {
                return Err(format!(
                    "it carries per-vector scales, which codec {} does not read",
                    codec.name()
                ))
            }
            (false, _) => None,
        };
        let windows_entry = match sections.get(&SectionKind::SlotWindows) {
            Some(entry) if entry.elem_count != header.slot_count || !entry.is_blocksummed() => {
                return Err(format!(
                    "SLOT_WINDOWS holds {} window(s) for {} slot(s), or is not covered by block \
                     CRCs",
                    entry.elem_count, header.slot_count
                ))
            }
            entry => entry.copied(),
        };

        if sections
            .get(&SectionKind::Occurrences)
            .is_some_and(|entry| !entry.is_critical() || !entry.is_blocksummed())
        {
            return Err("OCCURRENCES must be critical and block-summed".to_string());
        }
        let key_len = key_len(header.format_version);
        if let Some(entry) = sections.get(&SectionKind::Occurrences) {
            let row_len = occurrence_len(header.format_version);
            // Format 3 gives the first reserved byte a role: an alias, of window 0.
            let aliases = header.format_version >= super::format::SEGMENT_FORMAT_V3;
            // Window 255 is only ever an alias's role, never a window byte.
            let sound = |row: &[u8]| match row[row_len - 3..] {
                _ if row[row_len - 4] == super::occurrence::ALIAS_WINDOW => false,
                [0, 0, 0] => true,
                [role, 0, 0] => {
                    aliases && role == super::occurrence::ROLE_ALIAS && row[row_len - 4] == 0
                }
                _ => false,
            };
            if !map[range(entry)].chunks_exact(row_len).all(sound) {
                return Err(
                    "an occurrence has nonzero reserved bytes or an unknown role".to_string(),
                );
            }
        }
        if sections.get(&SectionKind::BookHashes).is_some_and(|entry| {
            entry.elem_count != u64::from(header.book_count)
                || !entry.is_critical()
                || !entry.is_blocksummed()
        }) {
            return Err(
                "BOOK_HASHES must contain one critical block-summed hash per book".to_string(),
            );
        }
        let blocksummed: Vec<(SectionKind, Range<usize>)> = header
            .sections
            .iter()
            .filter(|entry| entry.is_blocksummed())
            .map(|entry| {
                (
                    entry.known_kind().unwrap_or(SectionKind::Tier0Pq),
                    range(entry),
                )
            })
            .collect();
        let blocks: u64 = header
            .sections
            .iter()
            .filter(|entry| entry.is_blocksummed())
            .map(DirectoryEntry::block_count)
            .sum();
        if crcs_entry.elem_count != blocks {
            return Err(format!(
                "it carries {} block CRC(s) for {blocks} block(s)",
                crcs_entry.elem_count
            ));
        }
        for kind in [
            SectionKind::SlotHints,
            SectionKind::SlotKeys,
            SectionKind::Vectors,
        ] {
            if !sections[&kind].is_blocksummed() {
                return Err(format!("{} is not covered by block CRCs", kind.name()));
            }
        }

        // The book table: names sorted and inside BOOK_NAMES; slot, extra and foreign
        // ranges tiling their sections in order.
        let names = &map[range(&names_entry)];
        let table = &map[range(&books_entry)];
        let mut books = Vec::with_capacity(header.book_count as usize);
        let (mut slots, mut extras, mut foreign) = (0u64, 0u64, 0u64);
        let mut previous: Option<&[u8]> = None;
        for (index, bytes) in table.as_chunks::<BOOK_ENTRY_LEN>().0.iter().enumerate() {
            let entry = BookEntry::decode(bytes);
            let start = entry.name_off as usize;
            let name = names
                .get(start..start + entry.name_len as usize)
                .ok_or_else(|| format!("book {index}'s name lies outside BOOK_NAMES"))?;
            if name.is_empty() || previous.is_some_and(|previous| name <= previous) {
                return Err(format!("book {index}'s name is empty or out of byte order"));
            }
            previous = Some(name);
            let name = std::str::from_utf8(name)
                .map_err(|_| format!("book {index}'s name is not UTF-8"))?;
            for (label, at, count, cursor) in [
                ("slot", entry.slot_start, entry.slot_count, &mut slots),
                ("extra", entry.extra_start, entry.extra_count, &mut extras),
                (
                    "foreign",
                    entry.foreign_start,
                    entry.foreign_count,
                    &mut foreign,
                ),
            ] {
                if u64::from(at) != *cursor {
                    return Err(format!(
                        "book {index}'s {label} range starts at {at}, and the previous book's \
                         ends at {cursor}"
                    ));
                }
                *cursor += u64::from(count);
            }
            books.push(SegmentBook {
                name: Arc::from(name),
                slots: entry.slot_start..entry.slot_start + entry.slot_count,
                extras: entry.extra_start..entry.extra_start + entry.extra_count,
                foreign: entry.foreign_start..entry.foreign_start + entry.foreign_count,
            });
        }
        for (label, cursor, total) in [
            ("slot", slots, header.slot_count),
            ("extra", extras, header.extra_count),
            ("foreign", foreign, header.foreign_count),
        ] {
            if cursor != total {
                return Err(format!(
                    "the books' {label} ranges cover {cursor}, and the segment holds {total}"
                ));
            }
        }

        let segment = Self {
            path: path.to_path_buf(),
            kind,
            codec,
            key_len,
            books: books.into_boxed_slice(),
            hints: range(&hints_entry),
            keys: range(&keys_entry),
            vectors: range(&vectors_entry),
            scales: scales_entry.as_ref().map(range),
            windows: windows_entry.as_ref().map(range),
            occurrences: sections.get(&SectionKind::Occurrences).map(range),
            book_hashes: sections.get(&SectionKind::BookHashes).map(range),
            extras: range(&extras_entry),
            extras_by_slot: range(&by_slot_entry),
            foreign: range(&foreign_entry),
            tombstones: range(&tombstones_entry),
            block_crcs: range(&crcs_entry),
            blocksummed,
            header,
            map,
        };
        segment.check_records()?;
        let mut last = None;
        for occurrence in segment.occurrences() {
            if occurrence.book as usize >= segment.books.len() || occurrence.hint > HINT_MASK {
                return Err("an occurrence names an invalid book or hint".to_string());
            }
            if occurrence.window == 0 && occurrence.anchor != occurrence.key {
                return Err("a first-window occurrence must name its own anchor".to_string());
            }
            if last.is_some_and(|last| last >= occurrence) {
                return Err("occurrences must ascend strictly".to_string());
            }
            last = Some(occurrence);
        }
        Ok(segment)
    }

    /// The small record sections against each other: every extra names a slot that exists
    /// and is listed once by EXTRAS_BY_SLOT in slot order, every foreign record names its
    /// own book, and the tombstones are sorted.
    fn check_records(&self) -> Result<(), String> {
        let slots = self.slot_count();
        let extras = self.extra_count();
        for book in self.books.iter() {
            let mut previous_hint = None;
            for index in book.extras.clone() {
                let (slot, hint) = self.extra(index);
                if slot >= slots || hint > HINT_MASK {
                    return Err(format!("extra {index} names slot {slot} or hint {hint}"));
                }
                if previous_hint.is_some_and(|previous| hint < previous) {
                    return Err(format!(
                        "book {:?}'s extras are not in hint order",
                        book.name
                    ));
                }
                previous_hint = Some(hint);
            }
        }
        let mut seen = vec![false; extras as usize];
        let mut previous_slot = 0;
        for position in 0..extras {
            let index = self.extra_by_slot(position);
            let slot_seen = seen
                .get_mut(index as usize)
                .ok_or_else(|| format!("EXTRAS_BY_SLOT names extra {index} of {extras}"))?;
            if std::mem::replace(slot_seen, true) {
                return Err(format!("EXTRAS_BY_SLOT names extra {index} twice"));
            }
            let (slot, _) = self.extra(index);
            if slot < previous_slot {
                return Err("EXTRAS_BY_SLOT is not in slot order".into());
            }
            previous_slot = slot;
        }
        for (book_index, book) in self.books.iter().enumerate() {
            for index in book.foreign.clone() {
                let (_, book, hint) = self.foreign_record(index);
                if book as usize != book_index || hint > HINT_MASK {
                    return Err(format!("foreign record {index} names book {book}"));
                }
            }
        }
        let tombstones: Vec<&[u8]> = self.map[self.tombstones.clone()]
            .chunks_exact(self.key_len)
            .collect();
        if tombstones.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("its tombstones are not sorted and unique".into());
        }
        Ok(())
    }

    /// Read every block-summed section and compare each 1 MiB block with its CRC: the
    /// check [`Self::open`] leaves out, run at install and by a scrub. `progress` hears the
    /// bytes checked after each block; `cancel` is looked at before each one.
    pub fn verify_blocks(
        &self,
        cancel: &CancellationToken,
        mut progress: impl FnMut(u64),
    ) -> Result<(), VectorStoreError> {
        let crcs = &self.map[self.block_crcs.clone()];
        let mut index = 0usize;
        for (kind, section) in &self.blocksummed {
            for (block_number, block) in self.map[section.clone()].chunks(BLOCK_SIZE).enumerate() {
                if cancel.is_cancelled() {
                    return Err(VectorStoreError::Cancelled);
                }
                let expected = le_u32(crcs, index * 4);
                let actual = crc32fast::hash(block);
                if actual != expected {
                    return Err(VectorStoreError::Corrupted {
                        reason: format!(
                            "segment {}: block {block_number} of {} hashes to CRC {actual:08x}, \
                             and the segment declares {expected:08x}",
                            self.path.display(),
                            kind.name()
                        ),
                    });
                }
                index += 1;
                progress(block.len() as u64);
            }
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn kind(&self) -> PackageKind {
        self.kind
    }

    pub fn codec(&self) -> &Codec {
        &self.codec
    }

    /// The segment's format: 2, or 3 for 8-byte keys.
    pub fn format_version(&self) -> u32 {
        self.header.format_version
    }

    /// The bytes of a key as the segment stores it: 16, or 8 in format 3.
    pub fn key_len(&self) -> usize {
        self.key_len
    }

    pub fn segment_id(&self) -> [u8; 16] {
        self.header.segment_id
    }

    pub fn identity_digest(&self) -> [u8; 32] {
        self.header.identity_digest
    }

    pub fn codec_params_sha256(&self) -> [u8; 32] {
        self.header.codec_params_sha256
    }

    pub fn from_library_version(&self) -> u32 {
        self.header.from_library_version
    }

    pub fn to_library_version(&self) -> u32 {
        self.header.to_library_version
    }

    pub fn library_release_tag(&self) -> &str {
        &self.header.library_release_tag
    }

    pub fn counts(&self) -> PackageCounts {
        PackageCounts {
            books: self.header.book_count,
            slots: self.header.slot_count,
            extras: self.header.extra_count,
            foreign: self.header.foreign_count,
            tombstones: self.header.tombstone_count,
        }
    }

    /// The file's length.
    pub fn size(&self) -> u64 {
        self.map.len() as u64
    }

    /// Every byte of the file, as mapped when it was opened — whatever its path holds now.
    pub(crate) fn file_bytes(&self) -> &[u8] {
        &self.map
    }

    pub fn slot_count(&self) -> u32 {
        self.header.slot_count as u32
    }

    pub fn extra_count(&self) -> u32 {
        self.header.extra_count as u32
    }

    pub fn foreign_count(&self) -> u32 {
        self.header.foreign_count as u32
    }

    pub fn tombstone_count(&self) -> u32 {
        self.header.tombstone_count as u32
    }

    /// The books, in name order.
    pub fn books(&self) -> &[SegmentBook] {
        &self.books
    }

    /// The book whose primary range holds `slot`.
    pub fn book_of_slot(&self, slot: u32) -> usize {
        self.books.partition_point(|book| book.slots.end <= slot)
    }

    /// The book whose extra range holds extra `index`.
    pub fn book_of_extra(&self, index: u32) -> usize {
        self.books.partition_point(|book| book.extras.end <= index)
    }

    /// Slot `slot`'s key; in format 3, its eight bytes as [`ChunkKey::narrowed`]. Compare
    /// it with a computed key by [`ChunkKey::same_text`].
    pub fn key(&self, slot: u32) -> ChunkKey {
        self.key_at(self.keys.start + slot as usize * self.key_len)
    }

    /// Every slot's key, as [`Self::key`] reads it, in slot order.
    pub fn keys(&self) -> impl Iterator<Item = ChunkKey> + '_ {
        self.map[self.keys.clone()]
            .chunks_exact(self.key_len)
            .map(|bytes| {
                let mut key = [0u8; 16];
                key[..bytes.len()].copy_from_slice(bytes);
                ChunkKey(key)
            })
    }

    /// Every slot's key, as the section stores them: [`Self::key_len`] bytes each, in slot
    /// order.
    pub fn key_bytes(&self) -> &[u8] {
        &self.map[self.keys.clone()]
    }

    fn key_at(&self, at: usize) -> ChunkKey {
        let mut key = [0u8; 16];
        key[..self.key_len].copy_from_slice(&self.map[at..at + self.key_len]);
        ChunkKey(key)
    }

    /// The record ordinal a slot was built at.
    pub fn hint(&self, slot: u32) -> u32 {
        self.raw_hint(slot) & HINT_MASK
    }

    /// Whether the slot's key has records in [`Self::extras_of_slot`].
    pub fn has_extras(&self, slot: u32) -> bool {
        self.raw_hint(slot) & HAS_EXTRAS != 0
    }

    fn raw_hint(&self, slot: u32) -> u32 {
        le_u32(&self.map, self.hints.start + slot as usize * 4)
    }

    /// The slot's encoded vector.
    pub fn vector(&self, slot: u32) -> &[u8] {
        let width = self.codec.bytes_per_vector();
        let at = self.vectors.start + slot as usize * width;
        &self.map[at..at + width]
    }

    /// Slot `slot`'s own scale, for a codec with one per vector.
    pub fn vector_scale(&self, slot: u32) -> Option<f32> {
        let scales = self.scales.as_ref()?;
        let at = scales.start + slot as usize * 4;
        Some(f32::from_le_bytes(
            self.map[at..at + 4].try_into().expect("four bytes"),
        ))
    }

    /// Every slot's scale, little-endian `f32`s in slot order, for a codec with them.
    pub fn vector_scale_bytes(&self) -> Option<&[u8]> {
        self.scales.as_ref().map(|scales| &self.map[scales.clone()])
    }

    /// The least window ordinal of the slot's key: 0 for a line's first window, and for
    /// every slot of a segment without SLOT_WINDOWS.
    /// Source text hash of this book, or None for older/unknown metadata.
    pub fn book_source_hash(&self, name: &str) -> Option<u64> {
        let index = self
            .books
            .binary_search_by(|book| book.name.as_ref().cmp(name))
            .ok()?;
        let range = self.book_hashes.as_ref()?;
        let at = range.start + index * 8;
        let hash = u64::from_le_bytes(self.map[at..at + 8].try_into().expect("book hash"));
        (hash != 0).then_some(hash)
    }

    pub fn occurrences(&self) -> impl Iterator<Item = super::occurrence::Occurrence> + '_ {
        let key_len = self.key_len;
        self.occurrences
            .as_ref()
            .map_or(&[][..], |range| &self.map[range.clone()])
            .chunks_exact(occurrence_len(self.header.format_version))
            .map(move |row| super::occurrence::Occurrence::decode_with(row, key_len))
    }

    /// A complete snapshot for this key, or None when this segment does not update it.
    pub fn key_occurrences(
        &self,
        key: ChunkKey,
    ) -> Option<impl Iterator<Item = super::occurrence::Occurrence> + '_> {
        let bytes = self
            .occurrences
            .as_ref()
            .map(|range| &self.map[range.clone()])?;
        let key_len = self.key_len;
        // A key of the other width finds nothing: the set holds its keys at one width.
        if key_len < 16 && !key.is_narrow() {
            return None;
        }
        let row_len = occurrence_len(self.header.format_version);
        let row = move |index: usize| &bytes[index * row_len..(index + 1) * row_len];
        let count = bytes.len() / row_len;
        let wanted = key.0;
        let start = partition(count, |index| row(index)[..key_len] < wanted[..key_len]);
        let end = partition(count, |index| row(index)[..key_len] <= wanted[..key_len]);
        (end > start).then(move || {
            (start..end)
                .map(move |index| super::occurrence::Occurrence::decode_with(row(index), key_len))
        })
    }

    pub fn slot_window(&self, slot: u32) -> u8 {
        self.windows
            .as_ref()
            .map_or(0, |windows| self.map[windows.start + slot as usize])
    }

    /// Every slot's window, in slot order, when the segment has SLOT_WINDOWS.
    pub fn slot_window_bytes(&self) -> Option<&[u8]> {
        self.windows
            .as_ref()
            .map(|windows| &self.map[windows.clone()])
    }

    /// Every encoded vector, in slot order.
    pub fn vector_bytes(&self) -> &[u8] {
        &self.map[self.vectors.clone()]
    }

    /// Extra record `index`: `(slot, hint)`.
    pub fn extra(&self, index: u32) -> (u32, u32) {
        let at = self.extras.start + index as usize * EXTRA_LEN;
        (le_u32(&self.map, at), le_u32(&self.map, at + 4))
    }

    fn extra_by_slot(&self, position: u32) -> u32 {
        le_u32(&self.map, self.extras_by_slot.start + position as usize * 4)
    }

    /// The extra records of one slot's key, as indexes into the extras, in book order.
    pub fn extras_of_slot(&self, slot: u32) -> impl Iterator<Item = u32> + '_ {
        let count = self.extra_count();
        // The first position whose extra names a slot at or past `slot`.
        let (mut low, mut high) = (0u32, count);
        while low < high {
            let middle = low + (high - low) / 2;
            if self.extra(self.extra_by_slot(middle)).0 < slot {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        (low..count)
            .map(|position| self.extra_by_slot(position))
            .take_while(move |index| self.extra(*index).0 == slot)
    }

    /// Foreign record `index`: `(key, book, hint)`.
    pub fn foreign_record(&self, index: u32) -> (ChunkKey, u32, u32) {
        let at = self.foreign.start + index as usize * foreign_len(self.header.format_version);
        (
            self.key_at(at),
            le_u32(&self.map, at + self.key_len),
            le_u32(&self.map, at + self.key_len + 4),
        )
    }

    /// Tombstone `index`, in key order.
    pub fn tombstone(&self, index: u32) -> ChunkKey {
        self.key_at(self.tombstones.start + index as usize * self.key_len)
    }
}

/// Why a file that is a sound segment — its magic and its header's CRC — cannot be read by
/// this build: a format or a codec newer than it. `None` for anything else, which
/// [`Segment::check`] judges.
fn unsupported(bytes: &[u8]) -> Option<String> {
    if bytes.len() < HEADER_LEN
        || bytes[..8] != MAGIC
        || le_u32(bytes, HEADER_CRC_OFFSET) != crc32fast::hash(&bytes[..HEADER_CRC_OFFSET])
    {
        return None;
    }
    let version = le_u32(bytes, 8);
    if !READABLE_FORMATS.contains(&version) {
        return Some(format!(
            "it is segment format {version}, which this build does not read: it reads \
             {READABLE_FORMATS:?}"
        ));
    }
    let codec = le_u16(bytes, 22);
    crate::semantic::oxv::codec::codec_name(codec)
        .is_none()
        .then(|| format!("its codec {codec:#06x} is not one this build implements"))
}

/// The first index in `0..count` at which `before` is false, for a `before` that is true
/// and then false.
fn partition(count: usize, before: impl Fn(usize) -> bool) -> usize {
    let (mut low, mut high) = (0, count);
    while low < high {
        let middle = low + (high - low) / 2;
        if before(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn range(entry: &DirectoryEntry) -> Range<usize> {
    entry.offset as usize..(entry.offset + entry.length) as usize
}
