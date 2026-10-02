//! The bytes of a segment file: its header, its section directory and the fixed-size
//! records its sections hold. Nothing here reads a file — [`super::reader`] maps one and
//! [`super::writer`] writes one, and both go through these definitions, so the two cannot
//! disagree about an offset.
//!
//! Every multi-byte field is little-endian, and every offset is from the start of the file.

use crate::distribution::package::PackageKind;

/// `OXVSEG\r\n`: the CR-LF catches a file that went through a text-mode transfer.
pub(crate) const MAGIC: [u8; 8] = *b"OXVSEG\r\n";

/// The segment format this build reads and writes — `store.store_format_version`.
pub const SEGMENT_FORMAT_VERSION: u32 = 2;

pub(crate) const HEADER_LEN: usize = 4096;

/// Every section that carries per-block checksums is cut into blocks of this size.
pub const BLOCK_SIZE: usize = 1 << 20;

/// The directory has room for this many sections.
pub(crate) const MAX_SECTIONS: usize = 32;

const DIRECTORY_OFFSET: usize = 224;
const DIRECTORY_ENTRY_LEN: usize = 40;
const HEADER_CRC_OFFSET: usize = 4088;
const RELEASE_TAG_OFFSET: usize = 160;
pub(crate) const RELEASE_TAG_LEN: usize = 64;

/// Every section starts on a cache line, so each fixed-size record can be read in place.
pub(crate) const SECTION_ALIGN: u64 = 64;
/// VECTORS starts on a page, so a 256-byte vector never straddles one.
pub(crate) const VECTORS_ALIGN: u64 = 4096;

/// Bit 31 of a slot's hint: the slot's key has further records in [`SectionKind::Extras`].
pub(crate) const HAS_EXTRAS: u32 = 1 << 31;
/// The ordinal half of a hint.
pub(crate) const HINT_MASK: u32 = !HAS_EXTRAS;

// Header flags.
pub(crate) const FLAG_BASE: u32 = 1 << 0;
pub(crate) const FLAG_DELTA: u32 = 1 << 1;
pub(crate) const FLAG_COMPACTED: u32 = 1 << 2;
pub(crate) const FLAG_HAS_FOREIGN: u32 = 1 << 3;
pub(crate) const FLAG_HAS_TOMBSTONES: u32 = 1 << 4;
const KNOWN_FLAGS: u32 =
    FLAG_BASE | FLAG_DELTA | FLAG_COMPACTED | FLAG_HAS_FOREIGN | FLAG_HAS_TOMBSTONES;

// Directory entry flags.
/// A reader that does not know the section's kind must refuse the file.
pub(crate) const SECTION_CRITICAL: u16 = 1 << 0;
/// The section is checked by the per-block CRCs in [`SectionKind::BlockCrcs`] rather than by
/// one CRC of its own.
pub(crate) const SECTION_BLOCKSUMMED: u16 = 1 << 1;

/// What a section holds. The numbering is the format's; a reader ignores an ancillary kind
/// it does not know and refuses a critical one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum SectionKind {
    /// 40-byte [`BookEntry`]s, sorted by name bytes.
    Books,
    /// The UTF-8 names the book entries point into.
    BookNames,
    /// The codec's parameters, whose SHA-256 the header carries.
    CodecParams,
    /// One 16-byte key per slot.
    SlotKeys,
    /// One `u32` per slot: the record's ordinal, with [`HAS_EXTRAS`] in bit 31.
    SlotHints,
    /// One encoded vector per slot.
    Vectors,
    /// One `f32` per slot, for a codec that scales each vector. Reserved.
    VectorScales,
    /// `{slot u32, hint u32}`: a further record of a slot's key, in some book.
    Extras,
    /// `u32` indexes into [`Self::Extras`], ordered by slot.
    ExtrasBySlot,
    /// `{key [16], book u32, hint u32}`: a record whose vector an older segment holds.
    Foreign,
    /// Sorted 16-byte keys the library no longer holds.
    Tombstones,
    /// One sign bit per dimension, for a future first-pass scan. Ancillary, not produced.
    Tier0Sign,
    /// Product-quantization codes, likewise. Ancillary, not produced.
    Tier0Pq,
    /// One CRC-32 per block of every block-summed section, in directory order.
    BlockCrcs,
}

impl SectionKind {
    pub(crate) fn code(self) -> u16 {
        match self {
            Self::Books => 0x0001,
            Self::BookNames => 0x0002,
            Self::CodecParams => 0x0003,
            Self::SlotKeys => 0x0010,
            Self::SlotHints => 0x0011,
            Self::Vectors => 0x0012,
            Self::VectorScales => 0x0013,
            Self::Extras => 0x0020,
            Self::ExtrasBySlot => 0x0021,
            Self::Foreign => 0x0022,
            Self::Tombstones => 0x0023,
            Self::Tier0Sign => 0x0030,
            Self::Tier0Pq => 0x0031,
            Self::BlockCrcs => 0x007F,
        }
    }

    pub(crate) fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            0x0001 => Self::Books,
            0x0002 => Self::BookNames,
            0x0003 => Self::CodecParams,
            0x0010 => Self::SlotKeys,
            0x0011 => Self::SlotHints,
            0x0012 => Self::Vectors,
            0x0013 => Self::VectorScales,
            0x0020 => Self::Extras,
            0x0021 => Self::ExtrasBySlot,
            0x0022 => Self::Foreign,
            0x0023 => Self::Tombstones,
            0x0030 => Self::Tier0Sign,
            0x0031 => Self::Tier0Pq,
            0x007F => Self::BlockCrcs,
            _ => return None,
        })
    }

    /// The bytes one element takes, for the sections made of fixed-size elements.
    pub(crate) fn fixed_elem_size(self) -> Option<u32> {
        match self {
            Self::Books => Some(BOOK_ENTRY_LEN as u32),
            Self::SlotKeys | Self::Tombstones => Some(16),
            Self::SlotHints | Self::ExtrasBySlot | Self::BlockCrcs | Self::VectorScales => Some(4),
            Self::Extras => Some(EXTRA_LEN as u32),
            Self::Foreign => Some(FOREIGN_LEN as u32),
            Self::BookNames
            | Self::CodecParams
            | Self::Vectors
            | Self::Tier0Sign
            | Self::Tier0Pq => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Books => "BOOKS",
            Self::BookNames => "BOOK_NAMES",
            Self::CodecParams => "CODEC_PARAMS",
            Self::SlotKeys => "SLOT_KEYS",
            Self::SlotHints => "SLOT_HINTS",
            Self::Vectors => "VECTORS",
            Self::VectorScales => "VECTOR_SCALES",
            Self::Extras => "EXTRAS",
            Self::ExtrasBySlot => "EXTRAS_BY_SLOT",
            Self::Foreign => "FOREIGN",
            Self::Tombstones => "TOMBSTONES",
            Self::Tier0Sign => "TIER0_SIGN",
            Self::Tier0Pq => "TIER0_PQ",
            Self::BlockCrcs => "BLOCK_CRCS",
        }
    }
}

/// One entry of the section directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectoryEntry {
    /// The raw kind, kept even when this build does not know it: an unknown ancillary
    /// section is skipped, not forgotten.
    pub kind: u16,
    pub flags: u16,
    pub elem_size: u32,
    pub offset: u64,
    pub length: u64,
    pub elem_count: u64,
    /// CRC-32 of the section's bytes, or 0 for a block-summed section.
    pub crc32: u32,
}

impl DirectoryEntry {
    pub(crate) fn known_kind(&self) -> Option<SectionKind> {
        SectionKind::from_code(self.kind)
    }

    pub(crate) fn is_critical(&self) -> bool {
        self.flags & SECTION_CRITICAL != 0
    }

    pub(crate) fn is_blocksummed(&self) -> bool {
        self.flags & SECTION_BLOCKSUMMED != 0
    }

    pub(crate) fn end(&self) -> u64 {
        self.offset.saturating_add(self.length)
    }

    /// How many blocks the per-block CRCs cut this section into.
    pub(crate) fn block_count(&self) -> u64 {
        self.length.div_ceil(BLOCK_SIZE as u64)
    }

    fn encode(&self, out: &mut [u8]) {
        out[0..2].copy_from_slice(&self.kind.to_le_bytes());
        out[2..4].copy_from_slice(&self.flags.to_le_bytes());
        out[4..8].copy_from_slice(&self.elem_size.to_le_bytes());
        out[8..16].copy_from_slice(&self.offset.to_le_bytes());
        out[16..24].copy_from_slice(&self.length.to_le_bytes());
        out[24..32].copy_from_slice(&self.elem_count.to_le_bytes());
        out[32..36].copy_from_slice(&self.crc32.to_le_bytes());
        out[36..40].fill(0);
    }

    fn decode(bytes: &[u8]) -> Self {
        Self {
            kind: le_u16(bytes, 0),
            flags: le_u16(bytes, 2),
            elem_size: le_u32(bytes, 4),
            offset: le_u64(bytes, 8),
            length: le_u64(bytes, 16),
            elem_count: le_u64(bytes, 24),
            crc32: le_u32(bytes, 32),
        }
    }
}

/// The first 4,096 bytes of a segment, parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Header {
    pub flags: u32,
    pub dim: u16,
    pub codec_id: u16,
    pub block_size: u32,
    pub segment_id: [u8; 16],
    pub identity_digest: [u8; 32],
    pub codec_params_sha256: [u8; 32],
    pub slot_count: u64,
    pub extra_count: u64,
    pub foreign_count: u64,
    pub tombstone_count: u64,
    pub book_count: u32,
    pub from_library_version: u32,
    pub to_library_version: u32,
    pub library_release_tag: String,
    pub sections: Vec<DirectoryEntry>,
}

impl Header {
    /// Which of base, delta and compacted the flags say — `None` unless exactly one.
    pub(crate) fn kind(&self) -> Option<PackageKind> {
        match self.flags & (FLAG_BASE | FLAG_DELTA | FLAG_COMPACTED) {
            FLAG_BASE => Some(PackageKind::Base),
            FLAG_DELTA => Some(PackageKind::Delta),
            FLAG_COMPACTED => Some(PackageKind::Compacted),
            _ => None,
        }
    }

    pub(crate) fn kind_flag(kind: PackageKind) -> u32 {
        match kind {
            PackageKind::Base => FLAG_BASE,
            PackageKind::Delta => FLAG_DELTA,
            PackageKind::Compacted => FLAG_COMPACTED,
        }
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        assert!(
            self.sections.len() <= MAX_SECTIONS,
            "the directory holds 32"
        );
        let mut out = vec![0u8; HEADER_LEN];
        out[0..8].copy_from_slice(&MAGIC);
        out[8..12].copy_from_slice(&SEGMENT_FORMAT_VERSION.to_le_bytes());
        out[12..16].copy_from_slice(&(HEADER_LEN as u32).to_le_bytes());
        out[16..20].copy_from_slice(&self.flags.to_le_bytes());
        out[20..22].copy_from_slice(&self.dim.to_le_bytes());
        out[22..24].copy_from_slice(&self.codec_id.to_le_bytes());
        out[24..28].copy_from_slice(&self.block_size.to_le_bytes());
        out[28..32].copy_from_slice(&(self.sections.len() as u32).to_le_bytes());
        out[32..48].copy_from_slice(&self.segment_id);
        out[48..80].copy_from_slice(&self.identity_digest);
        out[80..112].copy_from_slice(&self.codec_params_sha256);
        out[112..120].copy_from_slice(&self.slot_count.to_le_bytes());
        out[120..128].copy_from_slice(&self.extra_count.to_le_bytes());
        out[128..136].copy_from_slice(&self.foreign_count.to_le_bytes());
        out[136..144].copy_from_slice(&self.tombstone_count.to_le_bytes());
        out[144..148].copy_from_slice(&self.book_count.to_le_bytes());
        out[148..152].copy_from_slice(&self.from_library_version.to_le_bytes());
        out[152..156].copy_from_slice(&self.to_library_version.to_le_bytes());
        let tag = self.library_release_tag.as_bytes();
        assert!(tag.len() <= RELEASE_TAG_LEN, "a release tag fits 64 bytes");
        out[RELEASE_TAG_OFFSET..RELEASE_TAG_OFFSET + tag.len()].copy_from_slice(tag);
        for (index, entry) in self.sections.iter().enumerate() {
            let at = DIRECTORY_OFFSET + index * DIRECTORY_ENTRY_LEN;
            entry.encode(&mut out[at..at + DIRECTORY_ENTRY_LEN]);
        }
        let crc = crc32fast::hash(&out[..HEADER_CRC_OFFSET]);
        out[HEADER_CRC_OFFSET..HEADER_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        out
    }

    /// Parse and check what can be checked from the header alone: the magic, the format
    /// version, the header's own CRC, the flags and the directory's size. The directory's
    /// *contents* are the reader's to check, against the file they describe.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_LEN {
            return Err(format!(
                "the file holds {} byte(s), fewer than a header's {HEADER_LEN}",
                bytes.len()
            ));
        }
        let bytes = &bytes[..HEADER_LEN];
        if bytes[0..8] != MAGIC {
            return Err("it does not start with the segment magic".to_string());
        }
        let version = le_u32(bytes, 8);
        if version != SEGMENT_FORMAT_VERSION {
            return Err(format!(
                "it is segment format {version}, and this build reads {SEGMENT_FORMAT_VERSION}"
            ));
        }
        let header_len = le_u32(bytes, 12);
        if header_len as usize != HEADER_LEN {
            return Err(format!(
                "its header is {header_len} bytes, not {HEADER_LEN}"
            ));
        }
        let stored = le_u32(bytes, HEADER_CRC_OFFSET);
        let actual = crc32fast::hash(&bytes[..HEADER_CRC_OFFSET]);
        if stored != actual {
            return Err(format!(
                "its header CRC is {actual:08x}, and the header declares {stored:08x}"
            ));
        }
        let flags = le_u32(bytes, 16);
        if flags & !KNOWN_FLAGS != 0 {
            return Err(format!("its header carries unknown flags {flags:#x}"));
        }
        let section_count = le_u32(bytes, 28) as usize;
        if section_count > MAX_SECTIONS {
            return Err(format!(
                "its directory claims {section_count} sections, and holds at most {MAX_SECTIONS}"
            ));
        }
        let tag_bytes = &bytes[RELEASE_TAG_OFFSET..RELEASE_TAG_OFFSET + RELEASE_TAG_LEN];
        let tag_len = tag_bytes
            .iter()
            .position(|b| *b == 0)
            .unwrap_or(RELEASE_TAG_LEN);
        let library_release_tag = std::str::from_utf8(&tag_bytes[..tag_len])
            .map_err(|_| "its release tag is not UTF-8".to_string())?
            .to_string();
        let sections = (0..section_count)
            .map(|index| {
                let at = DIRECTORY_OFFSET + index * DIRECTORY_ENTRY_LEN;
                DirectoryEntry::decode(&bytes[at..at + DIRECTORY_ENTRY_LEN])
            })
            .collect();
        let header = Self {
            flags,
            dim: le_u16(bytes, 20),
            codec_id: le_u16(bytes, 22),
            block_size: le_u32(bytes, 24),
            segment_id: bytes[32..48].try_into().expect("16 bytes"),
            identity_digest: bytes[48..80].try_into().expect("32 bytes"),
            codec_params_sha256: bytes[80..112].try_into().expect("32 bytes"),
            slot_count: le_u64(bytes, 112),
            extra_count: le_u64(bytes, 120),
            foreign_count: le_u64(bytes, 128),
            tombstone_count: le_u64(bytes, 136),
            book_count: le_u32(bytes, 144),
            from_library_version: le_u32(bytes, 148),
            to_library_version: le_u32(bytes, 152),
            library_release_tag,
            sections,
        };
        if header.kind().is_none() {
            return Err(format!(
                "its flags {flags:#x} name no single kind of segment"
            ));
        }
        Ok(header)
    }
}

/// One book of a segment: where its name is, and where its records are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct BookEntry {
    pub name_off: u32,
    pub name_len: u16,
    pub flags: u16,
    pub slot_start: u32,
    pub slot_count: u32,
    pub extra_start: u32,
    pub extra_count: u32,
    pub foreign_start: u32,
    pub foreign_count: u32,
}

pub(crate) const BOOK_ENTRY_LEN: usize = 40;
pub(crate) const EXTRA_LEN: usize = 8;
pub(crate) const FOREIGN_LEN: usize = 24;

impl BookEntry {
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.name_off.to_le_bytes());
        out.extend_from_slice(&self.name_len.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.slot_start.to_le_bytes());
        out.extend_from_slice(&self.slot_count.to_le_bytes());
        out.extend_from_slice(&self.extra_start.to_le_bytes());
        out.extend_from_slice(&self.extra_count.to_le_bytes());
        out.extend_from_slice(&self.foreign_start.to_le_bytes());
        out.extend_from_slice(&self.foreign_count.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
    }

    pub(crate) fn decode(bytes: &[u8]) -> Self {
        Self {
            name_off: le_u32(bytes, 0),
            name_len: le_u16(bytes, 4),
            flags: le_u16(bytes, 6),
            slot_start: le_u32(bytes, 8),
            slot_count: le_u32(bytes, 12),
            extra_start: le_u32(bytes, 16),
            extra_count: le_u32(bytes, 20),
            foreign_start: le_u32(bytes, 24),
            foreign_count: le_u32(bytes, 28),
        }
    }
}

pub(crate) fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
}

pub(crate) fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

pub(crate) fn le_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

pub(crate) fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Header {
        Header {
            flags: FLAG_DELTA | FLAG_HAS_TOMBSTONES,
            dim: 256,
            codec_id: 0x0101,
            block_size: BLOCK_SIZE as u32,
            segment_id: [7; 16],
            identity_digest: [8; 32],
            codec_params_sha256: [9; 32],
            slot_count: 5,
            extra_count: 4,
            foreign_count: 3,
            tombstone_count: 2,
            book_count: 1,
            from_library_version: 29,
            to_library_version: 30,
            library_release_tag: "v30-20260930120000".to_string(),
            sections: vec![DirectoryEntry {
                kind: SectionKind::Books.code(),
                flags: SECTION_CRITICAL,
                elem_size: 40,
                offset: 4096,
                length: 40,
                elem_count: 1,
                crc32: 0xDEAD_BEEF,
            }],
        }
    }

    #[test]
    fn a_header_round_trips_and_its_crc_covers_every_field() {
        let header = sample();
        let bytes = header.encode();
        assert_eq!(bytes.len(), HEADER_LEN);
        assert_eq!(Header::decode(&bytes).unwrap(), header);
        assert_eq!(header.kind(), Some(PackageKind::Delta));

        // Every byte the CRC covers, flipped one at a time, is a refusal.
        for at in (0..HEADER_CRC_OFFSET).step_by(7) {
            let mut damaged = bytes.clone();
            damaged[at] ^= 0x40;
            assert!(Header::decode(&damaged).is_err(), "byte {at}");
        }
    }

    #[test]
    fn a_header_says_what_is_wrong_with_it() {
        let bytes = sample().encode();
        let reason = |bytes: &[u8]| Header::decode(bytes).unwrap_err();

        assert!(reason(&bytes[..100]).contains("fewer than a header"));
        let mut text_mode = bytes.clone();
        text_mode[6] = b'\n';
        assert!(reason(&text_mode).contains("magic"));

        let mut future = sample();
        future.flags = FLAG_BASE | FLAG_DELTA;
        assert!(reason(&future.encode()).contains("no single kind"));

        let mut version = bytes.clone();
        version[8] = 3;
        assert!(reason(&version).contains("segment format 3"));
    }

    #[test]
    fn section_codes_round_trip_and_alignment_rounds_up() {
        for kind in [
            SectionKind::Books,
            SectionKind::BookNames,
            SectionKind::CodecParams,
            SectionKind::SlotKeys,
            SectionKind::SlotHints,
            SectionKind::Vectors,
            SectionKind::VectorScales,
            SectionKind::Extras,
            SectionKind::ExtrasBySlot,
            SectionKind::Foreign,
            SectionKind::Tombstones,
            SectionKind::Tier0Sign,
            SectionKind::Tier0Pq,
            SectionKind::BlockCrcs,
        ] {
            assert_eq!(SectionKind::from_code(kind.code()), Some(kind));
        }
        assert_eq!(SectionKind::from_code(0x4242), None);
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);

        let entry = BookEntry {
            name_off: 1,
            name_len: 2,
            flags: 0,
            slot_start: 3,
            slot_count: 4,
            extra_start: 5,
            extra_count: 6,
            foreign_start: 7,
            foreign_count: 8,
        };
        let mut bytes = Vec::new();
        entry.encode(&mut bytes);
        assert_eq!(bytes.len(), BOOK_ENTRY_LEN);
        assert_eq!(BookEntry::decode(&bytes), entry);
    }
}
