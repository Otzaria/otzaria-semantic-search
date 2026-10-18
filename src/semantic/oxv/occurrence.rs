//! Location metadata belongs to an occurrence, independently of its shared vector.
use crate::semantic::chunk_key::ChunkKey;

/// The `window` of an alias occurrence: a short line that shares the vector of a neighbour
/// in its book, at `hint`, whose own key is `anchor`. In a row it is window 0 with
/// [`ROLE_ALIAS`] in the first reserved byte; segment format 3 only.
pub const ALIAS_WINDOW: u8 = u8::MAX;
/// The role byte of an alias row.
pub const ROLE_ALIAS: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Occurrence {
    pub key: ChunkKey,
    pub book: u32,
    pub hint: u32,
    pub anchor: ChunkKey,
    pub window: u8,
}
impl Occurrence {
    pub const SIZE: usize = 44;
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[..16].copy_from_slice(&self.key.0);
        bytes[16..20].copy_from_slice(&self.book.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.hint.to_le_bytes());
        bytes[24..40].copy_from_slice(&self.anchor.0);
        bytes[40] = self.window;
        bytes
    }
    /// The row a segment of key width `key_len` stores — [`Self::to_bytes`] at 16, the keys'
    /// first eight bytes at 8.
    pub fn encode_with(&self, key_len: usize, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.key.0[..key_len]);
        out.extend_from_slice(&self.book.to_le_bytes());
        out.extend_from_slice(&self.hint.to_le_bytes());
        out.extend_from_slice(&self.anchor.0[..key_len]);
        if self.is_alias() {
            out.extend_from_slice(&[0, ROLE_ALIAS, 0, 0]);
        } else {
            out.extend_from_slice(&[self.window, 0, 0, 0]);
        }
    }

    /// A row of key width `key_len`; an 8-byte key reads as [`ChunkKey::narrowed`].
    pub fn decode_with(bytes: &[u8], key_len: usize) -> Self {
        let key = |at: usize| {
            let mut key = [0u8; 16];
            key[..key_len].copy_from_slice(&bytes[at..at + key_len]);
            ChunkKey(key)
        };
        let at = key_len;
        Self {
            key: key(0),
            book: u32::from_le_bytes(bytes[at..at + 4].try_into().expect("book")),
            hint: u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("hint")),
            anchor: key(at + 8),
            window: if bytes[2 * key_len + 9] == ROLE_ALIAS {
                ALIAS_WINDOW
            } else {
                bytes[2 * key_len + 8]
            },
        }
    }

    /// Whether it is an alias: see [`ALIAS_WINDOW`]. Only a format-3 row's role byte reads
    /// as one: no recipe makes window 255, and a reader refuses a row whose window byte is
    /// 255, so a format-2 row never is one.
    pub fn is_alias(&self) -> bool {
        self.window == ALIAS_WINDOW
    }

    pub fn decode(bytes: &[u8]) -> Self {
        Self {
            key: ChunkKey(bytes[..16].try_into().expect("key")),
            book: u32::from_le_bytes(bytes[16..20].try_into().expect("book")),
            hint: u32::from_le_bytes(bytes[20..24].try_into().expect("hint")),
            anchor: ChunkKey(bytes[24..40].try_into().expect("anchor")),
            window: bytes[40],
        }
    }
}
