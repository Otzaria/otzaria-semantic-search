//! The key a vector is stored under: what was embedded, not where it sits.
//!
//! A vector is a function of the model, the recipe and the exact string that reached the
//! model — and of nothing positional. So the store addresses it by that string: a
//! [`ChunkKey`] is the first 128 bits of the SHA-256 of the embedded text, the same digest
//! [`compute_chunk_hash`](crate::semantic::chunker::compute_chunk_hash) has always written
//! into every record, now in binary. A line keeps its vector across a library update for as
//! long as the text the recipe builds for it is unchanged, whichever id, book position or
//! catalogue order it has by then.
//!
//! Both sides compute the key with the same code, and that is the contract:
//!
//! * the build machine, through [`Chunker::chunk_book`](crate::semantic::chunker::Chunker::chunk_book),
//!   when it decides which vectors exist;
//! * the application's index, through [`Chunker::chunk_keys`](crate::semantic::chunker::Chunker::chunk_keys)
//!   over the lines it stores, and through
//!   [`Chunker::embedded_text`](crate::semantic::chunker::Chunker::embedded_text) when it
//!   re-checks one line from its stored text and its ±2 neighbours.
//!
//! [`KEY_VERSION`] versions the function. A change to it — another digest, another width,
//! another input — is a change to every key, and is refused by identity rather than
//! discovered as an index in which nothing resolves.

use sha2::{Digest, Sha256};
use std::fmt;

/// Version of the key function: [`ChunkKey::of`] over the string the recipe embeds.
///
/// Carried as `text.key_version` in every vector set's identity, beside the line-text
/// version of the index the keys are computed against.
pub const KEY_VERSION: u32 = 1;

/// The first 16 bytes of the SHA-256 of an embedded text.
///
/// Ordered bytewise, which is the order a segment's tombstones are sorted in and the order
/// ties between equal scores are broken in — so two devices holding the same vectors in
/// different segment layouts rank them identically.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkKey(pub [u8; 16]);

impl ChunkKey {
    /// The key of `embedded_text` — the exact string the model is given, role prefix
    /// included. `to_hex()` is what [`compute_chunk_hash`](crate::semantic::chunker::compute_chunk_hash)
    /// returns for the same text.
    pub fn of(embedded_text: &str) -> Self {
        let digest = Sha256::digest(embedded_text.as_bytes());
        let mut key = [0u8; 16];
        key.copy_from_slice(&digest[..16]);
        Self(key)
    }

    /// The value the application's index stores for a line in its `chunkKey` column: the
    /// key's first eight bytes, big-endian, so the column orders as the keys do.
    ///
    /// `0` is reserved there for "this line is not embedded". A key whose first eight bytes
    /// are zero would read as that — once in 2⁶⁴ keys, and harmlessly: every displayed
    /// result is verified against the full 128 bits.
    pub fn column_value(self) -> u64 {
        u64::from_be_bytes(self.0[..8].try_into().expect("a key holds 16 bytes"))
    }

    /// 32 lowercase hex digits.
    pub fn to_hex(self) -> String {
        use fmt::Write;
        let mut hex = String::with_capacity(32);
        for byte in self.0 {
            let _ = write!(hex, "{byte:02x}");
        }
        hex
    }

    /// Read 32 hex digits back, either case. `None` for anything else.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 32 || !hex.is_ascii() {
            return None;
        }
        let mut key = [0u8; 16];
        let (pairs, _) = hex.as_bytes().as_chunks::<2>();
        for (byte, pair) in key.iter_mut().zip(pairs) {
            let pair = std::str::from_utf8(pair).ok()?;
            *byte = u8::from_str_radix(pair, 16).ok()?;
        }
        Some(Self(key))
    }
}

impl fmt::Debug for ChunkKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChunkKey({})", self.to_hex())
    }
}

impl fmt::Display for ChunkKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// One line as the recipe sees it: its text, and which section it belongs to.
///
/// The text is the line as the index stores it — normalized by the application, never by
/// this crate. Only *equality* of `section` matters: a short line borrows context from
/// neighbours with the same value and from no others, so any numbering that gives the lines
/// of one heading block one value, and the next block another, describes the same book.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRef<'a> {
    pub text: &'a str,
    pub section: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Digests computed outside this crate, with Python's hashlib, so neither the function
    /// nor its expected values can drift with the code.
    #[test]
    fn a_key_is_the_sha256_prefix_python_computes() {
        let key = ChunkKey::of("[PASSAGE] a line that stands alone");
        assert_eq!(key.to_hex(), "0429557a367f9f5e6681647f940ff66e");
        assert_eq!(key.column_value(), 299_864_833_585_553_246);

        let key = ChunkKey::of("a line that stands alone");
        assert_eq!(key.to_hex(), "d5e703cd670772730f9f85bd4e17353c");
        assert_eq!(key.column_value(), 15_413_292_430_430_532_211);
    }

    #[test]
    fn the_column_orders_as_the_keys_do() {
        let low = ChunkKey([0x00, 0xff, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9]);
        let high = ChunkKey([0x01, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(low < high);
        assert!(low.column_value() < high.column_value());
    }

    #[test]
    fn hex_round_trips_and_refuses_what_is_not_a_key() {
        let key = ChunkKey::of("שורה");
        assert_eq!(ChunkKey::from_hex(&key.to_hex()), Some(key));
        assert_eq!(ChunkKey::from_hex(&key.to_hex().to_uppercase()), Some(key));
        for bad in ["", "abc", &"g".repeat(32), &"a".repeat(31), &"a".repeat(33)] {
            assert_eq!(ChunkKey::from_hex(bad), None, "{bad:?}");
        }
        // Two bytes of UTF-8 that would pass a length check made of characters.
        assert_eq!(ChunkKey::from_hex(&format!("{}é", "a".repeat(30))), None);
        assert_eq!(format!("{key}"), key.to_hex());
        assert_eq!(format!("{key:?}"), format!("ChunkKey({})", key.to_hex()));
    }
}
