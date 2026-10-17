//! Location metadata belongs to an occurrence, independently of its shared vector.
use crate::semantic::chunk_key::ChunkKey;
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
