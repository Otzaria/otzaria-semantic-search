//! What the store's tests build segments from: temporary directories, deterministic unit
//! vectors, and a one-call segment writer.

use crate::distribution::package::PackageKind;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::codec::Codec;
use crate::semantic::oxv::writer::{SegmentBuilder, SegmentSpec, WrittenSegment};
use std::path::{Path, PathBuf};

pub(crate) struct TempDir(PathBuf);

impl TempDir {
    pub(crate) fn new(name: &str) -> Self {
        // The clock alone collided: macOS ticks coarser than a test takes to start.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "otzaria_oxv_{name}_{}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub(crate) fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// splitmix64, for vectors a failing test can be rerun on.
pub(crate) struct Random(pub u64);

impl Random {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    pub(crate) fn signed(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    /// A unit vector, roughly uniform on the sphere.
    pub(crate) fn unit(&mut self, dim: usize) -> Vec<f32> {
        loop {
            let vector: Vec<f32> = (0..dim).map(|_| self.signed()).collect();
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-3 {
                return vector.into_iter().map(|x| x / norm).collect();
            }
        }
    }
}

/// The key a test names `n` by.
pub(crate) fn key(n: u64) -> ChunkKey {
    ChunkKey::of(&format!("[PASSAGE] passage {n}"))
}

/// The identity digest every test segment declares unless it means to differ.
pub(crate) const TEST_IDENTITY: [u8; 32] = [0x5A; 32];

pub(crate) fn spec(kind: PackageKind, from: u32, to: u32) -> SegmentSpec {
    SegmentSpec {
        kind,
        identity_digest: TEST_IDENTITY,
        from_library_version: from,
        to_library_version: to,
        library_release_tag: format!("v{to}-20260930120000"),
    }
}

/// One book for [`write_segment`]: primary records with their vectors, extras as
/// `(hint, slot)`, foreign records as `(key, hint)`.
#[derive(Debug, Clone, Default)]
pub(crate) struct TestBook {
    pub name: String,
    pub primary: Vec<(ChunkKey, u32, Vec<f32>)>,
    pub extras: Vec<(u32, u32)>,
    pub foreign: Vec<(ChunkKey, u32)>,
}

impl TestBook {
    pub(crate) fn named(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Self::default()
        }
    }
}

/// Write `books` — already in name order — and `tombstones` into a new segment at `path`.
pub(crate) fn write_segment(
    path: &Path,
    spec: SegmentSpec,
    codec: Codec,
    books: &[TestBook],
    tombstones: &[ChunkKey],
) -> WrittenSegment {
    let mut builder = SegmentBuilder::new(spec, codec);
    let mut vectors = Vec::new();
    for book in books {
        let primary: Vec<(ChunkKey, u32)> = book.primary.iter().map(|(k, h, _)| (*k, *h)).collect();
        builder
            .add_book(&book.name, &primary, &book.extras, &book.foreign)
            .unwrap();
        vectors.extend(book.primary.iter().map(|(_, _, v)| v.clone()));
    }
    builder.set_tombstones(tombstones.to_vec());
    let mut sink = builder.write(path).unwrap();
    for vector in &vectors {
        sink.push_f32(vector).unwrap();
    }
    sink.finish().unwrap()
}

/// `count` books of `per_book` fresh slots each, every vector random — the shape of a
/// small base. Book `b`'s slot `i` has key `key(b · per_book + i + offset)` and hint `i`.
pub(crate) fn random_books(
    random: &mut Random,
    count: usize,
    per_book: usize,
    dim: usize,
    offset: u64,
) -> Vec<TestBook> {
    (0..count)
        .map(|b| TestBook {
            name: format!("id:{:05}", b),
            primary: (0..per_book)
                .map(|i| {
                    (
                        key(offset + (b * per_book + i) as u64),
                        i as u32,
                        random.unit(dim),
                    )
                })
                .collect(),
            ..TestBook::default()
        })
        .collect()
}
