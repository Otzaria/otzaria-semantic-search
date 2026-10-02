//! What every build file shares: digests, atomic writes, and the errors around them.

use crate::errors::PackError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::hash::{BuildHasher, Hasher};
use std::io::{self, Read, Write};
use std::path::Path;

/// A file a manifest names: its SHA-256 and its size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDigest {
    pub sha256: String,
    pub size: u64,
}

impl FileDigest {
    /// Hash `path`, streamed.
    pub fn of(path: &Path) -> Result<Self, PackError> {
        let mut file = File::open(path).map_err(io_error(format!("reading {}", path.display())))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        let mut size = 0u64;
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(io_error(format!("reading {}", path.display())))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            size += read as u64;
        }
        Ok(Self {
            sha256: format!("{:x}", hasher.finalize()),
            size,
        })
    }

    /// Refuse `path` unless it is this file.
    pub fn verify(&self, path: &Path) -> Result<(), PackError> {
        let actual = Self::of(path)?;
        if actual != *self {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "{} is {} bytes with SHA-256 {}, and its manifest declares {} bytes with {}",
                    path.display(),
                    actual.size,
                    actual.sha256,
                    self.size,
                    self.sha256
                ),
            });
        }
        Ok(())
    }
}

/// Write `bytes` under a temporary name, flush them through the handle that wrote them —
/// Windows refuses to flush a handle opened for reading — and rename them into place.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), PackError> {
    let partial = partial_path(path);
    (|| {
        let mut file = File::create(&partial)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&partial, path)?;
        match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                crate::distribution::package::sync_dir(parent)
            }
            _ => Ok(()),
        }
    })()
    .map_err(io_error(format!("writing {}", path.display())))
}

/// `name.partial`, beside `path`.
pub(crate) fn partial_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    path.with_file_name(name)
}

/// Serialize `value` as pretty JSON and write it atomically.
pub(crate) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), PackError> {
    let json = serde_json::to_vec_pretty(value).map_err(|error| PackError::MalformedInput {
        reason: format!("{} could not be serialized: {error}", path.display()),
    })?;
    write_atomically(path, &json)
}

/// Read a JSON document, naming the file in every failure.
pub(crate) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, PackError> {
    let bytes = std::fs::read(path).map_err(io_error(format!("reading {}", path.display())))?;
    serde_json::from_slice(&bytes).map_err(|error| PackError::MalformedInput {
        reason: format!(
            "{} is not the document this build reads: {error}",
            path.display()
        ),
    })
}

pub(crate) fn io_error(context: String) -> impl FnOnce(io::Error) -> PackError {
    move |source| PackError::Io { context, source }
}

pub(crate) fn malformed(reason: impl Into<String>) -> PackError {
    PackError::MalformedInput {
        reason: reason.into(),
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    crate::semantic::versioning::hex(bytes)
}

pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// A hasher for keys that are already uniformly random — SHA-256 prefixes — which takes
/// their first eight bytes as the hash rather than hashing them again.
#[derive(Clone, Copy, Default)]
pub(crate) struct KeyHash;

pub(crate) struct KeyHasher(u64);

impl BuildHasher for KeyHash {
    type Hasher = KeyHasher;
    fn build_hasher(&self) -> KeyHasher {
        KeyHasher(0)
    }
}

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        // Arrays hash their bytes in one call; the first eight are as random as any.
        let mut word = [0u8; 8];
        let take = bytes.len().min(8);
        word[..take].copy_from_slice(&bytes[..take]);
        self.0 ^= u64::from_le_bytes(word);
    }
    fn write_usize(&mut self, _: usize) {
        // The length prefix `Hash for [u8; N]` does not write; slices would, and add
        // nothing to a key of fixed width.
    }
}

pub(crate) type KeyMap<K, V> = std::collections::HashMap<K, V, KeyHash>;
pub(crate) type KeySet<K> = std::collections::HashSet<K, KeyHash>;
