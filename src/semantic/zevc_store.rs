use crate::errors::VectorStoreError;
use crate::semantic::store_backend::VectorStoreBackend;
use crate::semantic::types::{SearchFilters, SemanticCandidate, VectorMetadata};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

#[derive(Debug, Clone)]
pub struct ZevcStoreConfig {
    pub db_path: PathBuf,
    pub embedding_dim: u32,
    pub collection_name: String,
    /// Whether to persist after every batch or defer to explicit flush.
    pub auto_persist: bool,
}

#[derive(Debug, Clone)]
pub struct StoredVectorRecord {
    pub metadata: VectorMetadata,
    pub vector: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct PersistedMetadata {
    metadata: VectorMetadata,
    metadata_sha256: String,
    vector_sha256: String,
}

#[derive(Serialize, Deserialize)]
struct PersistedBookIndex {
    format_version: u32,
    embedding_dim: u32,
    collection_name: String,
    books: HashMap<String, Vec<String>>,
}

#[derive(Default)]
struct StoreState {
    records: HashMap<String, StoredVectorRecord>,
    book_index: HashMap<String, Vec<String>>,
}

struct ScoredEntry {
    score: f32,
    semantic_id: String,
}

impl PartialEq for ScoredEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ScoredEntry {}
impl PartialOrd for ScoredEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.semantic_id.cmp(&other.semantic_id))
    }
}

pub struct ZevcStore {
    config: ZevcStoreConfig,
    state: RwLock<StoreState>,
    persist_lock: Mutex<()>,
}

impl ZevcStore {
    pub fn open_or_create(config: ZevcStoreConfig) -> Result<Self, VectorStoreError> {
        if config.embedding_dim == 0 {
            return Err(VectorStoreError::OpenFailed {
                reason: "embedding_dim must be greater than zero".to_string(),
            });
        }
        if config.collection_name.trim().is_empty() {
            return Err(VectorStoreError::OpenFailed {
                reason: "collection_name must not be empty".to_string(),
            });
        }
        if let Err(e) = std::fs::create_dir_all(&config.db_path) {
            return Err(VectorStoreError::OpenFailed {
                reason: format!("Failed to create DB directory: {e}"),
            });
        }

        let store = Self {
            config,
            state: RwLock::new(StoreState::default()),
            persist_lock: Mutex::new(()),
        };

        store.load_from_disk()?;

        log::info!(
            "ZevcStore (persistent) initialized at: {}",
            store.config.db_path.display()
        );
        Ok(store)
    }

    fn vectors_path(&self) -> PathBuf {
        self.config.db_path.join("vectors.bin")
    }

    fn metadata_path(&self) -> PathBuf {
        self.config.db_path.join("metadata.jsonl")
    }

    fn book_index_path(&self) -> PathBuf {
        self.config.db_path.join("book_index.json")
    }

    pub fn save_to_disk(&self) -> Result<(), VectorStoreError> {
        let _persist = self
            .persist_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = self
            .state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let v_path = self.vectors_path();
        let m_path = self.metadata_path();
        let b_path = self.book_index_path();

        let v_tmp = v_path.with_extension("bin.tmp");
        let m_tmp = m_path.with_extension("jsonl.tmp");
        let b_tmp = b_path.with_extension("json.tmp");

        let mut v_file =
            BufWriter::new(
                File::create(&v_tmp).map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?,
            );
        let mut m_file =
            BufWriter::new(
                File::create(&m_tmp).map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?,
            );

        for record in state.records.values() {
            let mut vector_bytes = Vec::with_capacity(record.vector.len() * 4);
            for value in &record.vector {
                vector_bytes.extend_from_slice(&value.to_le_bytes());
            }
            let persisted = PersistedMetadata {
                metadata: record.metadata.clone(),
                metadata_sha256: format!(
                    "{:x}",
                    Sha256::digest(serde_json::to_vec(&record.metadata).map_err(|e| {
                        VectorStoreError::CommitFailed {
                            reason: e.to_string(),
                        }
                    })?)
                ),
                vector_sha256: format!("{:x}", Sha256::digest(&vector_bytes)),
            };
            let meta_json =
                serde_json::to_string(&persisted).map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?;
            writeln!(m_file, "{meta_json}").map_err(|e| VectorStoreError::CommitFailed {
                reason: e.to_string(),
            })?;

            v_file
                .write_all(&vector_bytes)
                .map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?;
        }

        let b_file = File::create(&b_tmp).map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        let persisted_book_index = PersistedBookIndex {
            format_version: 1,
            embedding_dim: self.config.embedding_dim,
            collection_name: self.config.collection_name.clone(),
            books: state.book_index.clone(),
        };
        serde_json::to_writer(&b_file, &persisted_book_index).map_err(|e| {
            VectorStoreError::CommitFailed {
                reason: e.to_string(),
            }
        })?;

        v_file.flush().map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        m_file.flush().map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        v_file
            .get_ref()
            .sync_all()
            .and_then(|_| m_file.get_ref().sync_all())
            .and_then(|_| b_file.sync_all())
            .map_err(|e| VectorStoreError::CommitFailed {
                reason: e.to_string(),
            })?;

        // Windows refuses to rename an open file. Flushing is not enough: all
        // handles must be closed before the snapshot swap begins.
        drop(v_file);
        drop(m_file);
        drop(b_file);

        replace_snapshot(&[(&v_tmp, &v_path), (&m_tmp, &m_path), (&b_tmp, &b_path)])?;

        Ok(())
    }

    fn load_from_disk(&self) -> Result<(), VectorStoreError> {
        let v_path = self.vectors_path();
        let m_path = self.metadata_path();
        let b_path = self.book_index_path();

        let present = [v_path.exists(), m_path.exists(), b_path.exists()];
        if present.iter().all(|exists| !exists) {
            return Ok(());
        }
        if !present.iter().all(|exists| *exists) {
            return Err(VectorStoreError::Corrupted {
                reason: "persistent store is incomplete".to_string(),
            });
        }

        let m_file = File::open(&m_path).map_err(|e| VectorStoreError::OpenFailed {
            reason: e.to_string(),
        })?;
        let mut v_file = File::open(&v_path).map_err(|e| VectorStoreError::OpenFailed {
            reason: e.to_string(),
        })?;

        let m_reader = BufReader::new(m_file);

        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.records.clear();
        state.book_index.clear();

        let dim = self.config.embedding_dim as usize;
        let vec_byte_size = dim * 4;

        for line in m_reader.lines() {
            let line = line.map_err(|e| VectorStoreError::OpenFailed {
                reason: e.to_string(),
            })?;
            if line.is_empty() {
                continue;
            }
            let persisted: PersistedMetadata = serde_json::from_str(&line).map_err(|e| {
                log::warn!("Corrupted metadata: {e}");
                VectorStoreError::Corrupted {
                    reason: e.to_string(),
                }
            })?;

            let metadata_hash = format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&persisted.metadata).map_err(|e| {
                    VectorStoreError::Corrupted {
                        reason: e.to_string(),
                    }
                })?)
            );
            if metadata_hash != persisted.metadata_sha256 {
                return Err(VectorStoreError::Corrupted {
                    reason: format!(
                        "metadata checksum mismatch for {}",
                        persisted.metadata.semantic_id
                    ),
                });
            }

            let mut buf = vec![0u8; vec_byte_size];
            if let Err(e) = v_file.read_exact(&mut buf) {
                log::warn!("Corrupted vectors file: {e}");
                return Err(VectorStoreError::Corrupted {
                    reason: e.to_string(),
                });
            }

            let actual_hash = format!("{:x}", Sha256::digest(&buf));
            if actual_hash != persisted.vector_sha256 {
                return Err(VectorStoreError::Corrupted {
                    reason: format!(
                        "vector checksum mismatch for {}",
                        persisted.metadata.semantic_id
                    ),
                });
            }

            let mut vector = Vec::with_capacity(dim);
            for i in 0..dim {
                let bytes: [u8; 4] = buf[i * 4..i * 4 + 4].try_into().unwrap();
                vector.push(f32::from_le_bytes(bytes));
            }

            let semantic_id = persisted.metadata.semantic_id.clone();
            if state.records.contains_key(&semantic_id) {
                return Err(VectorStoreError::Corrupted {
                    reason: format!("duplicate semantic id in metadata: {semantic_id}"),
                });
            }
            state.records.insert(
                semantic_id,
                StoredVectorRecord {
                    metadata: persisted.metadata,
                    vector,
                },
            );
        }

        let mut trailing = [0u8; 1];
        if v_file
            .read(&mut trailing)
            .map_err(|e| VectorStoreError::OpenFailed {
                reason: e.to_string(),
            })?
            != 0
        {
            return Err(VectorStoreError::Corrupted {
                reason: "vectors file contains trailing data".to_string(),
            });
        }

        let b_file = File::open(&b_path).map_err(|e| VectorStoreError::OpenFailed {
            reason: e.to_string(),
        })?;
        let persisted_book_index: PersistedBookIndex =
            serde_json::from_reader(b_file).map_err(|e| {
                log::warn!("Corrupted book index: {e}");
                VectorStoreError::Corrupted {
                    reason: e.to_string(),
                }
            })?;

        if persisted_book_index.format_version != 1
            || persisted_book_index.embedding_dim != self.config.embedding_dim
            || persisted_book_index.collection_name != self.config.collection_name
        {
            return Err(VectorStoreError::Corrupted {
                reason: "store configuration does not match its persisted header".to_string(),
            });
        }

        let mut rebuilt_book_index: HashMap<String, Vec<String>> = HashMap::new();
        for record in state.records.values() {
            rebuilt_book_index
                .entry(record.metadata.source_book_key.clone())
                .or_default()
                .push(record.metadata.semantic_id.clone());
        }
        for ids in rebuilt_book_index.values_mut() {
            ids.sort();
        }
        let mut stored_book_index = persisted_book_index.books;
        for ids in stored_book_index.values_mut() {
            ids.sort();
        }
        if stored_book_index != rebuilt_book_index {
            return Err(VectorStoreError::Corrupted {
                reason: "book index disagrees with vector metadata".to_string(),
            });
        }
        state.book_index = rebuilt_book_index;

        Ok(())
    }
}

fn replace_snapshot(files: &[(&Path, &Path)]) -> Result<(), VectorStoreError> {
    for (_, destination) in files {
        let backup = destination.with_extension("previous");
        if backup.exists() {
            return Err(VectorStoreError::CommitFailed {
                reason: format!("refusing to overwrite recovery file {}", backup.display()),
            });
        }
    }

    let mut parked = Vec::new();
    for (_, destination) in files {
        if destination.exists() {
            let backup = destination.with_extension("previous");
            if let Err(error) = fs::rename(destination, &backup) {
                for (parked_destination, parked_backup) in parked.iter().rev() {
                    let _ = fs::rename(parked_backup, parked_destination);
                }
                return Err(VectorStoreError::CommitFailed {
                    reason: error.to_string(),
                });
            }
            parked.push((destination.to_path_buf(), backup));
        }
    }

    let mut installed = Vec::new();
    for (temp, destination) in files {
        if let Err(error) = fs::rename(temp, destination) {
            for path in installed.iter().rev() {
                let _ = fs::remove_file(path);
            }
            for (parked_destination, parked_backup) in parked.iter().rev() {
                let _ = fs::rename(parked_backup, parked_destination);
            }
            return Err(VectorStoreError::CommitFailed {
                reason: error.to_string(),
            });
        }
        installed.push(destination.to_path_buf());
    }

    for (_, backup) in parked {
        if let Err(error) = fs::remove_file(&backup) {
            log::warn!(
                "Store committed, but could not remove recovery file {}: {error}",
                backup.display()
            );
        }
    }
    Ok(())
}

impl VectorStoreBackend for ZevcStore {
    fn backend_id(&self) -> &'static str {
        "zevc-persistent-v1"
    }

    fn is_persistent(&self) -> bool {
        true
    }

    fn embedding_dim(&self) -> u32 {
        self.config.embedding_dim
    }

    fn count(&self) -> u32 {
        self.state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .records
            .len()
            .min(u32::MAX as usize) as u32
    }

    fn insert_batch(
        &self,
        mut batch: Vec<(VectorMetadata, Vec<f32>)>,
    ) -> Result<u32, VectorStoreError> {
        for (meta, vector) in &mut batch {
            if vector.len() as u32 != self.config.embedding_dim {
                return Err(VectorStoreError::DimensionMismatch {
                    store_dim: self.config.embedding_dim,
                    vector_dim: vector.len() as u32,
                });
            }
            if let Some(position) = vector.iter().position(|value| !value.is_finite()) {
                return Err(VectorStoreError::InsertFailed {
                    reason: format!(
                        "vector for {} has a non-finite component at {position}",
                        meta.semantic_id
                    ),
                });
            }
            let norm = vector
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>()
                .sqrt();
            if !norm.is_finite() || norm <= f64::EPSILON {
                return Err(VectorStoreError::InsertFailed {
                    reason: format!("vector for {} has no usable direction", meta.semantic_id),
                });
            }
            let reciprocal = (1.0 / norm) as f32;
            for value in vector {
                *value *= reciprocal;
            }
        }

        let inserted = batch.len().min(u32::MAX as usize) as u32;
        let mut state = self.state.write().unwrap_or_else(|poisoned| {
            log::warn!("ZevcStore lock was poisoned; recovering state");
            poisoned.into_inner()
        });

        for (meta, vector) in batch {
            let id = meta.semantic_id.clone();
            let record = StoredVectorRecord {
                metadata: meta.clone(),
                vector,
            };

            match state.records.insert(id.clone(), record) {
                None => {
                    state
                        .book_index
                        .entry(meta.source_book_key.clone())
                        .or_default()
                        .push(id);
                }
                Some(previous) => {
                    let previous_book = previous.metadata.source_book_key;
                    if previous_book != meta.source_book_key {
                        if let Some(ids) = state.book_index.get_mut(&previous_book) {
                            ids.retain(|existing| existing != &id);
                            if ids.is_empty() {
                                state.book_index.remove(&previous_book);
                            }
                        }
                        state
                            .book_index
                            .entry(meta.source_book_key.clone())
                            .or_default()
                            .push(id);
                    }
                }
            }
        }

        drop(state); // Drop lock before saving

        if self.config.auto_persist {
            self.save_to_disk()?;
        }

        Ok(inserted)
    }

    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, VectorStoreError> {
        if query_vector.len() as u32 != self.config.embedding_dim {
            return Err(VectorStoreError::DimensionMismatch {
                store_dim: self.config.embedding_dim,
                vector_dim: query_vector.len() as u32,
            });
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if query_vector.iter().any(|value| !value.is_finite()) {
            return Ok(Vec::new());
        }
        let norm = query_vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Ok(Vec::new());
        }
        let reciprocal = (1.0 / norm) as f32;
        let query_vector: Vec<f32> = query_vector
            .iter()
            .map(|value| value * reciprocal)
            .collect();

        let filters = filters.and_then(SearchFilters::compile);
        let state = self
            .state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut heap: BinaryHeap<ScoredEntry> = BinaryHeap::with_capacity(top_k + 1);

        for record in state.records.values() {
            if let Some(compiled) = filters.as_ref() {
                if !compiled.matches(&record.metadata) {
                    continue;
                }
            }

            let score = dot_product(&query_vector, &record.vector);
            if score.is_nan() {
                continue;
            }

            let entry = ScoredEntry {
                score,
                semantic_id: record.metadata.semantic_id.clone(),
            };
            if heap.len() < top_k {
                heap.push(entry);
            } else if let Some(weakest) = heap.peek() {
                if entry < *weakest {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }

        let mut candidates: Vec<SemanticCandidate> = heap
            .into_iter()
            .filter_map(|entry| {
                state
                    .records
                    .get(&entry.semantic_id)
                    .map(|record| SemanticCandidate {
                        metadata: record.metadata.clone(),
                        similarity_score: entry.score,
                    })
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.similarity_score
                .total_cmp(&a.similarity_score)
                .then_with(|| a.metadata.semantic_id.cmp(&b.metadata.semantic_id))
        });

        Ok(candidates)
    }

    fn remove_by_book(&self, source_book_key: &str) -> Result<u32, VectorStoreError> {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut deleted = 0u32;
        if let Some(ids) = state.book_index.remove(source_book_key) {
            for id in ids {
                if state.records.remove(&id).is_some() {
                    deleted += 1;
                }
            }
        }

        drop(state);
        if self.config.auto_persist && deleted > 0 {
            self.save_to_disk()?;
        }

        Ok(deleted)
    }

    fn clear(&self) -> Result<u32, VectorStoreError> {
        let _persist = self
            .persist_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = state.records.len() as u32;
        state.records.clear();
        state.book_index.clear();

        drop(state);

        for path in [
            self.vectors_path(),
            self.metadata_path(),
            self.book_index_path(),
        ] {
            if let Err(error) = fs::remove_file(path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    return Err(VectorStoreError::CommitFailed {
                        reason: error.to_string(),
                    });
                }
            }
        }

        Ok(removed)
    }

    fn book_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .book_index
            .keys()
            .cloned()
            .collect();
        keys.sort();
        keys
    }
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);

    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for lane in 0..LANES {
            acc[lane] += x[lane] * y[lane];
        }
    }

    let mut sum = acc.iter().sum::<f32>();
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        sum += x * y;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_zevc_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = std::fs::create_dir_all(&path);
            Self(path)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample_metadata(id: &str, book: &str) -> VectorMetadata {
        VectorMetadata {
            semantic_id: id.to_string(),
            source_book_key: book.to_string(),
            source_doc_key: format!("{book}:{id}"),
            line_id: 1,
            section_id: 10,
            line_hash: 100,
            chunk_hash: "hash".to_string(),
            content_hash: 555,
            reference: "Ref 1".to_string(),
            segment: 0,
            is_pdf: false,
            title: "Test Book".to_string(),
            facets: vec![],
        }
    }

    #[test]
    fn test_zevc_store_crud_and_persistence() {
        let dir = TempDir::new("crud");
        let config = ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
            auto_persist: true,
        };

        let store = ZevcStore::open_or_create(config.clone()).unwrap();

        let batch = vec![
            (
                sample_metadata("id1", "book1.txt"),
                vec![1.0, 0.0, 0.0, 0.0],
            ),
            (
                sample_metadata("id2", "book1.txt"),
                vec![0.0, 1.0, 0.0, 0.0],
            ),
        ];
        store.insert_batch(batch).unwrap();
        assert_eq!(store.count(), 2);

        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].metadata.semantic_id, "id1");

        // Test persistence by opening a new instance
        let store2 = ZevcStore::open_or_create(config.clone()).unwrap();
        assert_eq!(store2.count(), 2);
        let hits2 = store2.search(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits2.len(), 2);

        // Test remove by book
        store2.remove_by_book("book1.txt").unwrap();
        assert_eq!(store2.count(), 0);

        // Test clear
        store2
            .insert_batch(vec![(
                sample_metadata("id3", "b2"),
                vec![0.0, 0.0, 1.0, 0.0],
            )])
            .unwrap();
        assert_eq!(store2.count(), 1);
        store2.clear().unwrap();
        assert_eq!(store2.count(), 0);
    }

    #[test]
    fn test_corrupted_files() {
        let dir = TempDir::new("corrupt");
        let config = ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
            auto_persist: true,
        };

        let store = ZevcStore::open_or_create(config.clone()).unwrap();
        store
            .insert_batch(vec![(
                sample_metadata("id1", "book1.txt"),
                vec![1.0, 0.0, 0.0, 0.0],
            )])
            .unwrap();

        // Corrupt vectors
        let mut v_file = fs::OpenOptions::new()
            .write(true)
            .open(store.vectors_path())
            .unwrap();
        v_file.write_all(b"bad").unwrap();

        let res = ZevcStore::open_or_create(config.clone());
        assert!(matches!(res, Err(VectorStoreError::Corrupted { .. })));
    }

    #[test]
    fn invalid_batch_does_not_partially_mutate_the_store() {
        let dir = TempDir::new("atomic_batch");
        let store = ZevcStore::open_or_create(ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
            auto_persist: false,
        })
        .unwrap();

        let result = store.insert_batch(vec![
            (sample_metadata("valid", "book"), vec![1.0, 0.0, 0.0, 0.0]),
            (sample_metadata("invalid", "book"), vec![1.0, 0.0]),
        ]);

        assert!(result.is_err());
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn snapshot_replacement_rolls_back_every_file_on_error() {
        let dir = TempDir::new("snapshot_rollback");
        let old_a = dir.path().join("a.bin");
        let old_b = dir.path().join("b.bin");
        let new_a = dir.path().join("a.tmp");
        let missing_new_b = dir.path().join("missing.tmp");
        fs::write(&old_a, b"old-a").unwrap();
        fs::write(&old_b, b"old-b").unwrap();
        fs::write(&new_a, b"new-a").unwrap();

        let result = replace_snapshot(&[
            (new_a.as_path(), old_a.as_path()),
            (missing_new_b.as_path(), old_b.as_path()),
        ]);

        assert!(result.is_err());
        assert_eq!(fs::read(old_a).unwrap(), b"old-a");
        assert_eq!(fs::read(old_b).unwrap(), b"old-b");
    }

    #[test]
    fn concurrent_saves_produce_one_readable_snapshot() {
        let dir = TempDir::new("concurrent_saves");
        let config = ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
            auto_persist: false,
        };
        let store = std::sync::Arc::new(ZevcStore::open_or_create(config.clone()).unwrap());
        store
            .insert_batch(vec![(
                sample_metadata("id", "book"),
                vec![1.0, 0.0, 0.0, 0.0],
            )])
            .unwrap();

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let store = std::sync::Arc::clone(&store);
                std::thread::spawn(move || store.save_to_disk())
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }

        assert_eq!(ZevcStore::open_or_create(config).unwrap().count(), 1);
    }
}
