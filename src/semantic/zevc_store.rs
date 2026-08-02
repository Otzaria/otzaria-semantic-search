use crate::errors::VectorStoreError;
use crate::semantic::store_backend::VectorStoreBackend;
use crate::semantic::types::{SearchFilters, SemanticCandidate, VectorMetadata};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::RwLock;

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
}

impl ZevcStore {
    pub fn open_or_create(config: ZevcStoreConfig) -> Result<Self, VectorStoreError> {
        if config.embedding_dim == 0 {
            return Err(VectorStoreError::OpenFailed {
                reason: "embedding_dim must be greater than zero".to_string(),
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
        let state = self.state.read().unwrap();

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

        for (_, record) in &state.records {
            let meta_json = serde_json::to_string(&record.metadata).map_err(|e| {
                VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                }
            })?;
            writeln!(m_file, "{}", meta_json).map_err(|e| VectorStoreError::CommitFailed {
                reason: e.to_string(),
            })?;

            let mut bytes = Vec::with_capacity(record.vector.len() * 4);
            for f in &record.vector {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            v_file
                .write_all(&bytes)
                .map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?;
        }

        let b_file = File::create(&b_tmp).map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        serde_json::to_writer(b_file, &state.book_index).map_err(|e| {
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

        fs::rename(v_tmp, v_path).map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        fs::rename(m_tmp, m_path).map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        fs::rename(b_tmp, b_path).map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;

        Ok(())
    }

    pub fn load_from_disk(&self) -> Result<(), VectorStoreError> {
        let v_path = self.vectors_path();
        let m_path = self.metadata_path();
        let b_path = self.book_index_path();

        if !v_path.exists() || !m_path.exists() || !b_path.exists() {
            return Ok(());
        }

        let m_file = File::open(&m_path).map_err(|e| VectorStoreError::OpenFailed {
            reason: e.to_string(),
        })?;
        let mut v_file = File::open(&v_path).map_err(|e| VectorStoreError::OpenFailed {
            reason: e.to_string(),
        })?;

        let m_reader = BufReader::new(m_file);

        let mut state = self.state.write().unwrap();
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
            let metadata: VectorMetadata = serde_json::from_str(&line).map_err(|e| {
                log::warn!("Corrupted metadata: {}", e);
                VectorStoreError::Corrupted {
                    reason: e.to_string(),
                }
            })?;

            let mut buf = vec![0u8; vec_byte_size];
            if let Err(e) = v_file.read_exact(&mut buf) {
                log::warn!("Corrupted vectors file: {}", e);
                return Err(VectorStoreError::Corrupted {
                    reason: e.to_string(),
                });
            }

            let mut vector = Vec::with_capacity(dim);
            for i in 0..dim {
                let bytes: [u8; 4] = buf[i * 4..i * 4 + 4].try_into().unwrap();
                vector.push(f32::from_le_bytes(bytes));
            }

            state.records.insert(
                metadata.semantic_id.clone(),
                StoredVectorRecord { metadata, vector },
            );
        }

        let b_file = File::open(&b_path).map_err(|e| VectorStoreError::OpenFailed {
            reason: e.to_string(),
        })?;
        state.book_index = serde_json::from_reader(b_file).map_err(|e| {
            log::warn!("Corrupted book index: {}", e);
            VectorStoreError::Corrupted {
                reason: e.to_string(),
            }
        })?;

        Ok(())
    }
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
        self.state.read().unwrap().records.len() as u32
    }

    fn insert_batch(
        &self,
        batch: Vec<(VectorMetadata, Vec<f32>)>,
    ) -> Result<u32, VectorStoreError> {
        let mut state = self.state.write().unwrap();
        let mut inserted = 0;

        for (meta, vector) in batch {
            if vector.len() as u32 != self.config.embedding_dim {
                return Err(VectorStoreError::DimensionMismatch {
                    store_dim: self.config.embedding_dim,
                    vector_dim: vector.len() as u32,
                });
            }

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
            inserted += 1;
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

        let filters = filters.and_then(SearchFilters::compile);
        let state = self.state.read().unwrap();
        let mut heap: BinaryHeap<ScoredEntry> = BinaryHeap::with_capacity(top_k + 1);

        for record in state.records.values() {
            if let Some(compiled) = filters.as_ref() {
                if !compiled.matches(&record.metadata) {
                    continue;
                }
            }

            let score = dot_product(query_vector, &record.vector);
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
        let mut state = self.state.write().unwrap();
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
        let mut state = self.state.write().unwrap();
        let removed = state.records.len() as u32;
        state.records.clear();
        state.book_index.clear();

        drop(state);

        let _ = fs::remove_file(self.vectors_path());
        let _ = fs::remove_file(self.metadata_path());
        let _ = fs::remove_file(self.book_index_path());

        Ok(removed)
    }

    fn book_keys(&self) -> Vec<String> {
        self.state
            .read()
            .unwrap()
            .book_index
            .keys()
            .cloned()
            .collect()
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
}
