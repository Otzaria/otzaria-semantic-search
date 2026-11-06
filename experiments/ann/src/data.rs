use otzaria_semantic_search::cancellation::CancellationToken;
use otzaria_semantic_search::distribution::package::PackageKind;
use otzaria_semantic_search::semantic::chunk_key::{ChunkKey, KEY_VERSION};
use otzaria_semantic_search::semantic::oxv::codec::{Codec, CodecSpec};
use otzaria_semantic_search::semantic::oxv::writer::{SegmentBuilder, SegmentSpec, WrittenSegment};
use otzaria_semantic_search::semantic::resolve::BookSet;
use otzaria_semantic_search::semantic::segment_set::{
    InstallExpectation, InstallSource, ReleaseManifest, STORE_BACKEND_ID, install_package,
};
use otzaria_semantic_search::semantic::versioning::{
    EmbeddingWorker, IndexVersion, ModelIdentity, ModelPackage, StoreIdentity, TextIdentity,
    VectorProvenance,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::Result;

pub const BOOKS: usize = 100;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub schema: u32,
    pub vectors: usize,
    pub dim: usize,
    pub queries: usize,
    pub top_k: usize,
    pub codec: String,
    pub distribution: String,
    pub connectivity: usize,
    pub expansion_add: usize,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema: 1,
            vectors: 20_000,
            dim: 256,
            queries: 100,
            top_k: 10,
            codec: "i8-sym-vec".to_string(),
            distribution: "clustered".to_string(),
            connectivity: 16,
            expansion_add: 128,
            seed: 0xA11_2026_1105,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.schema != 1 || self.vectors < BOOKS || self.vectors > u32::MAX as usize / 2 {
            return Err("schema must be 1 and vector count must be 100..=u32::MAX/2".into());
        }
        if self.dim == 0 || self.dim > 4096 || self.queries == 0 || self.top_k == 0 {
            return Err("dim must be 1..=4096; queries and top-k must be positive".into());
        }
        if self.top_k > self.vectors {
            return Err("top-k must not exceed the total vector count".into());
        }
        if self.connectivity == 0 || self.expansion_add == 0 {
            return Err("connectivity and expansion-add must be positive".into());
        }
        if !matches!(self.distribution.as_str(), "uniform" | "clustered") {
            return Err("distribution must be uniform or clustered".into());
        }
        if self.codec == "rot192-i8i4" {
            if self.dim < 192 {
                return Err("rot192-i8i4 requires at least 192 dimensions".into());
            }
        } else {
            CodecSpec::parse(&self.codec, 0.9999)?;
        }
        Ok(())
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn per_book(&self) -> usize {
        self.vectors.div_ceil(BOOKS)
    }

    pub fn book(&self, key: u64) -> usize {
        ((key - 1) as usize / self.per_book()) % BOOKS
    }

    pub fn vector(&self, key: u64) -> Vec<f32> {
        self.sample(key, false)
    }

    pub fn query(&self, query: usize) -> Vec<f32> {
        // Separate noise seeds, never a stored row queried against itself.
        self.sample(query as u64 + 1, true)
    }

    fn sample(&self, key: u64, query: bool) -> Vec<f32> {
        let mut state = mix(key ^ self.seed ^ if query { 0xD15A_0A17_F00D } else { 0 });
        let mut noise = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            state = mix(state);
            noise.push((state >> 40) as f32 / 8_388_608.0 - 1.0);
        }
        normalize(&mut noise);
        if self.distribution == "clustered" {
            let mut center = mix((key % 128) ^ self.seed ^ 0xCE17_EA);
            let mut values = Vec::with_capacity(self.dim);
            for _ in 0..self.dim {
                center = mix(center);
                values.push((center >> 40) as f32 / 8_388_608.0 - 1.0);
            }
            normalize(&mut values);
            for (value, n) in values.iter_mut().zip(noise) {
                *value = 0.65 * *value + 0.35 * n;
            }
            normalize(&mut values);
            values
        } else {
            noise
        }
    }

    pub fn fit_codec(&self) -> Result<Codec> {
        // Bounded calibration sample; a calibration time is not a search time.
        let sample: Vec<Vec<f32>> = (1..=self.vectors.min(4096) as u64)
            .map(|key| self.vector(key))
            .collect();
        let refs: Vec<&[f32]> = sample.iter().map(Vec::as_slice).collect();
        // The production name parser deliberately rejects rotated codecs: they
        // need a fitted epoch, not CodecSpec's named/per-dimension calibration.
        if self.codec == "rot192-i8i4" {
            Ok(Codec::fit_rot_i8i4(&refs, 128, 64)?)
        } else {
            Ok(CodecSpec::parse(&self.codec, 0.9999)?.build(self.dim, &refs)?)
        }
    }

    pub fn ann(&self, expansion: usize) -> Result<Index> {
        Ok(Index::new(&IndexOptions {
            dimensions: self.dim,
            // OXV ranks dot products of decoded vectors without renormalizing
            // their quantization error away. Cosine would change that contract.
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            connectivity: self.connectivity,
            expansion_add: self.expansion_add,
            expansion_search: expansion,
            multi: false,
        })?)
    }
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    for value in vector {
        *value /= norm;
    }
}

fn mix(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^ (state >> 31)
}

pub fn chunk_key(key: u64) -> ChunkKey {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&key.to_be_bytes());
    ChunkKey(bytes)
}

pub fn key_id(key: ChunkKey) -> u64 {
    u64::from_be_bytes(key.0[..8].try_into().expect("a ChunkKey has 16 bytes"))
}

#[derive(Clone, Copy, Debug)]
pub enum Filter {
    All,
    FivePercent,
    OnePercent,
    Empty,
}

impl Filter {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "all" => Ok(Self::All),
            "5pct" => Ok(Self::FivePercent),
            "1pct" => Ok(Self::OnePercent),
            "empty" => Ok(Self::Empty),
            _ => Err("filter must be all, 5pct, 1pct, or empty".into()),
        }
    }

    pub fn admits_book(self, book: usize) -> bool {
        match self {
            Self::All => true,
            Self::FivePercent => book.is_multiple_of(20),
            Self::OnePercent => book == 0,
            Self::Empty => false,
        }
    }

    pub fn admits(self, config: &Config, key: u64) -> bool {
        // Defensive for FFI results; no arithmetic underflow on an invalid key.
        key > 0 && self.admits_book(config.book(key))
    }

    pub fn books(self) -> Option<BookSet> {
        if matches!(self, Self::All) {
            None
        } else {
            Some(
                (0..BOOKS)
                    .filter(|book| self.admits_book(*book))
                    .map(book_name)
                    .collect(),
            )
        }
    }
}

fn book_name(book: usize) -> String {
    format!("synthetic:{book:04}")
}

pub fn identity(config: &Config, codec: &Codec) -> IndexVersion {
    IndexVersion {
        text: TextIdentity {
            line_text_version: 1,
            key_version: KEY_VERSION,
        },
        model: ModelIdentity {
            family_id: "ann-synthetic@1".to_string(),
            tokenizer_checksum: "7".repeat(64),
            embedding_dim: config.dim as u32,
            pooling: "in-graph".to_string(),
            max_tokens: 256,
            query_max_tokens: None,
            embedding_text_version: 2,
            normalization_version: 1,
            chunking_identity: 1,
            query_packages: vec![ModelPackage {
                checksum: "a".repeat(64),
                quantization: "synthetic".to_string(),
            }],
        },
        store: StoreIdentity {
            backend_id: STORE_BACKEND_ID.to_string(),
            store_format_version: if codec.name() == "rot192-i8i4" { 3 } else { 2 },
            vector_precision: codec.name().to_string(),
        },
    }
}

pub fn write_segment(
    config: &Config,
    path: &Path,
    codec: &Codec,
    keys: &[u64],
    deleted: &[u64],
    kind: PackageKind,
    decode_from: Option<&Codec>,
) -> Result<WrittenSegment> {
    let identity = identity(config, codec);
    let is_base = kind == PackageKind::Base;
    let mut builder = SegmentBuilder::new(
        SegmentSpec {
            kind,
            identity_digest: identity.identity_digest(),
            from_library_version: if is_base { 0 } else { 1 },
            to_library_version: if is_base { 1 } else { 2 },
            library_release_tag: "ann-synthetic-experiment".to_string(),
        },
        codec.clone(),
    );
    builder.set_format_version(identity.store.store_format_version)?;
    let mut books: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    for &key in keys {
        books.entry(config.book(key)).or_default().push(key);
    }
    let mut ordered_keys = Vec::with_capacity(keys.len());
    for (book, mut keys) in books {
        keys.sort_unstable();
        let primary: Vec<_> = keys
            .iter()
            .map(|&key| (chunk_key(key), key as u32))
            .collect();
        builder.add_book(&book_name(book), &primary, &[], &[])?;
        ordered_keys.extend(keys);
    }
    builder.set_tombstones(deleted.iter().copied().map(chunk_key).collect());
    let mut sink = builder.write(path)?;
    let mut decoded = vec![0f32; config.dim];
    let mut bytes = vec![0u8; decode_from.map_or(0, Codec::bytes_per_vector)];
    for key in ordered_keys {
        let raw = config.vector(key);
        if let Some(source) = decode_from {
            let encoded = source.encode(&raw, &mut bytes);
            source.decode(&bytes, encoded.scale, &mut decoded);
            sink.push_f32(&decoded)?;
        } else {
            sink.push_f32(&raw)?;
        }
    }
    Ok(sink.finish()?)
}

pub fn install(
    config: &Config,
    dir: &Path,
    path: &Path,
    written: &WrittenSegment,
    codec: &Codec,
) -> Result<()> {
    let identity = identity(config, codec);
    let manifest = ReleaseManifest::for_segment(
        written,
        &identity,
        codec.params_sha256(),
        VectorProvenance {
            passage_package: identity.model.query_packages[0].clone(),
            worker: EmbeddingWorker {
                backend: "synthetic-experiment".to_string(),
                device: "cpu".to_string(),
            },
        },
        "2026-11-05T00:00:00Z".to_string(),
    )
    .to_json();
    install_package(
        dir,
        &InstallSource {
            segment: path,
            manifest_json: &manifest,
        },
        &InstallExpectation {
            identity,
            published_manifest_sha256: None,
        },
        &CancellationToken::new(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_are_independent_reproducible_unit_vectors() {
        for distribution in ["uniform", "clustered"] {
            let config = Config {
                distribution: distribution.to_string(),
                dim: 32,
                ..Config::default()
            };
            assert_eq!(config.vector(1), config.vector(1));
            assert_eq!(config.query(0), config.query(0));
            assert_ne!(config.vector(1), config.query(0));
            for values in [config.vector(1), config.query(0)] {
                assert!((values.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn filter_and_book_ranges_agree_even_for_small_and_uneven_corpora() {
        for vectors in [100, 101, 999, 20_000] {
            let config = Config {
                vectors,
                ..Config::default()
            };
            for filter in [Filter::FivePercent, Filter::OnePercent, Filter::Empty] {
                let books = filter.books().unwrap();
                for key in 1..=vectors as u64 {
                    assert_eq!(
                        filter.admits(&config, key),
                        books.contains(&book_name(config.book(key)))
                    );
                    assert_eq!(key_id(chunk_key(key)), key);
                }
            }
        }
    }
}
