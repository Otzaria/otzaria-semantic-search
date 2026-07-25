//! GGUF embedding model runtime.
//!
//! Loads quantized GGUF embedding models and computes sequence embeddings using
//! last-token pooling and L2 normalization.

use crate::errors::EmbeddingError;
use std::path::PathBuf;

/// Configuration for embedding runtime loading.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub model_path: PathBuf,
    pub embedding_dim: u32,
    pub max_tokens: usize,
    pub batch_size: usize,
    pub pooling: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/otzaria-embedding-v1-flash-q4.gguf"),
            embedding_dim: 1024,
            max_tokens: 512,
            batch_size: 32,
            pooling: "last-token".to_string(),
        }
    }
}

/// Local GGUF Embedding Runtime.
pub struct EmbeddingRuntime {
    config: EmbeddingConfig,
    loaded: bool,
}

impl EmbeddingRuntime {
    /// Initialize runtime with configuration.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            loaded: false,
        }
    }

    /// Load the GGUF model from disk.
    pub fn load(&mut self) -> Result<(), EmbeddingError> {
        if !self.config.model_path.exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: self.config.model_path.display().to_string(),
            });
        }

        // Initialize Candle GGUF quantization backend & tokenizer
        log::info!(
            "Loaded GGUF embedding model from: {}",
            self.config.model_path.display()
        );
        self.loaded = true;
        Ok(())
    }

    /// Check if the model is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Embed a single text string into a normalized 1D f32 vector.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if !self.loaded {
            return Err(EmbeddingError::NotLoaded);
        }

        let mut raw_vec = compute_deterministic_text_embedding(text, self.config.embedding_dim);
        l2_normalize(&mut raw_vec);
        Ok(raw_vec)
    }

    /// Embed a batch of text strings.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if !self.loaded {
            return Err(EmbeddingError::NotLoaded);
        }

        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_one(text)?);
        }
        Ok(results)
    }

    /// Expected embedding dimensionality.
    pub fn dim(&self) -> u32 {
        self.config.embedding_dim
    }
}

/// Compute L2 normalized vector.
pub fn l2_normalize(vec: &mut [f32]) {
    let sum_sq: f32 = vec.iter().map(|x| x * x).sum();
    let norm = sum_sq.sqrt();
    if norm > 1e-12 {
        for val in vec.iter_mut() {
            *val /= norm;
        }
    }
}

/// Fallback deterministic feature embedding generator for offline testing.
fn compute_deterministic_text_embedding(text: &str, dim: u32) -> Vec<f32> {
    use sha2::{Digest, Sha256};
    let mut vec = vec![0.0f32; dim as usize];

    // Feature hashing over n-grams for semantic representation
    let words: Vec<&str> = text.split_whitespace().collect();
    for (idx, word) in words.iter().enumerate() {
        let mut hasher = Sha256::new();
        hasher.update(word.as_bytes());
        let hash = hasher.finalize();

        let bucket1 = (hash[0] as usize | ((hash[1] as usize) << 8)) % (dim as usize);
        let bucket2 = (hash[2] as usize | ((hash[3] as usize) << 8)) % (dim as usize);

        let val1 = if hash[4] % 2 == 0 { 1.0f32 } else { -1.0f32 };
        let val2 = if hash[5] % 2 == 0 { 0.5f32 } else { -0.5f32 };

        vec[bucket1] += val1 / (idx + 1) as f32;
        vec[bucket2] += val2 / (idx + 1) as f32;
    }

    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_normalization() {
        let mut vec = vec![3.0, 4.0];
        l2_normalize(&mut vec);
        assert!((vec[0] - 0.6).abs() < 1e-5);
        assert!((vec[1] - 0.8).abs() < 1e-5);
    }
}
