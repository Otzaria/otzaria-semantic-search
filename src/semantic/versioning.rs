use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexVersion {
    pub schema_version: u32,
    pub model_id: String,
    pub embedding_dim: u32,
    pub pooling: String,
    pub max_tokens: usize,
    pub normalization_version: u32,
    pub chunking_identity: String,
    pub store_backend: String,
    pub vector_precision: String,
}

impl IndexVersion {
    pub fn current_schema_version() -> u32 {
        1
    }

    pub fn is_compatible(&self, other: &IndexVersion) -> bool {
        self.describe_incompatibilities(other).is_empty()
    }

    pub fn describe_incompatibilities(&self, other: &IndexVersion) -> Vec<String> {
        let mut diffs = Vec::new();
        if self.schema_version != other.schema_version {
            diffs.push(format!("schema_version: {} vs {}", self.schema_version, other.schema_version));
        }
        if self.model_id != other.model_id {
            diffs.push(format!("model_id: {} vs {}", self.model_id, other.model_id));
        }
        if self.embedding_dim != other.embedding_dim {
            diffs.push(format!("embedding_dim: {} vs {}", self.embedding_dim, other.embedding_dim));
        }
        if self.pooling != other.pooling {
            diffs.push(format!("pooling: {} vs {}", self.pooling, other.pooling));
        }
        if self.max_tokens != other.max_tokens {
            diffs.push(format!("max_tokens: {} vs {}", self.max_tokens, other.max_tokens));
        }
        if self.normalization_version != other.normalization_version {
            diffs.push(format!("normalization_version: {} vs {}", self.normalization_version, other.normalization_version));
        }
        if self.chunking_identity != other.chunking_identity {
            diffs.push(format!("chunking_identity: {} vs {}", self.chunking_identity, other.chunking_identity));
        }
        if self.store_backend != other.store_backend {
            diffs.push(format!("store_backend: {} vs {}", self.store_backend, other.store_backend));
        }
        if self.vector_precision != other.vector_precision {
            diffs.push(format!("vector_precision: {} vs {}", self.vector_precision, other.vector_precision));
        }
        diffs
    }
}

impl std::fmt::Display for IndexVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "v{}: {} (dim={}, {}, {} tok), backend: {}, precision: {}",
            self.schema_version,
            self.model_id,
            self.embedding_dim,
            self.pooling,
            self.max_tokens,
            self.store_backend,
            self.vector_precision
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_version() -> IndexVersion {
        IndexVersion {
            schema_version: 1,
            model_id: "test-model".to_string(),
            embedding_dim: 1024,
            pooling: "last-token".to_string(),
            max_tokens: 512,
            normalization_version: 1,
            chunking_identity: "chunk-v1".to_string(),
            store_backend: "zevc-persistent-v1".to_string(),
            vector_precision: "f32".to_string(),
        }
    }

    #[test]
    fn test_compatibility_exact_match() {
        let v1 = base_version();
        let v2 = base_version();
        assert!(v1.is_compatible(&v2));
        assert!(v1.describe_incompatibilities(&v2).is_empty());
    }

    #[test]
    fn test_describe_incompatibilities() {
        let v1 = base_version();
        let mut v2 = base_version();
        v2.model_id = "other-model".to_string();
        v2.embedding_dim = 512;

        let diffs = v1.describe_incompatibilities(&v2);
        assert!(!v1.is_compatible(&v2));
        assert_eq!(diffs.len(), 2);
        assert!(diffs[0].contains("test-model vs other-model"));
        assert!(diffs[1].contains("1024 vs 512"));
    }
}
