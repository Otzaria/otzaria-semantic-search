use crate::cloud::package::IndexPackage;
use crate::semantic::versioning::IndexVersion;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    pub source_path: PathBuf,
    pub target_store_path: PathBuf,
    pub verify_checksums: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    pub books_imported: u32,
    pub vectors_imported: u32,
    pub import_duration_ms: u128,
}

pub struct IndexImporter {
    config: ImportConfig,
}

impl IndexImporter {
    pub fn new(config: ImportConfig) -> Self {
        Self { config }
    }

    pub fn import(&self, current_version: &IndexVersion) -> io::Result<ImportResult> {
        let start_time = Instant::now();

        let package = IndexPackage::read(&self.config.source_path)?;

        if self.config.verify_checksums && !package.verify_checksums(&self.config.source_path)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Checksum verification failed",
            ));
        }

        if !package.manifest.version.is_compatible(current_version) {
            let diffs = package
                .manifest
                .version
                .describe_incompatibilities(current_version);
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Incompatible index version: {:?}", diffs),
            ));
        }

        fs::create_dir_all(&self.config.target_store_path)?;

        for filename in package.checksums.keys() {
            let src = self.config.source_path.join(filename);
            let dest = self.config.target_store_path.join(filename);
            if src.exists() {
                fs::copy(&src, &dest)?;
            }
        }

        let manifest_dest = self.config.target_store_path.join("manifest.json");
        fs::write(manifest_dest, package.manifest_json)?;

        Ok(ImportResult {
            books_imported: package.manifest.book_count,
            vectors_imported: package.manifest.vector_count,
            import_duration_ms: start_time.elapsed().as_millis(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::package::{IndexPackage, PackageManifest};
    use std::collections::HashMap;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_imp_{name}_{}",
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

    fn sample_version() -> IndexVersion {
        IndexVersion {
            schema_version: 1,
            model_id: "test".to_string(),
            embedding_dim: 128,
            pooling: "last".to_string(),
            max_tokens: 512,
            normalization_version: 1,
            chunking_identity: "chunk-v1".to_string(),
            store_backend: "zevc".to_string(),
            vector_precision: "f32".to_string(),
        }
    }

    #[test]
    fn test_import_compatible() {
        let dir = TempDir::new("compat");
        let source_path = dir.path().join("source");
        let target_path = dir.path().join("target");

        let manifest = PackageManifest {
            version: sample_version(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            book_count: 5,
            vector_count: 50,
            total_size_bytes: 500,
        };
        let manifest_json = serde_json::to_string(&manifest).unwrap();

        let mut checksums = HashMap::new();
        // Just mock a file
        let bin_path = source_path.join("vectors.bin");
        fs::create_dir_all(&source_path).unwrap();
        fs::write(&bin_path, b"data").unwrap();

        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"data");
        checksums.insert(
            "vectors.bin".to_string(),
            format!("{:x}", hasher.finalize()),
        );

        let pkg = IndexPackage {
            manifest,
            manifest_json,
            checksums,
        };
        IndexPackage::write(&source_path, &pkg).unwrap();

        let importer = IndexImporter::new(ImportConfig {
            source_path,
            target_store_path: target_path.clone(),
            verify_checksums: true,
        });

        let res = importer.import(&sample_version()).unwrap();
        assert_eq!(res.books_imported, 5);
        assert!(target_path.join("vectors.bin").exists());
        assert!(target_path.join("manifest.json").exists());
    }

    #[test]
    fn test_import_incompatible() {
        let dir = TempDir::new("incompat");
        let source_path = dir.path().join("source");

        let mut version = sample_version();
        version.embedding_dim = 256;

        let manifest = PackageManifest {
            version,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            book_count: 5,
            vector_count: 50,
            total_size_bytes: 500,
        };
        let manifest_json = serde_json::to_string(&manifest).unwrap();

        let pkg = IndexPackage {
            manifest,
            manifest_json,
            checksums: HashMap::new(),
        };
        IndexPackage::write(&source_path, &pkg).unwrap();

        let importer = IndexImporter::new(ImportConfig {
            source_path,
            target_store_path: dir.path().join("target"),
            verify_checksums: false,
        });

        let res = importer.import(&sample_version());
        assert!(res.is_err());
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("Incompatible index version"));
    }
}
