use crate::semantic::versioning::IndexVersion;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    pub version: IndexVersion,
    pub created_at: String,
    pub book_count: u32,
    pub vector_count: u32,
    pub total_size_bytes: u64,
}

pub struct IndexPackage {
    pub manifest: PackageManifest,
    pub manifest_json: String,
    pub checksums: HashMap<String, String>,
}

impl IndexPackage {
    pub fn write(path: &Path, package: &IndexPackage) -> io::Result<()> {
        fs::create_dir_all(path)?;

        let manifest_path = path.join("manifest.json");
        let mut manifest_file = File::create(manifest_path)?;
        manifest_file.write_all(package.manifest_json.as_bytes())?;

        let checksums_path = path.join("checksums.json");
        let mut checksums_file = File::create(checksums_path)?;
        let checksums_json = serde_json::to_string_pretty(&package.checksums)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        checksums_file.write_all(checksums_json.as_bytes())?;

        Ok(())
    }

    pub fn read(path: &Path) -> io::Result<Self> {
        let manifest_path = path.join("manifest.json");
        let mut manifest_file = File::open(manifest_path)?;
        let mut manifest_json = String::new();
        manifest_file.read_to_string(&mut manifest_json)?;

        let manifest: PackageManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let checksums_path = path.join("checksums.json");
        let mut checksums_file = File::open(checksums_path)?;
        let mut checksums_json = String::new();
        checksums_file.read_to_string(&mut checksums_json)?;

        let checksums: HashMap<String, String> = serde_json::from_str(&checksums_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Ok(Self {
            manifest,
            manifest_json,
            checksums,
        })
    }

    pub fn verify_checksums(&self, base_path: &Path) -> io::Result<bool> {
        for (filename, expected_hash) in &self.checksums {
            let file_path = base_path.join(filename);
            if !file_path.exists() {
                return Ok(false);
            }

            let mut file = File::open(file_path)?;
            let mut hasher = Sha256::new();
            io::copy(&mut file, &mut hasher)?;
            let hash_bytes = hasher.finalize();
            let actual_hash = format!("{:x}", hash_bytes);

            if actual_hash != *expected_hash {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_pkg_{name}_{}",
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
    fn test_package_read_write() {
        let dir = TempDir::new("rw");
        let manifest = PackageManifest {
            version: sample_version(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            book_count: 10,
            vector_count: 100,
            total_size_bytes: 1000,
        };
        let manifest_json = serde_json::to_string(&manifest).unwrap();

        let mut checksums = HashMap::new();
        checksums.insert("vectors.bin".to_string(), "hash123".to_string());

        let pkg = IndexPackage {
            manifest,
            manifest_json,
            checksums,
        };

        IndexPackage::write(dir.path(), &pkg).unwrap();

        let pkg2 = IndexPackage::read(dir.path()).unwrap();
        assert_eq!(pkg2.manifest.book_count, 10);
        assert_eq!(pkg2.checksums.get("vectors.bin").unwrap(), "hash123");
    }

    #[test]
    fn test_checksum_verification() {
        let dir = TempDir::new("checksum");
        let file_path = dir.path().join("test.bin");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"hello world").unwrap();

        let mut hasher = Sha256::new();
        hasher.update(b"hello world");
        let hash = format!("{:x}", hasher.finalize());

        let manifest = PackageManifest {
            version: sample_version(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            book_count: 1,
            vector_count: 1,
            total_size_bytes: 11,
        };
        let manifest_json = serde_json::to_string(&manifest).unwrap();

        let mut checksums = HashMap::new();
        checksums.insert("test.bin".to_string(), hash);

        let pkg = IndexPackage {
            manifest,
            manifest_json,
            checksums,
        };

        assert!(pkg.verify_checksums(dir.path()).unwrap());
    }
}
