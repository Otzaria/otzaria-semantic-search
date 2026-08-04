use crate::cloud::package::{validate_payload_name, IndexPackage};
use crate::semantic::versioning::IndexVersion;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
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
                format!("Incompatible index version: {diffs:?}"),
            ));
        }

        let target = &self.config.target_store_path;
        let parent = target.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "target has no parent directory",
            )
        })?;
        let target_name = target.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "target has no directory name")
        })?;
        fs::create_dir_all(parent)?;

        let source = fs::canonicalize(&self.config.source_path)?;
        let parent = fs::canonicalize(parent)?;
        let absolute_target = parent.join(target_name);
        if absolute_target == source || absolute_target.starts_with(&source) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "target store must not be the package directory or one of its children",
            ));
        }
        if target.exists() && !target.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "target store exists but is not a directory",
            ));
        }

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let staging = parent.join(format!(".{}.import-{nonce}", target_name.to_string_lossy()));
        let backup = parent.join(format!(
            ".{}.previous-import-{nonce}",
            target_name.to_string_lossy()
        ));
        fs::create_dir(&staging)?;

        let import_result = (|| {
            let mut filenames: Vec<&String> = package.checksums.keys().collect();
            filenames.sort();
            for filename in filenames {
                validate_payload_name(filename)?;
                let src = source.join(filename);
                let metadata = fs::symlink_metadata(&src)?;
                if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("package payload is not a regular file: {filename:?}"),
                    ));
                }
                fs::copy(src, staging.join(filename))?;
            }
            if !package.verify_checksums(&staging)? {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "staged payload checksum verification failed",
                ));
            }

            let manifest_json = serde_json::to_vec_pretty(&package.manifest)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            fs::write(staging.join("manifest.json"), manifest_json)?;
            let checksums_json = serde_json::to_vec_pretty(&package.checksums)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            fs::write(staging.join("checksums.json"), checksums_json)?;
            replace_directory(&staging, &absolute_target, &backup)
        })();

        if import_result.is_err() && staging.exists() {
            let _ = fs::remove_dir_all(&staging);
        }
        import_result?;

        Ok(ImportResult {
            books_imported: package.manifest.book_count,
            vectors_imported: package.manifest.vector_count,
            import_duration_ms: start_time.elapsed().as_millis(),
        })
    }
}

fn replace_directory(staging: &Path, target: &Path, backup: &Path) -> io::Result<()> {
    if !target.exists() {
        return fs::rename(staging, target);
    }

    if backup.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("import backup already exists: {}", backup.display()),
        ));
    }
    fs::rename(target, backup)?;
    match fs::rename(staging, target) {
        Ok(()) => {
            if let Err(error) = fs::remove_dir_all(backup) {
                log::warn!("Imported index, but could not remove old backup: {error}");
            }
            Ok(())
        }
        Err(error) => {
            let _ = fs::rename(backup, target);
            Err(error)
        }
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
        fs::create_dir_all(&source_path).unwrap();
        fs::write(source_path.join("vectors.bin"), b"data").unwrap();
        use sha2::{Digest, Sha256};
        let mut checksums = HashMap::new();
        checksums.insert(
            "vectors.bin".to_string(),
            format!("{:x}", Sha256::digest(b"data")),
        );

        let pkg = IndexPackage {
            manifest,
            checksums,
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
