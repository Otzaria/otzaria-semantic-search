use crate::semantic::types::BookForIndexing;
use std::io;
use std::path::PathBuf;

pub struct BuildConfig {
    pub model_path: PathBuf,
    pub embedding_dim: u32,
    pub output_path: PathBuf,
    pub batch_size: usize,
}

pub struct BuildResult {
    pub package_path: PathBuf,
    pub books_processed: u32,
    pub vectors_generated: u32,
    pub build_duration_ms: u128,
}

pub struct CloudIndexBuilder {
    config: BuildConfig,
}

impl CloudIndexBuilder {
    pub fn new(config: BuildConfig) -> Self {
        Self { config }
    }

    /// Build a cloud index package from a set of books.
    ///
    /// The intended flow is:
    /// 1. Initialize embedding model.
    /// 2. Initialize a ZevcStore at a temporary directory.
    /// 3. For each book, chunk and embed the contents.
    /// 4. Insert vectors into the ZevcStore.
    /// 5. Commit the store.
    /// 6. Compute checksums of all the generated files.
    /// 7. Create an IndexPackage with a PackageManifest.
    /// 8. Write the package to `config.output_path`.
    pub fn build(&self, _books: &[BookForIndexing]) -> io::Result<BuildResult> {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "CloudIndexBuilder::build is not yet implemented",
        ))
    }
}
