//! What a model path names on disk: which container format it is, and — for ONNX —
//! which files around it make up the package that produces the vectors.
//!
//! Always compiled, and free of any inference dependency: the format decides which
//! backend [`select_backend`](crate::semantic::backend::select_backend) may walk to, and
//! that decision has to be made identically in a build that can serve the format and in
//! one that can only say which feature is missing.

use std::path::{Path, PathBuf};

/// The container a model path names.
///
/// Decided by the path alone, before anything is opened, because it selects the
/// *validator* as well as the backend: sniffing the bytes first would mean reading a
/// file with a parser chosen by guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// A llama.cpp GGUF container: one self-contained file, tokenizer included.
    Gguf,
    /// An ONNX graph. The path names the graph file; the package is the graph plus
    /// `tokenizer.json` (and any external-data file) in the same directory — see
    /// [`onnx_package_root`].
    Onnx,
}

impl ModelFormat {
    /// Every format, in the order error messages list them.
    pub const ALL: [Self; 2] = [Self::Gguf, Self::Onnx];

    /// `.onnx` in any ASCII case is ONNX; **every other path is GGUF**, including a path
    /// with no extension at all.
    ///
    /// The default is GGUF and not "unknown" so that every model path that worked before
    /// ONNX existed keeps meaning what it meant, byte for byte — the production model has
    /// always been configured by path, and nothing ever required its extension to be
    /// `.gguf`.
    pub fn of(model_path: &Path) -> Self {
        match model_path.extension() {
            Some(extension) if extension.eq_ignore_ascii_case("onnx") => Self::Onnx,
            _ => Self::Gguf,
        }
    }

    /// The format's name as messages spell it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::Onnx => "ONNX",
        }
    }

    /// The cargo feature that compiles in this format's real backend — what a message
    /// that no backend serves the format has to name, since the fix is a rebuild.
    pub const fn backend_feature(self) -> &'static str {
        match self {
            Self::Gguf => "llama-backend",
            Self::Onnx => "onnx-backend",
        }
    }
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The file name the tokenizer must have, directly under the package root.
pub const ONNX_TOKENIZER_FILE: &str = "tokenizer.json";

/// The root of the ONNX package `graph` belongs to: the directory holding the graph.
///
/// A bare file name has the empty path as its parent, which already means the current
/// directory; the only paths with no parent at all name no file (`/`, the empty path),
/// and for those the current directory is as good an answer as any — the graph itself
/// will then fail to open.
pub fn onnx_package_root(graph: &Path) -> PathBuf {
    graph
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

/// Where the tokenizer of the package `graph` belongs to must be: `tokenizer.json` at
/// the package root, and nowhere else. Not searched for, because a tokenizer found
/// somewhere else is a tokenizer the package checksum does not cover.
pub fn onnx_tokenizer_path(graph: &Path) -> PathBuf {
    onnx_package_root(graph).join(ONNX_TOKENIZER_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_onnx_extension_in_any_ascii_case_is_onnx() {
        for path in [
            "model.onnx",
            "model.ONNX",
            "model.OnNx",
            "dir/seforim-embed-round2-fp32.onnx",
            "/abs/dir.with.dots/graph.onnx",
        ] {
            assert_eq!(
                ModelFormat::of(Path::new(path)),
                ModelFormat::Onnx,
                "{path} names an ONNX graph"
            );
        }
    }

    /// Every path that worked before ONNX existed must keep meaning GGUF, including the
    /// ones that never had a `.gguf` extension.
    #[test]
    fn every_other_path_is_gguf() {
        for path in [
            "model.gguf",
            "models/otzaria-embedding-v1-flash-q4.gguf",
            "model",
            "model.bin",
            "model.onnx.part",
            "model.onnx.gguf",
            "onnx",
            ".onnx",
            "model.onnx ",
            "",
        ] {
            assert_eq!(
                ModelFormat::of(Path::new(path)),
                ModelFormat::Gguf,
                "{path:?} must stay GGUF"
            );
        }
    }

    #[test]
    fn each_format_names_the_feature_that_serves_it() {
        assert_eq!(ModelFormat::Gguf.backend_feature(), "llama-backend");
        assert_eq!(ModelFormat::Onnx.backend_feature(), "onnx-backend");
        assert_eq!(ModelFormat::Gguf.to_string(), "GGUF");
        assert_eq!(ModelFormat::Onnx.to_string(), "ONNX");
    }

    #[test]
    fn the_tokenizer_sits_beside_the_graph() {
        assert_eq!(
            onnx_tokenizer_path(Path::new("/models/meivin/graph.onnx")),
            Path::new("/models/meivin/tokenizer.json")
        );
        assert_eq!(
            onnx_tokenizer_path(Path::new("graph.onnx")),
            Path::new("tokenizer.json")
        );
        assert_eq!(onnx_package_root(Path::new("/")), Path::new("."));
    }
}
