//! What a model path names on disk: which container format it is, and — for ONNX —
//! which files around it make up the package that produces the vectors, whether each of
//! them is whole, and the one checksum that names them all.
//!
//! Always compiled, and free of any inference dependency: the format decides which
//! backend [`select_backend`](crate::semantic::backend::select_backend) may walk to and
//! which validator [`EmbeddingRuntime::load`](crate::semantic::embedding::EmbeddingRuntime::load)
//! runs, and both decisions have to be made identically in a build that can serve the
//! format and in one that can only say which feature is missing.
//!
//! # The ONNX package
//!
//! ```text
//! <root>/                       the graph's own directory
//! ├── <graph>.onnx              what `model_path` names
//! ├── tokenizer.json            required: the package's own tokenizer
//! └── <external data>           whatever the graph's tensors name, relative to <root>
//! ```
//!
//! Nothing else in the directory belongs to the package. A README, a licence, an export
//! script, a second graph or an ONNX Runtime library placed beside the graph never
//! reaches a vector, so none of them may change the checksum.
//!
//! # The checksum (`otzaria-onnx-package-v1`)
//!
//! ```text
//! manifest       = "otzaria-onnx-package-v1\n"
//!                + one line per file, ordered by relpath bytewise:
//!                  relpath "\t" size_in_bytes "\t" sha256_lowercase_hex "\n"
//! model_checksum = sha256(manifest as UTF-8), 64 lowercase hex digits
//! ```
//!
//! `relpath` is relative to the root and `/`-separated. The definition is part of the
//! artifact contract (`docs/ARTIFACT_CONTRACT.md` §4.2), and the build tooling computes it
//! independently: `the_golden_package_has_the_documented_checksum` pins the value both
//! have to reach.

use crate::errors::EmbeddingError;
use crate::semantic::embedding::{
    hex_encode, validate_and_checksum_gguf, HashingReader, ReadError,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Seek};
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

/// The first line of the canonical package manifest. A change to what the checksum
/// covers or how it is written is a new name here, never a new value under this one:
/// every artifact built so far declares a checksum computed under it.
pub const ONNX_PACKAGE_MANIFEST_VERSION: &str = "otzaria-onnx-package-v1";

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

/// A model that validated, and the `model_checksum` that names it.
#[derive(Debug, Clone)]
pub enum ValidatedModel {
    /// A GGUF container. Its checksum is the file's own SHA-256.
    Gguf { checksum: String },
    /// An ONNX package. Its checksum is the package checksum over every file in it.
    Onnx(OnnxPackage),
}

impl ValidatedModel {
    pub fn format(&self) -> ModelFormat {
        match self {
            Self::Gguf { .. } => ModelFormat::Gguf,
            Self::Onnx(_) => ModelFormat::Onnx,
        }
    }

    /// What the manifest records as `model_checksum`.
    pub fn checksum(&self) -> &str {
        match self {
            Self::Gguf { checksum } => checksum,
            Self::Onnx(package) => package.checksum(),
        }
    }
}

/// Validate the model `model_path` names and compute its `model_checksum`, by the format
/// the path names — the one entry point
/// [`EmbeddingRuntime::load`](crate::semantic::embedding::EmbeddingRuntime::load) calls.
///
/// A GGUF path goes to [`validate_and_checksum_gguf`], unchanged; an ONNX path to
/// [`validate_onnx_package`].
///
/// # Errors
///
/// Whatever the format's validator refuses the model with.
pub fn validate_model(model_path: &Path) -> Result<ValidatedModel, EmbeddingError> {
    match ModelFormat::of(model_path) {
        ModelFormat::Gguf => {
            validate_and_checksum_gguf(model_path).map(|checksum| ValidatedModel::Gguf { checksum })
        }
        ModelFormat::Onnx => validate_onnx_package(model_path).map(ValidatedModel::Onnx),
    }
}

/// One file of an ONNX package, as the checksum describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageFile {
    /// Relative to the package root, `/`-separated, as the manifest spells it.
    pub relpath: String,
    /// Where it was read from.
    pub path: PathBuf,
    pub size: u64,
    /// Lowercase hex SHA-256 of the contents.
    pub sha256: String,
}

/// What the graph walk learned beyond the fact that the graph is whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphFacts {
    pub ir_version: i64,
    pub opset_imports: u64,
    /// Declared by the main graph, not by nested ones.
    pub graph_inputs: u64,
    pub graph_outputs: u64,
    /// Tensors whose data lives in an external file, counted per reference.
    pub external_tensors: u64,
}

/// An ONNX model package that validated: every file in it, whole, and its checksum.
#[derive(Debug, Clone)]
pub struct OnnxPackage {
    root: PathBuf,
    graph: PathBuf,
    tokenizer: PathBuf,
    /// Ordered by `relpath`, bytewise — the order the manifest lists them in.
    files: Vec<PackageFile>,
    checksum: String,
    facts: GraphFacts,
}

impl OnnxPackage {
    /// The directory the package lives in.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The graph file, as the configuration named it.
    pub fn graph_path(&self) -> &Path {
        &self.graph
    }

    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer
    }

    /// Every file of the package, in manifest order: the graph, `tokenizer.json`, and
    /// every external-data file the graph names.
    pub fn files(&self) -> &[PackageFile] {
        &self.files
    }

    /// The package checksum — what the manifest records as `model_checksum`.
    pub fn checksum(&self) -> &str {
        &self.checksum
    }

    /// The exact text the checksum is the SHA-256 of.
    pub fn manifest_text(&self) -> String {
        onnx_package_manifest(&self.files)
    }

    pub fn graph_facts(&self) -> GraphFacts {
        self.facts
    }
}

/// The canonical manifest text for `files`, in any order: they are sorted by `relpath`,
/// bytewise, before being written.
///
/// Public because it is the definition, and tooling that has to reproduce the checksum
/// is better served comparing texts than comparing digests.
pub fn onnx_package_manifest(files: &[PackageFile]) -> String {
    use std::fmt::Write;
    let mut ordered: Vec<&PackageFile> = files.iter().collect();
    ordered.sort_by(|a, b| a.relpath.as_bytes().cmp(b.relpath.as_bytes()));

    let mut manifest = format!("{ONNX_PACKAGE_MANIFEST_VERSION}\n");
    for file in ordered {
        // `\n`, not the platform's line ending: `writeln!` always writes a bare `\n`.
        let _ = writeln!(manifest, "{}\t{}\t{}", file.relpath, file.size, file.sha256);
    }
    manifest
}

/// The package checksum of `files`: SHA-256 of [`onnx_package_manifest`], lowercase hex.
pub fn onnx_package_checksum(files: &[PackageFile]) -> String {
    hex_encode(&Sha256::digest(onnx_package_manifest(files).as_bytes()))
}

/// Refused rather than read: a real tokenizer is a few megabytes, and this one is read
/// whole to be checked.
const MAX_TOKENIZER_BYTES: u64 = 256 << 20;

/// How deeply a `tokenizer.json` may nest its arrays and objects: far past any real one
/// (about five levels), and `serde_json`'s own recursion limit. Checked by a scan before
/// the file is parsed, identically in `tools/onnx_package_checksum.py`, so that neither
/// parser's recursion decides which packages are accepted — see `read_tokenizer`.
const MAX_TOKENIZER_JSON_DEPTH: usize = 128;

/// How many bytes of the graph's start are examined to name a *kind* of non-ONNX file —
/// a Git LFS pointer, an error page — rather than only refusing it.
const SNIFF_BYTES: usize = 64;

/// Validate the ONNX package whose graph is `graph`, and compute its checksum — every
/// file read exactly once.
///
/// In order, cheapest refusal first:
///
/// 1. `tokenizer.json` exists beside the graph — a missing one fails in microseconds
///    rather than after hashing the graph;
/// 2. the graph's first bytes are not a known kind of *other* file;
/// 3. the graph is walked as protobuf and hashed in the same pass — see `GraphWalk`,
///    which refuses a truncated or malformed file, and then the model itself if it
///    declares no `ir_version`, no `opset_import`, no graph, or a graph without an input
///    or an output;
/// 4. every external-data file the graph's tensors name is inside the package, present,
///    and at least as long as its references need; each is hashed;
/// 5. `tokenizer.json` is hashed and is a JSON object.
///
/// **Not download verification**, exactly as for GGUF: a checksum computed from the
/// files cannot attest to them. It detects that the bytes behind a model path changed.
///
/// # Errors
///
/// [`EmbeddingError::ModelNotFound`] for a graph that does not exist,
/// [`EmbeddingError::TokenizerNotFound`] for a missing `tokenizer.json`,
/// [`EmbeddingError::InvalidModelFile`] — naming the graph, and in its reason the file
/// at fault — for anything incomplete, malformed or unsafe, and
/// [`EmbeddingError::LoadFailed`] for a read that failed.
pub fn validate_onnx_package(graph: &Path) -> Result<OnnxPackage, EmbeddingError> {
    let invalid = |reason: String| EmbeddingError::InvalidModelFile {
        path: graph.display().to_string(),
        reason,
    };
    let unreadable = |what: &Path, e: std::io::Error| EmbeddingError::LoadFailed {
        reason: format!("cannot read {}: {e}", what.display()),
    };

    let graph_metadata = match std::fs::metadata(graph) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(EmbeddingError::ModelNotFound {
                path: graph.display().to_string(),
            })
        }
        Err(e) => return Err(unreadable(graph, e)),
    };
    if !graph_metadata.is_file() {
        return Err(invalid(
            "it is not a regular file; model_path must name the graph file itself".to_string(),
        ));
    }
    let graph_relpath = graph
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            invalid(
                "its file name is not UTF-8, and the package checksum names every file in \
                 UTF-8; rename the graph"
                    .to_string(),
            )
        })?
        .to_string();
    if let Some(reason) = unsafe_in_manifest(&graph_relpath) {
        return Err(invalid(format!(
            "its file name {graph_relpath:?} contains {reason}; rename the graph"
        )));
    }

    let root = onnx_package_root(graph);
    // The empty path is where a bare file name lives, but no file-system call accepts it.
    let fs_root = if root.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        root.clone()
    };

    // ── 1. the tokenizer is there at all ──
    let tokenizer = root.join(ONNX_TOKENIZER_FILE);
    match std::fs::metadata(&tokenizer) {
        Ok(metadata) if metadata.is_file() => {}
        Ok(_) => {
            return Err(invalid(format!(
                "its tokenizer {} is not a regular file",
                tokenizer.display()
            )))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(EmbeddingError::TokenizerNotFound {
                path: tokenizer.display().to_string(),
            })
        }
        Err(e) => return Err(unreadable(&tokenizer, e)),
    }

    // ── 2 and 3. the graph, sniffed, then walked and hashed in one pass ──
    let walked = walk_graph_file(graph).map_err(|failure| match failure {
        GraphFailure::Invalid(reason) => invalid(reason),
        GraphFailure::Unreadable(e) => unreadable(graph, e),
    })?;
    let facts = walked.facts().map_err(invalid)?;

    // ── 4. external data ──
    let external = resolve_external_data(
        &fs_root,
        graph,
        &graph_relpath,
        &tokenizer,
        &walked.summary.external,
    )
    .map_err(|failure| match failure {
        GraphFailure::Invalid(reason) => invalid(reason),
        GraphFailure::Unreadable(e) => unreadable(&root, e),
    })?;

    // ── 5. the tokenizer, whole ──
    let tokenizer_file = read_tokenizer(&tokenizer).map_err(|failure| match failure {
        GraphFailure::Invalid(reason) => invalid(reason),
        GraphFailure::Unreadable(e) => unreadable(&tokenizer, e),
    })?;

    let mut files = Vec::with_capacity(2 + external.len());
    files.push(PackageFile {
        relpath: graph_relpath,
        path: graph.to_path_buf(),
        size: walked.size,
        sha256: walked.sha256,
    });
    files.push(tokenizer_file);
    files.extend(external);
    files.sort_by(|a, b| a.relpath.as_bytes().cmp(b.relpath.as_bytes()));
    let checksum = onnx_package_checksum(&files);

    Ok(OnnxPackage {
        root,
        graph: graph.to_path_buf(),
        tokenizer,
        files,
        checksum,
        facts,
    })
}

/// Why a package file was refused: something wrong with the file, or a read that
/// failed. Only the first condemns the model.
enum GraphFailure {
    Invalid(String),
    Unreadable(std::io::Error),
}

impl From<std::io::Error> for GraphFailure {
    fn from(e: std::io::Error) -> Self {
        Self::Unreadable(e)
    }
}

/// What in a package-relative name would make the manifest ambiguous or the package
/// unportable, or `None`.
///
/// A tab or a newline would let one file's line pass for two; the rest are the
/// characters Windows refuses in a file name, so a package that used one would install
/// on one platform and fail to open on another.
fn unsafe_in_manifest(relpath: &str) -> Option<&'static str> {
    if relpath.chars().any(char::is_control) {
        return Some("a control character");
    }
    if relpath.contains('\\') {
        return Some(
            "a backslash, which Windows reads as a separator and every other platform as a \
             character — write the path with '/'",
        );
    }
    if relpath.contains([':', '<', '>', '"', '|', '?', '*']) {
        return Some("a character Windows does not allow in a file name");
    }
    None
}

/// The graph file, walked: its size and hash, and what the walk collected.
struct WalkedGraph {
    size: u64,
    sha256: String,
    summary: GraphSummary,
}

impl WalkedGraph {
    /// The model-level requirements, checked once the whole file has been read.
    fn facts(&self) -> Result<GraphFacts, String> {
        let summary = &self.summary;
        let ir_version = match summary.ir_version {
            None => {
                return Err(
                    "it declares no ir_version, which every ONNX model carries; it is \
                     not an ONNX model, or not all of one"
                        .to_string(),
                )
            }
            Some(version) if version <= 0 => {
                return Err(format!(
                    "its ir_version is {version}; every ONNX IR version is positive"
                ))
            }
            Some(version) => version,
        };
        if summary.opset_imports == 0 {
            return Err(
                "it declares no opset_import, so none of its operators can be resolved; if it \
                 was downloaded, the download may have stopped early"
                    .to_string(),
            );
        }
        if let Some(version) = summary.invalid_opset_version {
            return Err(format!(
                "it imports an operator set at version {version}; every operator set version \
                 is at least 1"
            ));
        }
        if summary.graphs == 0 {
            return Err("it holds no graph".to_string());
        }
        if summary.graph_inputs == 0 {
            return Err("its graph declares no input, so there is nothing to feed it".to_string());
        }
        if summary.graph_outputs == 0 {
            return Err(
                "its graph declares no output, so there is no vector to take from it".to_string(),
            );
        }
        Ok(GraphFacts {
            ir_version,
            opset_imports: summary.opset_imports,
            graph_inputs: summary.graph_inputs,
            graph_outputs: summary.graph_outputs,
            external_tensors: summary.external.len() as u64,
        })
    }
}

/// Name the kind of file `prefix` begins, when it is a kind people mistake for a model.
///
/// Only a better message: every prefix recognized here is also refused by the walk
/// within its first byte, because none of them begins with a field a `ModelProto` has,
/// in the wire type that field has — so this can never refuse a real graph.
fn sniff_non_onnx(prefix: &[u8], file_len: u64) -> Option<&'static str> {
    if file_len == 0 {
        return Some("the file is empty");
    }
    if prefix.starts_with(b"version https://git-lfs.github.com/spec/") {
        return Some(
            "it is a Git LFS pointer, not the model: the repository was cloned without its \
             LFS objects. Run `git lfs pull` in it, or download the file itself",
        );
    }
    if prefix.starts_with(b"GGUF") {
        return Some(
            "it is a GGUF container, not an ONNX graph. Only a path ending in .onnx is read \
             as ONNX; give a GGUF model its own .gguf name",
        );
    }
    if prefix.starts_with(b"PK\x03\x04") {
        return Some("it is a ZIP archive; extract the model from it first");
    }
    match prefix.trim_ascii_start().first() {
        Some(b'<') => Some(
            "it is an HTML or XML document — typically an error page saved in place of the \
             model; download it again",
        ),
        Some(b'{' | b'[') => Some(
            "it is a JSON document — typically an error response saved in place of the model; \
             download it again",
        ),
        _ => None,
    }
}

/// Sniff, walk and hash the graph file.
fn walk_graph_file(graph: &Path) -> Result<WalkedGraph, GraphFailure> {
    let mut file = std::fs::File::open(graph)?;
    let file_len = file.metadata()?.len();

    let mut prefix = [0u8; SNIFF_BYTES];
    let mut sniffed = 0;
    while sniffed < prefix.len() {
        match file.read(&mut prefix[sniffed..])? {
            0 => break,
            read => sniffed += read,
        }
    }
    if let Some(kind) = sniff_non_onnx(&prefix[..sniffed], file_len) {
        return Err(GraphFailure::Invalid(kind.to_string()));
    }
    file.rewind()?;

    let mut walk = GraphWalk {
        reader: HashingReader::new(file),
        file_len,
        messages: 0,
        regions: 0,
        summary: GraphSummary::default(),
    };
    walk.model().map_err(|error| match error {
        WalkError::Io(e) => GraphFailure::Unreadable(e),
        other => GraphFailure::Invalid(other.describe(file_len)),
    })?;

    let GraphWalk {
        reader, summary, ..
    } = walk;
    let (sha256, size) = reader.finish()?;
    if size != file_len {
        return Err(GraphFailure::Invalid(format!(
            "it changed while it was read ({file_len} bytes when opened, {size} read); \
             validate it again once nothing is writing to it"
        )));
    }
    Ok(WalkedGraph {
        size,
        sha256,
        summary,
    })
}

// ── the protobuf walk ─────────────────────────────────────────────────────────────
//
// Field numbers from `onnx/onnx.proto3` (onnx/onnx, main at adf47535). Only what the
// package needs is read — the model-level requirements, and every `TensorProto` that can
// keep its data in an external file — and everything else is skipped, hashed, and never
// held. Nothing is ever loaded whole: `raw_data`, the bulk of a real graph, is skipped
// in 64 KiB chunks.

/// Deeper than any real model nests — model, graph, node, attribute, graph… is three
/// levels per subgraph — and shallow enough that recursion cannot exhaust a test
/// thread's stack.
const MAX_MESSAGE_DEPTH: u32 = 64;

/// A real model holds thousands of messages; a count past this is not a model.
const MAX_MESSAGES: u64 = 1 << 24;

/// Tensors with external data, counted per reference.
const MAX_EXTERNAL_TENSORS: usize = 1 << 20;

/// Distinct external-data files one package may name.
const MAX_EXTERNAL_FILES: usize = 1 << 12;

/// An external-data key or value, e.g. a location.
const MAX_EXTERNAL_DATA_ENTRY_BYTES: u64 = 4096;

/// External-data entries on one tensor. The specification names four keys.
const MAX_EXTERNAL_DATA_ENTRIES: usize = 64;

/// How much of a tensor's name is kept, for messages only.
const MAX_KEPT_NAME_BYTES: u64 = 256;

/// Dimensions one tensor may declare.
const MAX_TENSOR_DIMS: usize = 256;

/// `TensorProto.DataLocation.EXTERNAL`.
const DATA_LOCATION_EXTERNAL: u64 = 1;

/// `TensorProto.DataType.STRING` and `UNDEFINED`: no fixed width, so no size floor.
const DATA_TYPE_UNDEFINED: i32 = 0;
const DATA_TYPE_STRING: i32 = 8;

/// A protobuf wire type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wire {
    Varint,
    Fixed64,
    Len,
    StartGroup,
    EndGroup,
    Fixed32,
}

impl std::fmt::Display for Wire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Varint => "a varint",
            Self::Fixed64 => "a fixed64",
            Self::Len => "a length-delimited value",
            Self::StartGroup => "a group",
            Self::EndGroup => "an end-group marker",
            Self::Fixed32 => "a fixed32",
        })
    }
}

/// The messages the walk descends into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Message {
    Model,
    Graph,
    Node,
    Attribute,
    Tensor,
    SparseTensor,
    TrainingInfo,
    Function,
    OperatorSetId,
    StringStringEntry,
}

impl Message {
    fn name(self) -> &'static str {
        match self {
            Self::Model => "ModelProto",
            Self::Graph => "GraphProto",
            Self::Node => "NodeProto",
            Self::Attribute => "AttributeProto",
            Self::Tensor => "TensorProto",
            Self::SparseTensor => "SparseTensorProto",
            Self::TrainingInfo => "TrainingInfoProto",
            Self::Function => "FunctionProto",
            Self::OperatorSetId => "OperatorSetIdProto",
            Self::StringStringEntry => "StringStringEntryProto",
        }
    }

    /// The wire types a field this message declares may arrive in, or `None` for a
    /// field it does not declare — skipped, as every protobuf parser skips an unknown
    /// field, so a model from a newer ONNX still validates.
    ///
    /// Every field is listed, not only the ones the walk reads: a declared field in the
    /// wrong wire type is not a newer ONNX but bytes that are not ONNX at all, and
    /// refusing it is what makes garbage fail early. A repeated scalar may arrive packed
    /// or not, and both are accepted, as protobuf requires.
    fn allowed(self, field: u32) -> Option<&'static [Wire]> {
        use Wire::{Fixed32, Fixed64, Len, Varint};
        const LEN: &[Wire] = &[Len];
        const VARINT: &[Wire] = &[Varint];
        const FIXED32: &[Wire] = &[Fixed32];
        const REPEATED_VARINT: &[Wire] = &[Varint, Len];
        const REPEATED_FIXED32: &[Wire] = &[Fixed32, Len];
        const REPEATED_FIXED64: &[Wire] = &[Fixed64, Len];

        Some(match (self, field) {
            // ir_version, model_version | producer_name, producer_version, domain,
            // doc_string, graph, opset_import, metadata_props, training_info, functions,
            // configuration
            (Self::Model, 1 | 5) => VARINT,
            (Self::Model, 2 | 3 | 4 | 6 | 7 | 8 | 14 | 20 | 25 | 26) => LEN,
            // node, name, initializer, doc_string, input, output, value_info,
            // quantization_annotation, sparse_initializer, metadata_props
            (Self::Graph, 1 | 2 | 5 | 10 | 11 | 12 | 13 | 14 | 15 | 16) => LEN,
            // input, output, name, op_type, attribute, doc_string, domain, overload,
            // metadata_props, device_configurations
            (Self::Node, 1..=10) => LEN,
            // f | i, type | floats | ints | name, s, t, g, strings, tensors, graphs,
            // doc_string, tp, type_protos, ref_attr_name, sparse_tensor, sparse_tensors
            (Self::Attribute, 2) => FIXED32,
            (Self::Attribute, 3 | 20) => VARINT,
            (Self::Attribute, 7) => REPEATED_FIXED32,
            (Self::Attribute, 8) => REPEATED_VARINT,
            (Self::Attribute, 1 | 4 | 5 | 6 | 9 | 10 | 11 | 13 | 14 | 15 | 21 | 22 | 23) => LEN,
            // dims, int32_data, int64_data, uint64_data | data_type, data_location |
            // float_data | double_data | segment, string_data, name, raw_data, doc_string,
            // external_data, metadata_props
            (Self::Tensor, 1 | 5 | 7 | 11) => REPEATED_VARINT,
            (Self::Tensor, 2 | 14) => VARINT,
            (Self::Tensor, 4) => REPEATED_FIXED32,
            (Self::Tensor, 10) => REPEATED_FIXED64,
            (Self::Tensor, 3 | 6 | 8 | 9 | 12 | 13 | 16) => LEN,
            // values, indices | dims
            (Self::SparseTensor, 1 | 2) => LEN,
            (Self::SparseTensor, 3) => REPEATED_VARINT,
            // initialization, algorithm, initialization_binding, update_binding
            (Self::TrainingInfo, 1..=4) => LEN,
            // name, input, output, attribute, node, doc_string, opset_import, domain,
            // attribute_proto, value_info, overload, metadata_props (2 and 3 are reserved)
            (Self::Function, 1 | 4..=14) => LEN,
            // domain | version
            (Self::OperatorSetId, 1) => LEN,
            (Self::OperatorSetId, 2) => VARINT,
            // key, value
            (Self::StringStringEntry, 1 | 2) => LEN,
            _ => return None,
        })
    }
}

/// One tensor whose data lives in another file, as the graph declares it.
#[derive(Debug, Clone)]
struct ExternalReference {
    /// What the tensor is, for messages: `initializer 'encoder.weight'`.
    tensor: String,
    location: String,
    /// The byte the tensor's data must reach: `offset + length`, or, with no length,
    /// `offset` plus one bit per element — below every ONNX data type, so it can only
    /// under-demand.
    needs_bytes: u64,
}

/// What the walk collects: the model-level facts, and every external reference.
#[derive(Debug, Default)]
struct GraphSummary {
    /// The last one wins, as protobuf defines for a repeated scalar field.
    ir_version: Option<i64>,
    opset_imports: u64,
    /// The first operator set whose version is below 1.
    invalid_opset_version: Option<i64>,
    /// Occurrences of `ModelProto.graph`; protobuf merges repeats into one graph.
    graphs: u64,
    graph_inputs: u64,
    graph_outputs: u64,
    external: Vec<ExternalReference>,
}

/// Why the walk stopped.
enum WalkError {
    /// The file ends before something it declared: an incomplete download.
    Truncated {
        at: u64,
        wanted: &'static str,
        needs: u64,
    },
    /// Bytes that are not a valid ONNX protobuf.
    Malformed(String),
    /// A valid-looking file past what this validator accepts.
    Limit(String),
    Io(std::io::Error),
}

impl WalkError {
    fn describe(self, file_len: u64) -> String {
        match self {
            Self::Truncated { at, wanted, needs } => format!(
                "the file is {file_len} bytes, and {wanted} starting at byte {at} runs to byte \
                 {needs} — the download is incomplete; download the model again"
            ),
            Self::Malformed(reason) => format!("it is not a valid ONNX model: {reason}"),
            Self::Limit(reason) => format!("it is past what this validator accepts: {reason}"),
            Self::Io(e) => format!("it could not be read: {e}"),
        }
    }
}

/// A forward-only protobuf walk over the graph file, hashing every byte it passes.
///
/// Every read is bounded by the end of the message it belongs to, and every message by
/// the one holding it. Only the outermost message is bounded by the end of the *file*,
/// so only there does a read that runs past its bound prove a truncated download. A
/// nested message's own length was checked against the file before it was entered, so
/// a field that runs past *it* is a malformed file — even when that message happens to
/// end exactly where the file does, and even when the field would run past the file
/// too.
struct GraphWalk {
    reader: HashingReader,
    file_len: u64,
    messages: u64,
    /// How many length-bounded messages the walk is inside; 1 is the file's own
    /// `ModelProto`.
    regions: u32,
    summary: GraphSummary,
}

impl GraphWalk {
    fn pos(&self) -> u64 {
        self.reader.consumed()
    }

    fn io(&self, error: ReadError, wanted: &'static str) -> WalkError {
        match error {
            // The length was checked against the file first, so this is a file that
            // shrank while it was read.
            ReadError::Eof { at } => WalkError::Truncated {
                at,
                wanted,
                needs: self.file_len,
            },
            ReadError::Io(e) => WalkError::Io(e),
        }
    }

    /// Refuse a read of `bytes` that would pass `end`: as truncation in the outermost
    /// message, whose bound is the end of the file, and as malformation anywhere else.
    fn within(&self, bytes: u64, end: u64, wanted: &'static str) -> Result<(), WalkError> {
        let at = self.pos();
        let Some(needs) = at.checked_add(bytes) else {
            return Err(WalkError::Malformed(format!(
                "{wanted} at byte {at} declares {bytes} bytes, which no file can hold"
            )));
        };
        if needs <= end {
            return Ok(());
        }
        if self.regions <= 1 {
            return Err(WalkError::Truncated { at, wanted, needs });
        }
        Err(WalkError::Malformed(format!(
            "{wanted} at byte {at} runs past the end of the message holding it (byte {end})"
        )))
    }

    fn byte(&mut self, end: u64, wanted: &'static str) -> Result<u8, WalkError> {
        self.within(1, end, wanted)?;
        let mut byte = [0u8; 1];
        self.reader
            .fill(&mut byte)
            .map_err(|e| self.io(e, wanted))?;
        Ok(byte[0])
    }

    fn varint(&mut self, end: u64, wanted: &'static str) -> Result<u64, WalkError> {
        let mut value = 0u64;
        for index in 0..10u32 {
            let byte = self.byte(end, wanted)?;
            // The tenth byte carries bit 63 alone; anything more is a varint no encoder
            // writes.
            if index == 9 && byte > 1 {
                return Err(WalkError::Malformed(format!(
                    "{wanted} ending at byte {} is a varint wider than 64 bits",
                    self.pos()
                )));
            }
            value |= u64::from(byte & 0x7f) << (7 * index);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(WalkError::Malformed(format!(
            "{wanted} ending at byte {} is a varint longer than ten bytes",
            self.pos()
        )))
    }

    fn skip(&mut self, bytes: u64, end: u64, wanted: &'static str) -> Result<(), WalkError> {
        self.within(bytes, end, wanted)?;
        self.reader.skip(bytes).map_err(|e| self.io(e, wanted))
    }

    fn tag(&mut self, end: u64) -> Result<(u32, Wire), WalkError> {
        let at = self.pos();
        let raw = self.varint(end, "a field tag")?;
        let raw = u32::try_from(raw).map_err(|_| {
            WalkError::Malformed(format!("the field tag at byte {at} is wider than 32 bits"))
        })?;
        let field = raw >> 3;
        if field == 0 {
            return Err(WalkError::Malformed(format!(
                "the field tag at byte {at} names field 0, which protobuf does not have"
            )));
        }
        let wire = match raw & 7 {
            0 => Wire::Varint,
            1 => Wire::Fixed64,
            2 => Wire::Len,
            3 => Wire::StartGroup,
            4 => Wire::EndGroup,
            5 => Wire::Fixed32,
            other => {
                return Err(WalkError::Malformed(format!(
                    "the field tag at byte {at} has wire type {other}, which protobuf does not \
                     have"
                )))
            }
        };
        Ok((field, wire))
    }

    /// Read a length prefix and return where the value it announces ends.
    fn value_end(&mut self, end: u64, wanted: &'static str) -> Result<u64, WalkError> {
        let len = self.varint(end, wanted)?;
        self.within(len, end, wanted)?;
        Ok(self.pos() + len)
    }

    fn enter(&mut self, depth: u32) -> Result<(), WalkError> {
        if depth > MAX_MESSAGE_DEPTH {
            return Err(WalkError::Limit(format!(
                "its messages nest deeper than {MAX_MESSAGE_DEPTH} levels at byte {}",
                self.pos()
            )));
        }
        self.messages += 1;
        if self.messages > MAX_MESSAGES {
            return Err(WalkError::Limit(format!(
                "it holds more than {MAX_MESSAGES} messages"
            )));
        }
        Ok(())
    }

    /// Skip one value whose tag has been read.
    fn skip_value(
        &mut self,
        field: u32,
        wire: Wire,
        end: u64,
        depth: u32,
    ) -> Result<(), WalkError> {
        match wire {
            Wire::Varint => self.varint(end, "a varint field").map(drop),
            Wire::Fixed64 => self.skip(8, end, "a fixed64 field"),
            Wire::Fixed32 => self.skip(4, end, "a fixed32 field"),
            Wire::Len => {
                let stop = self.value_end(end, "a length-delimited field")?;
                self.skip(stop - self.pos(), stop, "a length-delimited field")
            }
            Wire::StartGroup => self.skip_group(field, end, depth + 1),
            Wire::EndGroup => Err(WalkError::Malformed(format!(
                "an end-group marker at byte {} closes no group",
                self.pos()
            ))),
        }
    }

    /// Skip a (deprecated, but valid) group up to the end marker with its field number.
    fn skip_group(&mut self, field: u32, end: u64, depth: u32) -> Result<(), WalkError> {
        self.enter(depth)?;
        loop {
            let (inner, wire) = self.tag(end)?;
            if wire == Wire::EndGroup {
                if inner == field {
                    return Ok(());
                }
                return Err(WalkError::Malformed(format!(
                    "group {field} is closed as group {inner} at byte {}",
                    self.pos()
                )));
            }
            self.skip_value(inner, wire, end, depth)?;
        }
    }

    /// Walk the fields of one `message` running to `end`, handing each to `visit`, which
    /// returns whether it consumed the value; an unconsumed one is skipped.
    fn fields<F>(
        &mut self,
        message: Message,
        end: u64,
        depth: u32,
        mut visit: F,
    ) -> Result<(), WalkError>
    where
        F: FnMut(&mut Self, u32, Wire, u64) -> Result<bool, WalkError>,
    {
        self.enter(depth)?;
        // Left raised on an error: the walk is abandoned then, not resumed.
        self.regions += 1;
        while self.pos() < end {
            let at = self.pos();
            let (field, wire) = self.tag(end)?;
            if wire == Wire::EndGroup {
                return Err(WalkError::Malformed(format!(
                    "an end-group marker at byte {at} closes no group"
                )));
            }
            if let Some(allowed) = message.allowed(field) {
                if !allowed.contains(&wire) {
                    return Err(WalkError::Malformed(format!(
                        "field {field} of {} at byte {at} is encoded as {wire}, which that \
                         field never is",
                        message.name()
                    )));
                }
            }
            if !visit(self, field, wire, end)? {
                self.skip_value(field, wire, end, depth)?;
            }
        }
        self.regions -= 1;
        Ok(())
    }

    /// The file: one `ModelProto`, running to the end.
    fn model(&mut self) -> Result<(), WalkError> {
        let end = self.file_len;
        let mut first = true;
        self.fields(Message::Model, end, 0, |walk, field, _wire, end| {
            // A newer ONNX may add fields, but never before every field an older one
            // knows: a file whose first field is not a ModelProto field is not a model.
            if std::mem::take(&mut first) && Message::Model.allowed(field).is_none() {
                return Err(WalkError::Malformed(format!(
                    "it does not begin like an ONNX model: its first field is number \
                     {field}, which ModelProto does not have"
                )));
            }
            match field {
                1 => {
                    // int64 on the wire: a negative value is its two's complement.
                    walk.summary.ir_version = Some(walk.varint(end, "ir_version")? as i64);
                    Ok(true)
                }
                7 => {
                    let stop = walk.value_end(end, "the graph")?;
                    walk.summary.graphs += 1;
                    walk.graph(stop, 1, true)?;
                    Ok(true)
                }
                8 => {
                    let stop = walk.value_end(end, "an opset_import")?;
                    walk.opset(stop, 1)?;
                    Ok(true)
                }
                20 => {
                    let stop = walk.value_end(end, "a training_info")?;
                    walk.training_info(stop, 1)?;
                    Ok(true)
                }
                25 => {
                    let stop = walk.value_end(end, "a function")?;
                    walk.function(stop, 1)?;
                    Ok(true)
                }
                _ => Ok(false),
            }
        })
    }

    fn opset(&mut self, end: u64, depth: u32) -> Result<(), WalkError> {
        let mut version = 0i64;
        self.fields(
            Message::OperatorSetId,
            end,
            depth,
            |walk, field, _wire, end| {
                if field == 2 {
                    version = walk.varint(end, "an operator set version")? as i64;
                    return Ok(true);
                }
                Ok(false)
            },
        )?;
        self.summary.opset_imports += 1;
        if version < 1 && self.summary.invalid_opset_version.is_none() {
            self.summary.invalid_opset_version = Some(version);
        }
        Ok(())
    }

    /// A graph: the main one when `main`, whose inputs and outputs are counted, or one
    /// nested in an attribute or a training step, walked only for its tensors.
    fn graph(&mut self, end: u64, depth: u32, main: bool) -> Result<(), WalkError> {
        self.fields(
            Message::Graph,
            end,
            depth,
            |walk, field, _wire, end| match field {
                1 => {
                    let stop = walk.value_end(end, "a node")?;
                    walk.node(stop, depth + 1)?;
                    Ok(true)
                }
                5 => {
                    let stop = walk.value_end(end, "an initializer")?;
                    walk.tensor(stop, depth + 1, "initializer")?;
                    Ok(true)
                }
                15 => {
                    let stop = walk.value_end(end, "a sparse initializer")?;
                    walk.sparse_tensor(stop, depth + 1, "sparse initializer")?;
                    Ok(true)
                }
                // Counted, then skipped like any other field.
                11 if main => {
                    walk.summary.graph_inputs += 1;
                    Ok(false)
                }
                12 if main => {
                    walk.summary.graph_outputs += 1;
                    Ok(false)
                }
                _ => Ok(false),
            },
        )
    }

    fn node(&mut self, end: u64, depth: u32) -> Result<(), WalkError> {
        self.fields(Message::Node, end, depth, |walk, field, _wire, end| {
            if field == 5 {
                let stop = walk.value_end(end, "a node attribute")?;
                walk.attribute(stop, depth + 1)?;
                return Ok(true);
            }
            Ok(false)
        })
    }

    /// An attribute: where a constant tensor, a sparse one or a whole subgraph (the body
    /// of an `If`, a `Loop` or a `Scan`) can hide external data.
    fn attribute(&mut self, end: u64, depth: u32) -> Result<(), WalkError> {
        self.fields(Message::Attribute, end, depth, |walk, field, _wire, end| {
            match field {
                5 | 10 => {
                    let stop = walk.value_end(end, "an attribute tensor")?;
                    walk.tensor(stop, depth + 1, "attribute tensor")?;
                }
                6 | 11 => {
                    let stop = walk.value_end(end, "an attribute graph")?;
                    walk.graph(stop, depth + 1, false)?;
                }
                22 | 23 => {
                    let stop = walk.value_end(end, "an attribute sparse tensor")?;
                    walk.sparse_tensor(stop, depth + 1, "attribute sparse tensor")?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        })
    }

    fn sparse_tensor(&mut self, end: u64, depth: u32, role: &'static str) -> Result<(), WalkError> {
        self.fields(
            Message::SparseTensor,
            end,
            depth,
            |walk, field, _wire, end| {
                if field == 1 || field == 2 {
                    let stop = walk.value_end(end, "a sparse tensor's values or indices")?;
                    walk.tensor(stop, depth + 1, role)?;
                    return Ok(true);
                }
                Ok(false)
            },
        )
    }

    fn training_info(&mut self, end: u64, depth: u32) -> Result<(), WalkError> {
        self.fields(
            Message::TrainingInfo,
            end,
            depth,
            |walk, field, _wire, end| {
                if field == 1 || field == 2 {
                    let stop = walk.value_end(end, "a training graph")?;
                    walk.graph(stop, depth + 1, false)?;
                    return Ok(true);
                }
                Ok(false)
            },
        )
    }

    fn function(&mut self, end: u64, depth: u32) -> Result<(), WalkError> {
        self.fields(Message::Function, end, depth, |walk, field, _wire, end| {
            match field {
                7 => {
                    let stop = walk.value_end(end, "a function node")?;
                    walk.node(stop, depth + 1)?;
                }
                11 => {
                    let stop = walk.value_end(end, "a function attribute")?;
                    walk.attribute(stop, depth + 1)?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        })
    }

    /// A tensor: kept only if its data lives elsewhere, and then as its external
    /// reference. `raw_data` — the bulk of a real graph — is skipped, never held.
    fn tensor(&mut self, end: u64, depth: u32, role: &'static str) -> Result<(), WalkError> {
        let mut name = String::new();
        let mut dims: Vec<i64> = Vec::new();
        let mut data_type = DATA_TYPE_UNDEFINED;
        let mut location = 0u64;
        let mut entries: Vec<(Option<String>, Option<String>)> = Vec::new();

        self.fields(Message::Tensor, end, depth, |walk, field, wire, end| {
            match field {
                1 => walk.dims(wire, end, &mut dims)?,
                2 => {
                    // int32 on the wire, sign-extended to 64 bits when negative.
                    data_type = walk.varint(end, "a tensor data_type")? as i64 as i32;
                }
                8 => name = walk.kept_name(end)?,
                13 => {
                    if entries.len() >= MAX_EXTERNAL_DATA_ENTRIES {
                        return Err(WalkError::Limit(format!(
                            "a tensor carries more than {MAX_EXTERNAL_DATA_ENTRIES} \
                             external_data entries"
                        )));
                    }
                    let stop = walk.value_end(end, "an external_data entry")?;
                    entries.push(walk.string_entry(stop, depth + 1)?);
                }
                14 => location = walk.varint(end, "a tensor data_location")?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;

        match location {
            0 => Ok(()),
            DATA_LOCATION_EXTERNAL => {
                let tensor = if name.is_empty() {
                    format!("an unnamed {role}")
                } else {
                    format!("{role} '{name}'")
                };
                let reference = external_reference(tensor, &entries, &dims, data_type)?;
                if self.summary.external.len() >= MAX_EXTERNAL_TENSORS {
                    return Err(WalkError::Limit(format!(
                        "more than {MAX_EXTERNAL_TENSORS} tensors keep their data externally"
                    )));
                }
                self.summary.external.push(reference);
                Ok(())
            }
            other => Err(WalkError::Malformed(format!(
                "a tensor's data_location is {other}, which is neither DEFAULT (0) nor \
                 EXTERNAL (1)"
            ))),
        }
    }

    /// `dims`, packed or not.
    fn dims(&mut self, wire: Wire, end: u64, dims: &mut Vec<i64>) -> Result<(), WalkError> {
        let push = |dims: &mut Vec<i64>, value: u64| {
            if dims.len() >= MAX_TENSOR_DIMS {
                return Err(WalkError::Limit(format!(
                    "a tensor declares more than {MAX_TENSOR_DIMS} dimensions"
                )));
            }
            dims.push(value as i64);
            Ok(())
        };
        match wire {
            Wire::Varint => {
                let value = self.varint(end, "a tensor dimension")?;
                push(dims, value)
            }
            Wire::Len => {
                let stop = self.value_end(end, "packed tensor dimensions")?;
                while self.pos() < stop {
                    let value = self.varint(stop, "a packed tensor dimension")?;
                    push(dims, value)?;
                }
                Ok(())
            }
            other => Err(WalkError::Malformed(format!(
                "tensor dimensions arrive as {other}"
            ))),
        }
    }

    /// A tensor name, kept to [`MAX_KEPT_NAME_BYTES`] for messages; the rest is skipped.
    fn kept_name(&mut self, end: u64) -> Result<String, WalkError> {
        let stop = self.value_end(end, "a tensor name")?;
        let len = stop - self.pos();
        let kept = len.min(MAX_KEPT_NAME_BYTES);
        let mut bytes = vec![0u8; kept as usize];
        self.reader
            .fill(&mut bytes)
            .map_err(|e| self.io(e, "a tensor name"))?;
        self.skip(len - kept, stop, "a tensor name")?;
        let mut name = String::from_utf8_lossy(&bytes).into_owned();
        if kept < len {
            name.push('…');
        }
        Ok(name)
    }

    /// A bounded UTF-8 string.
    fn string(&mut self, end: u64, wanted: &'static str) -> Result<String, WalkError> {
        let stop = self.value_end(end, wanted)?;
        let len = stop - self.pos();
        if len > MAX_EXTERNAL_DATA_ENTRY_BYTES {
            return Err(WalkError::Limit(format!(
                "{wanted} is {len} bytes, and at most {MAX_EXTERNAL_DATA_ENTRY_BYTES} are \
                 accepted"
            )));
        }
        let mut bytes = vec![0u8; len as usize];
        self.reader
            .fill(&mut bytes)
            .map_err(|e| self.io(e, wanted))?;
        String::from_utf8(bytes).map_err(|_| WalkError::Malformed(format!("{wanted} is not UTF-8")))
    }

    fn string_entry(
        &mut self,
        end: u64,
        depth: u32,
    ) -> Result<(Option<String>, Option<String>), WalkError> {
        let mut key = None;
        let mut value = None;
        self.fields(
            Message::StringStringEntry,
            end,
            depth,
            |walk, field, _wire, end| match field {
                1 => {
                    key = Some(walk.string(end, "an external_data key")?);
                    Ok(true)
                }
                2 => {
                    value = Some(walk.string(end, "an external_data value")?);
                    Ok(true)
                }
                _ => Ok(false),
            },
        )?;
        Ok((key, value))
    }
}

/// Resolve one external tensor's entries into what the package must hold for it.
///
/// Stricter than the specification requires in one respect: a key given twice is
/// refused, since which of two locations a runtime reads is its own business, and a
/// validator that guessed differently would hash the wrong file. Keys other than the
/// four the specification names are ignored — none of them can move the data.
fn external_reference(
    tensor: String,
    entries: &[(Option<String>, Option<String>)],
    dims: &[i64],
    data_type: i32,
) -> Result<ExternalReference, WalkError> {
    let mut location = None;
    let mut offset = None;
    let mut length = None;
    for (key, value) in entries {
        let value = value.clone().unwrap_or_default();
        let slot = match key.as_deref() {
            Some("location") => &mut location,
            Some("offset") => &mut offset,
            Some("length") => &mut length,
            _ => continue,
        };
        if slot.is_some() {
            return Err(WalkError::Malformed(format!(
                "{tensor} names its external-data {} twice",
                key.as_deref().unwrap_or_default()
            )));
        }
        *slot = Some(value);
    }

    let Some(location) = location else {
        return Err(WalkError::Malformed(format!(
            "{tensor} keeps its data externally but names no location"
        )));
    };
    let number = |what: &str, value: Option<String>| -> Result<Option<u64>, WalkError> {
        let Some(value) = value else {
            return Ok(None);
        };
        // Digits only: no sign, no space, no base prefix — the specification says "an
        // integer stored as a string", and every exporter writes plain decimal.
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(WalkError::Malformed(format!(
                "{tensor} gives its external-data {what} as {value:?}, which is not a \
                 non-negative integer"
            )));
        }
        value.parse::<u64>().map(Some).map_err(|_| {
            WalkError::Malformed(format!(
                "{tensor} gives its external-data {what} as {value}, which no file can reach"
            ))
        })
    };
    let offset = number("offset", offset)?.unwrap_or(0);
    let length = number("length", length)?;

    let payload = match length {
        Some(length) => length,
        // One bit per element: the smallest ONNX type is two bits wide, so this cannot
        // demand a byte a valid file lacks. Strings have no fixed width at all.
        None if data_type == DATA_TYPE_STRING || data_type == DATA_TYPE_UNDEFINED => 0,
        None => {
            let mut elements = 1u64;
            for &dim in dims {
                let Ok(dim) = u64::try_from(dim) else {
                    return Err(WalkError::Malformed(format!(
                        "{tensor} declares the dimension {dim}"
                    )));
                };
                elements = elements.saturating_mul(dim);
            }
            elements.div_ceil(8)
        }
    };
    let needs_bytes = offset.checked_add(payload).ok_or_else(|| {
        WalkError::Malformed(format!(
            "{tensor}'s external-data offset {offset} plus its length overflows"
        ))
    })?;
    Ok(ExternalReference {
        tensor,
        location,
        needs_bytes,
    })
}

/// The package-relative, `/`-separated form of an external-data `location`, or why it
/// cannot be one.
///
/// `.` and empty components are dropped — `./weights.bin` and `weights.bin` are one file,
/// and must be one line of the manifest — while `..` is refused outright, even where it
/// would stay inside the package: a path that has to be resolved to be judged is a path
/// judged by whoever resolves it.
fn package_relpath(location: &str) -> Result<String, String> {
    if location.is_empty() {
        return Err("an empty location".to_string());
    }
    if let Some(reason) = unsafe_in_manifest(location) {
        return Err(reason.to_string());
    }
    if location.starts_with('/') {
        return Err("an absolute path".to_string());
    }
    if location.ends_with('/') {
        return Err("a trailing '/', which names a directory".to_string());
    }
    let mut components = Vec::new();
    for component in location.split('/') {
        match component {
            "" | "." => {}
            ".." => return Err("a '..' component".to_string()),
            component => components.push(component),
        }
    }
    if components.is_empty() {
        return Err("no file name".to_string());
    }
    Ok(components.join("/"))
}

/// Check every external-data file the graph names, and hash each once.
fn resolve_external_data(
    root: &Path,
    graph: &Path,
    graph_relpath: &str,
    tokenizer: &Path,
    references: &[ExternalReference],
) -> Result<Vec<PackageFile>, GraphFailure> {
    if references.is_empty() {
        return Ok(Vec::new());
    }
    let invalid = GraphFailure::Invalid;

    // Grouped by the file, keeping the reference that reaches furthest into it.
    let mut needed: BTreeMap<String, &ExternalReference> = BTreeMap::new();
    for reference in references {
        let relpath = package_relpath(&reference.location).map_err(|reason| {
            invalid(format!(
                "{} keeps its data in {:?}, which contains {reason}; an external-data location \
                 must be a relative path inside the package directory",
                reference.tensor, reference.location
            ))
        })?;
        if relpath == graph_relpath || relpath == ONNX_TOKENIZER_FILE {
            return Err(invalid(format!(
                "{} keeps its data in {:?}, which is the package's own {}",
                reference.tensor,
                reference.location,
                if relpath == graph_relpath {
                    "graph"
                } else {
                    "tokenizer"
                }
            )));
        }
        let slot = needed.entry(relpath).or_insert(reference);
        if reference.needs_bytes > slot.needs_bytes {
            *slot = reference;
        }
    }
    if needed.len() > MAX_EXTERNAL_FILES {
        return Err(invalid(format!(
            "it names {} external-data files, and at most {MAX_EXTERNAL_FILES} are accepted",
            needed.len()
        )));
    }

    let canonical_root = std::fs::canonicalize(root)?;
    let canonical_graph = std::fs::canonicalize(graph)?;
    let canonical_tokenizer = std::fs::canonicalize(tokenizer)?;

    let mut files = Vec::with_capacity(needed.len());
    for (relpath, reference) in needed {
        let path = root.join(&relpath);
        let canonical = match std::fs::canonicalize(&path) {
            Ok(canonical) => canonical,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(invalid(format!(
                    "{} keeps its data in {relpath:?}, which is missing from the package \
                     directory {}; put that file beside the graph",
                    reference.tensor,
                    root.display()
                )))
            }
            Err(e) => return Err(GraphFailure::Unreadable(e)),
        };
        // Through a symlink, `weights.bin` can be anything the process can read. The
        // checksum would then describe a file outside the package, and a package that
        // installs by copying its directory would arrive without it.
        if !canonical.starts_with(&canonical_root) {
            return Err(invalid(format!(
                "{} keeps its data in {relpath:?}, which resolves to {} — outside the package \
                 directory; copy the file into the package instead of linking it",
                reference.tensor,
                canonical.display()
            )));
        }
        if canonical == canonical_graph || canonical == canonical_tokenizer {
            return Err(invalid(format!(
                "{} keeps its data in {relpath:?}, which leads to the package's own {}",
                reference.tensor,
                if canonical == canonical_graph {
                    "graph"
                } else {
                    "tokenizer"
                }
            )));
        }
        let metadata = std::fs::metadata(&canonical)?;
        if !metadata.is_file() {
            return Err(invalid(format!(
                "{} keeps its data in {relpath:?}, which is not a regular file",
                reference.tensor
            )));
        }

        let (sha256, size) = HashingReader::new(std::fs::File::open(&canonical)?).finish()?;
        if size < reference.needs_bytes {
            return Err(invalid(format!(
                "its external-data file {relpath:?} is {size} bytes, but {} needs it to reach \
                 byte {} — the download is incomplete; download the model again",
                reference.tensor, reference.needs_bytes
            )));
        }
        files.push(PackageFile {
            relpath,
            path,
            size,
            sha256,
        });
    }
    Ok(files)
}

/// Read `tokenizer.json` whole, hash it, and require a JSON object — which is also what
/// catches a truncated download of it, since JSON that stops early does not parse.
fn read_tokenizer(path: &Path) -> Result<PackageFile, GraphFailure> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_TOKENIZER_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_TOKENIZER_BYTES {
        return Err(GraphFailure::Invalid(format!(
            "its tokenizer {} is larger than {MAX_TOKENIZER_BYTES} bytes, which no tokenizer is",
            path.display()
        )));
    }
    let not_json = |why: String| {
        GraphFailure::Invalid(format!(
            "its tokenizer {} is not a JSON object ({why}); if it was downloaded, download it \
             again",
            path.display()
        ))
    };
    // The whole file as UTF-8, before the parse: the parse below skips every value it
    // does not read without looking at its bytes, so a file the backend cannot even read
    // as text would pass it — and `tools/onnx_package_checksum.py`, which decodes first,
    // refuses that file. The two implementations must refuse the same packages.
    let text = std::str::from_utf8(&bytes).map_err(|e| not_json(e.to_string()))?;
    // For the same reason, bounded before either parser runs: this one skips a nested
    // value iteratively and would accept any depth, Python's recurses and fails at one
    // its interpreter decides. A real tokenizer nests about five levels.
    let depth = json_nesting_depth(&bytes);
    if depth > MAX_TOKENIZER_JSON_DEPTH {
        return Err(GraphFailure::Invalid(format!(
            "its tokenizer {} nests {depth} levels deep, and at most \
             {MAX_TOKENIZER_JSON_DEPTH} are accepted; no tokenizer nests that deep",
            path.display()
        )));
    }
    // Values ignored, not built: whether the tokenizer is one the backend can load is the
    // backend's question. This one is whether the file is whole. A top-level key is read
    // as a `String`, which is what refuses one holding a lone surrogate escape.
    serde_json::from_str::<BTreeMap<String, serde::de::IgnoredAny>>(text)
        .map_err(|e| not_json(e.to_string()))?;
    Ok(PackageFile {
        relpath: ONNX_TOKENIZER_FILE.to_string(),
        path: path.to_path_buf(),
        size: bytes.len() as u64,
        sha256: hex_encode(&Sha256::digest(&bytes)),
    })
}

/// How deeply `json`'s arrays and objects nest: every `[` or `{` outside a string opens a
/// level, the top-level object being level 1. A linear scan over the bytes, not a parse —
/// whether the text is JSON at all is the parser's to say, afterwards — and exactly the
/// scan `tools/onnx_package_checksum.py` makes: inside a string a backslash takes the next
/// byte with it, whatever it is.
fn json_nesting_depth(json: &[u8]) -> usize {
    let (mut depth, mut deepest) = (0usize, 0usize);
    let (mut in_string, mut escaped) = (false, false);
    for &byte in json {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'[' | b'{' => {
                depth += 1;
                deepest = deepest.max(depth);
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    deepest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::embedding::mock::onnx::{self, Dim};
    use crate::semantic::embedding::mock::{self, proto};

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            // The clock alone collided: macOS ticks coarser than a test takes to start.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_model_package_{name}_{}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A package of `graph` bytes and the stub tokenizer, in `dir`; returns the graph.
    fn package_with(dir: &TempDir, graph: &[u8]) -> PathBuf {
        let path = dir.path().join("model.onnx");
        std::fs::write(&path, graph).unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), mock::STUB_TOKENIZER_JSON).unwrap();
        path
    }

    /// A model whose graph declares the encoder's inputs, one output, and `initializers`.
    fn model_with_initializers(initializers: &[Vec<u8>]) -> Vec<u8> {
        let output = onnx::value_info("out", onnx::FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)]);
        onnx::model(
            8,
            &onnx::graph("g", &onnx::encoder_inputs(), &[output], initializers),
            &[onnx::opset("", 17)],
        )
    }

    /// A model whose graph holds `nodes` (encoded `NodeProto`s) and nothing external.
    fn model_with_nodes(nodes: &[Vec<u8>]) -> Vec<u8> {
        let output = onnx::value_info("out", onnx::FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)]);
        let mut graph = Vec::new();
        for node in nodes {
            proto::bytes(&mut graph, 1, node);
        }
        graph.extend(onnx::graph("g", &onnx::encoder_inputs(), &[output], &[]));
        onnx::model(8, &graph, &[onnx::opset("", 17)])
    }

    /// A node `op_type` with one attribute: `attribute_field` of an `AttributeProto`
    /// holding `payload`.
    fn node_with_attribute(op_type: &str, attribute_field: u32, payload: &[u8]) -> Vec<u8> {
        let mut attribute = Vec::new();
        proto::string(&mut attribute, 1, "attr");
        proto::bytes(&mut attribute, attribute_field, payload);
        let mut node = Vec::new();
        proto::string(&mut node, 4, op_type);
        proto::bytes(&mut node, 5, &attribute);
        node
    }

    /// The reason an invalid package was refused with; anything else is a failure.
    fn refusal(graph: &Path) -> String {
        match validate_onnx_package(graph) {
            Err(EmbeddingError::InvalidModelFile { path, reason }) => {
                assert_eq!(
                    path,
                    graph.display().to_string(),
                    "a package refusal names the graph, whichever file was at fault"
                );
                reason
            }
            other => panic!("expected InvalidModelFile, got {other:?}"),
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        hex_encode(&Sha256::digest(bytes))
    }

    // ── format and layout ──

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

    // ── the package as a whole ──

    #[test]
    fn the_stub_package_validates_and_lists_exactly_its_two_files() {
        let dir = TempDir::new("stub");
        let graph = mock::write_stub_onnx_package(dir.path());
        let package = validate_onnx_package(&graph).expect("the stub package is valid");

        let relpaths: Vec<&str> = package.files().iter().map(|f| f.relpath.as_str()).collect();
        assert_eq!(relpaths, ["model.onnx", "tokenizer.json"]);

        let graph_bytes = std::fs::read(&graph).unwrap();
        let expected_manifest = format!(
            "otzaria-onnx-package-v1\nmodel.onnx\t{}\t{}\ntokenizer.json\t{}\t{}\n",
            graph_bytes.len(),
            sha256_hex(&graph_bytes),
            mock::STUB_TOKENIZER_JSON.len(),
            sha256_hex(mock::STUB_TOKENIZER_JSON.as_bytes()),
        );
        assert_eq!(package.manifest_text(), expected_manifest);
        assert_eq!(
            package.checksum(),
            sha256_hex(expected_manifest.as_bytes()),
            "the checksum is the SHA-256 of the manifest text, nothing more"
        );
        assert_eq!(package.checksum().len(), 64);

        assert_eq!(
            package.graph_facts(),
            GraphFacts {
                ir_version: 8,
                opset_imports: 1,
                graph_inputs: 2,
                graph_outputs: 1,
                external_tensors: 0,
            }
        );
        assert_eq!(package.graph_path(), graph);
        assert_eq!(package.tokenizer_path(), dir.path().join("tokenizer.json"));
    }

    #[test]
    fn validate_model_dispatches_on_the_format_the_path_names() {
        let dir = TempDir::new("dispatch");
        let gguf = dir.path().join("model.gguf");
        mock::write_stub_gguf(&gguf, 3).unwrap();
        match validate_model(&gguf).unwrap() {
            ValidatedModel::Gguf { checksum } => assert_eq!(
                checksum,
                validate_and_checksum_gguf(&gguf).unwrap(),
                "GGUF keeps the file's own SHA-256"
            ),
            other => panic!("a .gguf path is GGUF, got {other:?}"),
        }

        let onnx = mock::write_stub_onnx_package(&dir.path().join("onnx"));
        let validated = validate_model(&onnx).unwrap();
        assert_eq!(validated.format(), ModelFormat::Onnx);
        assert_eq!(
            validated.checksum(),
            validate_onnx_package(&onnx).unwrap().checksum()
        );

        // The same bytes under a GGUF name are read as GGUF, and refused as GGUF.
        let misnamed = dir.path().join("model.bin");
        std::fs::copy(&onnx, &misnamed).unwrap();
        assert!(matches!(
            validate_model(&misnamed),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
    }

    /// The value the build tooling has to reproduce, bit for bit, from these exact
    /// bytes. Changing it is changing the checksum of every ONNX artifact ever built.
    #[test]
    fn the_golden_package_has_the_documented_checksum() {
        const GRAPH_HEX: &str = "0808120c6f747a617269612d737475623ad8011206676f6c64656e2a4b\
            080408041001420b70726f6a2e7765696768746a190a086c6f636174696f6e120d776569676874\
            732f772e62696e6a0b0a066f66667365741201306a0c0a066c656e6774681202363470015a2a0a\
            09696e7075745f696473121d0a1b080712170a0208010a11120f73657175656e63655f6c656e67\
            74685a2f0a0e617474656e74696f6e5f6d61736b121d0a1b080712170a0208010a11120f736571\
            75656e63655f6c656e67746862240a1273656e74656e63655f656d62656464696e67120e0a0c08\
            0112080a0208010a02080442040a001011";
        const TOKENIZER: &str =
            r#"{"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#;
        const MANIFEST: &str = "otzaria-onnx-package-v1\n\
            model.onnx\t241\t7a655f52bd5e7d6d72a0581505690c874d349505aa8a5a87019db3ba581add55\n\
            tokenizer.json\t70\t5181aefd3b938b58bde0afbd009afd4c8cc0db4a61ad07dfe94868b112be8b37\n\
            weights/w.bin\t64\tfdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108\n";
        const DIGEST: &str = "d8eea75d4348089f9ed6735709c95f24e8654fb6c5301ace41e03daea1f70699";

        let graph_bytes: Vec<u8> = (0..GRAPH_HEX.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&GRAPH_HEX[at..at + 2], 16).unwrap())
            .collect();
        assert_eq!(graph_bytes.len(), 241);

        let dir = TempDir::new("golden");
        let graph = dir.path().join("model.onnx");
        std::fs::write(&graph, &graph_bytes).unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), TOKENIZER).unwrap();
        std::fs::create_dir_all(dir.path().join("weights")).unwrap();
        std::fs::write(
            dir.path().join("weights/w.bin"),
            (0u8..64).collect::<Vec<_>>(),
        )
        .unwrap();
        // Beside the package, not in it.
        std::fs::write(dir.path().join("README.md"), "not part of the package").unwrap();

        let package = validate_onnx_package(&graph).unwrap();
        assert_eq!(package.manifest_text(), MANIFEST);
        assert_eq!(package.checksum(), DIGEST);
        assert_eq!(sha256_hex(MANIFEST.as_bytes()), DIGEST);
        assert_eq!(package.graph_facts().external_tensors, 1);
    }

    #[test]
    fn the_manifest_orders_files_bytewise_whatever_order_they_arrive_in() {
        let file = |relpath: &str, size: u64| PackageFile {
            relpath: relpath.to_string(),
            path: PathBuf::from(relpath),
            size,
            sha256: "0".repeat(64),
        };
        let files = [
            file("weights/w.bin", 3),
            file("tokenizer.json", 2),
            file("Model.onnx", 1),
            file("model.onnx.data", 4),
        ];
        let zeros = "0".repeat(64);
        assert_eq!(
            onnx_package_manifest(&files),
            format!(
                "otzaria-onnx-package-v1\nModel.onnx\t1\t{zeros}\nmodel.onnx.data\t4\t{zeros}\n\
                 tokenizer.json\t2\t{zeros}\nweights/w.bin\t3\t{zeros}\n"
            ),
            "uppercase sorts before lowercase, bytewise, as the Python side sorts too"
        );
        let mut reversed = files.to_vec();
        reversed.reverse();
        assert_eq!(
            onnx_package_checksum(&files),
            onnx_package_checksum(&reversed)
        );
    }

    #[test]
    fn the_checksum_is_stable_across_runs() {
        let dir = TempDir::new("stable");
        let graph = mock::write_stub_onnx_package(dir.path());
        let first = validate_onnx_package(&graph)
            .unwrap()
            .checksum()
            .to_string();
        for _ in 0..3 {
            assert_eq!(validate_onnx_package(&graph).unwrap().checksum(), first);
        }

        // And across directories: the root is not part of what is hashed.
        let elsewhere = TempDir::new("stable_elsewhere");
        let moved = mock::write_stub_onnx_package(elsewhere.path());
        assert_eq!(validate_onnx_package(&moved).unwrap().checksum(), first);
    }

    /// One byte anywhere in the package — graph, external data or tokenizer — is a
    /// different checksum.
    #[test]
    fn one_changed_byte_in_any_package_file_changes_the_checksum() {
        let dir = TempDir::new("sensitivity");
        let graph_bytes =
            model_with_initializers(&[onnx::external_tensor("w", &[4], "w.bin", None, Some(16))]);
        let graph = package_with(&dir, &graph_bytes);
        std::fs::write(dir.path().join("w.bin"), [7u8; 16]).unwrap();
        let original = validate_onnx_package(&graph)
            .unwrap()
            .checksum()
            .to_string();

        // The graph: one byte of the producer name, which nothing else reads.
        let mut edited = graph_bytes.clone();
        let at = edited
            .windows(b"otzaria-stub".len())
            .position(|window| window == b"otzaria-stub")
            .unwrap();
        edited[at] = b'O';
        std::fs::write(&graph, &edited).unwrap();
        let changed = validate_onnx_package(&graph)
            .unwrap()
            .checksum()
            .to_string();
        assert_ne!(
            changed, original,
            "a changed graph byte must change the checksum"
        );
        std::fs::write(&graph, &graph_bytes).unwrap();

        // The external data.
        let mut data = [7u8; 16];
        data[15] = 8;
        std::fs::write(dir.path().join("w.bin"), data).unwrap();
        assert_ne!(
            validate_onnx_package(&graph).unwrap().checksum(),
            original,
            "a changed external-data byte must change the checksum"
        );
        std::fs::write(dir.path().join("w.bin"), [7u8; 16]).unwrap();

        // The tokenizer.
        let tokenizer = mock::STUB_TOKENIZER_JSON.replace("[UNK]\":0", "[UNK]\":9");
        assert_eq!(tokenizer.len(), mock::STUB_TOKENIZER_JSON.len());
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer).unwrap();
        assert_ne!(
            validate_onnx_package(&graph).unwrap().checksum(),
            original,
            "a changed tokenizer byte must change the checksum"
        );
        std::fs::write(dir.path().join("tokenizer.json"), mock::STUB_TOKENIZER_JSON).unwrap();

        assert_eq!(
            validate_onnx_package(&graph).unwrap().checksum(),
            original,
            "restoring every byte restores the checksum"
        );
    }

    /// What never reaches a vector is not the package: the model card, the author's
    /// manifest, the export script, the other graph, and an ONNX Runtime library
    /// placed beside the graph for the backend to load.
    #[test]
    fn files_beside_the_package_do_not_change_the_checksum() {
        let dir = TempDir::new("outside");
        let graph = mock::write_stub_onnx_package(dir.path());
        let original = validate_onnx_package(&graph)
            .unwrap()
            .checksum()
            .to_string();

        for (name, contents) in [
            ("README.md", "# model card".as_bytes()),
            ("LICENSE.md", b"CC BY-NC-SA 4.0"),
            ("manifest.json", b"{\"dims\": 256}"),
            ("export_round2_onnx.py", b"print('export')"),
            (".gitattributes", b"*.onnx filter=lfs"),
            ("model-int8.onnx", &onnx::stub_graph()),
            ("libonnxruntime.dylib", b"\xcf\xfa\xed\xfe"),
            ("libonnxruntime.so", b"\x7fELF"),
            ("onnxruntime.dll", b"MZ"),
        ] {
            std::fs::write(dir.path().join(name), contents).unwrap();
            assert_eq!(
                validate_onnx_package(&graph).unwrap().checksum(),
                original,
                "{name} is not part of the package"
            );
        }
        std::fs::write(dir.path().join("README.md"), "# a different model card").unwrap();
        assert_eq!(validate_onnx_package(&graph).unwrap().checksum(), original);
    }

    // ── the graph ──

    /// A download cut short at any byte is refused — and once the cut falls inside a
    /// value the file declared, it is refused as the incomplete download it is.
    #[test]
    fn a_graph_cut_off_at_any_byte_is_refused() {
        let dir = TempDir::new("truncated");
        let whole = onnx::stub_graph();
        let graph = package_with(&dir, &whole);
        validate_onnx_package(&graph).expect("the whole graph is valid");

        let mut incomplete = 0;
        for cut in 0..whole.len() {
            std::fs::write(&graph, &whole[..cut]).unwrap();
            let reason = refusal(&graph);
            if reason.contains("incomplete") {
                incomplete += 1;
            }
        }
        // The graph field's own length is declared at byte 17; every cut after it and
        // before the opset_import is inside a declared length.
        assert!(
            incomplete > whole.len() / 2,
            "most cuts land inside a declared length and must say so ({incomplete} of {})",
            whole.len()
        );

        // Cut inside the graph: the length says more is coming.
        std::fs::write(&graph, &whole[..whole.len() / 2]).unwrap();
        let reason = refusal(&graph);
        assert!(reason.contains("incomplete"), "{reason}");
        assert!(
            reason.contains("download"),
            "the fix must be named: {reason}"
        );
    }

    /// The case the stub is too small for: a download cut off inside megabytes of
    /// weights, far past the read buffer.
    #[test]
    fn a_graph_cut_off_inside_a_large_initializer_is_incomplete() {
        let dir = TempDir::new("truncated_weights");
        let mut weights = Vec::new();
        proto::int(&mut weights, 1, 3 << 18);
        proto::uint(&mut weights, 2, onnx::FLOAT);
        proto::string(&mut weights, 8, "big");
        proto::bytes(&mut weights, 9, &vec![0x5a; 3 << 20]);
        let whole = model_with_initializers(&[weights]);
        let graph = package_with(&dir, &whole);
        let package = validate_onnx_package(&graph).expect("the whole graph is valid");
        assert_eq!(package.files()[0].size, whole.len() as u64);
        assert_eq!(package.files()[0].sha256, sha256_hex(&whole));

        std::fs::write(&graph, &whole[..whole.len() - 1]).unwrap();
        assert!(refusal(&graph).contains("incomplete"));
        std::fs::write(&graph, &whole[..2 << 20]).unwrap();
        assert!(refusal(&graph).contains("incomplete"));
    }

    #[test]
    fn bytes_that_are_not_onnx_are_refused_by_what_they_are() {
        let dir = TempDir::new("not_onnx");
        let graph = package_with(&dir, b"");

        for (contents, expected) in [
            (&b""[..], "empty"),
            (
                b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 168177986\n",
                "git lfs pull",
            ),
            (b"<!DOCTYPE html><html>Access denied</html>", "HTML"),
            (b"\n  <html>", "HTML"),
            (b"{\"error\":\"Access to model is restricted\"}", "JSON"),
            (b"[1, 2, 3]", "JSON"),
            (b"GGUF\x03\x00\x00\x00", "GGUF"),
            (b"PK\x03\x04rest-of-a-zip", "ZIP"),
        ] {
            std::fs::write(&graph, contents).unwrap();
            let reason = refusal(&graph);
            assert!(
                reason.contains(expected),
                "{:?}… must be recognized as {expected}, got: {reason}",
                String::from_utf8_lossy(&contents[..contents.len().min(12)])
            );
        }
    }

    #[test]
    fn garbage_is_refused_within_its_first_bytes() {
        let dir = TempDir::new("garbage");
        let graph = package_with(&dir, b"");

        // A deterministic scramble, so a failure is reproducible.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let scrambled: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect();

        for garbage in [
            vec![0u8; 1024],
            vec![0xffu8; 1024],
            b"plain text that is not a model at all".to_vec(),
            scrambled,
        ] {
            std::fs::write(&graph, &garbage).unwrap();
            let reason = refusal(&graph);
            assert!(
                reason.contains("not a valid ONNX model")
                    || reason.contains("incomplete")
                    || reason.contains("accepts"),
                "unexpected reason: {reason}"
            );
        }
    }

    #[test]
    fn a_model_without_its_required_parts_is_refused_by_name() {
        let dir = TempDir::new("requirements");
        let output = onnx::value_info("out", onnx::FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)]);
        let full_graph = onnx::graph(
            "g",
            &onnx::encoder_inputs(),
            std::slice::from_ref(&output),
            &[],
        );
        let opset = onnx::opset("", 17);

        let mut no_ir_version = Vec::new();
        proto::bytes(&mut no_ir_version, 7, &full_graph);
        proto::bytes(&mut no_ir_version, 8, &opset);

        let mut no_graph = Vec::new();
        proto::uint(&mut no_graph, 1, 8);
        proto::bytes(&mut no_graph, 8, &opset);

        let mut negative_ir_version = Vec::new();
        proto::int(&mut negative_ir_version, 1, -3);
        proto::bytes(&mut negative_ir_version, 7, &full_graph);
        proto::bytes(&mut negative_ir_version, 8, &opset);

        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            ("no ir_version", no_ir_version, "no ir_version"),
            (
                "ir_version 0",
                onnx::model(0, &full_graph, std::slice::from_ref(&opset)),
                "ir_version is 0",
            ),
            (
                "a negative ir_version",
                negative_ir_version,
                "ir_version is -3",
            ),
            (
                "no opset_import",
                onnx::model(8, &full_graph, &[]),
                "no opset_import",
            ),
            (
                "an opset at version 0",
                onnx::model(8, &full_graph, &[onnx::opset("", 0)]),
                "version 0",
            ),
            ("no graph", no_graph, "no graph"),
            (
                "a graph with no input",
                onnx::model(
                    8,
                    &onnx::graph("g", &[], std::slice::from_ref(&output), &[]),
                    std::slice::from_ref(&opset),
                ),
                "no input",
            ),
            (
                "a graph with no output",
                onnx::model(
                    8,
                    &onnx::graph("g", &onnx::encoder_inputs(), &[], &[]),
                    &[opset],
                ),
                "no output",
            ),
        ];
        for (name, bytes, expected) in cases {
            let graph = package_with(&dir, &bytes);
            let reason = refusal(&graph);
            assert!(reason.contains(expected), "{name}: {reason}");
        }
    }

    /// Fields this build does not know are skipped as protobuf skips them — a model from
    /// a newer ONNX must still validate — in every wire type, groups included.
    #[test]
    fn unknown_fields_in_every_wire_type_are_skipped() {
        let dir = TempDir::new("unknown_fields");
        let mut model = onnx::stub_graph();
        proto::uint(&mut model, 99, 12345);
        proto::key(&mut model, 98, proto::FIXED64);
        model.extend_from_slice(&[1u8; 8]);
        proto::key(&mut model, 97, proto::FIXED32);
        model.extend_from_slice(&[2u8; 4]);
        proto::bytes(&mut model, 96, b"a future field");
        proto::key(&mut model, 95, proto::START_GROUP);
        proto::uint(&mut model, 1, 7);
        proto::key(&mut model, 94, proto::START_GROUP);
        proto::bytes(&mut model, 2, b"nested");
        proto::key(&mut model, 94, proto::END_GROUP);
        proto::key(&mut model, 95, proto::END_GROUP);

        let graph = package_with(&dir, &model);
        let package = validate_onnx_package(&graph).expect("unknown fields are skipped");
        assert_eq!(package.graph_facts().graph_inputs, 2);
    }

    #[test]
    fn a_group_that_never_closes_or_closes_wrongly_is_refused() {
        let dir = TempDir::new("groups");

        let mut unterminated = onnx::stub_graph();
        proto::key(&mut unterminated, 95, proto::START_GROUP);
        proto::uint(&mut unterminated, 1, 7);
        let graph = package_with(&dir, &unterminated);
        assert!(refusal(&graph).contains("incomplete"));

        let mut mismatched = onnx::stub_graph();
        proto::key(&mut mismatched, 95, proto::START_GROUP);
        proto::key(&mut mismatched, 93, proto::END_GROUP);
        let graph = package_with(&dir, &mismatched);
        assert!(refusal(&graph).contains("closed as group 93"));

        let mut stray = onnx::stub_graph();
        proto::key(&mut stray, 93, proto::END_GROUP);
        let graph = package_with(&dir, &stray);
        assert!(refusal(&graph).contains("closes no group"));
    }

    /// A declared field in the wrong wire type is not a newer ONNX; it is not ONNX.
    #[test]
    fn a_known_field_in_the_wrong_wire_type_is_refused() {
        let dir = TempDir::new("wire_type");

        let mut ir_version_as_bytes = Vec::new();
        proto::bytes(&mut ir_version_as_bytes, 1, b"\x08");
        ir_version_as_bytes.extend(onnx::stub_graph());
        let graph = package_with(&dir, &ir_version_as_bytes);
        let reason = refusal(&graph);
        assert!(reason.contains("field 1 of ModelProto"), "{reason}");

        // Deeper: a graph input (GraphProto field 11) as a varint.
        let mut graph_bytes = Vec::new();
        proto::uint(&mut graph_bytes, 11, 1);
        let bad = onnx::model(8, &graph_bytes, &[onnx::opset("", 17)]);
        let graph = package_with(&dir, &bad);
        assert!(refusal(&graph).contains("field 11 of GraphProto"));
    }

    #[test]
    fn a_file_that_does_not_begin_with_a_model_field_is_refused() {
        let dir = TempDir::new("first_field");
        let mut model = Vec::new();
        proto::uint(&mut model, 99, 1);
        model.extend(onnx::stub_graph());
        let graph = package_with(&dir, &model);
        assert!(refusal(&graph).contains("does not begin like an ONNX model"));
    }

    #[test]
    fn a_varint_longer_than_ten_bytes_is_refused() {
        let dir = TempDir::new("long_varint");
        let mut model = vec![0x08];
        model.extend_from_slice(&[0x80; 10]);
        model.push(0x01);
        let graph = package_with(&dir, &model);
        assert!(refusal(&graph).contains("varint"));
    }

    #[test]
    fn a_nested_length_that_overruns_its_message_is_malformed_not_truncated() {
        let dir = TempDir::new("overrun");
        // An opset_import whose domain claims 50 bytes inside a 4-byte message, with
        // plenty of file after it.
        let mut opset = Vec::new();
        proto::key(&mut opset, 1, proto::LEN);
        proto::varint(&mut opset, 50);
        opset.extend_from_slice(b"ab");
        let mut model = onnx::stub_graph();
        proto::bytes(&mut model, 8, &opset);
        proto::bytes(&mut model, 96, &[0u8; 128]);
        let graph = package_with(&dir, &model);
        let reason = refusal(&graph);
        assert!(
            reason.contains("runs past the end of the message"),
            "{reason}"
        );
        assert!(!reason.contains("incomplete"), "{reason}");
    }

    /// Only the outermost message is bounded by the end of the file. A graph whose own
    /// length fits the file exactly, holding a field that claims more, is malformed —
    /// downloading it again would change nothing.
    #[test]
    fn an_overrun_inside_a_message_that_ends_at_the_end_of_the_file_is_malformed() {
        let dir = TempDir::new("overrun_at_eof");
        let mut graph = onnx::graph("g", &onnx::encoder_inputs(), &[], &[]);
        proto::key(&mut graph, 12, proto::LEN);
        proto::varint(&mut graph, 5000); // an output claiming far more than is left
        graph.extend_from_slice(b"xy");
        let mut model = Vec::new();
        proto::uint(&mut model, 1, 8);
        proto::bytes(&mut model, 8, &onnx::opset("", 17));
        proto::bytes(&mut model, 7, &graph); // last: it ends exactly at the end of the file

        let path = package_with(&dir, &model);
        let reason = refusal(&path);
        assert!(
            reason.contains("runs past the end of the message"),
            "{reason}"
        );
        assert!(!reason.contains("incomplete"), "{reason}");
    }

    #[test]
    fn nesting_past_the_bound_is_refused() {
        let dir = TempDir::new("depth");
        // Subgraphs inside attributes inside nodes, deeper than any real model.
        let output = onnx::value_info("out", onnx::FLOAT, &[Dim::Fixed(1)]);
        let mut inner = onnx::graph("leaf", &onnx::encoder_inputs(), &[output], &[]);
        for _ in 0..30 {
            let node = node_with_attribute("If", 6, &inner);
            inner = Vec::new();
            proto::bytes(&mut inner, 1, &node);
        }
        let deep = onnx::model(8, &inner, &[onnx::opset("", 17)]);
        let graph = package_with(&dir, &deep);
        assert!(refusal(&graph).contains("nest deeper"));

        // A handful of levels is an ordinary model.
        let output = onnx::value_info("out", onnx::FLOAT, &[Dim::Fixed(1)]);
        let body = onnx::graph("body", &[], &[], &[]);
        let shallow = model_with_nodes(&[node_with_attribute("If", 6, &body)]);
        let graph = package_with(&dir, &shallow);
        validate_onnx_package(&graph).expect("one level of subgraph is fine");
        drop(output);
    }

    // ── external data ──

    #[test]
    fn external_data_inside_the_package_is_hashed_as_part_of_it() {
        let dir = TempDir::new("external_ok");
        let graph = package_with(
            &dir,
            &model_with_initializers(&[
                onnx::external_tensor("a", &[4, 4], "weights/data.bin", Some(0), Some(64)),
                onnx::external_tensor("b", &[4], "./weights/data.bin", Some(64), Some(16)),
                onnx::external_tensor("c", &[2], "weights//data.bin", Some(80), None),
            ]),
        );
        std::fs::create_dir_all(dir.path().join("weights")).unwrap();
        let data = vec![3u8; 81];
        std::fs::write(dir.path().join("weights/data.bin"), &data).unwrap();

        let package = validate_onnx_package(&graph).unwrap();
        let relpaths: Vec<&str> = package.files().iter().map(|f| f.relpath.as_str()).collect();
        assert_eq!(
            relpaths,
            ["model.onnx", "tokenizer.json", "weights/data.bin"],
            "three references to one file are one file of the package"
        );
        let external = &package.files()[2];
        assert_eq!(external.size, 81);
        assert_eq!(external.sha256, sha256_hex(&data));
        assert_eq!(package.graph_facts().external_tensors, 3);
    }

    #[test]
    fn missing_external_data_is_refused_and_named() {
        let dir = TempDir::new("external_missing");
        let graph = package_with(
            &dir,
            &model_with_initializers(&[onnx::external_tensor(
                "encoder.weight",
                &[4],
                "model.onnx.data",
                None,
                Some(16),
            )]),
        );
        let reason = refusal(&graph);
        assert!(reason.contains("model.onnx.data"), "{reason}");
        assert!(reason.contains("missing"), "{reason}");
        assert!(reason.contains("encoder.weight"), "{reason}");
    }

    #[test]
    fn external_data_shorter_than_its_references_is_incomplete() {
        let dir = TempDir::new("external_short");
        let graph = package_with(
            &dir,
            &model_with_initializers(&[
                onnx::external_tensor("a", &[4], "w.bin", Some(0), Some(16)),
                onnx::external_tensor("b", &[4], "w.bin", Some(4096), Some(16)),
            ]),
        );
        std::fs::write(dir.path().join("w.bin"), vec![0u8; 4111]).unwrap();
        let reason = refusal(&graph);
        assert!(reason.contains("incomplete"), "{reason}");
        assert!(
            reason.contains("4112") && reason.contains("'b'"),
            "the reference that reaches furthest is the one named: {reason}"
        );

        std::fs::write(dir.path().join("w.bin"), vec![0u8; 4112]).unwrap();
        validate_onnx_package(&graph).expect("exactly long enough is enough");
    }

    /// With no `length`, the floor is one bit per element past the offset — below every
    /// ONNX data type, so it can refuse a file that is short but never one that is not.
    #[test]
    fn without_a_length_the_floor_is_one_bit_per_element() {
        let dir = TempDir::new("external_floor");
        // 1024 elements: at least 128 bytes whatever the type.
        let graph = package_with(
            &dir,
            &model_with_initializers(&[onnx::external_tensor(
                "a",
                &[32, 32],
                "w.bin",
                Some(8),
                None,
            )]),
        );
        std::fs::write(dir.path().join("w.bin"), vec![0u8; 8 + 127]).unwrap();
        assert!(refusal(&graph).contains("incomplete"));
        std::fs::write(dir.path().join("w.bin"), vec![0u8; 8 + 128]).unwrap();
        validate_onnx_package(&graph).unwrap();
    }

    #[test]
    fn an_external_location_outside_the_package_is_refused() {
        let dir = TempDir::new("external_escape");
        std::fs::write(dir.path().join("outside.bin"), [0u8; 64]).unwrap();
        let inner = TempDir::new("external_escape_inner");

        for (location, expected) in [
            ("/etc/passwd", "absolute"),
            ("../outside.bin", "'..'"),
            ("weights/../../outside.bin", "'..'"),
            ("..", "'..'"),
            ("weights\\w.bin", "backslash"),
            ("C:/weights/w.bin", "Windows"),
            ("w.bin\nsecond line", "control character"),
            ("", "empty location"),
            (".", "no file name"),
            ("./", "trailing '/'"),
            ("weights/", "trailing '/'"),
        ] {
            let graph = package_with(
                &inner,
                &model_with_initializers(&[onnx::external_tensor(
                    "a",
                    &[4],
                    location,
                    None,
                    Some(16),
                )]),
            );
            let reason = refusal(&graph);
            assert!(
                reason.contains(expected),
                "{location:?} must be refused for {expected}: {reason}"
            );
        }
    }

    #[test]
    fn an_external_location_naming_the_graph_or_the_tokenizer_is_refused() {
        let dir = TempDir::new("external_self");
        for (location, expected) in [
            ("model.onnx", "graph"),
            ("./model.onnx", "graph"),
            ("tokenizer.json", "tokenizer"),
        ] {
            let graph = package_with(
                &dir,
                &model_with_initializers(&[onnx::external_tensor(
                    "a",
                    &[4],
                    location,
                    None,
                    Some(1),
                )]),
            );
            let reason = refusal(&graph);
            assert!(reason.contains(expected), "{location}: {reason}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_the_package_is_refused_and_one_inside_it_is_not() {
        let outside = TempDir::new("symlink_target");
        std::fs::write(outside.path().join("stolen.bin"), [1u8; 64]).unwrap();
        let dir = TempDir::new("symlink");
        let graph = package_with(
            &dir,
            &model_with_initializers(&[onnx::external_tensor("a", &[4], "w.bin", None, Some(16))]),
        );

        std::os::unix::fs::symlink(outside.path().join("stolen.bin"), dir.path().join("w.bin"))
            .unwrap();
        let reason = refusal(&graph);
        assert!(reason.contains("outside the package"), "{reason}");

        std::fs::remove_file(dir.path().join("w.bin")).unwrap();
        std::fs::create_dir_all(dir.path().join("blobs")).unwrap();
        std::fs::write(dir.path().join("blobs/real.bin"), [2u8; 16]).unwrap();
        std::os::unix::fs::symlink(dir.path().join("blobs/real.bin"), dir.path().join("w.bin"))
            .unwrap();
        let package = validate_onnx_package(&graph).expect("a link inside the package is fine");
        let external = package
            .files()
            .iter()
            .find(|file| file.relpath == "w.bin")
            .expect("listed under the name the graph uses");
        assert_eq!(external.sha256, sha256_hex(&[2u8; 16]));

        // A link to the tokenizer is the tokenizer.
        std::fs::remove_file(dir.path().join("w.bin")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("tokenizer.json"), dir.path().join("w.bin"))
            .unwrap();
        assert!(refusal(&graph).contains("tokenizer"));
    }

    /// External data can hide in a subgraph, a sparse initializer, a function or a
    /// training step, and the walk has to find it in every one: a missing file there is
    /// the proof it looked.
    #[test]
    fn external_data_is_found_wherever_a_tensor_can_be() {
        let dir = TempDir::new("external_nested");
        let tensor = onnx::external_tensor("hidden", &[4], "hidden.bin", None, Some(16));

        let mut subgraph = Vec::new();
        proto::bytes(&mut subgraph, 5, &tensor);
        let in_subgraph = model_with_nodes(&[node_with_attribute("If", 6, &subgraph)]);

        let in_attribute_tensor = model_with_nodes(&[node_with_attribute("Constant", 5, &tensor)]);
        let in_attribute_tensors = model_with_nodes(&[node_with_attribute("X", 10, &tensor)]);
        let in_attribute_graphs = model_with_nodes(&[node_with_attribute("Loop", 11, &subgraph)]);

        let mut sparse = Vec::new();
        proto::bytes(&mut sparse, 1, &tensor);
        proto::int(&mut sparse, 3, 4);
        let in_sparse_attribute = model_with_nodes(&[node_with_attribute("C", 22, &sparse)]);
        let in_sparse_attributes = model_with_nodes(&[node_with_attribute("C", 23, &sparse)]);

        let output = onnx::value_info("out", onnx::FLOAT, &[Dim::Fixed(1)]);
        let mut graph_with_sparse = onnx::graph("g", &onnx::encoder_inputs(), &[output], &[]);
        proto::bytes(&mut graph_with_sparse, 15, &sparse);
        let in_sparse_initializer = onnx::model(8, &graph_with_sparse, &[onnx::opset("", 17)]);

        let mut function = Vec::new();
        proto::string(&mut function, 1, "f");
        proto::bytes(
            &mut function,
            7,
            &node_with_attribute("Constant", 5, &tensor),
        );
        let mut in_function = onnx::stub_graph();
        proto::bytes(&mut in_function, 25, &function);

        let mut function_default = Vec::new();
        proto::string(&mut function_default, 1, "f");
        let mut attribute = Vec::new();
        proto::string(&mut attribute, 1, "default");
        proto::bytes(&mut attribute, 5, &tensor);
        proto::bytes(&mut function_default, 11, &attribute);
        let mut in_function_default = onnx::stub_graph();
        proto::bytes(&mut in_function_default, 25, &function_default);

        let mut training = Vec::new();
        proto::bytes(&mut training, 1, &subgraph);
        let mut in_training = onnx::stub_graph();
        proto::bytes(&mut in_training, 20, &training);

        for (name, model) in [
            ("a subgraph attribute", in_subgraph),
            ("an attribute tensor", in_attribute_tensor),
            ("an attribute tensor list", in_attribute_tensors),
            ("a subgraph list attribute", in_attribute_graphs),
            ("a sparse tensor attribute", in_sparse_attribute),
            ("a sparse tensor list attribute", in_sparse_attributes),
            ("a sparse initializer", in_sparse_initializer),
            ("a function node", in_function),
            ("a function attribute default", in_function_default),
            ("a training graph", in_training),
        ] {
            let graph = package_with(&dir, &model);
            let reason = refusal(&graph);
            assert!(
                reason.contains("hidden.bin") && reason.contains("missing"),
                "external data in {name} was not found: {reason}"
            );

            std::fs::write(dir.path().join("hidden.bin"), [0u8; 16]).unwrap();
            let package = validate_onnx_package(&graph)
                .unwrap_or_else(|e| panic!("{name} with its data present: {e}"));
            assert!(
                package.files().iter().any(|f| f.relpath == "hidden.bin"),
                "{name}: the file belongs to the package"
            );
            std::fs::remove_file(dir.path().join("hidden.bin")).unwrap();
        }
    }

    /// Only `data_location = EXTERNAL` moves the data. Entries on a tensor stored inline
    /// are ignored, exactly as a runtime ignores them.
    #[test]
    fn external_data_entries_on_an_inline_tensor_are_ignored() {
        let dir = TempDir::new("inline_entries");
        let mut tensor = Vec::new();
        proto::int(&mut tensor, 1, 4);
        proto::uint(&mut tensor, 2, onnx::FLOAT);
        proto::bytes(&mut tensor, 13, &onnx::entry("location", "absent.bin"));
        proto::bytes(&mut tensor, 9, &[0u8; 16]);
        let graph = package_with(&dir, &model_with_initializers(&[tensor]));
        let package = validate_onnx_package(&graph).unwrap();
        assert_eq!(package.files().len(), 2);
    }

    #[test]
    fn malformed_external_data_entries_are_refused() {
        let dir = TempDir::new("external_entries");
        let with_entries = |entries: &[(&str, &str)], location: u64| {
            let mut tensor = Vec::new();
            proto::int(&mut tensor, 1, 4);
            proto::uint(&mut tensor, 2, onnx::FLOAT);
            proto::string(&mut tensor, 8, "t");
            for (key, value) in entries {
                proto::bytes(&mut tensor, 13, &onnx::entry(key, value));
            }
            proto::uint(&mut tensor, 14, location);
            model_with_initializers(&[tensor])
        };

        for (name, model, expected) in [
            (
                "no location",
                with_entries(&[("length", "16")], 1),
                "names no location",
            ),
            (
                "two locations",
                with_entries(&[("location", "a.bin"), ("location", "b.bin")], 1),
                "twice",
            ),
            (
                "a signed offset",
                with_entries(&[("location", "a.bin"), ("offset", "-1")], 1),
                "non-negative integer",
            ),
            (
                "a spaced length",
                with_entries(&[("location", "a.bin"), ("length", " 16")], 1),
                "non-negative integer",
            ),
            (
                "an offset past u64",
                with_entries(
                    &[("location", "a.bin"), ("offset", "99999999999999999999")],
                    1,
                ),
                "no file can reach",
            ),
            (
                "an overflowing reach",
                with_entries(
                    &[
                        ("location", "a.bin"),
                        ("offset", "18446744073709551615"),
                        ("length", "2"),
                    ],
                    1,
                ),
                "overflows",
            ),
            (
                "an unknown data_location",
                with_entries(&[("location", "a.bin")], 2),
                "neither DEFAULT",
            ),
        ] {
            let graph = package_with(&dir, &model);
            let reason = refusal(&graph);
            assert!(reason.contains(expected), "{name}: {reason}");
        }

        // The specification's fourth key, and keys it does not name, move nothing.
        std::fs::write(dir.path().join("a.bin"), [0u8; 16]).unwrap();
        let graph = package_with(
            &dir,
            &with_entries(
                &[
                    ("location", "a.bin"),
                    ("checksum", "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
                    ("future-key", "anything"),
                ],
                1,
            ),
        );
        validate_onnx_package(&graph).unwrap();
    }

    #[test]
    fn packed_dimensions_give_the_same_floor_as_unpacked_ones() {
        let dir = TempDir::new("packed_dims");
        let mut packed = Vec::new();
        let mut dims = Vec::new();
        proto::varint(&mut dims, 64);
        proto::varint(&mut dims, 64);
        proto::bytes(&mut packed, 1, &dims);
        proto::uint(&mut packed, 2, onnx::FLOAT);
        proto::bytes(&mut packed, 13, &onnx::entry("location", "w.bin"));
        proto::uint(&mut packed, 14, 1);
        let graph = package_with(&dir, &model_with_initializers(&[packed]));

        std::fs::write(dir.path().join("w.bin"), vec![0u8; 511]).unwrap();
        assert!(
            refusal(&graph).contains("incomplete"),
            "4096 elements need 512 bytes"
        );
        std::fs::write(dir.path().join("w.bin"), vec![0u8; 512]).unwrap();
        validate_onnx_package(&graph).unwrap();
    }

    // ── the tokenizer ──

    #[test]
    fn a_missing_tokenizer_is_refused_before_the_graph_is_read() {
        let dir = TempDir::new("no_tokenizer");
        let graph = dir.path().join("model.onnx");
        // Garbage: if the graph were read first, this would be the error.
        std::fs::write(&graph, b"not a model").unwrap();
        match validate_onnx_package(&graph) {
            Err(EmbeddingError::TokenizerNotFound { path }) => {
                assert_eq!(
                    path,
                    dir.path().join("tokenizer.json").display().to_string()
                );
            }
            other => panic!("expected TokenizerNotFound, got {other:?}"),
        }
    }

    #[test]
    fn a_tokenizer_that_is_not_a_whole_json_object_is_refused() {
        let dir = TempDir::new("bad_tokenizer");
        let graph = mock::write_stub_onnx_package(dir.path());
        let whole = mock::STUB_TOKENIZER_JSON;
        for contents in [
            &whole[..whole.len() / 2],
            &whole[..whole.len() - 1],
            "",
            "[1, 2, 3]",
            "\"a string\"",
            "{\"a\": 1} trailing",
        ] {
            std::fs::write(dir.path().join("tokenizer.json"), contents).unwrap();
            let reason = refusal(&graph);
            assert!(
                reason.contains("tokenizer") && reason.contains("JSON object"),
                "{contents:?}: {reason}"
            );
        }
    }

    /// Where the parse below `read_tokenizer` and Python's `json` could part, the crate
    /// refuses and accepts what `tools/onnx_package_checksum.py` does — its test suite
    /// holds the same cases. The parse skips every value it does not read, bytes unseen,
    /// and skips a nested value iteratively; Python decodes the whole file first and
    /// recurses. So the text must be UTF-8 throughout, and nest no deeper than the bound
    /// both check before either parser runs.
    #[test]
    fn a_tokenizer_is_refused_and_accepted_as_the_python_implementation_does() {
        let dir = TempDir::new("tokenizer_parity");
        let graph = mock::write_stub_onnx_package(dir.path());
        let nested = |levels: usize| {
            let mut json = "{\"a\":".to_string();
            json.push_str(&"[".repeat(levels - 1));
            json.push_str(&"]".repeat(levels - 1));
            json.push('}');
            json.into_bytes()
        };

        for (case, contents) in [
            (
                "invalid UTF-8 in a nested value",
                b"{\"a\":{\"b\":\"\xff\xfe\"}}".to_vec(),
            ),
            ("invalid UTF-8 in a value", b"{\"a\":\"\xc3\x28\"}".to_vec()),
            ("invalid UTF-8 in a key", b"{\"\xff\":1}".to_vec()),
            ("a lone surrogate in a key", br#"{"\ud800":1}"#.to_vec()),
            (
                "a lone trailing surrogate in a key",
                br#"{"\udc00":1}"#.to_vec(),
            ),
            (
                "one level past the bound",
                nested(MAX_TOKENIZER_JSON_DEPTH + 1),
            ),
            ("far past the bound", nested(100_000)),
        ] {
            std::fs::write(dir.path().join("tokenizer.json"), &contents).unwrap();
            let reason = refusal(&graph);
            assert!(reason.contains("tokenizer"), "{case}: {reason}");
        }

        for (case, contents) in [
            ("exactly the bound", nested(MAX_TOKENIZER_JSON_DEPTH)),
            (
                "brackets inside a string",
                br#"{"a":"[[[[{{{{\"]]]]"}"#.to_vec(),
            ),
            (
                "a lone surrogate inside a value",
                br#"{"a":"\ud800"}"#.to_vec(),
            ),
            (
                "a lone surrogate in a nested key",
                br#"{"a":{"\ud800":1}}"#.to_vec(),
            ),
            (
                "a surrogate pair in a key",
                br#"{"\ud83d\ude00":1}"#.to_vec(),
            ),
        ] {
            std::fs::write(dir.path().join("tokenizer.json"), &contents).unwrap();
            assert!(
                validate_onnx_package(&graph).is_ok(),
                "{case} must be accepted, as Python accepts it"
            );
        }
    }

    #[test]
    fn a_missing_graph_is_not_found_rather_than_invalid() {
        let dir = TempDir::new("no_graph");
        assert!(matches!(
            validate_onnx_package(&dir.path().join("absent.onnx")),
            Err(EmbeddingError::ModelNotFound { .. })
        ));
        assert!(matches!(
            validate_onnx_package(dir.path()),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
    }

    #[test]
    fn an_invalid_onnx_model_is_described_as_onnx_not_as_gguf() {
        let dir = TempDir::new("display");
        let graph = package_with(&dir, b"<html>");
        let message = validate_onnx_package(&graph).unwrap_err().to_string();
        assert!(
            message.starts_with("Not a valid ONNX model file ("),
            "{message}"
        );

        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"not gguf at all, nope").unwrap();
        let message = validate_and_checksum_gguf(&gguf).unwrap_err().to_string();
        assert!(
            message.starts_with("Not a valid GGUF model file ("),
            "GGUF keeps its exact wording: {message}"
        );
    }
}
