//! The artifact contract as a consumer of this crate sees it.
//!
//! `otzaria_search_engine` is the consumer: its builder writes a package and its
//! runtime installs and verifies one, both from outside this crate. These tests run
//! that path through the public API only — no `cfg(test)` helper, no private
//! constructor — so a field that cannot be filled from outside, or a check that cannot
//! be reached from outside, fails here.
//!
//! The payload is deliberately meaningless bytes. Nothing in this contract reads it;
//! reading it belongs to the store backend, whose format version is part of the
//! identity being verified.

use otzaria_semantic_search::distribution::importer::{
    previous_path, recover_interrupted_install, ImportConfig, IndexImporter,
};
use otzaria_semantic_search::distribution::package::{
    ArtifactExpectation, IndexPackage, PackageManifest, PayloadDescriptor, VerificationDepth,
};
use otzaria_semantic_search::errors::ArtifactError;
use otzaria_semantic_search::semantic::versioning::{
    CorpusIdentity, IdentityField, IdentityGroup, IndexVersion, ModelIdentity, StoreIdentity,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "otzaria_artifact_contract_{name}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A build that diverges from the installation in exactly one identity field.
type Divergence = (IdentityField, fn(&mut IndexVersion));

/// What the installation requires. `without_published_digest` is the honest form for a
/// locally built fixture: there is no published digest for an artifact built here.
fn expectation() -> ArtifactExpectation {
    ArtifactExpectation::without_published_digest(identity())
}

/// What a builder records, and what the installation later requires. Every value here
/// is a fact about a specific build — the dimension and precision are not this crate's
/// decision, which is why they are data.
fn identity() -> IndexVersion {
    IndexVersion {
        corpus: CorpusIdentity {
            corpus_id: "1f".repeat(32),
            library_version: "otzaria-library-2026-08".to_string(),
            tantivy_schema_version: 3,
            document_id_scheme_version: 1,
        },
        model: ModelIdentity {
            model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
            model_checksum: "ab".repeat(32),
            model_quantization: "Q4_K_M".to_string(),
            embedding_backend: "llama-cpp-2-0.1.153".to_string(),
            embedding_dim: 1024,
            pooling: "last-token".to_string(),
            max_tokens: 512,
            embedding_text_version: 1,
            normalization_version: 1,
            chunking_identity: 0x0BAD_C0DE,
        },
        store: StoreIdentity {
            backend_id: "zevc-persistent-v1".to_string(),
            store_format_version: 1,
            vector_precision: "f32".to_string(),
        },
    }
}

/// Build a package the way a builder would: write the payload, then the metadata that
/// describes it. Returns the digest a publisher would announce alongside it.
fn build_artifact(root: &Path, identity: IndexVersion, payload: &[u8]) -> String {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("vectors.bin"), payload).unwrap();

    let package = IndexPackage {
        manifest: PackageManifest::new(
            identity,
            "2026-08-06T00:00:00Z".to_string(),
            12,
            340,
            payload.len() as u64,
        ),
        payloads: BTreeMap::from([(
            "vectors.bin".to_string(),
            PayloadDescriptor::of_bytes(payload),
        )]),
    };
    IndexPackage::write(root, &package).unwrap();
    package.digest()
}

#[test]
fn a_matching_artifact_installs_and_verifies_again_from_the_installed_copy() {
    let dir = TempDir::new("install");
    let source = dir.path().join("build-output");
    let target = dir.path().join("semantic_index");
    build_artifact(&source, identity(), b"payload bytes");

    let result = IndexImporter::new(ImportConfig {
        source_path: source,
        target_store_path: target.clone(),
    })
    .import(&expectation())
    .unwrap();
    assert_eq!(result.books_imported, 12);
    assert_eq!(result.vectors_imported, 340);
    assert_eq!(result.bytes_imported, 13);

    // What a later session does: verify the installed copy on its own terms — metadata,
    // identity and payload presence, with nothing rebuilt and no payload byte read.
    let opened = IndexPackage::verify_for_open(&target, &expectation()).unwrap();
    assert_eq!(opened.identity(), &identity());
    assert_eq!(opened.payload_names(), ["vectors.bin"]);
    assert_eq!(opened.vector_count(), 340);
    assert_eq!(opened.depth(), VerificationDepth::MetadataAndPresence);

    // The deeper check is still available, and says so.
    assert_eq!(
        IndexPackage::verify_for_install(&target, &expectation())
            .unwrap()
            .depth(),
        VerificationDepth::FullPayload
    );
}

/// What a distributed artifact needs and a locally built one does not: a digest that
/// arrived from somewhere other than the package. Without it, a package rebuilt with
/// matching internal checksums is indistinguishable from the published one.
#[test]
fn a_published_digest_is_what_makes_a_distributed_artifact_verifiable() {
    let dir = TempDir::new("published");
    let official = dir.path().join("official");
    let published = build_artifact(&official, identity(), b"payload bytes");

    let target = dir.path().join("semantic_index");
    IndexImporter::new(ImportConfig {
        source_path: official,
        target_store_path: target.clone(),
    })
    .import(&ArtifactExpectation::with_published_digest(
        identity(),
        published.clone(),
    ))
    .unwrap();

    // The installed copy still answers to the published digest — the install rewrites
    // the metadata, so this is the property that makes the anchor usable at all.
    let opened = IndexPackage::verify_for_open(
        &target,
        &ArtifactExpectation::with_published_digest(identity(), published.clone()),
    )
    .unwrap();
    assert_eq!(opened.artifact_digest(), published);

    // A different artifact, internally consistent, same identity: refused only because
    // the digest was published.
    let impostor = dir.path().join("impostor");
    build_artifact(&impostor, identity(), b"payload bytez");
    assert!(
        IndexPackage::verify_for_install(&impostor, &expectation()).is_ok(),
        "in-package checks can only prove self-consistency"
    );
    assert!(matches!(
        IndexPackage::verify_for_install(
            &impostor,
            &ArtifactExpectation::with_published_digest(identity(), published)
        ),
        Err(ArtifactError::UnexpectedArtifactDigest { .. })
    ));
}

/// The crash window in the install: killed between the two renames, the target is gone
/// and the previous artifact is parked. A caller that opens the target without recovering
/// first would conclude the device has no artifact at all.
#[test]
fn an_install_interrupted_by_a_crash_is_recovered_to_the_previous_artifact() {
    let dir = TempDir::new("recovery");
    let source = dir.path().join("build-output");
    let target = dir.path().join("semantic_index");
    build_artifact(&source, identity(), b"payload bytes");

    IndexImporter::new(ImportConfig {
        source_path: source,
        target_store_path: target.clone(),
    })
    .import(&expectation())
    .unwrap();

    // Exactly what a kill between the two renames leaves behind.
    fs::rename(&target, previous_path(&target).unwrap()).unwrap();
    assert!(IndexPackage::verify_for_open(&target, &expectation()).is_err());

    let recovery = recover_interrupted_install(&target).unwrap();
    assert!(recovery.restored_previous);
    assert!(IndexPackage::verify_for_open(&target, &expectation()).is_ok());

    // And running it again on a healthy target changes nothing.
    assert!(!recover_interrupted_install(&target)
        .unwrap()
        .recovered_anything());
}

/// One field at a time, from outside the crate: an artifact that disagrees with the
/// installation on *anything* is refused by name before its payload is read.
#[test]
fn every_identity_group_can_refuse_an_artifact_and_says_which_field_disagreed() {
    let dir = TempDir::new("mismatch");

    let cases: Vec<Divergence> = vec![
        // Same library, one book inserted: the ids in the vectors now name other lines.
        (IdentityField::CorpusId, |identity| {
            identity.corpus.corpus_id = "2e".repeat(32)
        }),
        // Same model id, different weights behind it.
        (IdentityField::ModelChecksum, |identity| {
            identity.model.model_checksum = "cd".repeat(32)
        }),
        // A payload layout this build does not read.
        (IdentityField::StoreFormatVersion, |identity| {
            identity.store.store_format_version = 9
        }),
    ];

    for (field, diverge) in cases {
        let source = dir.path().join(format!("artifact-{field}"));
        let mut built = identity();
        diverge(&mut built);
        build_artifact(&source, built, b"payload bytes");

        match IndexPackage::verify_for_install(&source, &expectation()) {
            Err(ArtifactError::IdentityMismatch { mismatches }) => {
                assert_eq!(mismatches.len(), 1, "{field}");
                assert_eq!(mismatches[0].field, field);
            }
            other => panic!("{field} must refuse the artifact, got {other:?}"),
        }

        // And the same refusal blocks the install, so the target is never created.
        let target = dir.path().join(format!("target-{field}"));
        assert!(matches!(
            IndexImporter::new(ImportConfig {
                source_path: source,
                target_store_path: target.clone(),
            })
            .import(&expectation()),
            Err(ArtifactError::IdentityMismatch { .. })
        ));
        assert!(
            !target.exists(),
            "{field}: a refused install created a target"
        );
    }

    for group in [
        IdentityGroup::Corpus,
        IdentityGroup::Model,
        IdentityGroup::Store,
    ] {
        assert!(
            IdentityField::ALL.iter().any(|f| f.group() == group),
            "{group} must have comparable fields"
        );
    }
}

/// A damaged artifact and a foreign one are different problems: the first is fixed by
/// downloading this artifact again, the second by fetching the right one. The host
/// application shows different messages, so the crate must not collapse them.
#[test]
fn corruption_is_reported_separately_from_incompatibility() {
    let dir = TempDir::new("corrupt");
    let source = dir.path().join("build-output");
    build_artifact(&source, identity(), b"payload bytes");

    // A truncation changes the length, so even the cheap check sees it.
    fs::write(source.join("vectors.bin"), b"payload byte").unwrap();
    match IndexPackage::verify_for_open(&source, &expectation()) {
        Err(ArtifactError::ManifestDisagreesWithPayload { reason }) => {
            assert!(reason.contains("vectors.bin"), "{reason}")
        }
        other => panic!("expected a size rejection, got {other:?}"),
    }

    // A same-length edit needs the checksum.
    fs::write(source.join("vectors.bin"), b"payload byteZ").unwrap();
    match IndexPackage::verify_for_install(&source, &expectation()) {
        Err(ArtifactError::PayloadChecksumFailed { payload, .. }) => {
            assert_eq!(payload, "vectors.bin")
        }
        other => panic!("expected a checksum rejection, got {other:?}"),
    }

    let target = dir.path().join("semantic_index");
    assert!(IndexImporter::new(ImportConfig {
        source_path: source,
        target_store_path: target.clone(),
    })
    .import(&expectation())
    .is_err());
    assert!(!target.exists());
}

/// An installed artifact that is replaced under the runtime's feet does not stay
/// verified: the token describes the bytes that were checked, not the directory name.
///
/// And the two depths answer differently, which is the honest limit of opening without
/// hashing — recorded here so that nobody reads `verify_for_open` as tamper detection.
#[test]
fn an_installed_artifact_that_is_tampered_with_afterwards_stops_verifying() {
    let dir = TempDir::new("tamper");
    let source = dir.path().join("build-output");
    let target = dir.path().join("semantic_index");
    build_artifact(&source, identity(), b"payload bytes");

    IndexImporter::new(ImportConfig {
        source_path: source,
        target_store_path: target.clone(),
    })
    .import(&expectation())
    .unwrap();
    assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());

    fs::write(target.join("vectors.bin"), b"payload bytez").unwrap();
    assert!(matches!(
        IndexPackage::verify_for_install(&target, &expectation()),
        Err(ArtifactError::PayloadChecksumFailed { .. })
    ));
    assert!(
        IndexPackage::verify_for_open(&target, &expectation()).is_ok(),
        "a same-length edit is invisible without hashing; catching it while reading the \
         payload is the store backend's job, not this contract's"
    );
}
