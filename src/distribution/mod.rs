//! Packaging and installation of a prebuilt semantic index.
//!
//! Nothing here talks to a remote service. A package is a directory of static payload
//! files plus a manifest and a per-payload descriptor table; installing one verifies it
//! and swaps it into place. Downloading it — if it is downloaded at all rather than
//! copied — is the host application's job.
//!
//! The swap is **not one atomic operation**, and nothing here claims it is: replacing a
//! directory takes two renames, and between them the target does not exist. What the
//! install guarantees instead is that the state a crash leaves is recognizable and
//! recoverable — see [`importer`] and `docs/ARTIFACT_CONTRACT.md` §5.4.
//!
//! The module was named `cloud` before S0; the name implied a runtime dependency
//! on a server that does not exist. See `docs/PRODUCT_CONTRACT.md` §5.
//!
//! The build side lives here too, because producing a package and installing one are two
//! ends of the same contract: [`builder`] applies the embedding recipe to a [`corpus`],
//! embeds what it derives and writes the base segment, the [`package`] around it and the
//! release manifest an installation takes — so the vectors and the identity that
//! describes them come from a single pass over a single model. [`shard`] cuts the same
//! work in two, for a library embedded on machines that never see the corpus.
pub mod builder;
pub mod corpus;
pub(crate) mod files;
pub mod importer;
pub mod ledger;
pub mod package;
pub mod plan;
pub mod shard;

#[cfg(test)]
mod plan_tests;
#[cfg(test)]
pub(crate) mod testing;
