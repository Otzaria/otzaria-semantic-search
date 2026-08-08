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
//! ends of the same contract: [`packer`] turns ready-made vectors into a directory
//! [`package`] verifies and [`importer`] installs, and it joins every vector to the
//! [`corpus`] whose ids it claims before it writes anything.
pub mod corpus;
pub mod importer;
pub mod package;
pub mod packer;
