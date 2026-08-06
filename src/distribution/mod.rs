//! Packaging and installation of a prebuilt semantic index.
//!
//! Nothing here talks to a remote service. A package is a directory of static
//! payload files plus a manifest and a checksum list; importing one validates it
//! and swaps it into place atomically. Downloading it — if it is downloaded at
//! all rather than copied — is the host application's job.
//!
//! The module was named `cloud` before S0; the name implied a runtime dependency
//! on a server that does not exist. See `docs/PRODUCT_CONTRACT.md` §5.
pub mod importer;
pub mod package;
