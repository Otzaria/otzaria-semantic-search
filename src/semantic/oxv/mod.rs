//! The vector store's segment files: `.oxv`.
//!
//! A vector set is a few immutable segments — a base and the deltas applied since — each a
//! single file that is mapped rather than read. A segment holds, per **slot**, one encoded
//! vector, the 16-byte [`ChunkKey`](crate::semantic::chunk_key::ChunkKey) of the text it
//! was embedded from, and a hint: the ordinal of the line it was built at. Records — which
//! books hold which key, at which line — are stored apart from the vectors, so a vector
//! shared by many lines is stored once:
//!
//! * a **primary** record per slot, in the book whose slot range holds it;
//! * an **extra** record for each further occurrence of a key the segment ships;
//! * a **foreign** record, in a delta, for an occurrence of a key an older segment ships;
//! * a **tombstone**, in a delta, for a key the library no longer holds.
//!
//! Books are stored in byte order of their names and each book's primary slots are
//! contiguous, so a filter on books scans ranges. The layout, field by field, is in
//! `docs/ARTIFACT_CONTRACT.md`; [`format`](mod@format) holds it as code.
//!
//! | module | what it does |
//! |---|---|
//! | [`format`](mod@format) | the header, the section directory and the fixed-size records |
//! | [`codec`] | how a unit vector becomes a slot's bytes: `f32`, and int8 with per-dimension scales |
//! | [`writer`] | tables first, then the vectors streamed in slot order |
//! | [`reader`] | map a segment, check what is cheap to check, read records in place |
//! | `kernel` | the exact integer dot product, scalar, AVX2 and NEON |
//! | [`scan`] | every live slot scored, on several threads, under a book filter, the best `k` kept |

pub mod codec;
pub mod format;
pub(crate) mod kernel;
pub mod reader;
pub mod scan;
pub mod writer;

#[cfg(test)]
mod scan_tests;
#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod tests;
