//! The three recipe versions an artifact declares, as things this build either implements
//! or refuses.
//!
//! `embedding_text_version`, `normalization_version` and `chunking_version` are not facts
//! about a model file — they are versions of the code in this crate. That makes them the
//! one part of an artifact's identity nothing external can vouch for, and the one part the
//! code itself can enforce completely.
//!
//! Left as bare integers they enforce nothing. A build could declare
//! `embedding_text_version = 27`, run the only text recipe that exists, and produce an
//! artifact that verifies: `validate_complete` only asked for a number above zero, and the
//! installation compares 27 against its own 27 and agrees. The artifact would describe a
//! recipe nobody has written.
//!
//! So each version is a closed enum here, and the code that *does* the work dispatches on
//! it:
//!
//! * [`ChunkingAlgorithm`] selects how a book becomes chunks, in
//!   [`Chunker::chunk_book`](crate::semantic::chunker::Chunker::chunk_book).
//! * [`EmbeddingTextRecipe`] selects what text a chunk carries to the model, in the same
//!   place.
//! * [`TextNormalizationRecipe`] selects what is done to a text before it reaches the
//!   model — on the build side inside the chunker, and on the query side before the one
//!   string a search embeds.
//!
//! **It is text normalization, not vector normalization.** `normalization_version` has
//! meant "text preprocessing before embedding" since the manifest first carried it, and
//! that is what it still means. L2 normalization of a finished vector is not versioned and
//! must not be: it is the invariant that makes a dot product a cosine, every store applies
//! it unconditionally, and a "version" that some code paths ignored would be a label
//! again — which is the whole failure this module exists to prevent.
//!
//! Each dispatch is a `match` over the enum rather than a default with a comment, so adding
//! a variant does not compile until every path that behaves differently under it has been
//! written. That is the whole mechanism: **a version exists when there is code selected by
//! it, and not before.**
//!
//! # Where each one is checked
//!
//! | version | carried by | refused by |
//! |---|---|---|
//! | `embedding_text_version` | `ModelIdentity`, and the chunker configuration | [`IndexVersion::validate_complete`](crate::semantic::versioning::IndexVersion::validate_complete), so both a build and an install refuse it |
//! | `normalization_version` | `ModelIdentity`, and the chunker configuration | the same |
//! | `chunking_version` | the chunker configuration only — the artifact carries the *hash* of that configuration | [`EmbeddingRecipe::resolve`] and [`Chunker::new`](crate::semantic::chunker::Chunker::new), i.e. wherever a configuration exists |
//!
//! The asymmetry in the last row is not an oversight. An installation never chunks
//! anything: it embeds one query and compares it. It therefore has no chunker
//! configuration to check, and `chunking_identity` is a one-way hash it cannot open. What
//! protects it is that the hash must match — a build under an unimplemented
//! `chunking_version` is refused at the build, which is the only place that number is ever
//! more than an opaque input to a digest.

use crate::errors::ArtifactError;
use crate::semantic::chunker::ChunkerConfig;
use crate::semantic::versioning::ModelIdentity;

/// How a book becomes chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkingAlgorithm {
    /// One chunk per line, with a line too short to stand alone borrowing text from its
    /// neighbours inside the same section, and the result truncated on a character
    /// boundary.
    AnchoredLine,
}

/// What text a chunk carries to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingTextRecipe {
    /// The line itself, or the line surrounded by its neighbours when it is too short to
    /// carry meaning alone. No title prefix and no reference prefix — whether either helps
    /// is S1's measurement, and the recipe that wins there becomes a second variant here
    /// rather than an edit to this one.
    LineOrNeighbourContext,
}

/// What is done to a text before the model sees it.
///
/// Applied to **both** sides of a search, and that is why it is versioned: a stored vector
/// built from normalized text and a query embedded from raw text land in different places,
/// and nothing about either vector says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextNormalizationRecipe {
    /// Nothing. The text is embedded exactly as the corpus stores it.
    ///
    /// A choice rather than a placeholder: the lexical index already holds text that went
    /// through the engine's own normalization at indexing time, so a second pass here would
    /// embed something neither side holds. Whether stripping vowels, folding finals or
    /// removing punctuation helps retrieval is S1's measurement, and the answer becomes a
    /// second variant here.
    AsSuppliedByCorpus,
}

impl TextNormalizationRecipe {
    /// The text the model is given.
    ///
    /// Borrowed when the recipe changes nothing, so the identity case costs no allocation
    /// over six million lines.
    pub fn apply<'a>(self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match self {
            Self::AsSuppliedByCorpus => std::borrow::Cow::Borrowed(text),
        }
    }
}

/// Generate the version mapping, the supported list and the rejection for one recipe axis.
///
/// A macro rather than three hand-written copies, because the failure it prevents is one of
/// them drifting: a `from_version` that accepts a number `version()` never produces, or a
/// "supported" list in a rejection message that is not the list actually accepted.
macro_rules! recipe_versions {
    ($name:ident, $field:literal, { $($variant:ident => $version:literal),+ $(,)? }) => {
        impl $name {
            /// Every variant this build implements. What
            /// [`Self::from_version`] searches, so a variant left out cannot be read back.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// The number an artifact declares for this variant. Already written into
            /// artifacts; changing one invalidates every artifact carrying it.
            pub const fn version(self) -> u32 {
                match self {
                    $(Self::$variant => $version),+
                }
            }

            /// The variant a declared version names, or a refusal.
            ///
            /// A version with no variant is not a future this build can be lenient about:
            /// it names behaviour that does not exist here, and the vectors were either
            /// built by something else or labelled by hand.
            pub fn from_version(version: u32) -> Result<Self, ArtifactError> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|candidate| candidate.version() == version)
                    .ok_or_else(|| ArtifactError::UnsupportedRecipeVersion {
                        field: $field,
                        found: version,
                        supported: Self::ALL
                            .iter()
                            .map(|candidate| candidate.version().to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    })
            }
        }
    };
}

recipe_versions!(ChunkingAlgorithm, "chunking_version", {
    AnchoredLine => 1,
});
recipe_versions!(EmbeddingTextRecipe, "embedding_text_version", {
    LineOrNeighbourContext => 1,
});
recipe_versions!(TextNormalizationRecipe, "normalization_version", {
    AsSuppliedByCorpus => 1,
});

/// The three versions resolved together, once, before any of them is acted on.
///
/// Resolved as a set rather than one at a time because they are one decision: a build that
/// implements two of the three declared versions and not the third has not implemented the
/// recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingRecipe {
    pub chunking: ChunkingAlgorithm,
    pub text: EmbeddingTextRecipe,
    pub normalization: TextNormalizationRecipe,
}

impl EmbeddingRecipe {
    /// Resolve every version a build is about to act under, and refuse the recipe if the
    /// configuration and the identity disagree about the text.
    ///
    /// Two of the three are carried in both places — the configuration because the chunker
    /// needs them to choose a path, the identity because an installation compares identities
    /// and never sees a configuration. Two copies of one fact drift, so they are compared
    /// here, at the only point where both are in hand.
    pub fn resolve(chunking: &ChunkerConfig, model: &ModelIdentity) -> Result<Self, ArtifactError> {
        for (field, configured, declared) in [
            (
                "embedding_text_version",
                chunking.embedding_text_version,
                model.embedding_text_version,
            ),
            (
                "normalization_version",
                chunking.normalization_version,
                model.normalization_version,
            ),
        ] {
            if configured != declared {
                return Err(ArtifactError::RecipeDisagreesWithIdentity {
                    field,
                    configured,
                    declared,
                });
            }
        }
        Ok(Self {
            chunking: ChunkingAlgorithm::from_version(chunking.chunking_version)?,
            text: EmbeddingTextRecipe::from_version(chunking.embedding_text_version)?,
            normalization: TextNormalizationRecipe::from_version(chunking.normalization_version)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A version maps to exactly one variant and back. The macro writes both directions
    /// from one list, and this is what would catch a hand-edit that broke the pairing.
    #[test]
    fn every_variant_is_reachable_from_the_version_it_declares() {
        for algorithm in ChunkingAlgorithm::ALL {
            assert_eq!(
                ChunkingAlgorithm::from_version(algorithm.version()).unwrap(),
                *algorithm
            );
        }
        for recipe in EmbeddingTextRecipe::ALL {
            assert_eq!(
                EmbeddingTextRecipe::from_version(recipe.version()).unwrap(),
                *recipe
            );
        }
        for recipe in TextNormalizationRecipe::ALL {
            assert_eq!(
                TextNormalizationRecipe::from_version(recipe.version()).unwrap(),
                *recipe
            );
        }
    }

    /// The point of the whole module: a number nothing implements is a rejection, and the
    /// message says which axis and what this build does implement.
    #[test]
    fn a_version_no_variant_implements_is_refused_by_name() {
        for (field, error) in [
            (
                "chunking_version",
                ChunkingAlgorithm::from_version(2).unwrap_err(),
            ),
            (
                "embedding_text_version",
                EmbeddingTextRecipe::from_version(2).unwrap_err(),
            ),
            (
                "normalization_version",
                TextNormalizationRecipe::from_version(2).unwrap_err(),
            ),
        ] {
            match error {
                ArtifactError::UnsupportedRecipeVersion {
                    field: named,
                    found,
                    supported,
                } => {
                    assert_eq!(named, field);
                    assert_eq!(found, 2);
                    assert_eq!(supported, "1");
                }
                other => panic!("{field} 2 must be refused as unsupported, got {other:?}"),
            }
        }

        // Zero is refused too, and by this rather than only by the "is it filled in" check
        // — the two happen to agree today, and neither is the other's backstop.
        assert!(ChunkingAlgorithm::from_version(0).is_err());
    }

    /// Two copies of `embedding_text_version` exist because the chunker needs one and an
    /// installation can only see the other. Copies drift; this is where that is caught.
    #[test]
    fn a_configuration_that_contradicts_the_declared_text_version_is_refused() {
        let model = crate::semantic::versioning::test_identity().model;
        let chunking = ChunkerConfig {
            embedding_text_version: model.embedding_text_version + 1,
            ..ChunkerConfig::default()
        };

        match EmbeddingRecipe::resolve(&chunking, &model) {
            Err(ArtifactError::RecipeDisagreesWithIdentity {
                field,
                configured,
                declared,
            }) => {
                assert_eq!(field, "embedding_text_version");
                assert_eq!(configured, model.embedding_text_version + 1);
                assert_eq!(declared, model.embedding_text_version);
            }
            other => panic!("the two copies must be compared, got {other:?}"),
        }
    }

    #[test]
    fn the_default_configuration_resolves_to_what_this_build_implements() {
        let model = crate::semantic::versioning::test_identity().model;
        let recipe = EmbeddingRecipe::resolve(&ChunkerConfig::default(), &model).unwrap();
        assert_eq!(
            recipe,
            EmbeddingRecipe {
                chunking: ChunkingAlgorithm::AnchoredLine,
                text: EmbeddingTextRecipe::LineOrNeighbourContext,
                normalization: TextNormalizationRecipe::AsSuppliedByCorpus,
            }
        );
    }
}
