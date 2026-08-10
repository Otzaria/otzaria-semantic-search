//! Carrying vectors forward from one library release to the next.
//!
//! A vector is a function of the text that was embedded and nothing else — same model,
//! same backend, same recipe, same float. So the question "has this line already been
//! embedded?" has a complete answer that costs no inference: **is its
//! `embedding_text_sha256` in the previous release's ledger?**
//!
//! That key is deliberately not a line identity. Two things go wrong with one:
//!
//! * The runtime id (`document_id_scheme_version` 1) is positional, so inserting a book
//!   moves the ids of every book after it and a diff sees millions of changed lines where
//!   nothing changed.
//! * A stable id would be worse than useless here, because it would be *quietly* wrong:
//!   a line whose own text is untouched still gets a new embedding text when the
//!   neighbour it borrows context from changes. An id-keyed diff calls that unchanged and
//!   reuses a vector built from text the library no longer contains.
//!
//! The digest cannot make either mistake. It describes the string that was actually
//! embedded, after context and truncation, which is exactly what the vector depends on.
//!
//! ```text
//! plan_split   plan + previous ledger -> reuse.jsonl + embed.jsonl
//! embed_shard  embed.jsonl            -> vectors + records          (GPU)
//! assemble     reuse + base + shards  -> vectors + records + ledger
//! ```
//!
//! `assemble` writes in whatever order is cheapest, because
//! [`pack`](super::packer::pack) compares the *set* of ids it was handed against the
//! recipe's expected set and sorts internally. What it may not do is pair a vector with
//! the wrong record, which is why the two files are written in lockstep here as well.

use crate::distribution::packer::VectorInputRecord;
use crate::distribution::shard::PlannedChunk;
use crate::errors::PackError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Read, Seek, SeekFrom, Write};

/// One line of a ledger: a digest, and where its vector sits in that release's
/// `vectors.f32`.
///
/// Published beside an artifact and never shipped to a user — it exists so the *next*
/// build can decide what to skip, and a user's installation has nothing to decide.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub embedding_text_sha256: String,
    /// Index of the vector, not a byte offset: the width is the artifact's
    /// `embedding_dim`, and multiplying here would bake it into the file.
    pub offset: u64,
}

/// A planned line whose vector already exists, and where to find it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuseEntry {
    pub line_id: u64,
    pub source_line_sha256: String,
    pub embedding_text_sha256: String,
    pub base_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitReport {
    pub planned: usize,
    pub reused: usize,
    pub to_embed: usize,
}

/// One shard's vectors and the records that pair with them, already opened.
pub type ShardStreams = (Box<dyn Read>, Box<dyn BufRead>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembleReport {
    pub vectors: usize,
    pub reused: usize,
    pub embedded: usize,
    pub embedding_dim: usize,
}

/// Split a plan against a ledger: what can be copied, and what has to be embedded.
///
/// Streams the plan, so the memory cost is the ledger alone — 32 bytes of digest and 8 of
/// offset per previous vector, which for a six-million-line library is a few hundred
/// megabytes on a build machine and nothing on the GPU worker, which never sees it.
///
/// An empty ledger is not an error. It is the first build, and every line goes to the GPU.
///
/// # Errors
///
/// [`PackError::Corpus`] for a malformed ledger or plan line, or a sink that will not
/// accept bytes.
pub fn plan_split(
    plan: impl BufRead,
    ledger: impl BufRead,
    reuse_sink: &mut dyn Write,
    embed_sink: &mut dyn Write,
) -> Result<SplitReport, PackError> {
    let mut known: HashMap<String, u64> = HashMap::new();
    for line in ledger.lines() {
        let line = line.map_err(read_error)?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: LedgerEntry =
            serde_json::from_str(&line).map_err(|error| PackError::Corpus {
                reason: format!("a ledger entry is malformed: {error}"),
            })?;
        // Last writer wins, and it does not matter: two entries with one digest describe
        // the same text and therefore the same vector.
        known.insert(entry.embedding_text_sha256, entry.offset);
    }

    let (mut planned, mut reused) = (0usize, 0usize);
    for line in plan.lines() {
        let line = line.map_err(read_error)?;
        if line.trim().is_empty() {
            continue;
        }
        let chunk: PlannedChunk =
            serde_json::from_str(&line).map_err(|error| PackError::Corpus {
                reason: format!("a plan record is malformed: {error}"),
            })?;
        planned += 1;

        match known.get(&chunk.embedding_text_sha256) {
            Some(&base_offset) => {
                reused += 1;
                write_json(
                    reuse_sink,
                    &ReuseEntry {
                        line_id: chunk.line_id,
                        source_line_sha256: chunk.source_line_sha256,
                        embedding_text_sha256: chunk.embedding_text_sha256,
                        base_offset,
                    },
                )?;
            }
            // The plan record travels whole: the worker embeds `embedding_text` and can
            // carry no corpus, so anything dropped here cannot be recovered there.
            None => write_json(embed_sink, &chunk)?,
        }
    }

    reuse_sink.flush().map_err(read_error)?;
    embed_sink.flush().map_err(read_error)?;
    Ok(SplitReport {
        planned,
        reused,
        to_embed: planned - reused,
    })
}

/// Copy the reusable vectors out of the base artifact, append this run's shards, and
/// write the ledger the *next* release will split against.
///
/// Reads of the base file are issued in ascending offset order rather than in plan order.
/// The base is tens of gigabytes and the reuse set is most of it; in plan order those
/// reads are a random walk over the whole file, and sorting first turns them into a
/// forward scan. It costs one sort of 8-byte keys and changes nothing about the result,
/// because the pairing travels in the record written beside each vector.
///
/// # Errors
///
/// [`PackError::MalformedInput`] for a base file too short for an offset it was asked
/// for, and [`PackError::Corpus`] for unreadable input or an unwritable sink.
pub fn assemble(
    mut reuse: Vec<ReuseEntry>,
    base_vectors: Option<&mut (impl Read + Seek)>,
    shards: Vec<ShardStreams>,
    embedding_dim: usize,
    vectors_sink: &mut dyn Write,
    records_sink: &mut dyn Write,
    ledger_sink: &mut dyn Write,
) -> Result<AssembleReport, PackError> {
    let width = embedding_dim
        .checked_mul(4)
        .ok_or_else(|| PackError::MalformedInput {
            reason: format!("an embedding dimension of {embedding_dim} has no byte width"),
        })?;
    let mut offset = 0u64;
    let mut buffer = vec![0u8; width];

    let reused = reuse.len();
    if reused > 0 {
        let base = base_vectors.ok_or_else(|| PackError::MalformedInput {
            reason: format!("{reused} vector(s) are to be reused and no base was given"),
        })?;
        reuse.sort_unstable_by_key(|entry| entry.base_offset);
        for entry in reuse {
            let at = entry.base_offset.checked_mul(width as u64).ok_or_else(|| {
                PackError::MalformedInput {
                    reason: format!("offset {} is beyond any file", entry.base_offset),
                }
            })?;
            base.seek(SeekFrom::Start(at)).map_err(read_error)?;
            base.read_exact(&mut buffer)
                .map_err(|error| PackError::MalformedInput {
                    reason: format!(
                        "the base artifact has no vector at offset {}: {error}",
                        entry.base_offset
                    ),
                })?;
            vectors_sink.write_all(&buffer).map_err(read_error)?;
            write_json(
                records_sink,
                &VectorInputRecord {
                    line_id: entry.line_id,
                    source_line_sha256: entry.source_line_sha256,
                    embedding_text_sha256: entry.embedding_text_sha256.clone(),
                },
            )?;
            write_json(
                ledger_sink,
                &LedgerEntry {
                    embedding_text_sha256: entry.embedding_text_sha256,
                    offset,
                },
            )?;
            offset += 1;
        }
    }

    let mut embedded = 0usize;
    for (mut shard_vectors, shard_records) in shards {
        for line in shard_records.lines() {
            let line = line.map_err(read_error)?;
            if line.trim().is_empty() {
                continue;
            }
            let record: VectorInputRecord =
                serde_json::from_str(&line).map_err(|error| PackError::Corpus {
                    reason: format!("a shard record is malformed: {error}"),
                })?;
            // One vector per record, read in lockstep. A shard whose two files disagree
            // on length fails here rather than silently shifting every pairing after it.
            shard_vectors
                .read_exact(&mut buffer)
                .map_err(|error| PackError::MalformedInput {
                    reason: format!(
                        "a shard has {} record(s) and ran out of vectors: {error}",
                        embedded + 1
                    ),
                })?;
            vectors_sink.write_all(&buffer).map_err(read_error)?;
            write_json(
                ledger_sink,
                &LedgerEntry {
                    embedding_text_sha256: record.embedding_text_sha256.clone(),
                    offset,
                },
            )?;
            write_json(records_sink, &record)?;
            offset += 1;
            embedded += 1;
        }
        let mut tail = [0u8; 1];
        if shard_vectors.read(&mut tail).map_err(read_error)? != 0 {
            return Err(PackError::MalformedInput {
                reason: "a shard has more vector bytes than records".to_string(),
            });
        }
    }

    vectors_sink.flush().map_err(read_error)?;
    records_sink.flush().map_err(read_error)?;
    ledger_sink.flush().map_err(read_error)?;

    Ok(AssembleReport {
        vectors: reused + embedded,
        reused,
        embedded,
        embedding_dim,
    })
}

fn write_json<T: Serialize>(sink: &mut dyn Write, value: &T) -> Result<(), PackError> {
    let mut line = serde_json::to_vec(value).map_err(|error| PackError::Corpus {
        reason: format!("a record could not be written: {error}"),
    })?;
    line.push(b'\n');
    sink.write_all(&line).map_err(read_error)
}

fn read_error(error: std::io::Error) -> PackError {
    PackError::Corpus {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const DIM: usize = 2;

    fn planned(line_id: u64, text: &str) -> String {
        serde_json::to_string(&PlannedChunk {
            line_id,
            source_line_sha256: format!("{line_id:064}"),
            embedding_text_sha256: text.to_string(),
            embedding_text: format!("text of {line_id}"),
        })
        .unwrap()
            + "\n"
    }

    fn vectors_of(values: &[[f32; DIM]]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
            .collect()
    }

    /// The whole point, as one assertion: a line whose embedding text is unchanged does
    /// not reach the GPU, and one whose text is new does.
    #[test]
    fn only_the_digests_the_ledger_does_not_know_are_sent_to_be_embedded() {
        let plan = planned(1, "aa") + &planned(2, "bb") + &planned(3, "cc");
        let ledger = "{\"embedding_text_sha256\":\"aa\",\"offset\":7}\n\
                      {\"embedding_text_sha256\":\"cc\",\"offset\":0}\n";
        let (mut reuse, mut embed) = (Vec::new(), Vec::new());

        let report = plan_split(
            Cursor::new(plan),
            Cursor::new(ledger),
            &mut reuse,
            &mut embed,
        )
        .unwrap();

        assert_eq!((report.planned, report.reused, report.to_embed), (3, 2, 1));
        let reused: Vec<ReuseEntry> = String::from_utf8(reuse)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            reused
                .iter()
                .map(|e| (e.line_id, e.base_offset))
                .collect::<Vec<_>>(),
            vec![(1, 7), (3, 0)],
            "each reused line must name the offset its own digest had"
        );
        let embedded: Vec<PlannedChunk> = String::from_utf8(embed)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(embedded.len(), 1);
        assert_eq!(embedded[0].line_id, 2);
        assert_eq!(
            embedded[0].embedding_text, "text of 2",
            "the worker has no corpus, so the text has to travel with the record"
        );
    }

    /// A reused vector must be the base vector its offset names — the failure this whole
    /// mechanism risks is copying the wrong float and calling it unchanged.
    #[test]
    fn a_reused_vector_is_the_one_its_offset_points_at() {
        let base = vectors_of(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
        let reuse = vec![
            ReuseEntry {
                line_id: 10,
                source_line_sha256: "s3".to_string(),
                embedding_text_sha256: "d3".to_string(),
                base_offset: 3,
            },
            ReuseEntry {
                line_id: 11,
                source_line_sha256: "s1".to_string(),
                embedding_text_sha256: "d1".to_string(),
                base_offset: 1,
            },
        ];
        let (mut vectors, mut records, mut ledger) = (Vec::new(), Vec::new(), Vec::new());

        let report = assemble(
            reuse,
            Some(&mut Cursor::new(base)),
            Vec::new(),
            DIM,
            &mut vectors,
            &mut records,
            &mut ledger,
        )
        .unwrap();

        assert_eq!((report.vectors, report.reused, report.embedded), (2, 2, 0));
        // Ascending base offset, not the order the reuse list was given in.
        assert_eq!(vectors, vectors_of(&[[1.0, 1.0], [3.0, 3.0]]));
        let ids: Vec<u64> = String::from_utf8(records)
            .unwrap()
            .lines()
            .map(|l| {
                serde_json::from_str::<VectorInputRecord>(l)
                    .unwrap()
                    .line_id
            })
            .collect();
        assert_eq!(ids, vec![11, 10], "the record must follow its own vector");
        let entries: Vec<LedgerEntry> = String::from_utf8(ledger)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            entries
                .iter()
                .map(|e| (e.embedding_text_sha256.as_str(), e.offset))
                .collect::<Vec<_>>(),
            vec![("d1", 0), ("d3", 1)],
            "the new ledger must describe this artifact, not the base it copied from"
        );
    }

    /// A shard whose two files disagree on length would shift every pairing after it, and
    /// nothing downstream could see it: the ids would all be present and the count would
    /// be right.
    #[test]
    fn a_shard_with_more_vectors_than_records_is_refused() {
        let records =
            "{\"line_id\":1,\"source_line_sha256\":\"s\",\"embedding_text_sha256\":\"d\"}\n";
        let shards: Vec<(Box<dyn Read>, Box<dyn BufRead>)> = vec![(
            Box::new(Cursor::new(vectors_of(&[[1.0, 1.0], [2.0, 2.0]]))),
            Box::new(Cursor::new(records)),
        )];
        let (mut vectors, mut out_records, mut ledger) = (Vec::new(), Vec::new(), Vec::new());

        let outcome = assemble(
            Vec::new(),
            None::<&mut Cursor<Vec<u8>>>,
            shards,
            DIM,
            &mut vectors,
            &mut out_records,
            &mut ledger,
        );
        match outcome {
            Err(PackError::MalformedInput { reason }) => {
                assert!(
                    reason.contains("more vector bytes than records"),
                    "{reason}"
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
}
