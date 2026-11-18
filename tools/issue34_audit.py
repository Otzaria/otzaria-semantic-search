#!/usr/bin/env python3
"""Reproducible, read-only issue #34 audit of a schema-7 library.

Requires Python zstandard/tokenizers and rustc. The normalizer is compiled from
the engine source, rather than approximated by an HTML parser. Windowing is the
documented legacy production v2 recipe. Sampling excludes books with line_note
rows, since this tool deliberately does not port note rehydration.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import tempfile
import time

import zstandard
import tokenizers
from tokenizers import Tokenizer


def compile_normalizer(engine: Path, directory: Path) -> Path:
    source = (engine / "rust/src/hebrew_query.rs").read_text()
    # Source boundaries deliberately fail closed if the engine is reorganized.
    pieces = [
        source[source.index("pub(crate) fn is_attached_mark"):
               source.index("/// מסיר ניקוד וטעמים צמודים בלבד")],
        source[source.index("pub(crate) fn fold_presentation_form"):
               source.index("/// [`tag_breaks_word`] over a char vector")],
        source[source.index("/// The whitespace entities [`strip_html_for_indexing`]"):
               source.index("/// Both texts a line is indexed as, in one pass:")],
    ]
    main = r'''
use std::io::{BufRead, Write};
fn main() {
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let bytes = (0..line.len()).step_by(2)
            .map(|i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        let input = String::from_utf8(bytes).unwrap();
        let normalized = normalize_one_pass(&input, true, |_| false);
        for b in normalized.as_bytes() { write!(out, "{b:02x}").unwrap(); }
        writeln!(out).unwrap();
    }
}
'''
    path = directory / "normalize.rs"
    path.write_text("#![allow(dead_code)]\nconst CGJ: char = '\\u{034F}';\n"
                    + "\n".join(pieces) + main)
    binary = directory / "normalize"
    subprocess.run(["rustc", "--edition=2021", "-O", str(path), "-o", str(binary)],
                   check=True)
    return binary


def normalize(binary: Path, lines: list[str]) -> list[str]:
    payload = "".join(line.encode().hex() + "\n" for line in lines)
    result = subprocess.run([str(binary)], input=payload, text=True,
                            capture_output=True, check=True)
    output = [bytes.fromhex(line).decode() for line in result.stdout.splitlines()]
    if len(output) != len(lines):
        raise RuntimeError("normalizer changed row count")
    return output


def legacy_spans(text: str) -> list[tuple[int, int]]:
    """Exact v2 window_spans: 512 first, 512 next, 128 overlap, six maximum."""
    n = len(text)
    spans = [(0, min(n, 512))]
    while spans[-1][1] < n and len(spans) < 6:
        previous_start, previous_end = spans[-1]
        start = max(previous_end - 128, previous_start + 1)
        if text[start - 1] != " ":
            space = text.find(" ", start, previous_end)
            if space >= 0:
                start = space + 1
        end = min(n, start + 512)
        if end < n:
            space = text.rfind(" ", start + 256, end + 1)
            if space >= 0:
                end = space
        spans.append((start, end))
    return spans


def passages(lines: list[str], sections: list[int]):
    for i, text in enumerate(lines):
        text = text.strip()
        if len(text) < 5:
            continue
        if len(text) < 20:
            start, end = i, i
            while start > max(0, i - 2) and sections[start - 1] == sections[i]:
                start -= 1
            while end < min(len(lines) - 1, i + 2) and sections[end + 1] == sections[i]:
                end += 1
            context = " ".join(lines[start:end + 1]).strip()[:512].strip()
            anchor_start = len(" ".join(lines[start:i])) + (i > start)
            yield i, "short", context, anchor_start + len(text) > 512
        else:
            seen = set()
            for start, end in legacy_spans(text):
                passage = text[start:end].strip()
                if passage in seen:
                    continue
                seen.add(passage)
                yield i, "standalone", passage, False


def audit(args):
    started = time.monotonic()
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    dictionaries = {}
    for _, data in connection.execute("SELECT id,dict FROM zstd_dict"):
        dictionary = zstandard.ZstdCompressionDict(data)
        dictionaries[dictionary.dict_id()] = zstandard.ZstdDecompressor(dict_data=dictionary)
    plain = zstandard.ZstdDecompressor()

    def decode(value):
        if isinstance(value, str):
            return value
        dictionary_id = zstandard.get_frame_parameters(value).dict_id
        return dictionaries.get(dictionary_id, plain).decompress(value).decode("utf-8")

    books = connection.execute("SELECT id,title,totalLines FROM book ORDER BY id").fetchall()
    with_notes = {row[0] for row in connection.execute(
        "SELECT DISTINCT l.bookId FROM line_note n JOIN line l ON l.id=n.lineId")}
    eligible = [book for book in books if book[0] not in with_notes]
    sample = random.Random(args.seed).sample(eligible, min(args.books, len(eligible)))
    raw_counts = collections.Counter()
    examples = collections.defaultdict(list)
    sampled_counts = collections.Counter()
    token_lengths = []
    near_full_token_lengths = []
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.encode_special_tokens = False
    tokenizer.no_truncation()
    tokenizer.no_padding()
    metadata = {
        "db": str(args.db.resolve()), "db_size_bytes": args.db.stat().st_size,
        "schema_meta": dict(connection.execute("SELECT key,value FROM schema_meta")),
        "book_count": len(books), "declared_lines": sum(book[2] for book in books),
        "note_rows": connection.execute("SELECT count(*) FROM line_note").fetchone()[0],
        "books_with_notes": len(with_notes), "sample_seed": args.seed,
        "sample_book_ids": sorted(book[0] for book in sample),
        "sample_excludes_notes": True,
        "sample_data_uri_books_skipped": [],
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "python_tokenizers_version": tokenizers.__version__,
        "max_tokens_including_specials": args.max_tokens,
        "normalizer_source_sha256": hashlib.sha256(
            (args.engine / "rust/src/hebrew_query.rs").read_bytes()).hexdigest(),
        "legacy_recipe": {"chunking_version": 2, "embedding_text_version": 2,
                          "max_chunk_chars": 512, "window_chars": 512,
                          "overlap": 128, "max_windows": 6},
    }
    if args.full_raw_scan:
        first_rows = {}
        for line_id, book_id, line_index, value in connection.execute(
                "SELECT l.id,l.bookId,l.lineIndex,c.content FROM line l JOIN line_content c ON c.id=l.id ORDER BY l.id"):
            text = decode(value)
            raw_counts["lines"] += 1
            if (book_id not in first_rows
                    or (line_index, line_id) < first_rows[book_id][:2]):
                first_rows[book_id] = (line_index, line_id, text.startswith("\ufeff"))
            raw_counts["lines_with_bom"] += "\ufeff" in text
            scripts = list(re.finditer(r"<(sup|sub)\b([^>]*)>", text, re.I))
            raw_counts["lines_with_sup_or_sub"] += bool(scripts)
            raw_counts["sup_sub_tags"] += len(scripts)
            raw_counts["sup_sub_without_class"] += sum(
                not re.search(r"\bclass\s*=", match[2], re.I) for match in scripts)
            raw_counts["sup_sub_with_footnote_marker_word"] += sum(
                bool(re.search(r"\bfootnote-marker\b", match[2], re.I)) for match in scripts)
        raw_counts["first_rows"] = len(first_rows)
        raw_counts["first_row_bom"] = sum(row[2] for row in first_rows.values())
    with tempfile.TemporaryDirectory(prefix="otzaria-issue34-") as directory:
        binary = compile_normalizer(args.engine, Path(directory))
        assert normalize(binary, ["תורה<sup>30</sup>", "מְ<sup>נַ</sup>שֶּׁה"]) == ["תורה 30", "מנשה"]
        extreme = connection.execute(
            "SELECT id,bookId,lineIndex,charCount FROM line ORDER BY charCount DESC LIMIT 5").fetchall()
        extreme_raw = [decode(connection.execute(
            "SELECT content FROM line_content WHERE id=?", (row[0],)).fetchone()[0])
            for row in extreme]
        metadata["longest_visible_lines"] = [
            {"line_id": row[0], "book_id": row[1], "line_index": row[2],
             "visible_char_count": row[3], "normalized_chars_before_rehydration": len(text),
             "note_rows": connection.execute(
                 "SELECT count(*) FROM line_note WHERE lineId=?", (row[0],)).fetchone()[0]}
            for row, text in zip(extreme, normalize(binary, extreme_raw))]
        for book_id, title, _ in sorted(sample):
            rows = connection.execute(
                "SELECT l.id,l.lineIndex,c.content FROM line l JOIN line_content c ON c.id=l.id "
                "WHERE l.bookId=? ORDER BY l.lineIndex,l.id", (book_id,)).fetchall()
            raw = [decode(row[2]) for row in rows]
            # Avoid a second implementation of the app's whole-book URI cleanup.
            if any("data:" in text for text in raw):
                metadata["sample_data_uri_books_skipped"].append(book_id)
                continue
            lines = normalize(binary, raw)
            sections = []
            section = 0
            for text in raw:
                if text.startswith("<h"):
                    section += 1
                sections.append(section)
            sampled_counts["books"] += 1
            sampled_counts["lines"] += len(lines)
            for i, text in enumerate(lines):
                if len(text.strip()) >= 20:
                    n = len(text.strip())
                    end = legacy_spans(text.strip())[-1][1]
                    sampled_counts["standalone_lines"] += 1
                    sampled_counts["standalone_chars"] += n
                    sampled_counts["uncovered_tail_chars"] += n - end
                    sampled_counts["partially_covered_lines"] += end < n
                    sampled_counts["first_window_splits_word"] += (
                        n > 512 and text[511] != " " and text[512] != " ")
                    if end < n and len(examples["uncovered_tail"]) < 5:
                        examples["uncovered_tail"].append({"book_id": book_id, "title": title,
                            "line_id": rows[i][0], "line_index": rows[i][1],
                            "chars": n, "covered_to": end, "tail_chars": n - end})
            book_passages = list(passages(lines, sections))
            encoded = tokenizer.encode_batch(["[PASSAGE] " + p[2] for p in book_passages])
            for (i, kind, text, loses_anchor), encoding in zip(book_passages, encoded):
                n = len(encoding.ids)
                token_lengths.append(n)
                if len(text) >= 500:
                    near_full_token_lengths.append(n)
                sampled_counts["passages"] += 1
                sampled_counts["over_token_cap"] += n > args.max_tokens
                sampled_counts["near_full_passages"] += len(text) >= 500
                sampled_counts["near_full_over_token_cap"] += len(text) >= 500 and n > args.max_tokens
                if kind == "short":
                    sampled_counts["short_passages"] += 1
                    sampled_counts["short_anchor_not_fully_present"] += loses_anchor
                    sampled_counts["short_anchor_completely_absent_by_offset"] += loses_anchor and (
                        lines[i].strip() not in text)
                    if loses_anchor and len(examples["lost_short_anchor"]) < 5:
                        examples["lost_short_anchor"].append({"book_id": book_id, "title": title,
                            "line_id": rows[i][0], "line_index": rows[i][1],
                            "anchor": lines[i], "passage": text})
                if n > args.max_tokens and len(examples["token_overflow"]) < 5:
                    examples["token_overflow"].append({"book_id": book_id, "title": title,
                        "line_id": rows[i][0], "line_index": rows[i][1],
                        "chars": len(text), "tokens": n, "passage": text})
    token_lengths.sort()
    near_full_token_lengths.sort()
    return {"metadata": metadata, "full_raw_scan": dict(raw_counts),
            "sample": dict(sampled_counts), "examples": dict(examples),
            "tokens": {"median": token_lengths[len(token_lengths)//2] if token_lengths else 0,
                       "p95": token_lengths[int(len(token_lengths)*.95)] if token_lengths else 0,
                       "max": max(token_lengths, default=0),
                       "near_full_median": near_full_token_lengths[len(near_full_token_lengths)//2]
                           if near_full_token_lengths else 0,
                       "near_full_p95": near_full_token_lengths[int(len(near_full_token_lengths)*.95)]
                           if near_full_token_lengths else 0},
            "limitations": [
                "generation-7 OXV set unavailable: historical 34.8% missing-key rate not reproducible here",
                "sample excludes notes; raw scan counts include note-bearing books",
                "standalone tail coverage ignores tokenizer truncation; token overflow measured separately",
                "sample excludes whole books containing data: rather than approximating URI cleanup",
            ], "elapsed_seconds": round(time.monotonic()-started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--books", type=int, default=200)
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--full-raw-scan", action="store_true")
    args = parser.parse_args()
    result = audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "sample": result["sample"],
                      "full_raw_scan": result["full_raw_scan"],
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
