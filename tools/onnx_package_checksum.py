#!/usr/bin/env python3
"""The `model_checksum` of an ONNX model package, computed without the crate.

An ONNX model is a package, not a file: the graph `model_path` names, the
`tokenizer.json` beside it, and every external-data file the graph's tensors name. The
crate defines its checksum in `src/semantic/model_package.rs`:

    manifest       = "otzaria-onnx-package-v1\\n"
                   + one line per file, ordered by relpath bytewise:
                     relpath "\\t" size_in_bytes "\\t" sha256_lowercase_hex "\\n"
    model_checksum = sha256(manifest as UTF-8), 64 lowercase hex digits

`relpath` is relative to the graph's directory and `/`-separated. Nothing else in that
directory is covered -- a README, a licence, a second graph, an ONNX Runtime library --
because none of it reaches a vector.

This is the second, independent implementation the definition asks for. Whoever writes
`model.json` for a new model, and CI checking a download, must reach the value the crate
reaches from the same bytes; a checksum only one implementation can produce is one
nobody can check. `tools/test_onnx_package_checksum.py` reproduces the crate's golden
(`the_golden_package_has_the_documented_checksum`) from the same bytes.

Why the walk is hand-written rather than `onnx.load` + `onnx.external_data_helper`:

  * that helper finds external data in initializers and attribute tensors only -- not in
    sparse initializers, sparse attributes or training graphs, all of which the crate
    covers. For a package that uses one, it would name fewer files, i.e. produce a
    different checksum, and silently;
  * `onnx.load` parses the whole graph into memory, and a dependency is a version to pin.

So this mirrors the crate's bounded, forward-only protobuf walk -- the same fields (from
onnx/onnx.proto3 at adf47535), the same wire-type checks, the same limits, and the same
refusals: a package the crate refuses is refused here, and one it accepts has the same
checksum here. Standard library only; the graph is hashed while it is walked, and never
held in memory.

Usage:

    python3 tools/onnx_package_checksum.py path/to/graph.onnx
    python3 tools/onnx_package_checksum.py --manifest path/to/graph.onnx | shasum -a 256
    python3 tools/onnx_package_checksum.py --expect <64 hex> path/to/graph.onnx

The first prints the checksum alone on stdout, and the package's files on stderr. The
second prints the manifest exactly as it is hashed. The third exits 1 unless the
checksum is the expected one. A refused package exits 1 and says why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Tuple

MANIFEST_VERSION = "otzaria-onnx-package-v1"
TOKENIZER_FILE = "tokenizer.json"

# The crate's limits (model_package.rs), kept identical: a package one implementation
# accepts and the other refuses would have a checksum only one of them can confirm.
MAX_TOKENIZER_BYTES = 256 << 20
# Past any real tokenizer.json, and serde_json's own recursion limit; see _read_tokenizer.
MAX_TOKENIZER_JSON_DEPTH = 128
SNIFF_BYTES = 64
MAX_MESSAGE_DEPTH = 64
MAX_MESSAGES = 1 << 24
MAX_EXTERNAL_TENSORS = 1 << 20
MAX_EXTERNAL_FILES = 1 << 12
MAX_EXTERNAL_DATA_ENTRY_BYTES = 4096
MAX_EXTERNAL_DATA_ENTRIES = 64
MAX_KEPT_NAME_BYTES = 256
MAX_TENSOR_DIMS = 256

DATA_LOCATION_EXTERNAL = 1
DATA_TYPE_UNDEFINED = 0
DATA_TYPE_STRING = 8

U64_MAX = (1 << 64) - 1
READ_CHUNK = 1 << 20

# Protobuf wire types.
VARINT, FIXED64, LEN, START_GROUP, END_GROUP, FIXED32 = 0, 1, 2, 3, 4, 5
WIRE_NAMES = {
    VARINT: "a varint",
    FIXED64: "a fixed64",
    LEN: "a length-delimited value",
    START_GROUP: "a group",
    END_GROUP: "an end-group marker",
    FIXED32: "a fixed32",
}


def _fields(spec):
    """Expand {(field numbers...): wire types} into {field: wire types}."""
    table = {}
    for numbers, wires in spec.items():
        for number in numbers:
            table[number] = wires
    return table


_L, _V, _F32 = (LEN,), (VARINT,), (FIXED32,)
_RV, _RF32, _RF64 = (VARINT, LEN), (FIXED32, LEN), (FIXED64, LEN)

# Every field each walked message declares, and the wire types it may arrive in; a field
# not listed is unknown and skipped, as every protobuf parser skips one, so a model from
# a newer ONNX still validates. A declared field in the wrong wire type is not a newer
# ONNX but bytes that are not ONNX at all. A repeated scalar may arrive packed or not.
ALLOWED = {
    # ir_version, model_version | producer_name, producer_version, domain, doc_string,
    # graph, opset_import, metadata_props, training_info, functions, configuration
    "ModelProto": _fields({(1, 5): _V, (2, 3, 4, 6, 7, 8, 14, 20, 25, 26): _L}),
    # node, name, initializer, doc_string, input, output, value_info,
    # quantization_annotation, sparse_initializer, metadata_props
    "GraphProto": _fields({(1, 2, 5, 10, 11, 12, 13, 14, 15, 16): _L}),
    # input, output, name, op_type, attribute, doc_string, domain, overload,
    # metadata_props, device_configurations
    "NodeProto": _fields({tuple(range(1, 11)): _L}),
    # f | i, type | floats | ints | name, s, t, g, strings, tensors, graphs, doc_string,
    # tp, type_protos, ref_attr_name, sparse_tensor, sparse_tensors
    "AttributeProto": _fields(
        {
            (2,): _F32,
            (3, 20): _V,
            (7,): _RF32,
            (8,): _RV,
            (1, 4, 5, 6, 9, 10, 11, 13, 14, 15, 21, 22, 23): _L,
        }
    ),
    # dims, int32_data, int64_data, uint64_data | data_type, data_location | float_data
    # | double_data | segment, string_data, name, raw_data, doc_string, external_data,
    # metadata_props
    "TensorProto": _fields(
        {
            (1, 5, 7, 11): _RV,
            (2, 14): _V,
            (4,): _RF32,
            (10,): _RF64,
            (3, 6, 8, 9, 12, 13, 16): _L,
        }
    ),
    # values, indices | dims
    "SparseTensorProto": _fields({(1, 2): _L, (3,): _RV}),
    # initialization, algorithm, initialization_binding, update_binding
    "TrainingInfoProto": _fields({(1, 2, 3, 4): _L}),
    # name, input, output, attribute, node, doc_string, opset_import, domain,
    # attribute_proto, value_info, overload, metadata_props (2 and 3 are reserved)
    "FunctionProto": _fields({(1,) + tuple(range(4, 15)): _L}),
    # domain | version
    "OperatorSetIdProto": _fields({(1,): _L, (2,): _V}),
    # key, value
    "StringStringEntryProto": _fields({(1, 2): _L}),
}


class PackageRefused(Exception):
    """The package is not one the crate would load. The message says why, and the fix."""

    def __init__(self, path, reason: str, kind: str = "invalid"):
        super().__init__(f"{path}: {reason}")
        self.path = str(path)
        self.reason = reason
        # "invalid", "model-not-found", "tokenizer-not-found" or "unreadable" -- the
        # crate's InvalidModelFile, ModelNotFound, TokenizerNotFound and LoadFailed.
        self.kind = kind


class PackageFile(NamedTuple):
    relpath: str
    path: Path
    size: int
    sha256: str


class GraphFacts(NamedTuple):
    ir_version: int
    opset_imports: int
    graph_inputs: int
    graph_outputs: int
    external_tensors: int


class OnnxPackage(NamedTuple):
    root: Path
    graph: Path
    files: List[PackageFile]
    facts: GraphFacts
    manifest: str
    checksum: str


def manifest_text(files) -> str:
    """The canonical manifest for `files`, in any order: sorted by relpath, bytewise."""
    ordered = sorted(files, key=lambda f: f.relpath.encode("utf-8"))
    lines = [f"{MANIFEST_VERSION}\n"]
    lines.extend(f"{f.relpath}\t{f.size}\t{f.sha256}\n" for f in ordered)
    return "".join(lines)


def package_checksum(files) -> str:
    return hashlib.sha256(manifest_text(files).encode("utf-8")).hexdigest()


def unsafe_in_manifest(relpath: str) -> Optional[str]:
    """What in a package-relative name would make the manifest ambiguous or the package
    unportable, or None. Rust's `char::is_control` is exactly Unicode category Cc."""
    if any(ord(c) < 0x20 or 0x7F <= ord(c) <= 0x9F for c in relpath):
        return "a control character"
    if "\\" in relpath:
        return (
            "a backslash, which Windows reads as a separator and every other platform as "
            "a character -- write the path with '/'"
        )
    if any(c in relpath for c in ':<>"|?*'):
        return "a character Windows does not allow in a file name"
    return None


def package_relpath(location: str) -> str:
    """The package-relative, '/'-separated form of an external-data location. Raises
    ValueError naming what is wrong with it. `.` and empty components are dropped --
    `./weights.bin` and `weights.bin` are one file and one manifest line -- while `..` is
    refused outright, even where it would stay inside the package, and so is a trailing
    `/`, which names a directory."""
    if location == "":
        raise ValueError("an empty location")
    reason = unsafe_in_manifest(location)
    if reason is not None:
        raise ValueError(reason)
    if location.startswith("/"):
        raise ValueError("an absolute path")
    if location.endswith("/"):
        raise ValueError("a trailing '/', which names a directory")
    components = []
    for component in location.split("/"):
        if component in ("", "."):
            continue
        if component == "..":
            raise ValueError("a '..' component")
        components.append(component)
    if not components:
        raise ValueError("no file name")
    return "/".join(components)


def sniff_non_onnx(prefix: bytes, file_len: int) -> Optional[str]:
    """Name the kind of file `prefix` begins, when it is a kind people mistake for a
    model. Only a better message: the walk refuses every one of these in its first byte."""
    if file_len == 0:
        return "the file is empty"
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        return (
            "it is a Git LFS pointer, not the model: the repository was cloned without its "
            "LFS objects. Run `git lfs pull` in it, or download the file itself"
        )
    if prefix.startswith(b"GGUF"):
        return (
            "it is a GGUF container, not an ONNX graph. Only a path ending in .onnx is read "
            "as ONNX; give a GGUF model its own .gguf name"
        )
    if prefix.startswith(b"PK\x03\x04"):
        return "it is a ZIP archive; extract the model from it first"
    # Rust's trim_ascii_start: space, \t, \n, \x0c, \r -- not \x0b, which bytes.lstrip()
    # would also strip.
    first = prefix.lstrip(b" \t\n\x0c\r")[:1]
    if first == b"<":
        return (
            "it is an HTML or XML document -- typically an error page saved in place of "
            "the model; download it again"
        )
    if first in (b"{", b"["):
        return (
            "it is a JSON document -- typically an error response saved in place of the "
            "model; download it again"
        )
    return None


# ── the protobuf walk ─────────────────────────────────────────────────────────────────


class _Truncated(Exception):
    def __init__(self, at: int, wanted: str, needs: int):
        self.at, self.wanted, self.needs = at, wanted, needs


class _Malformed(Exception):
    pass


class _Limit(Exception):
    pass


class _HashingReader:
    """Forward reader over the graph file. Hashes every byte read from the file, and
    counts the bytes the walk has consumed; `finish` hashes whatever is left."""

    def __init__(self, fh):
        self._fh = fh
        self._hasher = hashlib.sha256()
        self._buf = b""
        self._off = 0
        self.consumed = 0
        self.read_total = 0

    def _refill(self) -> bool:
        chunk = self._fh.read(READ_CHUNK)
        if chunk:
            self._hasher.update(chunk)
            self.read_total += len(chunk)
        self._buf, self._off = chunk, 0
        return bool(chunk)

    def byte(self) -> Optional[int]:
        if self._off >= len(self._buf) and not self._refill():
            return None
        value = self._buf[self._off]
        self._off += 1
        self.consumed += 1
        return value

    def fill(self, n: int) -> Optional[bytes]:
        out = bytearray()
        while n > 0:
            if self._off >= len(self._buf) and not self._refill():
                return None
            take = min(n, len(self._buf) - self._off)
            out += self._buf[self._off : self._off + take]
            self._off += take
            self.consumed += take
            n -= take
        return bytes(out)

    def skip(self, n: int) -> bool:
        while n > 0:
            if self._off >= len(self._buf) and not self._refill():
                return False
            take = min(n, len(self._buf) - self._off)
            self._off += take
            self.consumed += take
            n -= take
        return True

    def finish(self) -> Tuple[str, int]:
        while self._refill():
            pass
        return self._hasher.hexdigest(), self.read_total


class _External(NamedTuple):
    tensor: str
    location: str
    needs_bytes: int


def _as_i64(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


def _as_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


class _GraphWalk:
    """A forward-only protobuf walk over the graph file, the crate's `GraphWalk`.

    Every read is bounded by the end of the message it belongs to, and every message by
    the one holding it. Only the outermost message is bounded by the end of the *file*,
    so only there does a read past its bound prove a truncated download. A nested
    message's own length was checked before it was entered, so a field that runs past
    *it* is a malformed file -- even when that message ends exactly where the file does.
    """

    def __init__(self, reader: _HashingReader, file_len: int):
        self.reader = reader
        self.file_len = file_len
        self.messages = 0
        # How many length-bounded messages the walk is inside; 1 is the file's own
        # ModelProto.
        self.regions = 0
        self.ir_version: Optional[int] = None
        self.opset_imports = 0
        self.invalid_opset_version: Optional[int] = None
        self.graphs = 0
        self.graph_inputs = 0
        self.graph_outputs = 0
        self.external: List[_External] = []

    def pos(self) -> int:
        return self.reader.consumed

    def _eof(self, wanted: str):
        # The length was checked against the file first: this is a file that shrank.
        return _Truncated(self.pos(), wanted, self.file_len)

    def within(self, n: int, end: int, wanted: str) -> None:
        """Refuse a read of `n` bytes that would pass `end`: as truncation in the
        outermost message, whose bound is the end of the file, and as malformation
        anywhere else."""
        at = self.pos()
        needs = at + n
        if needs > U64_MAX:
            raise _Malformed(f"{wanted} at byte {at} declares {n} bytes, which no file can hold")
        if needs <= end:
            return
        if self.regions <= 1:
            raise _Truncated(at, wanted, needs)
        raise _Malformed(
            f"{wanted} at byte {at} runs past the end of the message holding it (byte {end})"
        )

    def byte(self, end: int, wanted: str) -> int:
        self.within(1, end, wanted)
        value = self.reader.byte()
        if value is None:
            raise self._eof(wanted)
        return value

    def varint(self, end: int, wanted: str) -> int:
        value = 0
        for index in range(10):
            byte = self.byte(end, wanted)
            # The tenth byte carries bit 63 alone; anything more is a varint no encoder
            # writes.
            if index == 9 and byte > 1:
                raise _Malformed(
                    f"{wanted} ending at byte {self.pos()} is a varint wider than 64 bits"
                )
            value |= (byte & 0x7F) << (7 * index)
            if byte & 0x80 == 0:
                return value
        raise _Malformed(
            f"{wanted} ending at byte {self.pos()} is a varint longer than ten bytes"
        )

    def skip(self, n: int, end: int, wanted: str) -> None:
        self.within(n, end, wanted)
        if not self.reader.skip(n):
            raise self._eof(wanted)

    def fill(self, n: int, wanted: str) -> bytes:
        data = self.reader.fill(n)
        if data is None:
            raise self._eof(wanted)
        return data

    def tag(self, end: int) -> Tuple[int, int]:
        at = self.pos()
        raw = self.varint(end, "a field tag")
        if raw > 0xFFFFFFFF:
            raise _Malformed(f"the field tag at byte {at} is wider than 32 bits")
        field = raw >> 3
        if field == 0:
            raise _Malformed(
                f"the field tag at byte {at} names field 0, which protobuf does not have"
            )
        wire = raw & 7
        if wire not in WIRE_NAMES:
            raise _Malformed(
                f"the field tag at byte {at} has wire type {wire}, which protobuf does not "
                "have"
            )
        return field, wire

    def value_end(self, end: int, wanted: str) -> int:
        """Read a length prefix and return where the value it announces ends."""
        n = self.varint(end, wanted)
        self.within(n, end, wanted)
        return self.pos() + n

    def enter(self, depth: int) -> None:
        if depth > MAX_MESSAGE_DEPTH:
            raise _Limit(
                f"its messages nest deeper than {MAX_MESSAGE_DEPTH} levels at byte "
                f"{self.pos()}"
            )
        self.messages += 1
        if self.messages > MAX_MESSAGES:
            raise _Limit(f"it holds more than {MAX_MESSAGES} messages")

    def skip_value(self, field: int, wire: int, end: int, depth: int) -> None:
        if wire == VARINT:
            self.varint(end, "a varint field")
        elif wire == FIXED64:
            self.skip(8, end, "a fixed64 field")
        elif wire == FIXED32:
            self.skip(4, end, "a fixed32 field")
        elif wire == LEN:
            stop = self.value_end(end, "a length-delimited field")
            self.skip(stop - self.pos(), stop, "a length-delimited field")
        elif wire == START_GROUP:
            self.skip_group(field, end, depth + 1)
        else:
            raise _Malformed(f"an end-group marker at byte {self.pos()} closes no group")

    def skip_group(self, field: int, end: int, depth: int) -> None:
        """Skip a (deprecated, but valid) group up to the end marker of its field."""
        self.enter(depth)
        while True:
            inner, wire = self.tag(end)
            if wire == END_GROUP:
                if inner == field:
                    return
                raise _Malformed(f"group {field} is closed as group {inner} at byte {self.pos()}")
            self.skip_value(inner, wire, end, depth)

    def fields(
        self,
        message: str,
        end: int,
        depth: int,
        visit: Callable[[int, int, int], bool],
    ) -> None:
        """Walk the fields of one `message` running to `end`, handing each to `visit`,
        which returns whether it consumed the value; an unconsumed one is skipped."""
        self.enter(depth)
        # Left raised on an error: the walk is abandoned then, not resumed.
        self.regions += 1
        allowed = ALLOWED[message]
        while self.pos() < end:
            at = self.pos()
            field, wire = self.tag(end)
            if wire == END_GROUP:
                raise _Malformed(f"an end-group marker at byte {at} closes no group")
            wires = allowed.get(field)
            if wires is not None and wire not in wires:
                raise _Malformed(
                    f"field {field} of {message} at byte {at} is encoded as "
                    f"{WIRE_NAMES[wire]}, which that field never is"
                )
            if not visit(field, wire, end):
                self.skip_value(field, wire, end, depth)
        self.regions -= 1

    def model(self) -> None:
        """The file: one ModelProto, running to the end."""
        first = [True]

        def visit(field, _wire, end):
            # A newer ONNX may add fields, but never before every field an older one
            # knows: a file whose first field is not a ModelProto field is not a model.
            if first[0]:
                first[0] = False
                if field not in ALLOWED["ModelProto"]:
                    raise _Malformed(
                        "it does not begin like an ONNX model: its first field is number "
                        f"{field}, which ModelProto does not have"
                    )
            if field == 1:
                self.ir_version = _as_i64(self.varint(end, "ir_version"))
            elif field == 7:
                stop = self.value_end(end, "the graph")
                self.graphs += 1
                self.graph(stop, 1, True)
            elif field == 8:
                self.opset(self.value_end(end, "an opset_import"), 1)
            elif field == 20:
                self.training_info(self.value_end(end, "a training_info"), 1)
            elif field == 25:
                self.function(self.value_end(end, "a function"), 1)
            else:
                return False
            return True

        self.fields("ModelProto", self.file_len, 0, visit)

    def opset(self, end: int, depth: int) -> None:
        version = [0]

        def visit(field, _wire, end):
            if field == 2:
                version[0] = _as_i64(self.varint(end, "an operator set version"))
                return True
            return False

        self.fields("OperatorSetIdProto", end, depth, visit)
        self.opset_imports += 1
        if version[0] < 1 and self.invalid_opset_version is None:
            self.invalid_opset_version = version[0]

    def graph(self, end: int, depth: int, main: bool) -> None:
        """The main graph when `main`, whose inputs and outputs are counted, or one
        nested in an attribute or a training step, walked only for its tensors."""

        def visit(field, _wire, end):
            if field == 1:
                self.node(self.value_end(end, "a node"), depth + 1)
                return True
            if field == 5:
                self.tensor(self.value_end(end, "an initializer"), depth + 1, "initializer")
                return True
            if field == 15:
                self.sparse_tensor(
                    self.value_end(end, "a sparse initializer"), depth + 1, "sparse initializer"
                )
                return True
            # Counted, then skipped like any other field.
            if field == 11 and main:
                self.graph_inputs += 1
            elif field == 12 and main:
                self.graph_outputs += 1
            return False

        self.fields("GraphProto", end, depth, visit)

    def node(self, end: int, depth: int) -> None:
        def visit(field, _wire, end):
            if field == 5:
                self.attribute(self.value_end(end, "a node attribute"), depth + 1)
                return True
            return False

        self.fields("NodeProto", end, depth, visit)

    def attribute(self, end: int, depth: int) -> None:
        """Where a constant tensor, a sparse one or a whole subgraph (the body of an If,
        a Loop or a Scan) can hide external data."""

        def visit(field, _wire, end):
            if field in (5, 10):
                self.tensor(self.value_end(end, "an attribute tensor"), depth + 1, "attribute tensor")
            elif field in (6, 11):
                self.graph(self.value_end(end, "an attribute graph"), depth + 1, False)
            elif field in (22, 23):
                self.sparse_tensor(
                    self.value_end(end, "an attribute sparse tensor"),
                    depth + 1,
                    "attribute sparse tensor",
                )
            else:
                return False
            return True

        self.fields("AttributeProto", end, depth, visit)

    def sparse_tensor(self, end: int, depth: int, role: str) -> None:
        def visit(field, _wire, end):
            if field in (1, 2):
                self.tensor(
                    self.value_end(end, "a sparse tensor's values or indices"), depth + 1, role
                )
                return True
            return False

        self.fields("SparseTensorProto", end, depth, visit)

    def training_info(self, end: int, depth: int) -> None:
        def visit(field, _wire, end):
            if field in (1, 2):
                self.graph(self.value_end(end, "a training graph"), depth + 1, False)
                return True
            return False

        self.fields("TrainingInfoProto", end, depth, visit)

    def function(self, end: int, depth: int) -> None:
        def visit(field, _wire, end):
            if field == 7:
                self.node(self.value_end(end, "a function node"), depth + 1)
            elif field == 11:
                self.attribute(self.value_end(end, "a function attribute"), depth + 1)
            else:
                return False
            return True

        self.fields("FunctionProto", end, depth, visit)

    def tensor(self, end: int, depth: int, role: str) -> None:
        """Kept only if its data lives elsewhere, and then as its external reference.
        raw_data -- the bulk of a real graph -- is skipped, never held."""
        state = {"name": "", "data_type": DATA_TYPE_UNDEFINED, "location": 0}
        dims: List[int] = []
        entries: List[Tuple[Optional[str], Optional[str]]] = []

        def visit(field, wire, end):
            if field == 1:
                self.dims(wire, end, dims)
            elif field == 2:
                # int32 on the wire, sign-extended to 64 bits when negative.
                state["data_type"] = _as_i32(self.varint(end, "a tensor data_type"))
            elif field == 8:
                state["name"] = self.kept_name(end)
            elif field == 13:
                if len(entries) >= MAX_EXTERNAL_DATA_ENTRIES:
                    raise _Limit(
                        f"a tensor carries more than {MAX_EXTERNAL_DATA_ENTRIES} "
                        "external_data entries"
                    )
                stop = self.value_end(end, "an external_data entry")
                entries.append(self.string_entry(stop, depth + 1))
            elif field == 14:
                state["location"] = self.varint(end, "a tensor data_location")
            else:
                return False
            return True

        self.fields("TensorProto", end, depth, visit)

        location = state["location"]
        if location == 0:
            return
        if location != DATA_LOCATION_EXTERNAL:
            raise _Malformed(
                f"a tensor's data_location is {location}, which is neither DEFAULT (0) nor "
                "EXTERNAL (1)"
            )
        name = state["name"]
        tensor = f"an unnamed {role}" if name == "" else f"{role} '{name}'"
        reference = _external_reference(tensor, entries, dims, state["data_type"])
        if len(self.external) >= MAX_EXTERNAL_TENSORS:
            raise _Limit(f"more than {MAX_EXTERNAL_TENSORS} tensors keep their data externally")
        self.external.append(reference)

    def dims(self, wire: int, end: int, dims: List[int]) -> None:
        def push(value):
            if len(dims) >= MAX_TENSOR_DIMS:
                raise _Limit(f"a tensor declares more than {MAX_TENSOR_DIMS} dimensions")
            dims.append(_as_i64(value))

        if wire == VARINT:
            push(self.varint(end, "a tensor dimension"))
        elif wire == LEN:
            stop = self.value_end(end, "packed tensor dimensions")
            while self.pos() < stop:
                push(self.varint(stop, "a packed tensor dimension"))
        else:
            raise _Malformed(f"tensor dimensions arrive as {WIRE_NAMES[wire]}")

    def kept_name(self, end: int) -> str:
        """A tensor name, kept to MAX_KEPT_NAME_BYTES for messages; the rest skipped."""
        stop = self.value_end(end, "a tensor name")
        length = stop - self.pos()
        kept = min(length, MAX_KEPT_NAME_BYTES)
        data = self.fill(kept, "a tensor name")
        self.skip(length - kept, stop, "a tensor name")
        name = data.decode("utf-8", errors="replace")
        return name + "…" if kept < length else name

    def string(self, end: int, wanted: str) -> str:
        stop = self.value_end(end, wanted)
        length = stop - self.pos()
        if length > MAX_EXTERNAL_DATA_ENTRY_BYTES:
            raise _Limit(
                f"{wanted} is {length} bytes, and at most {MAX_EXTERNAL_DATA_ENTRY_BYTES} "
                "are accepted"
            )
        data = self.fill(length, wanted)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise _Malformed(f"{wanted} is not UTF-8") from None

    def string_entry(self, end: int, depth: int) -> Tuple[Optional[str], Optional[str]]:
        entry: List[Optional[str]] = [None, None]

        def visit(field, _wire, end):
            if field == 1:
                entry[0] = self.string(end, "an external_data key")
            elif field == 2:
                entry[1] = self.string(end, "an external_data value")
            else:
                return False
            return True

        self.fields("StringStringEntryProto", end, depth, visit)
        return entry[0], entry[1]


def _external_reference(tensor, entries, dims, data_type) -> _External:
    """Resolve one external tensor's entries into what the package must hold for it.

    A key given twice is refused: which of two locations a runtime reads is its own
    business, and a validator that guessed differently would hash the wrong file. Keys
    other than the ones the specification names are ignored -- none can move the data.
    """
    slots = {"location": None, "offset": None, "length": None}
    for key, value in entries:
        if key not in slots:
            continue
        if slots[key] is not None:
            raise _Malformed(f"{tensor} names its external-data {key} twice")
        slots[key] = value if value is not None else ""

    location = slots["location"]
    if location is None:
        raise _Malformed(f"{tensor} keeps its data externally but names no location")

    def number(what, value):
        if value is None:
            return None
        # Digits only: no sign, no space, no base prefix -- and ASCII digits only, which
        # str.isdigit() is not.
        if value == "" or not all("0" <= c <= "9" for c in value):
            raise _Malformed(
                f"{tensor} gives its external-data {what} as {value!r}, which is not a "
                "non-negative integer"
            )
        parsed = int(value)
        if parsed > U64_MAX:
            raise _Malformed(
                f"{tensor} gives its external-data {what} as {value}, which no file can reach"
            )
        return parsed

    offset = number("offset", slots["offset"]) or 0
    length = number("length", slots["length"])

    if length is not None:
        payload = length
    elif data_type in (DATA_TYPE_STRING, DATA_TYPE_UNDEFINED):
        payload = 0
    else:
        # One bit per element: the smallest ONNX type is two bits wide, so this cannot
        # demand a byte a valid file lacks.
        elements = 1
        for dim in dims:
            if dim < 0:
                raise _Malformed(f"{tensor} declares the dimension {dim}")
            elements = min(elements * dim, U64_MAX)
        payload = (elements + 7) // 8
    needs_bytes = offset + payload
    if needs_bytes > U64_MAX:
        raise _Malformed(f"{tensor}'s external-data offset {offset} plus its length overflows")
    return _External(tensor, location, needs_bytes)


def _hash_file(path: Path) -> Tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(READ_CHUNK), b""):
            hasher.update(block)
            size += len(block)
    return hasher.hexdigest(), size


def _walk_graph_file(graph: Path, refuse) -> Tuple[_GraphWalk, str, int]:
    """Sniff, walk and hash the graph file."""
    with open(graph, "rb") as fh:
        file_len = os.fstat(fh.fileno()).st_size
        kind = sniff_non_onnx(fh.read(SNIFF_BYTES), file_len)
        if kind is not None:
            raise refuse(kind)
        fh.seek(0)
        walk = _GraphWalk(_HashingReader(fh), file_len)
        try:
            walk.model()
        except _Truncated as e:
            raise refuse(
                f"the file is {file_len} bytes, and {e.wanted} starting at byte {e.at} runs to "
                f"byte {e.needs} -- the download is incomplete; download the model again"
            ) from None
        except _Malformed as e:
            raise refuse(f"it is not a valid ONNX model: {e}") from None
        except _Limit as e:
            raise refuse(f"it is past what this validator accepts: {e}") from None
        sha256, size = walk.reader.finish()
    if size != file_len:
        raise refuse(
            f"it changed while it was read ({file_len} bytes when opened, {size} read); "
            "validate it again once nothing is writing to it"
        )
    return walk, sha256, size


def _facts(walk: _GraphWalk, refuse) -> GraphFacts:
    """The model-level requirements, checked once the whole file has been read."""
    if walk.ir_version is None:
        raise refuse(
            "it declares no ir_version, which every ONNX model carries; it is not an ONNX "
            "model, or not all of one"
        )
    if walk.ir_version <= 0:
        raise refuse(f"its ir_version is {walk.ir_version}; every ONNX IR version is positive")
    if walk.opset_imports == 0:
        raise refuse(
            "it declares no opset_import, so none of its operators can be resolved; if it "
            "was downloaded, the download may have stopped early"
        )
    if walk.invalid_opset_version is not None:
        raise refuse(
            f"it imports an operator set at version {walk.invalid_opset_version}; every "
            "operator set version is at least 1"
        )
    if walk.graphs == 0:
        raise refuse("it holds no graph")
    if walk.graph_inputs == 0:
        raise refuse("its graph declares no input, so there is nothing to feed it")
    if walk.graph_outputs == 0:
        raise refuse("its graph declares no output, so there is no vector to take from it")
    return GraphFacts(
        walk.ir_version,
        walk.opset_imports,
        walk.graph_inputs,
        walk.graph_outputs,
        len(walk.external),
    )


def _within(path: Path, root: Path) -> bool:
    """Path::starts_with: component-wise, so `/pkg-other` is not inside `/pkg`."""
    return path.parts[: len(root.parts)] == root.parts


def _resolve_external_data(
    root: Path, graph: Path, graph_relpath: str, tokenizer: Path, references, refuse, unreadable
) -> List[PackageFile]:
    """Check every external-data file the graph names, and hash each once."""
    if not references:
        return []

    # Grouped by the file, keeping the reference that reaches furthest into it.
    needed = {}
    for reference in references:
        try:
            relpath = package_relpath(reference.location)
        except ValueError as e:
            raise refuse(
                f"{reference.tensor} keeps its data in {reference.location!r}, which contains "
                f"{e}; an external-data location must be a relative path inside the package "
                "directory"
            ) from None
        if relpath in (graph_relpath, TOKENIZER_FILE):
            which = "graph" if relpath == graph_relpath else "tokenizer"
            raise refuse(
                f"{reference.tensor} keeps its data in {reference.location!r}, which is the "
                f"package's own {which}"
            )
        kept = needed.get(relpath)
        if kept is None or reference.needs_bytes > kept.needs_bytes:
            needed[relpath] = reference
    if len(needed) > MAX_EXTERNAL_FILES:
        raise refuse(
            f"it names {len(needed)} external-data files, and at most {MAX_EXTERNAL_FILES} "
            "are accepted"
        )

    try:
        canonical_root = root.resolve(strict=True)
        canonical_graph = graph.resolve(strict=True)
        canonical_tokenizer = tokenizer.resolve(strict=True)
    except OSError as e:
        raise unreadable(root, e) from None

    files = []
    for relpath in sorted(needed, key=lambda r: r.encode("utf-8")):
        reference = needed[relpath]
        path = root / relpath
        try:
            canonical = path.resolve(strict=True)
        except FileNotFoundError:
            raise refuse(
                f"{reference.tensor} keeps its data in {relpath!r}, which is missing from the "
                f"package directory {root}; put that file beside the graph"
            ) from None
        except OSError as e:
            raise unreadable(path, e) from None
        # Through a symlink, `weights.bin` can be anything the process can read: the
        # checksum would describe a file outside the package, and a package installed
        # by copying its directory would arrive without it.
        if not _within(canonical, canonical_root):
            raise refuse(
                f"{reference.tensor} keeps its data in {relpath!r}, which resolves to "
                f"{canonical} -- outside the package directory; copy the file into the "
                "package instead of linking it"
            )
        if canonical in (canonical_graph, canonical_tokenizer):
            which = "graph" if canonical == canonical_graph else "tokenizer"
            raise refuse(
                f"{reference.tensor} keeps its data in {relpath!r}, which leads to the "
                f"package's own {which}"
            )
        if not canonical.is_file():
            raise refuse(f"{reference.tensor} keeps its data in {relpath!r}, which is not a regular file")
        try:
            sha256, size = _hash_file(canonical)
        except OSError as e:
            raise unreadable(canonical, e) from None
        if size < reference.needs_bytes:
            raise refuse(
                f"its external-data file {relpath!r} is {size} bytes, but "
                f"{reference.tensor} needs it to reach byte {reference.needs_bytes} -- the "
                "download is incomplete; download the model again"
            )
        files.append(PackageFile(relpath, path, size, sha256))
    return files


def _refuse_json_constant(name):
    raise ValueError(f"{name} is not JSON")


def json_nesting_depth(data: bytes) -> int:
    """How deeply `data`'s arrays and objects nest: every `[` or `{` outside a string opens
    a level, the top-level object being level 1. A linear scan over the bytes, not a parse,
    and exactly the crate's `json_nesting_depth`: inside a string a backslash takes the
    next byte with it, whatever it is."""
    depth = deepest = 0
    in_string = escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):  # [ {
            depth += 1
            deepest = max(deepest, depth)
        elif byte in (0x5D, 0x7D):  # ] }
            depth = max(depth - 1, 0)
    return deepest


def _read_tokenizer(path: Path, refuse, unreadable) -> PackageFile:
    """Read tokenizer.json whole, hash it, and require a JSON object -- which is also
    what catches a truncated download of it, since JSON that stops early does not parse."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_TOKENIZER_BYTES + 1)
    except OSError as e:
        raise unreadable(path, e) from None
    if len(data) > MAX_TOKENIZER_BYTES:
        raise refuse(
            f"its tokenizer {path} is larger than {MAX_TOKENIZER_BYTES} bytes, which no "
            "tokenizer is"
        )
    def not_json(why):
        return refuse(
            f"its tokenizer {path} is not a JSON object ({why}); if it was downloaded, "
            "download it again"
        )

    try:
        # The whole file as strict UTF-8, as the crate checks it before parsing.
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise not_json(e) from None
    # Bounded before json.loads, as in the crate: json recurses, and past a depth its
    # interpreter decides it raises RecursionError, while the crate's parser accepts any
    # depth. The same bound in both is what makes them agree.
    depth = json_nesting_depth(data)
    if depth > MAX_TOKENIZER_JSON_DEPTH:
        raise refuse(
            f"its tokenizer {path} nests {depth} levels deep, and at most "
            f"{MAX_TOKENIZER_JSON_DEPTH} are accepted; no tokenizer nests that deep"
        )
    try:
        # Strict JSON, as serde_json reads it: no byte-order mark, no NaN or Infinity.
        parsed = json.loads(text, parse_constant=_refuse_json_constant)
    except ValueError as e:
        raise not_json(e) from None
    if not isinstance(parsed, dict):
        raise not_json(f"it is a JSON {type(parsed).__name__}")
    # The crate reads each top-level key into a Rust String, which cannot hold a lone
    # surrogate: `{"\ud800": 1}` is refused there, and json alone would accept it. A key
    # deeper down, or a value, is not read as text by the crate, and is accepted by both.
    for key in parsed:
        try:
            key.encode("utf-8")
        except UnicodeEncodeError:
            raise not_json(f"the key {key!r} holds a lone surrogate") from None
    return PackageFile(TOKENIZER_FILE, path, len(data), hashlib.sha256(data).hexdigest())


def validate_onnx_package(graph) -> OnnxPackage:
    """Validate the ONNX package whose graph is `graph`, and compute its checksum --
    every file read exactly once, cheapest refusal first, in the crate's order.

    Raises PackageRefused for anything the crate would refuse.
    """
    graph = Path(graph)

    def refuse(reason, kind="invalid"):
        return PackageRefused(graph, reason, kind)

    def unreadable(what, error):
        return PackageRefused(graph, f"cannot read {what}: {error}", "unreadable")

    if not graph.name.lower().endswith(".onnx") or graph.name.lower() == ".onnx":
        raise refuse(
            "its name does not end in .onnx, and the crate reads every other path as a GGUF "
            "model; name the graph <something>.onnx"
        )
    try:
        is_file = graph.is_file()
        exists = graph.exists()
    except OSError as e:
        raise unreadable(graph, e) from None
    if not exists:
        raise refuse("no such file", "model-not-found")
    if not is_file:
        raise refuse("it is not a regular file; model_path must name the graph file itself")
    graph_relpath = graph.name
    try:
        graph_relpath.encode("utf-8")
    except UnicodeEncodeError:
        raise refuse(
            "its file name is not UTF-8, and the package checksum names every file in UTF-8; "
            "rename the graph"
        ) from None
    reason = unsafe_in_manifest(graph_relpath)
    if reason is not None:
        raise refuse(f"its file name {graph_relpath!r} contains {reason}; rename the graph")

    root = graph.parent
    tokenizer = root / TOKENIZER_FILE

    # 1. the tokenizer is there at all
    if not tokenizer.exists():
        raise PackageRefused(tokenizer, "the package's tokenizer.json is missing", "tokenizer-not-found")
    if not tokenizer.is_file():
        raise refuse(f"its tokenizer {tokenizer} is not a regular file")

    # 2 and 3. the graph, sniffed, then walked and hashed in one pass
    try:
        walk, graph_sha256, graph_size = _walk_graph_file(graph, refuse)
    except OSError as e:
        raise unreadable(graph, e) from None
    facts = _facts(walk, refuse)

    # 4. external data
    external = _resolve_external_data(
        root, graph, graph_relpath, tokenizer, walk.external, refuse, unreadable
    )

    # 5. the tokenizer, whole
    tokenizer_file = _read_tokenizer(tokenizer, refuse, unreadable)

    files = [PackageFile(graph_relpath, graph, graph_size, graph_sha256), tokenizer_file]
    files.extend(external)
    files.sort(key=lambda f: f.relpath.encode("utf-8"))
    manifest = manifest_text(files)
    return OnnxPackage(
        root=root,
        graph=graph,
        files=files,
        facts=facts,
        manifest=manifest,
        checksum=hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate an ONNX model package and print its model_checksum "
        "(otzaria-onnx-package-v1), exactly as the crate computes it."
    )
    ap.add_argument("graph", help="The graph file (*.onnx); tokenizer.json must be beside it.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--manifest",
        action="store_true",
        help="Print the manifest exactly as it is hashed, instead of the checksum.",
    )
    mode.add_argument(
        "--expect",
        metavar="HEX",
        help="Exit 1 unless the checksum is this one.",
    )
    args = ap.parse_args(argv)

    try:
        package = validate_onnx_package(args.graph)
    except PackageRefused as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1

    if args.manifest:
        sys.stdout.write(package.manifest)
        return 0

    facts = package.facts
    print(f"package root: {package.root}", file=sys.stderr)
    print(
        f"graph: IR {facts.ir_version}, {facts.opset_imports} opset import(s), "
        f"{facts.graph_inputs} input(s), {facts.graph_outputs} output(s), "
        f"{facts.external_tensors} external tensor reference(s)",
        file=sys.stderr,
    )
    for f in package.files:
        print(f"  {f.relpath}\t{f.size}\t{f.sha256}", file=sys.stderr)

    if args.expect is not None:
        if package.checksum != args.expect.strip().lower():
            print(
                f"MISMATCH: the package checksum is {package.checksum}, expected "
                f"{args.expect.strip().lower()}",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {package.checksum}", file=sys.stderr)
        return 0

    print(package.checksum)
    return 0


if __name__ == "__main__":
    sys.exit(main())
