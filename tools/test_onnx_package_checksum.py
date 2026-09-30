"""The Python package checksum reaches the crate's golden, and covers the files it covers.

`tools/onnx_package_checksum.py` is the second implementation of the ONNX package
checksum (`otzaria-onnx-package-v1`). The first is `src/semantic/model_package.rs`, and
the value both must reach from the same bytes is pinned there by
`the_golden_package_has_the_documented_checksum`: the graph bytes, the tokenizer, the
external-data file and the digest below are copied from that test, not recomputed. If
this file and that test disagree, one implementation changed the definition -- which is
changing the checksum of every ONNX artifact ever built.

The rest is the file-set rule, which is where two implementations drift apart quietly:
which files are in the package, which are beside it, and which references are refused.

Run with `python -m pytest tools/test_onnx_package_checksum.py`. Standard library and
pytest only; the one test that uses the `onnx` package as an independent parser skips
without it.
"""

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SCRIPT = HERE / "onnx_package_checksum.py"

spec = importlib.util.spec_from_file_location("onnx_package_checksum", SCRIPT)
checksum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checksum)

# ── the golden, from `the_golden_package_has_the_documented_checksum` ──

GOLDEN_GRAPH = bytes.fromhex(
    "0808120c6f747a617269612d737475623ad8011206676f6c64656e2a4b"
    "080408041001420b70726f6a2e7765696768746a190a086c6f636174696f6e120d776569676874"
    "732f772e62696e6a0b0a066f66667365741201306a0c0a066c656e6774681202363470015a2a0a"
    "09696e7075745f696473121d0a1b080712170a0208010a11120f73657175656e63655f6c656e67"
    "74685a2f0a0e617474656e74696f6e5f6d61736b121d0a1b080712170a0208010a11120f736571"
    "75656e63655f6c656e67746862240a1273656e74656e63655f656d62656464696e67120e0a0c08"
    "0112080a0208010a02080442040a001011"
)
GOLDEN_TOKENIZER = '{"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}'
GOLDEN_WEIGHTS = bytes(range(64))
GOLDEN_MANIFEST = (
    "otzaria-onnx-package-v1\n"
    "model.onnx\t241\t7a655f52bd5e7d6d72a0581505690c874d349505aa8a5a87019db3ba581add55\n"
    "tokenizer.json\t70\t5181aefd3b938b58bde0afbd009afd4c8cc0db4a61ad07dfe94868b112be8b37\n"
    "weights/w.bin\t64\tfdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108\n"
)
GOLDEN_DIGEST = "d8eea75d4348089f9ed6735709c95f24e8654fb6c5301ace41e03daea1f70699"


def write_golden(root: Path) -> Path:
    graph = root / "model.onnx"
    graph.write_bytes(GOLDEN_GRAPH)
    (root / "tokenizer.json").write_text(GOLDEN_TOKENIZER, encoding="utf-8")
    (root / "weights").mkdir()
    (root / "weights" / "w.bin").write_bytes(GOLDEN_WEIGHTS)
    # Beside the package, not in it.
    (root / "README.md").write_text("not part of the package", encoding="utf-8")
    return graph


# ── a minimal protobuf encoder, for graphs the tests shape themselves ──


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def uint(field: int, value: int) -> bytes:
    return varint(field << 3) + varint(value)


def message(field: int, payload: bytes) -> bytes:
    return varint(field << 3 | 2) + varint(len(payload)) + payload


def string(field: int, text: str) -> bytes:
    return message(field, text.encode("utf-8"))


def value_info(name: str, elem_type: int, dims) -> bytes:
    shape = b"".join(message(1, uint(1, d)) for d in dims)
    tensor_type = uint(1, elem_type) + message(2, shape)
    return string(1, name) + message(2, message(1, tensor_type))


def external_tensor(name, dims, location, offset=None, length=None) -> bytes:
    """A float TensorProto whose data lives in `location`."""
    tensor = b"".join(uint(1, d) for d in dims) + uint(2, 1) + string(8, name)
    for key, value in (("location", location), ("offset", offset), ("length", length)):
        if value is not None:
            tensor += message(13, string(1, key) + string(2, str(value)))
    return tensor + uint(14, checksum.DATA_LOCATION_EXTERNAL)


def graph(initializers=(), nodes=(), sparse_initializers=()) -> bytes:
    body = b"".join(message(1, node) for node in nodes)
    body += string(2, "g")
    body += b"".join(message(5, tensor) for tensor in initializers)
    body += message(11, value_info("input_ids", 7, [1, 8]))
    body += message(11, value_info("attention_mask", 7, [1, 8]))
    body += message(12, value_info("out", 1, [1, 4]))
    body += b"".join(message(15, sparse) for sparse in sparse_initializers)
    return body


def model(graph_body: bytes, functions=(), training=()) -> bytes:
    out = uint(1, 8) + string(2, "otzaria-stub") + message(7, graph_body)
    out += message(8, string(1, "") + uint(2, 17))
    out += b"".join(message(20, t) for t in training)
    out += b"".join(message(25, f) for f in functions)
    return out


def node_with_attribute(field: int, payload: bytes) -> bytes:
    """A NodeProto with one attribute whose `field` holds `payload`."""
    attribute = string(1, "attr") + message(field, payload)
    return string(4, "Op") + message(5, attribute)


def package(root: Path, graph_bytes: bytes, data=None) -> Path:
    """`graph_bytes` as `model.onnx`, the golden tokenizer, and `data` ({relpath: bytes})
    in `root`; returns the graph."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "model.onnx"
    path.write_bytes(graph_bytes)
    (root / "tokenizer.json").write_text(GOLDEN_TOKENIZER, encoding="utf-8")
    for relpath, contents in (data or {}).items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    return path


def refusal(graph_path: Path, kind: str = "invalid") -> str:
    with pytest.raises(checksum.PackageRefused) as refused:
        checksum.validate_onnx_package(graph_path)
    assert refused.value.kind == kind, refused.value
    return refused.value.reason


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── the checksum ──


def test_the_golden_package_has_the_documented_checksum(tmp_path):
    graph_path = write_golden(tmp_path)
    assert len(GOLDEN_GRAPH) == 241

    validated = checksum.validate_onnx_package(graph_path)
    assert validated.manifest == GOLDEN_MANIFEST
    assert validated.checksum == GOLDEN_DIGEST
    assert sha256(GOLDEN_MANIFEST.encode("utf-8")) == GOLDEN_DIGEST
    assert validated.facts == checksum.GraphFacts(
        ir_version=8, opset_imports=1, graph_inputs=2, graph_outputs=1, external_tensors=1
    )


def test_the_command_line_prints_the_checksum_and_the_manifest_it_hashes(tmp_path):
    graph_path = write_golden(tmp_path)

    def run(*args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args, str(graph_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    plain = run()
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout == GOLDEN_DIGEST + "\n", "stdout is the checksum and nothing else"

    manifest = run("--manifest")
    assert manifest.returncode == 0, manifest.stderr
    assert manifest.stdout.replace("\r\n", "\n") == GOLDEN_MANIFEST

    assert run("--expect", GOLDEN_DIGEST).returncode == 0
    wrong = run("--expect", "0" * 64)
    assert wrong.returncode == 1 and "MISMATCH" in wrong.stderr


def test_the_manifest_orders_files_bytewise_whatever_order_they_arrive_in():
    zeros = "0" * 64

    def entry(relpath, size):
        return checksum.PackageFile(relpath, Path(relpath), size, zeros)

    files = [
        entry("weights/w.bin", 3),
        entry("tokenizer.json", 2),
        entry("Model.onnx", 1),
        entry("model.onnx.data", 4),
    ]
    assert checksum.manifest_text(files) == (
        f"otzaria-onnx-package-v1\nModel.onnx\t1\t{zeros}\nmodel.onnx.data\t4\t{zeros}\n"
        f"tokenizer.json\t2\t{zeros}\nweights/w.bin\t3\t{zeros}\n"
    ), "uppercase sorts before lowercase, bytewise, as the crate sorts"
    assert checksum.package_checksum(files) == checksum.package_checksum(files[::-1])


def test_files_beside_the_package_do_not_change_the_checksum(tmp_path):
    graph_path = write_golden(tmp_path)
    for name, contents in [
        ("LICENSE.md", b"CC BY-NC-SA 4.0"),
        ("manifest.json", b'{"dims": 256}'),
        ("export_round2_onnx.py", b"print('export')"),
        (".gitattributes", b"*.onnx filter=lfs"),
        ("model-int8.onnx", GOLDEN_GRAPH),
        ("libonnxruntime.dylib", b"\xcf\xfa\xed\xfe"),
        ("libonnxruntime.so", b"\x7fELF"),
        ("onnxruntime.dll", b"MZ"),
        ("weights/other.bin", b"not referenced"),
    ]:
        (tmp_path / name).write_bytes(contents)
        assert checksum.validate_onnx_package(graph_path).checksum == GOLDEN_DIGEST, name


def test_one_changed_byte_in_any_package_file_changes_the_checksum(tmp_path):
    graph_path = write_golden(tmp_path)

    edited = GOLDEN_GRAPH.replace(b"otzaria-stub", b"Otzaria-stub")
    graph_path.write_bytes(edited)
    assert checksum.validate_onnx_package(graph_path).checksum != GOLDEN_DIGEST
    graph_path.write_bytes(GOLDEN_GRAPH)

    weights = tmp_path / "weights" / "w.bin"
    weights.write_bytes(GOLDEN_WEIGHTS[:-1] + b"\x00")
    assert checksum.validate_onnx_package(graph_path).checksum != GOLDEN_DIGEST
    weights.write_bytes(GOLDEN_WEIGHTS)

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(GOLDEN_TOKENIZER.replace('"[UNK]":0', '"[UNK]":9'), encoding="utf-8")
    assert checksum.validate_onnx_package(graph_path).checksum != GOLDEN_DIGEST
    tokenizer.write_text(GOLDEN_TOKENIZER, encoding="utf-8")

    assert checksum.validate_onnx_package(graph_path).checksum == GOLDEN_DIGEST


# ── the graph ──


def test_a_graph_cut_off_at_any_byte_is_refused(tmp_path):
    graph_path = write_golden(tmp_path)
    incomplete = 0
    for cut in range(len(GOLDEN_GRAPH)):
        graph_path.write_bytes(GOLDEN_GRAPH[:cut])
        reason = refusal(graph_path)
        incomplete += "download is incomplete" in reason
    # A cut between two whole fields leaves a shorter but well-formed model, refused for
    # what it lacks; every other cut is the incomplete download it is.
    assert incomplete > len(GOLDEN_GRAPH) // 2


@pytest.mark.parametrize(
    "contents, named",
    [
        (b"", "empty"),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:00\n", "Git LFS pointer"),
        (b"GGUF\x03\x00\x00\x00", "GGUF"),
        (b"PK\x03\x04rest", "ZIP"),
        (b"  \n<!DOCTYPE html><html>", "HTML"),
        (b'{"error": "Access to model is restricted"}', "JSON"),
        (b"\x0b<not html>", "not a valid ONNX model"),
        (b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff", "not a valid ONNX model"),
    ],
)
def test_bytes_that_are_not_onnx_are_refused_by_what_they_are(tmp_path, contents, named):
    graph_path = package(tmp_path, contents)
    assert named in refusal(graph_path)


def test_an_overrun_inside_a_message_that_ends_at_the_end_of_the_file_is_malformed(tmp_path):
    """Only the outermost message is bounded by the end of the file: a graph whose own
    length fits the file exactly, holding a field that claims more, is malformed --
    downloading it again would change nothing."""
    graph_body = graph() + varint(12 << 3 | 2) + varint(5000) + b"xy"
    opset = message(8, string(1, "") + uint(2, 17))
    reason = refusal(package(tmp_path, uint(1, 8) + opset + message(7, graph_body)))
    assert "runs past the end of the message" in reason
    assert "incomplete" not in reason

    # The same overrun at the top level is the incomplete download it looks like.
    reason = refusal(package(tmp_path / "top", uint(1, 8) + opset + varint(7 << 3 | 2) + varint(5000)))
    assert "download is incomplete" in reason


def test_a_model_without_its_required_parts_is_refused_by_name(tmp_path):
    no_opset = uint(1, 8) + message(7, graph())
    assert "no opset_import" in refusal(package(tmp_path / "a", no_opset))
    no_graph = uint(1, 8) + message(8, string(1, "") + uint(2, 17))
    assert "holds no graph" in refusal(package(tmp_path / "b", no_graph))
    no_ir = message(7, graph()) + message(8, string(1, "") + uint(2, 17))
    assert "no ir_version" in refusal(package(tmp_path / "c", no_ir))
    opset_zero = uint(1, 8) + message(7, graph()) + message(8, string(1, ""))
    assert "version 0" in refusal(package(tmp_path / "d", opset_zero))


def test_a_path_the_crate_reads_as_gguf_is_refused(tmp_path):
    write_golden(tmp_path)
    # Windows strips a trailing space from a file name, so it cannot hold the last one.
    names = ["model.bin", "model.onnx.part", ".onnx"] + ([] if os.name == "nt" else ["model.onnx "])
    for name in names:
        (tmp_path / name).write_bytes(GOLDEN_GRAPH)
        assert "does not end in .onnx" in refusal(tmp_path / name)
    (tmp_path / "MODEL.ONNX").write_bytes(GOLDEN_GRAPH)
    renamed = checksum.validate_onnx_package(tmp_path / "MODEL.ONNX")
    assert [f.relpath for f in renamed.files] == ["MODEL.ONNX", "tokenizer.json", "weights/w.bin"]


def test_a_missing_graph_is_not_found_rather_than_invalid(tmp_path):
    (tmp_path / "tokenizer.json").write_text(GOLDEN_TOKENIZER, encoding="utf-8")
    refusal(tmp_path / "absent.onnx", "model-not-found")


# ── the tokenizer ──


def test_a_missing_tokenizer_is_refused_before_the_graph_is_read(tmp_path):
    graph_path = tmp_path / "model.onnx"
    graph_path.write_bytes(b"not even a graph")
    refusal(graph_path, "tokenizer-not-found")


@pytest.mark.parametrize(
    "contents",
    [
        b'{"model": {"type": "WordLevel"',
        b"[1, 2, 3]",
        b"\xef\xbb\xbf{}",
        b'{"x": NaN}',
        b"{} trailing",
        b"",
    ],
)
def test_a_tokenizer_that_is_not_a_whole_json_object_is_refused(tmp_path, contents):
    graph_path = write_golden(tmp_path)
    (tmp_path / "tokenizer.json").write_bytes(contents)
    assert "not a JSON object" in refusal(graph_path)


# ── external data ──


@pytest.mark.parametrize(
    "location, contains",
    [
        ("../w.bin", "'..' component"),
        ("weights/../w.bin", "'..' component"),
        ("/etc/passwd", "absolute path"),
        ("weights\\w.bin", "backslash"),
        ("C:/w.bin", "Windows does not allow"),
        ("w\tbin", "control character"),
        ("", "empty location"),
        (".", "no file name"),
        ("./", "trailing '/'"),
        ("weights/", "trailing '/'"),
        ("model.onnx", "package's own graph"),
        ("./tokenizer.json", "package's own tokenizer"),
    ],
)
def test_an_external_location_that_is_not_a_file_inside_the_package_is_refused(
    tmp_path, location, contains
):
    graph_bytes = model(graph([external_tensor("w", [4], location, length=16)]))
    graph_path = package(tmp_path, graph_bytes, {"w.bin": b"\x07" * 16})
    assert contains in refusal(graph_path)


def test_dot_and_empty_components_name_one_file_and_one_manifest_line(tmp_path):
    graph_bytes = model(
        graph(
            [
                external_tensor("a", [4], "./weights//w.bin", offset=0, length=8),
                external_tensor("b", [4], "weights/w.bin", offset=8, length=8),
            ]
        )
    )
    graph_path = package(tmp_path, graph_bytes, {"weights/w.bin": b"\x01" * 16})
    validated = checksum.validate_onnx_package(graph_path)
    assert [f.relpath for f in validated.files] == ["model.onnx", "tokenizer.json", "weights/w.bin"]
    assert validated.facts.external_tensors == 2


def test_missing_or_short_external_data_is_refused(tmp_path):
    graph_bytes = model(graph([external_tensor("w", [4], "w.bin", offset=4, length=16)]))
    graph_path = package(tmp_path, graph_bytes)
    assert "missing from the package directory" in refusal(graph_path)

    (tmp_path / "w.bin").write_bytes(b"\x00" * 19)
    assert "download is incomplete" in refusal(graph_path)
    (tmp_path / "w.bin").write_bytes(b"\x00" * 20)
    checksum.validate_onnx_package(graph_path)

    # Without a length, the floor is one bit per element: 16 elements need 2 bytes.
    graph_path.write_bytes(model(graph([external_tensor("w", [4, 4], "w.bin")])))
    (tmp_path / "w.bin").write_bytes(b"\x00")
    assert "reach byte 2" in refusal(graph_path)


def test_a_symlink_out_of_the_package_is_refused_and_one_inside_it_is_not(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"\x00" * 16)
    root = tmp_path / "pkg"
    root.mkdir()
    graph_bytes = model(graph([external_tensor("w", [4], "w.bin", length=16)]))
    graph_path = package(root, graph_bytes)
    try:
        (root / "w.bin").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create symlinks here")
    assert "outside the package directory" in refusal(graph_path)

    (root / "w.bin").unlink()
    (root / "data").mkdir()
    (root / "data" / "real.bin").write_bytes(b"\x00" * 16)
    (root / "w.bin").symlink_to(root / "data" / "real.bin")
    validated = checksum.validate_onnx_package(graph_path)
    assert "w.bin" in [f.relpath for f in validated.files], "named by the link, as referenced"


def test_external_data_is_found_wherever_a_tensor_can_be(tmp_path):
    """Initializers, sparse initializers, node attributes, subgraphs, functions and
    training graphs -- every place the crate looks. `onnx.external_data_helper` looks
    in only some of them, which is why the walk does not use it."""

    def ext(name):
        return external_tensor(name, [2], f"{name}.bin", length=8)

    sparse = message(1, ext("sparse_values")) + message(2, ext("sparse_indices")) + uint(3, 2)
    subgraph = string(2, "body") + message(5, ext("subgraph_init"))
    function = string(1, "f") + message(7, node_with_attribute(5, ext("function_attr")))
    training = message(1, string(2, "init") + message(5, ext("training_init")))
    graph_bytes = model(
        graph(
            initializers=[ext("init")],
            nodes=[
                node_with_attribute(5, ext("attr_t")),
                node_with_attribute(10, ext("attr_tensors")),
                node_with_attribute(6, subgraph),
                node_with_attribute(22, message(1, ext("attr_sparse")) + uint(3, 2)),
            ],
            sparse_initializers=[sparse],
        ),
        functions=[function],
        training=[training],
    )
    names = [
        "init",
        "attr_t",
        "attr_tensors",
        "subgraph_init",
        "attr_sparse",
        "sparse_values",
        "sparse_indices",
        "function_attr",
        "training_init",
    ]
    graph_path = package(tmp_path, graph_bytes, {f"{n}.bin": b"\x00" * 8 for n in names})
    validated = checksum.validate_onnx_package(graph_path)
    assert sorted(f.relpath for f in validated.files) == sorted(
        [f"{n}.bin" for n in names] + ["model.onnx", "tokenizer.json"]
    )
    assert validated.facts.external_tensors == len(names)


def test_unknown_fields_are_skipped_and_known_fields_in_the_wrong_wire_type_are_not(tmp_path):
    # Field 99 of ModelProto, in every wire type, after the first field.
    unknown = (
        uint(99, 1)
        + varint(99 << 3 | 1) + b"\x00" * 8
        + message(99, b"abc")
        + varint(99 << 3 | 3) + uint(1, 5) + varint(99 << 3 | 4)
        + varint(99 << 3 | 5) + b"\x00" * 4
    )
    graph_bytes = model(graph())
    graph_path = package(tmp_path, graph_bytes[:2] + unknown + graph_bytes[2:])
    checksum.validate_onnx_package(graph_path)

    # ir_version (a varint) as a length-delimited value.
    graph_path.write_bytes(message(1, b"\x08") + graph_bytes[2:])
    assert "encoded as a length-delimited value" in refusal(graph_path)


def test_the_onnx_package_finds_no_external_file_the_walk_misses(tmp_path):
    """An independent parser, where one is installed: a model onnx itself saved with
    external data -- initializers and a Constant's tensor -- has exactly the files onnx
    says it references, and the checksum of them."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, external_data_helper, helper, numpy_helper

    np = pytest.importorskip("numpy")
    weight = numpy_helper.from_array(np.arange(64, dtype=np.float32).reshape(8, 8), "weight")
    bias = numpy_helper.from_array(np.ones(8, dtype=np.float32), "bias")
    constant = helper.make_node(
        "Constant",
        [],
        ["scale"],
        value=numpy_helper.from_array(np.full(8, 2.0, dtype=np.float32), "scale_value"),
    )
    nodes = [
        constant,
        helper.make_node("MatMul", ["x", "weight"], ["y"]),
        helper.make_node("Add", ["y", "bias"], ["z"]),
        helper.make_node("Mul", ["z", "scale"], ["out"]),
    ]
    g = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8])],
        initializer=[weight, bias],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    external_data_helper.convert_model_to_external_data(
        m, all_tensors_to_one_file=False, size_threshold=0, convert_attribute=True
    )
    onnx.save_model(m, str(tmp_path / "model.onnx"))
    (tmp_path / "tokenizer.json").write_text(GOLDEN_TOKENIZER, encoding="utf-8")

    reloaded = onnx.load(str(tmp_path / "model.onnx"), load_external_data=False)
    by_onnx = {
        entry.value
        for tensor in external_data_helper._get_all_tensors(reloaded)
        if external_data_helper.uses_external_data(tensor)
        for entry in tensor.external_data
        if entry.key == "location"
    }
    assert len(by_onnx) == 3, "two initializers and the Constant's tensor"

    validated = checksum.validate_onnx_package(tmp_path / "model.onnx")
    external = {f.relpath for f in validated.files} - {"model.onnx", "tokenizer.json"}
    assert external == by_onnx
    expected = [
        checksum.PackageFile(name, tmp_path / name, os.path.getsize(tmp_path / name), sha256((tmp_path / name).read_bytes()))
        for name in sorted(by_onnx | {"model.onnx", "tokenizer.json"})
    ]
    assert validated.checksum == checksum.package_checksum(expected)
