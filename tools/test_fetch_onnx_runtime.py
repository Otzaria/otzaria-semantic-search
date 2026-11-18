import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
import fetch_onnx_runtime as fetch


class RuntimeFetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "official.zip"
        with zipfile.ZipFile(self.bundle, "w") as archive:
            archive.writestr("onnxruntime/capi/onnxruntime.dll", b"runtime")
            archive.writestr("onnxruntime/capi/onnxruntime_providers_shared.dll", b"provider")
            archive.writestr("onnxruntime/capi/bindings.pyd", b"not needed")
            archive.writestr("../../escape.dll", b"do not extract")
        self.pins = {"1.28.0/windows-x64": dict(url="https://example.invalid/pinned.zip", sha256=hashlib.sha256(self.bundle.read_bytes()).hexdigest(), member="onnxruntime/capi/onnxruntime.dll")}
        self.out = self.root / "runtime"
        self.calls = 0

    def download(self, _url, destination):
        self.calls += 1
        destination.write_bytes(self.bundle.read_bytes())

    def run_fetch(self):
        return fetch.fetch_runtime("1.28.0", "windows-x64", self.out, pins=self.pins, downloader=self.download)

    def test_verified_native_only_extraction_and_cache(self):
        self.assertEqual(self.run_fetch().read_bytes(), b"runtime")
        self.assertTrue((self.out / "onnxruntime_providers_shared.dll").exists())
        self.assertFalse((self.out / "bindings.pyd").exists())
        self.assertFalse((self.root / "escape.dll").exists())
        self.run_fetch()
        self.assertEqual(self.calls, 1)

    def test_digest_mismatch_fails_before_extract(self):
        self.pins["1.28.0/windows-x64"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_fetch()
        self.assertFalse((self.out / "onnxruntime.dll").exists())
        self.assertFalse((self.out / "runtime.zip.partial").exists())

    def test_missing_main_library_fails(self):
        self.pins["1.28.0/windows-x64"]["member"] = "onnxruntime/capi/missing.dll"
        with self.assertRaisesRegex(ValueError, "has no"):
            self.run_fetch()

    def test_unknown_version_is_not_upgraded(self):
        with self.assertRaisesRegex(ValueError, "no pinned"):
            fetch.fetch_runtime("99", "windows-x64", self.out, pins=self.pins)

    def test_cli_failure_does_not_export_runtime(self):
        env = self.root / "github.env"
        with mock.patch.dict(fetch.os.environ, {"GITHUB_ENV": str(env)}):
            self.assertEqual(fetch.main(["--version", "99", "--platform", "windows-x64", "--out", str(self.out)]), 1)
        self.assertFalse(env.exists())


if __name__ == "__main__":
    unittest.main()
