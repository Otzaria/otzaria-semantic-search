#!/usr/bin/env python3
"""Fetch a pinned official native runtime without installing Python bindings.

The reference version remains 1.28.0 and the minimum remains 1.17.0. GitHub
release assets returned 404; Microsoft's PyPI wheels and NuGet minimum package
provide the same runtime versions through immutable, checksum-pinned artifacts.
Only native libraries beside the exact selected member are extracted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

PINS = Path(__file__).with_name("onnx_runtime_artifacts.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    subprocess.run([
        "curl.exe" if os.name == "nt" else "curl", "--fail", "--location",
        "--retry", "3", "--retry-all-errors", "--proto", "=https",
        "--output", str(destination), url,
    ], check=True)


def fetch_runtime(version: str, platform: str, out: Path, *, pins=None,
                  downloader=download) -> Path:
    pins = json.loads(PINS.read_text(encoding="utf-8")) if pins is None else pins
    key = f"{version}/{platform}"
    if key not in pins:
        raise ValueError(f"no pinned official runtime for {key}")
    pin = pins[key]
    member = PurePosixPath(pin["member"])
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("runtime member must be a relative archive path")
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "runtime.zip"
    partial = out / "runtime.zip.partial"
    if not archive.exists() or sha256(archive) != pin["sha256"]:
        try:
            downloader(pin["url"], partial)
            actual = sha256(partial)
            if actual != pin["sha256"]:
                raise ValueError(f"runtime SHA-256 {actual}, expected {pin['sha256']}")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as package:
        # Check the selected main library before extracting anything.
        if str(member) not in package.namelist():
            raise ValueError(f"runtime archive has no {member}")
        native = []
        for entry in package.infolist():
            name = PurePosixPath(entry.filename)
            if entry.is_dir() or name.parent != member.parent:
                continue
            library = name.name
            if (library.endswith(".dll") or library.startswith("libonnxruntime")
                    and (".so" in library or library.endswith(".dylib"))):
                native.append((entry, library))
        for entry, library in native:
            target = out / library
            with package.open(entry) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    library = (out / member.name).resolve()
    if not library.is_file():
        raise ValueError(f"runtime library missing after extraction: {library}")
    return library


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1.28.0")
    parser.add_argument("--platform", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        library = fetch_runtime(args.version, args.platform, args.out)
        value = str(library)
        if "\n" in value or "\r" in value:
            raise ValueError("runtime output path cannot contain line breaks")
        if os.environ.get("GITHUB_ENV"):
            with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as env:
                env.write(f"OTZARIA_ONNX_RUNTIME={value}\n")
        print(value)
        return 0
    except (ValueError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(f"Runtime fetch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
