# Vendored llama-cpp-2 0.1.153 (local patch)

Verbatim copy of `llama-cpp-2 0.1.153` from crates.io, wired in as a path
dependency from this crate's `Cargo.toml`.

Upstream: <https://github.com/utilityai/llama-cpp-rs>, MIT OR Apache-2.0.
The exact source commit is recorded in `.cargo_vcs_info.json`.

**The single functional change** is in `Cargo.toml` — both `llama-cpp-sys-2`
dependency declarations gain:

```toml
default-features = false   # upstream: omitted, so sys's default = ["common"] applies
```

Why: `llama-cpp-sys-2`'s `common` feature sets `LLAMA_BUILD_COMMON=ON`, which
makes llama.cpp build `common/` (chat templates, jinja, `download.cpp`,
`hf-cache.cpp`) plus its private `vendor/cpp-httplib`. Upstream's build script
scans only the CMake *install* dir for libraries to link; `cpp-httplib` is
never installed there, so no `cargo:rustc-link-lib` is emitted for it. The
resulting `staticlib` therefore carries `download.cpp.o` with unresolved
`httplib::*` symbols — and Cargokit's `-force_load` on Apple pulls that object
in, failing the app link.

None of `common` is reachable from here: this crate already sets
`default-features = false` on `llama-cpp-2`, so `llama-cpp-2`'s own `common`
feature — which gates every Rust binding to that C++ code — is off. Only the
sys-level dependency declaration leaked it back on. Turning it off restores the
intended configuration and drops ~13MB of unreachable C++ from the binary.

Re-check this patch when the pinned `llama-cpp-2` version moves.
