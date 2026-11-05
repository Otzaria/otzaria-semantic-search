//! Emit comparable tokens, vectors, neighbours and Rust allocation counts.
//! Run this same source against both dependency revisions in separate processes.

#[cfg(all(
    feature = "onnx-backend",
    any(
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "linux",
            target_env = "gnu",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "windows",
            target_env = "msvc",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    )
))]
mod probe {
    use otzaria_semantic_search::semantic::backend::{EmbeddingBackend, Pooling};
    use otzaria_semantic_search::semantic::onnx_backend::{OnnxBackend, OnnxBackendConfig};
    use serde_json::json;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;

    pub struct CountingAllocator;
    static COUNTING: AtomicBool = AtomicBool::new(false);
    static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if COUNTING.load(Ordering::Relaxed) {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if COUNTING.load(Ordering::Relaxed) {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            if COUNTING.load(Ordering::Relaxed) {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(size as u64, Ordering::Relaxed);
            }
            unsafe { System.realloc(ptr, layout, size) }
        }
    }

    fn normalized(vector: &[f32]) -> Vec<f32> {
        let norm = vector
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(norm.is_finite() && norm > 1e-12);
        vector
            .iter()
            .map(|&v| (f64::from(v) / norm) as f32)
            .collect()
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args().skip(1);
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/onnx_fixture");
        let model = args
            .next()
            .map_or_else(|| fixture.join("dynamic.onnx"), PathBuf::from);
        let tokenizer = args
            .next()
            .map_or_else(|| fixture.join("tokenizer.json"), PathBuf::from);
        let max_tokens: usize = args.next().map_or(Ok(32), |s| s.parse())?;
        let iterations: usize = args.next().map_or(Ok(1000), |s| s.parse())?;
        if args.next().is_some() || iterations == 0 {
            return Err(
                "usage: ort_parity [model tokenizer max_tokens iterations]; iterations > 0".into(),
            );
        }
        let runtime = std::env::var_os("OTZARIA_ONNX_RUNTIME")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .ok_or("OTZARIA_ONNX_RUNTIME must name a real shared library")?;
        let tuning = OnnxBackendConfig {
            intra_threads: 1,
            sessions: 1,
        };
        let started = Instant::now();
        let backend = OnnxBackend::open(
            &model,
            &tokenizer,
            max_tokens,
            Pooling::InGraph,
            &tuning,
            Some(&runtime),
        )?;
        let load_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut texts: Vec<String> = [
            "the quick brown fox",
            "the lazy dog",
            "the fox jumps over the dog",
            "quick brown",
            "שלום ספר",
            "ספר חדש",
            "שלום עולם",
            "",
            "123 456",
            "[QUERY] the quick fox",
            "[PASSAGE] the lazy dog",
            "the\nquick\tbrown",
            "the quick brown fox!",
            "שלום\u{200f} ספר",
            "fox dog fox",
            "dog fox dog",
            "the the the",
            "quick lazy quick",
            "brown dog",
            "jumps over",
            "ספר ספר",
            "א ב ג",
            "hello world",
            "unknown-token-42",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        texts.push("the quick brown fox ".repeat(max_tokens));
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let tokens: Vec<Vec<u32>> = refs
            .iter()
            .map(|text| backend.tokenize(text))
            .collect::<Result<_, _>>()?;
        let raw = backend.embed_batch_raw(&refs)?;
        let vectors: Vec<Vec<f32>> = raw.iter().map(|v| normalized(v)).collect();
        let neighbours: Vec<Vec<usize>> = vectors
            .iter()
            .enumerate()
            .map(|(own, query)| {
                let mut scored: Vec<(usize, f64)> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != own)
                    .map(|(i, vector)| {
                        (
                            i,
                            query
                                .iter()
                                .zip(vector)
                                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                .sum(),
                        )
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                scored.iter().take(5).map(|(i, _)| *i).collect()
            })
            .collect();
        for _ in 0..20 {
            backend.embed_batch_raw(&[refs[0]])?;
        }
        let mut durations = Vec::with_capacity(iterations);
        ALLOCATIONS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        for index in 0..iterations {
            let started = Instant::now();
            COUNTING.store(true, Ordering::Relaxed);
            let result = backend.embed_batch_raw(&[refs[index % refs.len()]]);
            COUNTING.store(false, Ordering::Relaxed);
            let result = result?;
            durations.push(started.elapsed().as_secs_f64() * 1_000_000.0);
            std::hint::black_box(result);
        }
        durations.sort_by(f64::total_cmp);
        let quantile = |p: f64| {
            durations[((iterations as f64 * p).ceil() as usize)
                .saturating_sub(1)
                .min(iterations - 1)]
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "schema": 1, "backend": backend.id(), "max_tokens": max_tokens,
                "texts": texts, "tokens": tokens, "raw_vectors": raw, "normalized_vectors": vectors,
                "neighbours_at_5": neighbours,
                "measurement": {"iterations": iterations, "load_ms": load_ms,
                    "p50_us": quantile(0.5), "p95_us": quantile(0.95), "p99_us": quantile(0.99),
                    "rust_allocations": ALLOCATIONS.load(Ordering::Relaxed),
                    "rust_requested_bytes": BYTES.load(Ordering::Relaxed),
                    "native_allocations_measured": false, "threads": 1, "sessions": 1}
            }))?
        );
        Ok(())
    }
}

#[cfg(all(
    feature = "onnx-backend",
    any(
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "linux",
            target_env = "gnu",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "windows",
            target_env = "msvc",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    )
))]
#[global_allocator]
static ALLOCATOR: probe::CountingAllocator = probe::CountingAllocator;

#[cfg(all(
    feature = "onnx-backend",
    any(
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "linux",
            target_env = "gnu",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "windows",
            target_env = "msvc",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    )
))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    probe::run()
}

#[cfg(not(all(
    feature = "onnx-backend",
    any(
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "linux",
            target_env = "gnu",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "windows",
            target_env = "msvc",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    )
)))]
fn main() {
    eprintln!("ort_parity requires --features onnx-backend on a supported desktop target");
    std::process::exit(2);
}
