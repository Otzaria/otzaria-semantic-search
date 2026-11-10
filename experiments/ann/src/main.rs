//! Experimental comparison only. Nothing in this executable is linked by the app.
mod data;
mod lifecycle;
mod measure;
mod native_checks;

use data::Config;
use otzaria_semantic_search::distribution::package::PackageKind;
use otzaria_semantic_search::semantic::oxv::codec::Codec;
use otzaria_semantic_search::semantic::oxv::reader::Segment;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("ann experiment: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args
        .next()
        .ok_or("expected build, measure, lifecycle, or threads")?;
    let mut options = BTreeMap::new();
    while let Some(key) = args.next() {
        if !key.starts_with("--") {
            return Err(format!("unexpected argument: {key}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if options.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option: {key}").into());
        }
    }
    let dir = PathBuf::from(options.remove("--dir").unwrap_or_default());
    if command != "threads" && dir.as_os_str().is_empty() {
        return Err("--dir is required".into());
    }
    let result = match command.as_str() {
        "build" => {
            let mut config = Config::default();
            config.vectors = number(&mut options, "--vectors", config.vectors)?;
            config.dim = number(&mut options, "--dim", config.dim)?;
            config.queries = number(&mut options, "--queries", config.queries)?;
            config.top_k = number(&mut options, "--top-k", config.top_k)?;
            config.connectivity = number(&mut options, "--connectivity", config.connectivity)?;
            config.expansion_add = number(&mut options, "--expansion-add", config.expansion_add)?;
            if let Some(codec) = options.remove("--codec") {
                config.codec = codec;
            }
            if let Some(distribution) = options.remove("--distribution") {
                config.distribution = distribution;
            }
            no_options(&options)?;
            config.validate()?;
            build(&dir, &config)?
        }
        "measure" => {
            let backend = options.remove("--backend").ok_or("--backend is required")?;
            let filter = options
                .remove("--filter")
                .unwrap_or_else(|| "all".to_string());
            let threads = number(&mut options, "--threads", 1)?;
            let expansion = number(&mut options, "--expansion", 128)?;
            let repeats = number(&mut options, "--repeats", 3)?;
            no_options(&options)?;
            if threads == 0 || expansion == 0 || repeats == 0 {
                return Err("threads, expansion, repeats must be positive".into());
            }
            measure::measure(&dir, &backend, &filter, threads, expansion, repeats)?
        }
        "lifecycle" => {
            no_options(&options)?;
            lifecycle::run(&dir)?
        }
        "threads" => {
            let samples = number(&mut options, "--samples", 100)?;
            if samples == 0 {
                return Err("samples must be positive".into());
            }
            no_options(&options)?;
            measure::thread_probe(samples)
        }
        _ => return Err(format!("unknown command: {command}").into()),
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

fn number(options: &mut BTreeMap<String, String>, key: &str, default: usize) -> Result<usize> {
    Ok(match options.remove(key) {
        Some(value) => value.parse()?,
        None => default,
    })
}

fn no_options(options: &BTreeMap<String, String>) -> Result<()> {
    if !options.is_empty() {
        return Err(format!("unknown options: {:?}", options.keys()).into());
    }
    Ok(())
}

fn build(dir: &Path, config: &Config) -> Result<serde_json::Value> {
    // Never remove or overwrite a supplied directory, especially a production set.
    std::fs::create_dir(dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(config)?)?;
    let started = Instant::now();
    let codec = config.fit_codec()?;
    let calibration_ms = started.elapsed().as_secs_f64() * 1000.0;
    let keys: Vec<_> = (1..=config.vectors as u64).collect();
    let started = Instant::now();
    let written = data::write_segment(
        config,
        &dir.join("encoded.oxv"),
        &codec,
        &keys,
        &[],
        PackageKind::Base,
        None,
    )?;
    data::install(
        config,
        &dir.join("lifecycle-set"),
        &dir.join("encoded.oxv"),
        &written,
        &codec,
    )?;
    let oxv_build_ms = started.elapsed().as_secs_f64() * 1000.0;
    let f32_codec = Codec::f32(config.dim)?;
    data::write_segment(
        config,
        &dir.join("raw.oxv"),
        &f32_codec,
        &keys,
        &[],
        PackageKind::Base,
        None,
    )?;
    data::write_segment(
        config,
        &dir.join("decoded.oxv"),
        &f32_codec,
        &keys,
        &[],
        PackageKind::Base,
        Some(&codec),
    )?;
    let source = Segment::open(&dir.join("encoded.oxv"))?;
    let started = Instant::now();
    let ann = config.ann(128)?;
    // Explicitly bound both build and search scratch contexts. All operations in
    // this tool are serial; USearch need not reserve one per host CPU.
    ann.reserve_capacity_and_threads(config.vectors, 1)?;
    let mut decoded = vec![0f32; config.dim];
    for slot in 0..source.slot_count() {
        codec.decode(source.vector(slot), source.vector_scale(slot), &mut decoded);
        // add copies its buffer. No buffer view or custom metric pointer survives this call.
        ann.add(data::key_id(source.key(slot)), &decoded)?;
    }
    let ann_build_ms = started.elapsed().as_secs_f64() * 1000.0;
    ann.save(path_str(&dir.join("index.usearch"))?)?;
    let result = json!({
        "schema": 1, "config": config, "calibration_ms": calibration_ms,
        "oxv_build_and_install_ms": oxv_build_ms, "ann_build_ms": ann_build_ms,
        "ann_memory_usage_bytes": ann.memory_usage(),
        "ann_hardware_acceleration": ann.hardware_acceleration(),
        "usearch_version": "2.26.6", "usearch_features": [],
        "oxv_bytes": source.size(),
        "raw_bytes": std::fs::metadata(dir.join("raw.oxv"))?.len(),
        "decoded_bytes": std::fs::metadata(dir.join("decoded.oxv"))?.len(),
        "ann_bytes": std::fs::metadata(dir.join("index.usearch"))?.len(),
        "memory": measure::memory(),
    });
    std::fs::write(dir.join("build.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| "USearch experiment requires a UTF-8 path".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use otzaria_semantic_search::semantic::segment_set::SegmentSet;

    struct Fixture(PathBuf);
    impl Fixture {
        fn build(config: &Config) -> Result<Self> {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let fixture = Self(
                std::env::temp_dir()
                    .join(format!("otzaria-ann-test-{}-{unique}", std::process::id())),
            );
            config.validate()?;
            build(&fixture.0, config)?;
            Ok(fixture)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            // Only the unique directory this test created, never a CLI path.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn real_oxv_and_native_ann_roundtrip_filters_delta_compaction_and_cancel() -> Result<()> {
        check_roundtrip(Config {
            vectors: 200,
            dim: 32,
            queries: 4,
            top_k: 10,
            ..Config::default()
        })
    }

    #[test]
    fn fitted_rotated_codec_roundtrip_filters_delta_compaction_and_cancel() -> Result<()> {
        check_roundtrip(Config {
            vectors: 200,
            dim: 256,
            queries: 4,
            top_k: 10,
            codec: "rot192-i8i4".to_string(),
            ..Config::default()
        })
    }

    #[test]
    fn tied_f32_scores_survive_delta_compaction_and_restore() -> Result<()> {
        check_roundtrip(Config {
            vectors: 200,
            dim: 1,
            queries: 4,
            top_k: 10,
            codec: "f32".to_string(),
            ..Config::default()
        })
    }

    #[test]
    fn native_contents_detects_corruption_with_unchanged_size_and_membership() -> Result<()> {
        let config = Config {
            vectors: 200,
            dim: 32,
            queries: 4,
            codec: "f32".to_string(),
            ..Config::default()
        };
        let fixture = Fixture::build(&config)?;
        let source = SegmentSet::open(&fixture.0.join("lifecycle-set"))?;
        let ann = config.ann(128)?;
        ann.load(path_str(&fixture.0.join("index.usearch"))?)?;
        native_checks::verify_contents(&ann, &source)?;
        let mut first = vec![0f32; config.dim];
        let mut second = vec![0f32; config.dim];
        assert_eq!(ann.get(1, &mut first)?, 1);
        assert_eq!(ann.get(2, &mut second)?, 1);
        assert_ne!(first, second);

        // A one-bit payload change must fail even when every ID is still present.
        let mut changed = first.clone();
        changed[0] = f32::from_bits(changed[0].to_bits() ^ 1);
        ann.remove(1)?;
        ann.add(1, &changed)?;
        assert_eq!(ann.size(), config.vectors);
        assert!((1..=config.vectors as u64).all(|key| ann.contains(key)));
        let error = native_checks::verify_contents(&ann, &source).unwrap_err();
        assert!(error.to_string().contains("vector content"));

        // Swapping two intact vectors also preserves size and membership.
        ann.remove(1)?;
        ann.remove(2)?;
        ann.add(1, &second)?;
        ann.add(2, &first)?;
        assert!(native_checks::verify_contents(&ann, &source).is_err());
        ann.remove(1)?;
        ann.remove(2)?;
        ann.add(1, &first)?;
        ann.add(2, &second)?;
        native_checks::verify_contents(&ann, &source)?;

        ann.remove(1)?;
        assert!(native_checks::verify_contents(&ann, &source).is_err());
        ann.add(config.vectors as u64 + 1, &first)?;
        assert_eq!(ann.size(), config.vectors);
        assert!(native_checks::verify_contents(&ann, &source).is_err());
        ann.remove(config.vectors as u64 + 1)?;
        ann.add(1, &first)?;
        ann.change_metric_kind(usearch::MetricKind::Cos)?;
        assert!(native_checks::verify_contents(&ann, &source).is_err());
        ann.change_metric_kind(usearch::MetricKind::IP)?;
        native_checks::verify_contents(&ann, &source)?;
        Ok(())
    }

    #[test]
    fn native_contents_detects_resurrected_tombstones() -> Result<()> {
        let config = Config {
            vectors: 200,
            dim: 1,
            queries: 4,
            codec: "f32".to_string(),
            ..Config::default()
        };
        let fixture = Fixture::build(&config)?;
        lifecycle::run(&fixture.0)?;
        let source = SegmentSet::open(&fixture.0.join("lifecycle-set"))?;
        let ann = config.ann(128)?;
        ann.load(path_str(&fixture.0.join("lifecycle-compacted.usearch"))?)?;
        native_checks::verify_contents(&ann, &source)?;
        assert!(!ann.contains(50));
        let mut vector = vec![0f32; config.dim];
        assert_eq!(ann.get(1, &mut vector)?, 1);
        ann.remove(1)?;
        ann.add(50, &vector)?;
        assert_eq!(ann.size() as u64, source.info().slots_live);
        assert!(native_checks::verify_contents(&ann, &source).is_err());
        Ok(())
    }

    fn check_roundtrip(config: Config) -> Result<()> {
        let fixture = Fixture::build(&config)?;
        for filter in ["all", "5pct", "1pct", "empty"] {
            let exact = measure::measure(&fixture.0, "oxv", filter, 1, 128, 1)?;
            let parallel = measure::measure(&fixture.0, "oxv", filter, 8, 128, 1)?;
            assert_eq!(exact["queries"], parallel["queries"]);
            let loaded = measure::measure(&fixture.0, "usearch-load", filter, 1, 512, 1)?;
            let viewed = measure::measure(&fixture.0, "usearch-view", filter, 1, 512, 1)?;
            assert_eq!(loaded["queries"], viewed["queries"]);
            let expected_count = match filter {
                "all" | "5pct" => 10,
                "1pct" => 2,
                _ => 0,
            };
            assert_eq!(
                exact["queries"][0]["keys"].as_array().unwrap().len(),
                expected_count
            );
            assert_eq!(
                loaded["queries"][0]["keys"].as_array().unwrap().len(),
                expected_count
            );
        }
        let lifecycle = lifecycle::run(&fixture.0)?;
        assert_eq!(lifecycle["status"], "passed");
        assert_eq!(lifecycle["live_after"], 208);
        assert_eq!(lifecycle["native_ann_abort_supported"], false);
        assert_eq!(lifecycle["ann_all_live_vectors_match_oxv"], true);
        assert!(
            lifecycle::run(&fixture.0).is_err(),
            "a second lifecycle must not overwrite its own generation history"
        );
        assert!(
            build(&fixture.0, &config).is_err(),
            "build must refuse an existing directory"
        );
        Ok(())
    }
}
