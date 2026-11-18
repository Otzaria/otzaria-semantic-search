//! Audit pending passage token lengths without loading ONNX Runtime.
#[cfg(any(
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
))]
fn main() {
    use otzaria_semantic_search::distribution::plan::EmbedManifest;
    use otzaria_semantic_search::distribution::token_audit::audit_plan_tokens;
    use otzaria_semantic_search::semantic::onnx_backend::PassageTokenCounter;
    use std::path::Path;
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 && !(args.len() == 4 && args[3] == "--allow-truncation") {
        eprintln!("Usage: audit_plan_tokens PLAN_DIRECTORY TOKENIZER_JSON [--allow-truncation]");
        std::process::exit(1);
    }
    let run = || -> Result<_, Box<dyn std::error::Error>> {
        let dir = Path::new(&args[1]);
        let tokenizer = Path::new(&args[2]);
        let manifest = EmbedManifest::read(dir)?;
        let counter = PassageTokenCounter::open(tokenizer, manifest.model.max_tokens)?;
        Ok(audit_plan_tokens(dir, tokenizer, |text| {
            counter.count(text)
        })?)
    };
    match run() {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            if report.over_budget_passages > 0 && args.len() != 4 {
                std::process::exit(2);
            }
        }
        Err(error) => {
            eprintln!("Token audit failed: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(any(
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
)))]
fn main() {
    eprintln!("Token audits require a desktop ONNX-backend target.");
    std::process::exit(1);
}
