use std::process::Command;

#[test]
fn test_cli_binary_execution() {
    // Determine target binary path
    let mut bin_path = std::env::current_exe().expect("failed to get current test exe path");
    bin_path.pop(); // remove test binary name
    if bin_path.ends_with("deps") {
        bin_path.pop(); // remove deps
    }
    bin_path.push("otzaria-semantic-search.exe");

    if !bin_path.exists() {
        // Fallback for non-windows / dev build naming
        bin_path.set_file_name("otzaria-semantic-search");
    }

    if !bin_path.exists() {
        // Skip if not built yet (cargo test runs unittests first)
        return;
    }

    // Run version command
    let output = Command::new(&bin_path)
        .arg("version")
        .output()
        .expect("Failed to execute CLI binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("otzaria-semantic-search CLI"));

    // Run status command
    let output_status = Command::new(&bin_path)
        .arg("status")
        .arg("--dir")
        .arg("./target/test_cli_db")
        .output()
        .expect("Failed to execute status command");

    assert!(output_status.status.success());
    let status_stdout = String::from_utf8_lossy(&output_status.stdout);
    assert!(status_stdout.contains("Otzaria Semantic Engine Status"));
}
