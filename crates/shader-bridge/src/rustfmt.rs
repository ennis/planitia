use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Gets the rustfmt path to rustfmt the generated bindings.
fn rustfmt_path<'a>() -> PathBuf {
    if let Ok(rustfmt) = env::var("RUSTFMT") {
        rustfmt.into()
    } else {
        // just assume that it is in path
        "rustfmt".into()
    }
}

/// Runs rustfmt on a file.
pub fn rustfmt_file(path: &Path) {
    let rustfmt = rustfmt_path();
    let mut cmd = Command::new(rustfmt);

    cmd.arg(path.as_os_str());
    match cmd.spawn() {
        Ok(mut child) => {
            if let Ok(exit_status) = child.wait() {
                if !exit_status.success() {
                    eprintln!("rustfmt failed (exit status = {exit_status})")
                }
            }
        }
        Err(err) => {
            eprintln!("failed to run rustfmt: {err}");
        }
    }
}
