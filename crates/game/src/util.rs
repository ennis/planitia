use arboard::Clipboard;
use log::error;
use std::env;
use std::future::pending;
use std::path::{Path, PathBuf};

/// An async future that never completes.
pub async fn forever() {
    pending::<()>().await;
}

/// Returns the value of an environment variable as a boolean flag.
/// The variable is considered true if its value is "1", "true", or "yes".
pub fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|v| v == "1" || v == "true" || v == "yes")
        .unwrap_or(false)
}

/// Copies the given text to the system clipboard.
///
/// If this fails, an error message is logged.
/// TODO move this to platform module
pub fn copy_to_clipboard(text: &str) {
    match Clipboard::new() {
        Ok(mut clipboard) => {
            if let Err(err) = clipboard.set_text(text) {
                error!("Failed to set clipboard text: {}", err);
            }
        }
        Err(err) => {
            error!("Failed to access clipboard: {}", err);
        }
    }
}

/// Gets the current text from the system clipboard.
/// TODO move this to platform module
pub fn read_clipboard() -> String {
    match Clipboard::new() {
        Ok(mut clipboard) => clipboard.get_text().unwrap_or_else(|err| {
            error!("Failed to get clipboard text: {}", err);
            String::new()
        }),
        Err(err) => {
            error!("Failed to access clipboard: {}", err);
            String::new()
        }
    }
}

/// Checks if a file or directory at the given path exists, and returns another path with a numeric suffix if it does.
///
/// For example, if "file.txt" exists, it will return "file (1).txt", and so on.
pub fn make_unique_path(path: &Path) -> PathBuf {
    let orig_path = Path::new(path);
    // split <directory>/<file_stem>.<extension>
    let extension = orig_path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    let parent = orig_path.parent().unwrap_or(Path::new(""));
    let file_stem = orig_path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("");

    let mut unique_path = orig_path.to_owned();
    let mut suffix = 1;

    while Path::new(&unique_path).exists() {
        unique_path = Path::new(parent).join(if extension.is_empty() {
            format!("{file_stem}({suffix})")
        } else {
            format!("{file_stem}({suffix}).{extension}")
        });
        suffix += 1;
    }

    unique_path
}
