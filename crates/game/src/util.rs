use std::env;
use std::future::pending;
use arboard::Clipboard;
use log::error;

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