//! Global configuration settings.

use std::sync::{LazyLock, OnceLock, RwLock, RwLockReadGuard};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

const CONFIG_FILE: &str = "editor_config.ron";

/// Global configuration settings.
#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    /// Whether the imgui debug panel should be visible.
    pub show_imgui: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { show_imgui: true }
    }
}

impl Config {
    /// Loads the configuration from a file, or returns the default configuration if the file does not exist.
    pub fn load() -> Self {
        let ron_str = std::fs::read_to_string(CONFIG_FILE).unwrap_or_default();
        ron::from_str(&ron_str).unwrap_or_default()
    }

    /// Saves the configuration to a file.
    pub fn save(&self) {
        if let Ok(s) = ron::ser::to_string_pretty(self, PrettyConfig::default()) {
            let _ = std::fs::write(CONFIG_FILE, s);
        }
    }
}

static CONFIG: LazyLock<RwLock<Config>> = LazyLock::new(|| RwLock::new(Config::load()));

/// Returns a read-only reference to the global configuration.
pub fn config() -> RwLockReadGuard<'static, Config> {
    CONFIG.read().unwrap()
}

/// Returns a mutable reference to the global configuration.
pub fn config_mut() -> std::sync::RwLockWriteGuard<'static, Config> {
    CONFIG.write().unwrap()
}