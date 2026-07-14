use std::path::PathBuf;
use std::time::SystemTime;
use libloading::Library;
use notify_debouncer_mini::{new_debouncer, Debouncer};
use notify_debouncer_mini::notify::RecommendedWatcher;

struct Module {
    /// Path to shared library file.
    path: PathBuf,
    /// Last modification time.
    last_modified: SystemTime,
    /// Handle to the loaded library.
    library: Option<Library>,

}

pub struct ModuleManager {
    modules: Vec<Module>,
    watcher: Debouncer<RecommendedWatcher>,
}

impl ModuleManager {
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            watcher: new_debouncer(std::time::Duration::from_secs(1), |_| {}).expect("Failed to create debouncer")
        }
    }

    pub fn insert_module(&mut self, path: PathBuf) {
        let last_modified = std::fs::metadata(&path)
            .expect("Failed to get metadata")
            .modified()
            .expect("Failed to get modification time");
        let module = Module { path, last_modified, library: None };
        self.modules.push(module);
    }


}