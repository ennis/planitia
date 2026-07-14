use gamelib::error::{ExcResult, OptionExt, ResultExt};
use libloading::Library;
use std::cell::{Cell, RefCell};
use std::env::temp_dir;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(thiserror::Error, Debug, Clone, Default)]
#[error("plugin load error")]
pub struct PluginLoadError;

#[derive(thiserror::Error, Debug, Clone, Default)]
#[error("plugin entry point procedure (`plugin_entry`) not found")]
pub struct PluginEntryPointNotFound;

#[derive(thiserror::Error, Debug, Clone, Default)]
#[error("failed to create temporary directory for plugin library")]
pub struct PluginTempDirCreationError;

pub type PluginEntryFn = unsafe extern "C" fn() -> ();

struct PluginLibrary {
    last_modified: SystemTime,
    tmpdir: tempfile::TempDir,
    library: Library,
    entry: unsafe extern "C" fn() -> (),
}

impl PluginLibrary {
    fn load(path: &Path) -> ExcResult<PluginLibrary, PluginLoadError> {
        let last_modified = std::fs::metadata(path)?.modified()?;
        let file_name = path.file_name().ok_or_raise_value(PluginLoadError)?;

        // Copy library to a temporary directory to avoid locking the original file, which would prevent recompilation.
        let tmpdir = tempfile::tempdir().raise(PluginTempDirCreationError).raise(PluginLoadError)?;
        let tmplib = tmpdir.path().join(file_name);

        debug!("copying plugin library from {} to {}", path.display(), tmplib.display());
        fs::copy(path, &tmplib)?;

        let library = unsafe { Library::new(&tmplib) }?;
        // Query entry point
        unsafe {
            // `get` returns a `Symbol` which is tied to the lifetime of `Library`,
            // but for function pointers we can clone the pointer without using unsafe,
            // so the usefulness of the lifetime in `Symbol` seems limited...
            // (https://github.com/nagisa/rust_libloading/issues/13)
            let entry = (*library
                .get::<PluginEntryFn>("plugin_entry")
                .raise(PluginEntryPointNotFound)
                .raise(PluginLoadError)?)
            .clone();
            Ok(PluginLibrary { last_modified, library, entry, tmpdir })
        }
    }

    fn load_or_log_error(path: &Path) -> Option<PluginLibrary> {
        match Self::load(path) {
            Ok(lib) => Some(lib),
            Err(err) => {
                err.log_error();
                None
            }
        }
    }
}

impl Drop for PluginLibrary {
    fn drop(&mut self) {
        debug!("unloading plugin library in directory: {}", self.tmpdir.path().display());
    }
}

struct Plugin {
    /// Path to shared library file.
    path: PathBuf,
    /// Handle to the loaded library.
    library: Option<PluginLibrary>,
}

thread_local! {
    static PLUGIN_HOST: &'static PluginHost = Box::leak(Box::new(PluginHost::new()));
}

pub struct PluginHost {
    plugins: RefCell<Vec<Plugin>>,
    last_reload_time: Cell<SystemTime>,
}

impl PluginHost {
    pub fn new() -> Self {
        Self { plugins: RefCell::new(Vec::new()), last_reload_time: Cell::new(SystemTime::UNIX_EPOCH) }
    }

    pub fn insert_module(&self, path: PathBuf) {
        debug!("loading plugin library: {}", path.display());
        let library = PluginLibrary::load_or_log_error(&path);
        self.plugins.borrow_mut().push(Plugin { path, library });
    }

    pub fn reload(&self) {
        let reload_start_time = SystemTime::now();

        for plugin in self.plugins.borrow_mut().iter_mut() {
            if let Some(ref lib) = plugin.library {
                let last_modified =
                    std::fs::metadata(&plugin.path).and_then(|meta| meta.modified()).unwrap_or(SystemTime::UNIX_EPOCH);

                if last_modified > lib.last_modified {
                    debug!(
                        "reloading plugin library: {} (last modified: {:?}, last reload: {:?})",
                        plugin.path.display(),
                        last_modified,
                        self.last_reload_time.get()
                    );

                    // Unload existing library first.
                    plugin.library = None;
                    plugin.library = PluginLibrary::load_or_log_error(&plugin.path);
                }
            }
        }

        self.last_reload_time.set(reload_start_time);
    }

    /// Sends an event to all loaded plugins.
    pub fn send_event(&self) {
        for plugin in self.plugins.borrow().iter() {
            if let Some(ref lib) = plugin.library {
                unsafe {
                    (lib.entry)();
                }
            }
        }
    }

    /// Returns a reference to the singleton instance of `PluginHost`.
    pub fn instance() -> &'static Self {
        // FIXME: we should check that this function is only called from the main thread.
        PLUGIN_HOST.with(|host| *host)
    }
}

pub fn load_test_plugin() {
    let plugin_host = PluginHost::instance();
    #[cfg(target_os = "windows")]
    {
        let exe_dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
        let path = exe_dir.join("hot_reload_test.dll");

        plugin_host.insert_module(path);
        plugin_host.send_event();
    }
}

pub fn reload_plugins() {
    let plugin_host = PluginHost::instance();
    plugin_host.reload();
    plugin_host.send_event();
}
