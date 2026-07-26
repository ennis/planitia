use crate::error::{ExcResult, OptionExt, ResultExt};
use crate::{watch_file, AppHandler, InputEvent};
use libloading::Library;
use serde::Serialize;
use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::env::temp_dir;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::SystemTime;
use std::{fs, mem, ptr};

#[derive(thiserror::Error, Debug, Clone, Default)]
#[error("plugin load error")]
pub enum PluginLoadError {
    #[error("plugin entry point procedure (`plugin_entry`) not found")]
    EntryPointNotFound,
    #[error("failed to create temporary directory for plugin library")]
    TempDirCreationError,
    #[error("other error")]
    #[default]
    Other,
}

/// Saved plugin data.
#[derive(Clone, Default)]
struct PluginData {
    data: String,
}

/// Context passed to plugin callbacks.
#[derive(Default)]
pub struct PluginCtx {
    user_ptr: Option<NonNull<()>>,
    data: PluginData,
}

impl PluginCtx {
    pub fn set_user_ptr(&mut self, ptr: Option<NonNull<()>>) -> Option<NonNull<()>> {
        let prev_ptr = self.user_ptr;
        self.user_ptr = ptr;
        prev_ptr
    }

    pub fn get_user_ptr(&self) -> Option<NonNull<()>> {
        self.user_ptr
    }

    pub fn save<T: serde::Serialize>(&mut self, s: &T) {
        self.data.data = ron::to_string(s).unwrap();
    }

    pub fn load<T: serde::de::DeserializeOwned>(&self, out: &mut T) {
        match ron::de::from_reader(self.data.data.as_bytes()) {
            Ok(value) => *out = value,
            Err(err) => {
                eprintln!("Failed to load plugin data: {}", err);
            }
        }
    }
}

/// Events sent to a plugin.
#[repr(C)]
#[derive(Copy,Clone)]
pub enum PluginEvent<'a> {
    /// VSync
    VSync,
    /// The plugin was just loaded.
    Init,
    /// The plugin is about to be unloaded.
    Deinit,
    /// A new frame should be rendered.
    Render(&'a gpu::Image),
    /// An input event occurred.
    Input(&'a InputEvent),
}

/// Result of the plugin event handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum PluginResult {
    /// Plugin should be called back on next vsync.
    WaitVSync,
    ///
    WaitInput,
}

pub type PluginEntryFn = for<'a> unsafe extern "C" fn(ctx: &mut PluginCtx, event: &PluginEvent) -> PluginResult;


/// Represents a loaded plugin library.
struct PluginLibrary {
    last_modified: SystemTime,
    tmpdir: tempfile::TempDir,
    library: Library,
    entry: PluginEntryFn,
}

impl PluginLibrary {
    fn load(path: &Path) -> ExcResult<PluginLibrary, PluginLoadError> {
        let last_modified = fs::metadata(path)?.modified()?;
        let file_name = path.file_name().ok_or_raise_value(PluginLoadError::Other)?;

        // Copy library to a temporary directory to avoid locking the original file, which would prevent recompilation.
        let tmpdir = tempfile::tempdir().raise(PluginLoadError::TempDirCreationError)?;
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
            let entry =
                (*library.get::<PluginEntryFn>("plugin_entry").raise(PluginLoadError::EntryPointNotFound)?).clone();

            // Initialize the plugin.
            //let mut ctx = PluginCtx { data: PluginData::default(), user_ptr: None };
            //let _result = entry(&mut ctx, &PluginEvent::Init);
            Ok(PluginLibrary { last_modified, library, entry, tmpdir })
        }
    }

    fn load_or_log_error(path: &Path) -> Option<PluginLibrary> {
        match Self::load(path) {
            Ok(lib) => Some(lib),
            Err(err) => {
                let err = err.with_info(format!("failed to load plugin library: {}", path.display()));
                err.log_to_stderr();
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

pub struct Plugin {
    /// Path to shared library file.
    path: PathBuf,
    /// Canonical path to the shared library file.
    canonical_path: PathBuf,
    /// Handle to the loaded library.
    library: Option<PluginLibrary>,
    ctx: PluginCtx,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    Registered,
    Loaded,
}

/// Manages loading and reloading of hot-reloadable plugin libraries.
pub struct PluginHost {
    plugins: RefCell<Vec<Plugin>>,
    last_reload_time: Cell<SystemTime>,
    current_plugin: Cell<Option<usize>>,
}

impl PluginHost {
    /// Creates a new `PluginHost` instance.
    pub fn new() -> Self {
        Self {
            plugins: RefCell::new(Vec::new()),
            last_reload_time: Cell::new(SystemTime::UNIX_EPOCH),
            current_plugin: Cell::new(None),
        }
    }

    /// Returns whether [register](PluginHost::register) has already been called with the given path,
    /// and if so, whether the plugin has been loaded successfully.
    ///
    /// # Return value
    /// * `None` if the plugin has not been inserted yet.
    /// * `Some(PluginState::Registered)` if the plugin has been inserted but not loaded yet.
    /// * `Some(PluginState::Loaded)` if the plugin has been inserted and loaded successfully.
    pub fn plugin_state(&self, path: &Path) -> Option<PluginState> {
        let Some(canonical_path) = fs::canonicalize(path).ok() else { return None };
        self.plugins.borrow().iter().find_map(|plugin| {
            if plugin.canonical_path == canonical_path {
                if plugin.library.is_some() { Some(PluginState::Loaded) } else { Some(PluginState::Registered) }
            } else {
                None
            }
        })
    }

    fn init_plugin(&self, plugin: &mut Plugin) {
        if let Some(ref library) = plugin.library {
            unsafe {
                (library.entry)(&mut plugin.ctx, &PluginEvent::Init);
            }
        }
    }

    fn deinit_plugin(&self, plugin: &mut Plugin) {
        if let Some(ref library) = plugin.library {
            unsafe {
                (library.entry)(&mut plugin.ctx, &PluginEvent::Deinit);
            }
        }
        plugin.library = None;
    }

    /// Iterates over available interfaces of the specified type.
    // Issue:
    //pub fn interfaces<Interface: ?Sized + Any>(&self)

    /// Registers a plugin library.
    ///
    /// This will try to load the library immediately if the file exists.
    pub fn register(&self, path: PathBuf) -> PluginState {
        match self.plugin_state(&path) {
            Some(state @ PluginState::Registered | state @ PluginState::Loaded) => {
                debug!("plugin library `{}` is already inserted", path.display());
                return state;
            }
            None => {}
        }

        match fs::exists(&path) {
            Ok(true) => {
                debug!("loading plugin library: {}", path.display());
                let _ = watch_file(&path);
                let library = PluginLibrary::load_or_log_error(&path);
                // Not sure what could cause canonicalize to fail here (the path exists), so unwrap.
                let canonical_path = fs::canonicalize(&path).unwrap();

                let mut plugin = Plugin { path, library, canonical_path, ctx: Default::default() };
                self.init_plugin(&mut plugin);
                self.plugins.borrow_mut().push(plugin);
                PluginState::Loaded
            }
            _ => {
                // Add a file watch on the parent directory if the library file isn't there yet, so that we can reload it when it is created.
                debug!("plugin library `{}` not found", path.display());
                if let Some(parent) = path.parent()
                    && fs::exists(parent).unwrap_or(false)
                {
                    let _ = watch_file(parent);
                }
                self.plugins.borrow_mut().push(Plugin {
                    path,
                    library: None,
                    canonical_path: PathBuf::new(),
                    ctx: Default::default(),
                });
                PluginState::Registered
            }
        }
    }

    fn path_is_newer(path: &Path, last_modified: SystemTime) -> bool {
        let new_last_modified = fs::metadata(&path).and_then(|meta| meta.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
        new_last_modified > last_modified
    }

    /// Reloads plugins whose shared library files have been modified since the last reload.
    pub fn reload(&self) {
        let reload_start_time = SystemTime::now();

        for plugin in self.plugins.borrow_mut().iter_mut() {
            let Some(exists) = fs::exists(&plugin.path).ok() else { continue };
            if !exists {
                // Library file doesn't exist; do nothing. We might be in the middle of
                // recompilation.
                continue;
            }

            if let Some(ref lib) = plugin.library {
                if Self::path_is_newer(&plugin.path, lib.last_modified) {
                    debug!("plugin library `{}` is new", plugin.path.display());

                    self.deinit_plugin(plugin);
                    plugin.library = PluginLibrary::load_or_log_error(&plugin.path);
                    // Update canonical path in case the file was replaced with a different file.
                    plugin.canonical_path = fs::canonicalize(&plugin.path).unwrap();
                    self.init_plugin(plugin);
                }
            } else {
                // Library is not loaded yet. Try to load it.
                plugin.library = PluginLibrary::load_or_log_error(&plugin.path);
                plugin.canonical_path = fs::canonicalize(&plugin.path).unwrap();
            }
        }

        self.last_reload_time.set(reload_start_time);
    }

    /// Sends an event to all loaded plugins.
    pub fn send_event(&self, event: PluginEvent) {
        for (_i, plugin) in self.plugins.borrow_mut().iter_mut().enumerate() {
            if let Some(ref mut lib) = plugin.library {
                unsafe {
                    (lib.entry)(&mut plugin.ctx, &event);
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

thread_local! {
    static PLUGIN_HOST: &'static PluginHost = Box::leak(Box::new(PluginHost::new()));
}

/// Registers a dynamically-loaded plugin library.
pub fn register_plugin<P: AsRef<Path>>(path: P) -> PluginState {
    let plugin_host = PluginHost::instance();
    #[cfg(target_os = "windows")]
    {
        let exe_dir = std::env::current_exe()
            .expect("failed to get current executable path")
            .parent()
            .expect("unexpected executable path")
            .to_path_buf();
        let path = exe_dir.join(path);
        plugin_host.register(path)
    }
}

/// Reloads all hot-reloadable plugins.
pub fn reload_plugins() {
    let plugin_host = PluginHost::instance();
    plugin_host.reload();
}

//--------------------------------------------------------------------------------------------------

pub trait PluginInterface {
    fn event(&mut self, event: &PluginEvent) -> PluginResult;
}

impl<T> PluginInterface for T where T: AppHandler {
    fn event(&mut self, event: &PluginEvent) -> PluginResult {
        match event {
            PluginEvent::VSync => {
                self.vsync();
            }
            PluginEvent::Render(_) => {
                self.render()
            }
            PluginEvent::Init => {}
            PluginEvent::Deinit => {}
        }
    }
}

#[macro_export]
macro_rules! register_plugin {
    ($init_fn:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn plugin_entry(
            ctx: &mut $crate::PluginCtx,
            event: &$crate::PluginEvent,
        ) -> $crate::PluginResult {

            fn cast<T>(ptr: ::std::ptr::NonNull<()>, _fn: fn() -> T) -> ::std::ptr::NonNull<T> {
                ::std::ptr::NonNull::new(ptr.as_ptr() as *mut T).unwrap()
            }

            // On initialization, allocate the plugin object.
            match event {
                $crate::PluginEvent::Init => {
                    ctx.set_user_ptr(Some(
                        ::std::ptr::NonNull::new(Box::into_raw(Box::new($init_fn())) as *mut ()).unwrap(),
                    ));
                }
                _ => {}
            }


            // Handle event.
            let result = if let Some(ptr) = ctx.get_user_ptr() {
                let t = cast(ptr, $init_fn);
                let t = unsafe { &mut *t.as_ptr() };
                match event {
                    $crate::PluginEvent::Init => {
                        ctx.load(t);
                        t.event(event)
                    }
                    $crate::PluginEvent::Deinit => {
                        let result = t.event(event);
                        ctx.save(t);
                        result
                    }
                    other => {
                        t.event(other)
                    }
                }
            } else {
                $crate::PluginResult::WaitVSync
            };

            match event {
                $crate::PluginEvent::Deinit => {
                    let prev = ctx.set_user_ptr(None);
                    if let Some(ptr) = prev {
                        unsafe {
                            let _ = Box::from_raw(ptr.as_ptr() as *mut _);
                       }
                    }
                }
                _ => {}
            }

            result
        }
    };
}
