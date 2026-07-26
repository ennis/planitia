use crate::error::{ExcResult, OptionExt, ResultExt};
use crate::platform::RenderTargetImage;
use crate::{AppHandler, InputEvent, UserEvent, WindowHandle, watch_file};
use egui::Context;
use libloading::Library;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
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

/// Result of the plugin event handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum PluginResult {
    /// Plugin should be called back on next vsync.
    WaitVSync,
    ///
    WaitInput,
}

pub type PluginInitFn = for<'a> unsafe extern "C" fn(ctx: &mut PluginCtx);
pub type PluginShutdownFn = for<'a> unsafe extern "C" fn(ctx: &mut PluginCtx);
pub type PluginEntryFn = for<'a> unsafe extern "C" fn(ctx: &mut PluginCtx, event: Box<PluginEvent>) -> PluginResult;

/// Represents a loaded plugin library.
struct PluginLibrary {
    last_modified: SystemTime,
    tmpdir: tempfile::TempDir,
    library: Library,
    entry: PluginEntryFn,
    init: PluginInitFn,
    shutdown: PluginShutdownFn,
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
            let shutdown =
                (*library.get::<PluginShutdownFn>("plugin_shutdown").raise(PluginLoadError::EntryPointNotFound)?)
                    .clone();
            let init =
                (*library.get::<PluginInitFn>("plugin_init").raise(PluginLoadError::EntryPointNotFound)?).clone();

            // Initialize the plugin.
            //let mut ctx = PluginCtx { data: PluginData::default(), user_ptr: None };
            //let _result = entry(&mut ctx, &PluginEvent::Init);
            Ok(PluginLibrary { last_modified, library, entry, tmpdir, init, shutdown })
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

pub fn get_plugin_library_path<P: AsRef<Path>>(path: P) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let exe_dir = std::env::current_exe()
            .expect("failed to get current executable path")
            .parent()
            .expect("unexpected executable path")
            .to_path_buf();
        exe_dir.join(path)
    }
    #[cfg(not(target_os = "windows"))]
    {
        path.as_ref().to_path_buf()
    }
}

/// Manages loading and reloading of hot-reloadable plugin libraries.
pub struct PluginHost {
    /// Path to shared library file.
    path: PathBuf,
    /// Canonical path to the shared library file.
    canonical_path: PathBuf,
    /// Handle to the loaded library.
    library: Option<PluginLibrary>,
    ctx: PluginCtx,
    last_reload_time: SystemTime,
}

impl PluginHost {
    /// Creates a new `PluginHost` instance.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        fn new_inner(path: &Path) -> PluginHost {
            let path = get_plugin_library_path(path);
            let last_reload_time = SystemTime::now();

            match fs::exists(&path) {
                Ok(true) => {
                    debug!("loading plugin library: {}", path.display());
                    let _ = watch_file(&path);
                    let library = PluginLibrary::load_or_log_error(&path);
                    // Not sure what could cause canonicalize to fail here (the path exists), so unwrap.
                    let canonical_path = fs::canonicalize(&path).unwrap();

                    let mut plugin =
                        PluginHost { path, library, canonical_path, ctx: Default::default(), last_reload_time };
                    plugin.init();
                    plugin
                }
                _ => {
                    // Add a file watch on the parent directory if the library file isn't there yet, so that we can reload it when it is created.
                    debug!("plugin library `{}` not found", path.display());

                    if let Some(parent) = path.parent()
                        && fs::exists(parent).unwrap_or(false)
                    {
                        let _ = watch_file(parent);
                    }

                    PluginHost {
                        path,
                        library: None,
                        canonical_path: PathBuf::new(),
                        ctx: Default::default(),
                        last_reload_time,
                    }
                }
            }
        }

        new_inner(path.as_ref())
    }

    fn init(&mut self) {
        if let Some(ref library) = self.library {
            unsafe {
                (library.init)(&mut self.ctx);
            }
        }
    }

    fn shutdown(&mut self) {
        if let Some(ref library) = self.library {
            unsafe {
                (library.shutdown)(&mut self.ctx);
            }
        }
        self.library = None;
    }

    fn path_is_newer(path: &Path, last_modified: SystemTime) -> bool {
        let new_last_modified = fs::metadata(&path).and_then(|meta| meta.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
        new_last_modified > last_modified
    }

    /// Reloads plugins whose shared library files have been modified since the last reload.
    pub fn reload(&mut self) {
        let reload_start_time = SystemTime::now();

        let Some(exists) = fs::exists(&self.path).ok() else { return };
        if !exists {
            // Library file doesn't exist; do nothing. We might be in the middle of
            // recompilation.
            return;
        }

        if let Some(ref lib) = self.library {
            if Self::path_is_newer(&self.path, lib.last_modified) {
                debug!("plugin library `{}` is new", self.path.display());
                self.send_event(PluginEvent::Unloading);
                self.shutdown();
                self.library = None;
                self.library = PluginLibrary::load_or_log_error(&self.path);
                // Update canonical path in case the file was replaced with a different file.
                self.canonical_path = fs::canonicalize(&self.path).unwrap();
                self.init();
                self.send_event(PluginEvent::Loaded);
            }
        } else {
            // Library is not loaded yet. Try to load it.
            self.library = PluginLibrary::load_or_log_error(&self.path);
            self.canonical_path = fs::canonicalize(&self.path).unwrap();
        }

        self.last_reload_time = reload_start_time;
    }

    fn send_event(&mut self, event: PluginEvent) -> PluginResult {
        if let Some(ref library) = self.library {
            unsafe { (library.entry)(&mut self.ctx, Box::new(event)) }
        } else {
            PluginResult::WaitVSync
        }
    }
}

/// Events sent to a plugin.
#[repr(C)]
pub enum PluginEvent<'a> {
    /// `AppHandler::unloading`
    Unloading,
    /// `AppHandler::loaded`
    Loaded,
    /// `AppHandler::started`
    Started,
    /// `AppHandler::render`
    Render(WindowHandle, RenderTargetImage<'a>),
    /// `AppHandler::input`
    Input(WindowHandle, &'a InputEvent),
    /// `AppHandler::event`
    UserEvent(UserEvent),
    /// `AppHandler::resized`
    Resized(WindowHandle, u32, u32),
    /// `AppHandler::vsync`
    VSync,
    /// `AppHandler::file_changed`
    FileChanged(&'a Path),
    /// `AppHandler::close_requested`
    CloseRequested(WindowHandle),
    /// `AppHandler::exiting`
    Exiting,
    /// `AppHandler::imgui`
    Imgui(&'a egui::Context),
}

impl AppHandler for PluginHost {
    fn started(&mut self) {
        self.send_event(PluginEvent::Started);
    }

    fn unloading(&mut self) {
        self.send_event(PluginEvent::Unloading);
    }

    fn loaded(&mut self) {
        self.send_event(PluginEvent::Loaded);
    }

    fn input(&mut self, window: WindowHandle, input_event: &InputEvent) {
        self.send_event(PluginEvent::Input(window, input_event));
    }

    fn event(&mut self, event: UserEvent) {
        self.send_event(PluginEvent::UserEvent(event));
    }

    fn resized(&mut self, window: WindowHandle, width: u32, height: u32) {
        self.send_event(PluginEvent::Resized(window, width, height));
    }

    fn vsync(&mut self) {
        self.send_event(PluginEvent::VSync);
    }

    fn render(&mut self, window: WindowHandle, image: RenderTargetImage<'_>) {
        self.send_event(PluginEvent::Render(window, image));
    }

    fn file_changed(&mut self, path: &Path) {
        self.send_event(PluginEvent::FileChanged(path));
    }

    fn close_requested(&mut self, window: WindowHandle) {
        self.send_event(PluginEvent::CloseRequested(window));
    }

    fn imgui(&mut self, ctx: &egui::Context) {
        self.send_event(PluginEvent::Imgui(ctx));
    }

    fn exiting(&mut self) {
        self.send_event(PluginEvent::Exiting);
    }
}

//--------------------------------------------------------------------------------------------------

#[doc(hidden)]
pub fn dispatch_plugin_event<T: AppHandler + Serialize + DeserializeOwned>(
    ctx: &mut PluginCtx,
    handler: &mut T,
    event: Box<PluginEvent>,
) -> PluginResult {
    match *event {
        PluginEvent::Started => {
            handler.started();
            PluginResult::WaitVSync
        }
        PluginEvent::Render(window, image) => {
            handler.render(window, image);
            PluginResult::WaitVSync
        }
        PluginEvent::Input(window, input_event) => {
            handler.input(window, input_event);
            PluginResult::WaitVSync
        }
        PluginEvent::UserEvent(user_event) => {
            handler.event(user_event);
            PluginResult::WaitVSync
        }
        PluginEvent::Resized(window, width, height) => {
            handler.resized(window, width, height);
            PluginResult::WaitVSync
        }
        PluginEvent::VSync => {
            handler.vsync();
            PluginResult::WaitVSync
        }
        PluginEvent::FileChanged(path) => {
            handler.file_changed(path);
            PluginResult::WaitVSync
        }
        PluginEvent::CloseRequested(window) => {
            handler.close_requested(window);
            PluginResult::WaitVSync
        }
        PluginEvent::Imgui(ctx) => {
            handler.imgui(ctx);
            PluginResult::WaitVSync
        }
        PluginEvent::Exiting => {
            handler.exiting();
            PluginResult::WaitVSync
        }
        PluginEvent::Unloading => {
            handler.unloading();
            ctx.save(&handler);
            PluginResult::WaitVSync
        }
        PluginEvent::Loaded => {
            ctx.load(handler);
            handler.loaded();
            PluginResult::WaitVSync
        }
    }
}

#[macro_export]
macro_rules! register_plugin {
    ($init_fn:expr) => {
        mod plugin {
            use super::*;

            fn cast<T>(ptr: ::std::ptr::NonNull<()>, _fn: fn() -> T) -> ::std::ptr::NonNull<T> {
                ::std::ptr::NonNull::new(ptr.as_ptr() as *mut T).unwrap()
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn plugin_init(ctx: &mut $crate::PluginCtx) {
                let mut handler = $init_fn();
                let _ = ctx.set_user_ptr(Some(::std::ptr::NonNull::new(Box::into_raw(Box::new(handler)) as *mut ()).unwrap()));
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn plugin_shutdown(ctx: &mut $crate::PluginCtx) {
                let handler = cast(ctx.set_user_ptr(None).expect("expected non-zero user pointer"), $init_fn);
                unsafe {
                    let _ = Box::from_raw(handler.as_ptr() as *mut _);
                }
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn plugin_entry(
                ctx: &mut $crate::PluginCtx,
                event: Box<$crate::PluginEvent>,
            ) -> $crate::PluginResult {
                let handler = cast(ctx.get_user_ptr().expect("expected non-zero user pointer"), $init_fn);
                let handler = unsafe { &mut *handler.as_ptr() };
                $crate::dispatch_plugin_event(ctx, handler, event)
            }
        }
    };
}
