//! Global context.
use crate::error::{ExcResult, ResultExt};
use crate::event::UserEvent;
use crate::executor::LocalExecutor;
use crate::imgui::ImguiContext;
use crate::input::InputEvent;
use crate::paint::{PaintScene, Painter, TextFormat};
use crate::platform::{LoopHandler, Platform, RenderTargetImage, WindowHandle};
use crate::plugin_host::PluginHost;
use crate::tweak::show_tweaks_gui;
use crate::util::env_flag;
use crate::{PluginEvent, imgui, span, wake_event_loop};
use color::Srgba8;
use color_print::cwriteln;
use env_logger::fmt::style::AnsiColor;
use futures::future::AbortHandle;
use gpu::vk::Handle;
use keyboard_types::{Key, KeyState, Modifiers, NamedKey};
use log::{debug, error, info, warn};
use math::{IVec2, Vec2, vec2};
use notify_debouncer_mini::notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{DebounceEventResult, DebouncedEvent, Debouncer, new_debouncer};
use renderdoc::{RenderDoc, V141};
use std::any::{Any, TypeId};
use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, OnceLock};
use std::{mem, ptr};
use threadbound::ThreadBound;

/// Tries to load the RenderDoc DLL.
fn load_renderdoc_dll() {
    #[cfg(target_os = "windows")]
    const DLL_PATH: &[&str] = &["renderdoc.dll", "C:\\Program Files\\RenderDoc\\renderdoc.dll"];

    unsafe {
        for &path in DLL_PATH {
            match libloading::Library::new(path) {
                Ok(library) => {
                    info!("Loaded RenderDoc DLL from {}", path);
                    mem::forget(library);
                    return;
                }
                Err(_) => {}
            }
        }
    }
}

/// Gets a pointer to a VkInstance for RenderDoc captures.
unsafe fn rdoc_instance_ptr() -> *mut c_void {
    unsafe {
        let instance = gpu::get_vulkan_instance().handle().as_raw() as *mut *mut c_void;
        ptr::read(instance)
    }
}

/// Initializes `env_logger` with a custom format.
fn setup_env_logger() {
    env_logger::builder()
        .parse_default_env()
        .format(|fmt, record| {
            use env_logger::fmt::style::{AnsiColor, Style};
            use std::io::Write;

            //let style = fmt.default_level_style(record.level());
            let args = record.args();

            let message_color = match record.level() {
                log::Level::Error => Some(AnsiColor::Red),
                log::Level::Warn => Some(AnsiColor::Yellow),
                log::Level::Info => None,
                log::Level::Debug => Some(AnsiColor::BrightBlack),
                log::Level::Trace => Some(AnsiColor::BrightBlack),
            };
            // let target = record.target();
            let mut msg_sty = Style::new();
            let mut target_sty = Style::new().italic();
            if let Some(color) = message_color {
                msg_sty = msg_sty.fg_color(Some(color.into()));
                target_sty = target_sty.fg_color(Some(AnsiColor::BrightBlack.into()));
            }
            writeln!(fmt, "{msg_sty}{args}{msg_sty:#} {target_sty} {target_sty:#}")
        })
        .format_target(false)
        .format_timestamp(None)
        .init();
}

//--------------------------------------------------------------------------------------------------

/// Application event handler.
#[allow(unused_variables)]
pub trait AppHandler {
    /// Called when the event loop starts running.
    fn started(&mut self) {}

    /// Called when the application is about to be reloaded (for hot-reloading).
    fn unloading(&mut self) {}

    /// Called when the application has been reloaded.
    fn loaded(&mut self) {}

    /// Called when the event loop receives an input event.
    fn input(&mut self, window: WindowHandle, input_event: &InputEvent);

    /// Called when the event loop receives a custom event.
    fn event(&mut self, event: UserEvent) {}

    /// Called when the window is resized.
    fn resized(&mut self, window: WindowHandle, width: u32, height: u32);

    /// Called when the vsync signal is received.
    fn vsync(&mut self);

    /// Renders the current frame.
    fn render(&mut self, window: WindowHandle, image: RenderTargetImage<'_>) {}

    /// Called when a watched file or directory has changed.
    fn file_changed(&mut self, path: &Path) {}

    fn close_requested(&mut self, window: WindowHandle) {}
    fn imgui(&mut self, ctx: &egui::Context) {}

    fn exiting(&mut self) {}
}

/// Dummy handler used during app initialization, before the actual handler is created.
struct DummyHandler;
impl AppHandler for DummyHandler {
    fn input(&mut self, _window: WindowHandle, _input_event: &InputEvent) {
        panic!("dummy handler called")
    }

    fn resized(&mut self, _window: WindowHandle, _width: u32, _height: u32) {
        panic!("dummy handler called")
    }

    fn vsync(&mut self) {
        panic!("dummy handler called")
    }
}

//--------------------------------------------------------------------------------------------------

/// Main application object.
///
/// This should be created as a static singleton. E.g:
/// ```
/// static APP: App<MyAppHandler> = App::new();
/// # fn main() {
/// APP.run(&AppOptions {
///     // options here
/// });
/// # }
/// ```
pub struct App<H: 'static>(OnceLock<ThreadBound<&'static MainThreadContext>>, PhantomData<fn() -> H>);

/// Application initialization options.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppOptions {
    // nothing for now
}

impl<H: AppHandler + Default + 'static> App<H> {
    pub const fn new() -> Self {
        App(OnceLock::new(), PhantomData)
    }

    pub fn run(&'static self, init_options: &AppOptions) {
        // Setup env_logger.
        setup_env_logger();

        // Load the renderdoc DLL asap
        if env_flag("RENDERDOC") {
            load_renderdoc_dll();
        }

        // Create the main thread context object.
        let main_thread_ctx = self.0.get_or_init(|| {
            let handler = RefCell::new(Box::new(DummyHandler));
            // This needs to be a leaked static because ThreadBound wouldn't be Send, and
            // OnceLock requires its contents to be Send + Sync in order to be Sync and usable
            // within a static.
            //
            // See https://users.rust-lang.org/t/how-to-migrate-to-lazylock-from-lazy-static/128921
            ThreadBound::new(Box::leak(Box::new(MainThreadContext::new(handler, init_options))))
        });

        let ctx = *main_thread_ctx.get_ref().unwrap();
        CURRENT_CTX.set(Some(ctx));

        // Create handler.
        let handler = Box::new(H::default());
        ctx.handler.replace(handler);

        // Schedule the first render on the next VSync.
        // TODO: we probably should render immediately?
        ctx.platform.wake_at_next_vsync();

        // Enter the event loop. This doesn't return until the application exits.
        ctx.run_event_loop();

        // The event loop has returned, the application is exiting. Do platform-specific cleanup.
        ctx.platform.teardown();
    }
}

/*pub fn lua(&self) -> &Lua {
    let context = self.ensure_running().get_context();
    &context.lua
}

/// Spawns a task with a name.
///
/// The name is only used for debugging purposes and is not required to be unique.
///
/// # Arguments
/// * `name` - The name of the task.
/// * `future` - The future to run as a task.
///
pub fn spawn_named<F>(&self, name: &str, future: F)
where
    F: Future<Output = ()> + 'static,
{
    let context = self.ensure_running().get_context();
    context.executor.spawn(name.to_string(), future);
}

/// Spawns a task on the global executor.
pub fn spawn<F>(&self, future: F)
where
    F: Future<Output = ()> + 'static,
{
    // For now, we don't do anything with the name, but it can be used for debugging.
    self.spawn_named("<anonymous>", future);
}

/// Spawns a task on the global executor and returns an abort handle.
pub fn spawn_abortable<F>(&self, future: F) -> AbortHandle
where
    F: Future<Output = ()> + 'static,
{
    let (future, handle) = futures::future::abortable(future);
    self.spawn(async move {
        // ignore abort result
        let _ = future.await;
    });
    handle
}*/
/*/// Spawns a task from a Lua script.
pub fn spawn_lua_task(&'static self, code: &str) -> AbortHandle {
    let context = self.ensure_running().get_context();
    let code = code.to_string();
    self.spawn_abortable(async move {
        let r = context.lua.load(code).exec_async().await;
        match r {
            Ok(_) => {}
            Err(err) => {
                error!("Error executing Lua script: {}", err);
            }
        }
    })
}*/

/// Represents an instance of the application.
///
/// Holds globally-accessible objects and systems.
/// Within the event loop, it is accessible via [`CURRENT_CTX`].
pub(crate) struct MainThreadContext {
    /// Platform instance.
    ///
    /// Manages graphics devices, windows, and the event loop.
    pub(crate) platform: Platform,
    /// Vector graphics context
    pub(crate) painter: RefCell<Painter>,
    /// Lua VM instance.
    #[cfg(feature = "lua")]
    pub(crate) lua: Lua,
    /// ImGui context.
    pub(crate) imgui: RefCell<ImguiContext>,
    /// Executor for async tasks.
    pub(crate) executor: LocalExecutor,
    /// RenderDoc connection.
    rdoc: Option<RefCell<RenderDoc<V141>>>,
    rdoc_capture_requested: Cell<bool>,
    rdoc_launch_replay_ui: Cell<bool>,
    debug_mark_counter: Cell<usize>,
    handler: RefCell<Box<dyn AppHandler + 'static>>,
    /// File watcher.
    watch: RefCell<Debouncer<RecommendedWatcher>>,
    /// Text overlay.
    text_overlay: RefCell<String>,
    // Registered global resources, ordered by type.
    //global_resources: RefCell<HashMap<TypeId, Box<dyn Any>>>,
    pub(crate) tracy_gpu_context: tracy_client::GpuContext,
    timestamp_query_counter: Cell<u16>,
}

impl MainThreadContext {
    /// Creates a new application instance and initializes the global systems.
    fn new(handler: RefCell<Box<dyn AppHandler + 'static>>, options: &AppOptions) -> Self {
        // Setup platform. This will also initialize the GPU device.
        let platform = Platform::new(options);

        // Setup tracy.
        let tracy_client = tracy_client::Client::running().unwrap();
        tracy_client.set_thread_name("main thread");
        info!("running with Tracy profiler enabled");

        // Setup tracy GPU context
        // This depends on the GPU device being initialized.
        let (device_timestamp, system_timestamp) = gpu::get_calibrated_timestamp_pair();
        let tracy_gpu_context = tracy_client
            .new_gpu_context(
                Some(&gpu::get_physical_device_name()),
                tracy_client::GpuContextType::Vulkan,
                device_timestamp as i64,
                gpu::get_timestamp_period(),
            )
            .expect("failed to create tracy GPU context");

        // Create a RenderDoc connection, if available.
        let rdoc = RenderDoc::new().ok();
        if rdoc.is_some() {
            info!("running with RenderDoc");
        } else {
            info!("not running with RenderDoc");
        }

        let executor = LocalExecutor::new();
        let imgui = RefCell::new(ImguiContext::new());

        // Create the file watcher.
        //
        // NOTE: `notify` spins a thread to watch for file changes, and calls the callback here.
        // It's annoying for us because a lot of things are only available on the main thread.
        // So when we receive an event, we forward it to the event loop on the main thread.
        // It's probably possible to do this without even spinning a separate thread,
        // by using the win32 API directly, but I don't have time for this.
        let watch = RefCell::new(
            new_debouncer(std::time::Duration::from_millis(250), move |events: DebounceEventResult| {
                match events {
                    Ok(events) => wake_event_loop(move || {
                        with_app_ctx(|ctx| {
                            ctx.handle_file_change_events(events);
                        });
                    }),
                    Err(err) => {
                        // Log, but otherwise ignore errors; not much we can do about them.
                        error!("error: {err}");
                    }
                }
            })
            .expect("failed to create file watcher"),
        );

        Self {
            platform,
            imgui,
            executor,
            painter: RefCell::new(Painter::new()),
            rdoc: rdoc.map(RefCell::new),
            rdoc_capture_requested: Cell::new(false),
            rdoc_launch_replay_ui: Cell::new(false),
            debug_mark_counter: Cell::new(0),
            handler,
            #[cfg(feature = "lua")]
            lua: Lua::new(),
            watch,
            text_overlay: RefCell::new(String::new()),
            tracy_gpu_context,
            timestamp_query_counter: Cell::new(0),
        }
    }

    /// Handles file change events from `notify`, and invokes the appropriate handlers.
    fn handle_file_change_events(&self, events: Vec<DebouncedEvent>) {
        for event in events {
            self.handler.borrow_mut().file_changed(&event.path)
        }
    }

    fn start_renderdoc_capture(&self) {
        if let Some(rdoc) = &self.rdoc {
            info!("starting RenderDoc capture");
            rdoc.borrow_mut().start_frame_capture(unsafe { rdoc_instance_ptr() }, std::ptr::null());
        }
    }

    fn end_renderdoc_capture(&self, launch_replay_ui: bool) {
        if let Some(rdoc) = &self.rdoc {
            let mut rdoc = rdoc.borrow_mut();
            if rdoc.is_frame_capturing() {
                info!("finishing RenderDoc capture");
                rdoc.end_frame_capture(unsafe { rdoc_instance_ptr() }, std::ptr::null());
                if launch_replay_ui {
                    let Some((path, _)) = rdoc.get_capture(0) else { return };

                    if let Err(err) = rdoc.launch_replay_ui(true, Some(path.to_string_lossy().as_ref())) {
                        error!("failed to launch renderdoc UI: {err}");
                    }
                }
            }
        }
    }

    fn run_event_loop(&'static self) {
        // Run the event loop.
        // This doesn't return until the application exits.
        let mut this = self;
        self.platform.run_event_loop(&mut this);
    }
}

impl LoopHandler for &'static MainThreadContext {
    fn started(&mut self) {
        self.handler.borrow_mut().started();
    }

    fn input(&mut self, window: WindowHandle, input_event: InputEvent) {
        let _span = span!("input");
        if self.imgui.borrow_mut().handle_input(&input_event) {
            // If the event was processed by egui, don't pass it to the application
            return;
        }

        if input_event.is_shortcut("F9") {
            self.rdoc_capture_requested.set(true);
        }

        if input_event.is_shortcut("Shift+F9") {
            self.rdoc_capture_requested.set(true);
            self.rdoc_launch_replay_ui.set(true);
        }

        if input_event.is_shortcut("F4") {
            let count = self.debug_mark_counter.get();
            self.debug_mark_counter.set(count + 1);
            info!("---------------------------------- MARK {count} ----------------------------------");
        }

        // Otherwise, pass the event to the inner handler
        self.handler.borrow_mut().input(window, &input_event);
    }

    fn event(&mut self, event: UserEvent) {
        let _span = span!("event");
        self.handler.borrow_mut().event(event);
    }

    fn resized(&mut self, window: WindowHandle, width: u32, height: u32) {
        let _span = span!("resized");
        self.handler.borrow_mut().resized(window, width, height)
    }

    fn vsync(&mut self) {
        let _span = span!("vsync");
        // invoke application vsync handler
        self.handler.borrow_mut().vsync();

        // update the GUI
        {
            let _span = span!("imgui");
            let _gpu_span = crate::gpu_span!("imgui");

            let mut cmd = gpu::CommandBuffer::new();
            self.imgui.borrow_mut().run(&mut cmd, |imgui_ctx| {
                egui::Window::new("Tweaks").show(imgui_ctx, |ui| {
                    show_tweaks_gui(ui);
                });
                self.handler.borrow_mut().imgui(imgui_ctx);
            });
            gpu::submit(cmd).unwrap();
        }

        // start frame capture if requested and RenderDoc is available
        if self.rdoc_capture_requested.get() {
            self.start_renderdoc_capture();
        }

        // render the frame (the application is expected to render the GUI as part of its rendering)
        {
            self.platform.render_all(&mut |window, render_target| {
                let _span = span!("render_window");
                let _gpu_span = crate::gpu_span!("render_window");
                self.handler.borrow_mut().render(window, render_target);

                // render text overlay
                {
                    let _span = span!("text_overlay");
                    let _gpu_span = crate::gpu_span!("text_overlay");
                    let text = self.text_overlay.take();
                    let mut scene = PaintScene::new(Srgba8::TRANSPARENT);
                    let pos = vec2(10.0, 10.0);
                    let shadow_pos = pos + vec2(1.0, 1.0);
                    // Draw shadow
                    let format = TextFormat { size: 20.0, ..Default::default() };
                    scene.draw_text(shadow_pos, &text, &format, Srgba8::BLACK);
                    scene.draw_text(pos, &text, &format, Srgba8::WHITE);
                    scene.render(render_target.image);
                }
            });
        }

        // end frame capture
        if self.rdoc_capture_requested.get() {
            self.rdoc_capture_requested.set(false);
            self.end_renderdoc_capture(self.rdoc_launch_replay_ui.replace(false));
        }

        // mark the end of the frame for tracy
        tracy_client::frame_mark();

        // cleanup expired GPU resources
        {
            let _span = span!("gpu_poll");
            gpu::end_frame();
        }

        // ask for a re-render on the next vsync
        self.platform.wake_at_next_vsync();
    }

    fn poll(&mut self) {
        let _span = span!("poll");
        // TODO
    }

    fn close_requested(&mut self, window: WindowHandle) {
        let _span = span!("close_requested");
        self.handler.borrow_mut().close_requested(window);
    }

    fn exiting(&mut self) {
        let _span = span!("exiting");
        self.imgui.borrow_mut().save_state();
        self.handler.borrow_mut().exiting();
    }
}

thread_local! {
    /// Thread-local pointer to the current application context.
    ///
    /// Some APIs need to access the platform instance, so we store the current app context in a
    /// thread-local to avoid having to pass it around everywhere.
    pub(crate) static CURRENT_CTX: RefCell<Option<&'static MainThreadContext>> = RefCell::new(None);
}

pub(crate) fn with_app_ctx<F, R>(f: F) -> R
where
    F: FnOnce(&'static MainThreadContext) -> R,
{
    CURRENT_CTX.with(|ctx| {
        let ctx = ctx.borrow();
        let ctx = ctx.as_ref().expect("app context is not available; either the app is not running, or this function was called from a different thread");
        f(ctx)
    })
}

/// Returns the current application context, if the application is running and the current thread is the main thread.
pub(crate) fn get_context() -> &'static MainThreadContext {
    with_app_ctx(|ctx| ctx)
}

/// Quits the application.
///
/// This causes the event loop to exit and `App::run` to return to the caller.
pub fn quit() {
    with_app_ctx(|ctx| {
        ctx.platform.quit();
    });
}

/// Render ImGui components in the specified render target.
pub fn render_imgui(command_stream: &mut gpu::CommandBuffer, image: &gpu::Image) {
    with_app_ctx(|ctx| {
        ctx.imgui.borrow_mut().render(command_stream, image);
    });
}

#[derive(thiserror::Error, Debug, Copy, Clone)]
#[error("failed to watch file")]
pub struct WatchFileError;

/// Watch for file changes at the specified path.
///
/// When a change occurs, the [`file_changed`] method of the currently running [`AppHandler`] will be called with the path of the changed file.
/// To stop watching a file, call [`unwatch_file`].
pub fn watch_file(path: &Path) -> ExcResult<(), WatchFileError> {
    debug!("watching file: {}", path.display());
    with_app_ctx(|ctx| {
        match ctx.watch.borrow_mut().watcher().watch(path, RecursiveMode::NonRecursive).raise(WatchFileError) {
            Ok(_) => Ok(()),
            Err(err) => {
                err.log_to_stderr();
                Err(err)
            }
        }
    })
}

/// Stop watching for file changes at the specified path.
pub fn unwatch_file(path: &Path) {
    debug!("unwatching file: {}", path.display());
    with_app_ctx(|ctx| {
        ctx.watch.borrow_mut().watcher().unwatch(path).expect("failed to unwatch file");
    });
}

/// Options for [`show_file_dialog`](show_file_dialog).
///
/// # Example
///
/// * Show a file picker for image files:
/// ```rust
/// let options = FileDialogOptions {
///     filters: &[("Image files", &["png", "jpg", "jpeg"])],
/// };
#[derive(Clone, Debug, Default)]
pub struct FileDialogOptions<'a> {
    /// File type filter.
    ///
    /// It's a list of (file_type_description, allowed_extensions) tuples.
    pub filters: &'a [(&'a str, &'a [&'a str])] = &[],
}

/// Shows a file picker dialog.
pub fn show_file_dialog(options: &FileDialogOptions<'_>) -> Option<std::path::PathBuf> {
    let mut dialog = rfd::FileDialog::new();
    for (name, extensions) in options.filters {
        dialog = dialog.add_filter(*name, extensions);
    }
    dialog.pick_file()
}

/// Shows a file picker dialog (shorthand for `show_file_dialog` with one filter).
///
/// # Arguments
/// * `file_type_description` - A description of the file type.
/// * `extensions` - A list of allowed file extensions (without the dot) (e.g. `["png", "jpg"]`).
pub fn pick_file(file_type_description: &str, extensions: &[&str]) -> Option<std::path::PathBuf> {
    let mut dialog = rfd::FileDialog::new();
    dialog = dialog.add_filter(file_type_description, extensions);
    dialog.pick_file()
}

/// Prints a message on screen.
pub fn print_message(message: impl AsRef<str>) {
    with_app_ctx(|ctx| {
        ctx.text_overlay.borrow_mut().push_str(message.as_ref());
    });
}

/// Formats a message on screen.
#[macro_export]
macro_rules! format_message {
    ($($arg:tt)*) => {{
        let message = format!($($arg)*);
        $crate::print_message(message);
    }};
}

/// Registers a global resource object.
pub fn register_resource<T: Any>(resource: T) {}

//--------------------------------------------------------------------------------------------------

impl MainThreadContext {
    pub(crate) fn begin_gpu_span(&self, span_location: &'static tracy_client::SpanLocation) {
        let query_id = self.timestamp_query_counter.get();
        // TODO: this breaks if there are more than u16::MAX pending queries: old queries
        //       will start to overwrite new ones with the same query_id. This can also happen before
        //       that if queries are not fulfilled in `query_id` order.
        //       To do that cleanly, you'd need a free list of query IDs but that's just too ridiculous
        //       for something that's supposed to be low overhead.
        self.timestamp_query_counter.set(query_id.wrapping_add(1));
        gpu::write_timestamp(move |ts| {
            with_app_ctx(|ctx| ctx.tracy_gpu_context.upload_gpu_timestamp(query_id, ts as i64))
        });
        // FIXME: this creates another command buffer, and thus another query pool
        //        meaning that we create one query pool per span...
        //gpu::flush().unwrap();
        self.tracy_gpu_context.begin_span(span_location, query_id);
    }

    pub(crate) fn end_gpu_span(&self) {
        let query_id = self.timestamp_query_counter.get();
        self.timestamp_query_counter.set(query_id.wrapping_add(1));
        gpu::write_timestamp(move |ts| {
            with_app_ctx(|ctx| ctx.tracy_gpu_context.upload_gpu_timestamp(query_id, ts as i64))
        });
        //gpu::flush().unwrap();
        self.tracy_gpu_context.end_span(query_id);
    }
}

pub struct TracyGpuSpanGuard;

impl Drop for TracyGpuSpanGuard {
    fn drop(&mut self) {
        tracy_end_gpu_span();
    }
}

#[doc(hidden)]
pub fn tracy_begin_gpu_span(location: &'static tracy_client::SpanLocation) {
    with_app_ctx(|app| {
        app.begin_gpu_span(location);
    })
}

pub fn tracy_end_gpu_span() {
    with_app_ctx(|app| {
        app.end_gpu_span();
    })
}

#[macro_export]
macro_rules! gpu_span {
    ($name:expr) => {{
        let location = $crate::tracy_client::span_location!($name);
        $crate::tracy_begin_gpu_span(location);
        $crate::TracyGpuSpanGuard
    }};
}
