//! Global context.
use crate::executor::LocalExecutor;
use crate::imgui;
use crate::imgui::ImguiContext;
use crate::input::InputEvent;
pub use crate::platform::LoopHandler;
use crate::platform::{EventToken, InitOptions, LoopEvent, Platform, PlatformHandler, RenderTargetImage};
use gpu::vk::Handle;
use keyboard_types::{Key, KeyState, NamedKey};
use log::{info, warn};
use renderdoc::{RenderDoc, V141};
use std::cell::{Cell, OnceCell, RefCell};
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

/// Holds the application's global objects and services.
///
/// This is a singleton only accessible on the main thread with `get_context()`.
pub(crate) struct Context {
    /// Platform object. As the name suggests, this is platform-specific code and state
    /// that manage the GPU context, the main window, and the event loop.
    pub(crate) platform: Platform,
    pub(crate) lua: mlua::Lua,
    imgui: RefCell<ImguiContext>,
    pub(crate) executor: LocalExecutor,
    quit_requested: Cell<bool>,
    rdoc: Option<RefCell<RenderDoc<V141>>>,
}

unsafe fn rdoc_instance_ptr() -> *mut c_void {
    unsafe {
        let instance = gpu::get_vulkan_instance().handle().as_raw() as *mut *mut c_void;
        ptr::read(instance)
    }
}

impl Context {
    fn new(options: &InitOptions) -> Self {
        let platform = Platform::new(options);
        let lua = mlua::Lua::new();
        let executor = LocalExecutor::new();
        let imgui = RefCell::new(ImguiContext::new(platform.get_gpu_device()));

        let rdoc = RenderDoc::new().ok();
        if rdoc.is_some() {
            info!("Running with RenderDoc");
        } else {
            info!("RenderDoc not loaded");
        }

        Self {
            platform,
            lua,
            executor,
            imgui,
            quit_requested: Cell::new(false),
            rdoc: rdoc.map(RefCell::new),
        }
    }

    fn teardown(&self) {
        self.platform.teardown();
    }

    fn render_imgui(&self, command_stream: &mut gpu::CommandStream, image: &gpu::Image) {
        self.imgui.borrow_mut().render(command_stream, image);
    }

    fn start_renderdoc_capture(&self) {
        if let Some(rdoc) = &self.rdoc {
            info!("Starting RenderDoc capture");
            rdoc.borrow_mut()
                .start_frame_capture(unsafe { rdoc_instance_ptr() }, std::ptr::null());
        }
    }

    fn end_renderdoc_capture(&self) {
        if let Some(rdoc) = &self.rdoc {
            if rdoc.borrow().is_frame_capturing() {
                rdoc.borrow_mut()
                    .end_frame_capture(unsafe { rdoc_instance_ptr() }, std::ptr::null());
            }
        }
    }
}

static INITIALIZED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static CONTEXT: OnceCell<&'static Context> = OnceCell::new();
}

/// Returns the global context.
pub(crate) fn get_context() -> &'static Context {
    CONTEXT.with(|g| *g.get().expect("a context should be active on this thread"))
}

//------------------------------------------------------------------------------------------------

#[allow(unused_variables)]
pub trait AppHandler {
    /// Called when the event loop receives an input event.
    fn input(&mut self, input_event: InputEvent);

    /// Called when the event loop receives a custom event.
    fn event(&mut self, token: EventToken);

    /// Called when the window is resized.
    fn resized(&mut self, width: u32, height: u32);

    /// Called when the vsync signal is received.
    fn vsync(&mut self);

    /// Renders the current frame.
    fn render(&mut self, image: RenderTargetImage<'_>) {}

    fn close_requested(&mut self) {}
    fn imgui(&mut self, ctx: &egui::Context) {}
}

/// A wrapper around a `LoopHandler` that handles input events for egui.
struct LoopHandlerWrapper<Inner> {
    inner: Inner,
    rdoc_capture_requested: bool,
}

impl<Inner> LoopHandler for LoopHandlerWrapper<Inner>
where
    Inner: AppHandler,
{
    fn input(&mut self, input_event: InputEvent) {
        let ctx = get_context();
        if ctx.imgui.borrow_mut().handle_input(&input_event) {
            // If the event was processed by egui, we can continue
            return;
        }

        match input_event {
            InputEvent::KeyboardEvent(ref ke) if ke.key == Key::Named(NamedKey::F12) && ke.state == KeyState::Down => {
                self.rdoc_capture_requested = true;
            }
            _ => {}
        }

        // Otherwise, pass the event to the inner handler
        self.inner.input(input_event);
    }

    fn event(&mut self, token: EventToken) {
        self.inner.event(token)
    }

    fn resized(&mut self, width: u32, height: u32) {
        self.inner.resized(width, height);
    }

    fn vsync(&mut self) {
        let ctx = get_context();

        // invoke application vsync handler
        self.inner.vsync();

        // update the GUI
        ctx.imgui.borrow_mut().run(|imgui_ctx| {
            self.inner.imgui(imgui_ctx);
        });

        // start frame capture if requested and RenderDoc is available
        if self.rdoc_capture_requested {
            ctx.start_renderdoc_capture();
        }

        // render the frame (the application is expected to render the GUI as part of its rendering)
        ctx.platform.render(&mut |render_target| {
            self.inner.render(render_target);
        });

        // end frame capture
        if self.rdoc_capture_requested {
            self.rdoc_capture_requested = false;
            ctx.end_renderdoc_capture();
        }

        // mark the end of the frame for tracy
        tracy_client::frame_mark();

        // cleanup expired GPU resources
        get_gpu_device().cleanup();
        // ask for a re-render on the next vsync
        ctx.platform.wake_at_next_vsync();
    }

    fn poll(&mut self) {
        // TODO
    }

    fn close_requested(&mut self) {
        self.inner.close_requested();
    }
}

fn initialize(options: &InitOptions) -> &'static Context {
    if INITIALIZED
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        warn!("already initialized");
        return get_context();
    }

    // tracy client initialization
    tracy_client::set_thread_name!("main thread");
    info!("Running with Tracy profiler enabled");

    CONTEXT.with(|g| *g.get_or_init(|| Box::leak(Box::new(Context::new(options)))))
}

//--------------------------------------------------------------------------------------------------

pub fn run<Handler: AppHandler + Default>(init_options: &InitOptions) {
    pretty_env_logger::init();
    initialize(init_options);
    let context = get_context();
    context.platform.wake_at_next_vsync();
    let handler = Handler::default();
    context.platform.run_event_loop(Box::new(LoopHandlerWrapper {
        inner: handler,
        rdoc_capture_requested: false,
    }));
    context.teardown();
}

/// Returns the GPU device.
pub fn get_gpu_device() -> &'static gpu::RcDevice {
    get_context().platform.get_gpu_device()
}

/// Requests the event loop to quit.
pub fn quit() {
    get_context().platform.quit();
}

/// Renders the ImGui UI into the given image using the provided command stream.
pub fn render_imgui(command_stream: &mut gpu::CommandStream, image: &gpu::Image) {
    get_context().render_imgui(command_stream, image);
}
