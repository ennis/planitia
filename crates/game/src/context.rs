//! Global context.
use crate::executor::LocalExecutor;
use crate::imgui;
use crate::imgui::ImguiContext;
use crate::input::InputEvent;
pub use crate::platform::LoopHandler;
use crate::platform::{EventToken, InitOptions, Platform, PlatformHandler, RenderTargetImage, UserEvent};
use futures::future::AbortHandle;
use gpu::vk::Handle;
use keyboard_types::{Key, KeyState, NamedKey};
use log::{error, info, warn};
use mlua::Lua;
use renderdoc::{RenderDoc, V141};
use std::cell::{Cell, OnceCell, RefCell};
use std::ffi::c_void;
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, OnceLock};
use threadbound::ThreadBound;

/// Holds the application's global objects and services.
pub(crate) struct Context {
    /// Platform object. As the name suggests, this is platform-specific code and state
    /// that manage the GPU context, the main window, and the event loop.
    pub(crate) platform: Platform,
    pub(crate) lua: mlua::Lua,
    imgui: RefCell<ImguiContext>,
    pub(crate) executor: LocalExecutor,
    quit_requested: Cell<bool>,
    rdoc: Option<RefCell<RenderDoc<V141>>>,
    rdoc_capture_requested: Cell<bool>,
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
        let imgui = RefCell::new(ImguiContext::new());

        let rdoc = RenderDoc::new().ok();
        if rdoc.is_some() {
            info!("running with RenderDoc");
        } else {
            info!("not running with RenderDoc");
        }

        Self {
            platform,
            lua,
            executor,
            imgui,
            quit_requested: Cell::new(false),
            rdoc: rdoc.map(RefCell::new),
            rdoc_capture_requested: Cell::new(false),
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
            info!("starting RenderDoc capture");
            rdoc.borrow_mut()
                .start_frame_capture(unsafe { rdoc_instance_ptr() }, std::ptr::null());
        }
    }

    fn end_renderdoc_capture(&self) {
        if let Some(rdoc) = &self.rdoc {
            if rdoc.borrow().is_frame_capturing() {
                info!("finishing RenderDoc capture");
                rdoc.borrow_mut()
                    .end_frame_capture(unsafe { rdoc_instance_ptr() }, std::ptr::null());
            }
        }
    }
}


//------------------------------------------------------------------------------------------------

#[allow(unused_variables)]
pub trait AppHandler {
    /// Called when the event loop receives an input event.
    fn input(&mut self, input_event: InputEvent);

    /// Called when the event loop receives a custom event.
    fn event(&mut self, event: UserEvent);

    /// Called when the window is resized.
    fn resized(&mut self, width: u32, height: u32);

    /// Called when the vsync signal is received.
    fn vsync(&mut self);

    /// Renders the current frame.
    fn render(&mut self, image: RenderTargetImage<'_>) {}

    /// Called when a watched file or directory has changed.
    fn file_system_event(&mut self, path: &Path) {}

    fn close_requested(&mut self) {}
    fn imgui(&mut self, ctx: &egui::Context) {}
}

//--------------------------------------------------------------------------------------------------

struct MainThreadContext<H: 'static> {
    handler: RefCell<H>,
    context: Context,
}

struct AppInner<H: 'static> {
    main_thread: ThreadBound<&'static MainThreadContext<H>>,
}

pub struct App<H: 'static>(OnceLock<AppInner<H>>);

impl<H: AppHandler + Default + 'static> App<H> {
    pub const fn new() -> Self {
        App(OnceLock::new())
    }

    pub fn run(&'static self, init_options: &InitOptions) {
        pretty_env_logger::init();
        tracy_client::set_thread_name!("main thread");
        info!("running with Tracy profiler enabled");

        let app = self.0.get_or_init(|| AppInner {
            // This needs to be a leaked static because ThreadBound wouldn't be Send, and
            // OnceLock requires its contents to be Send + Sync in order to be Sync and usable
            // within a static.
            //
            // See https://users.rust-lang.org/t/how-to-migrate-to-lazylock-from-lazy-static/128921
            main_thread: ThreadBound::new(Box::leak(Box::new(MainThreadContext {
                handler: RefCell::new(H::default()),
                context: Context::new(init_options),
            }))),
        });

        let main_thread = *app.main_thread.get_ref().unwrap();
        let context = &main_thread.context;
        context.platform.wake_at_next_vsync();
        context.platform.run_event_loop(Box::new(self.0.get().unwrap()));
        context.teardown();
    }

    fn ensure_running(&self) -> &AppInner<H> {
        self.0.get().expect("App::run should have been called")
    }

    pub fn lua(&self) -> &Lua {
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
    }

    /// Spawns a task from a Lua script.
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
    }

    pub fn quit(&self) {
        let context = self.ensure_running().get_context();
        context.platform.quit();
    }

    /// Renders the ImGui UI into the given image using the provided command stream.
    pub fn render_imgui(&self, command_stream: &mut gpu::CommandStream, image: &gpu::Image) {
        let context = self.ensure_running().get_context();
        context.render_imgui(command_stream, image);
    }
}

impl<H> AppInner<H> {
    fn get_context(&self) -> &'static Context {
        let main_thread = *self.main_thread.get_ref().unwrap();
        &main_thread.context
    }

    fn get_handler(&self) -> &'static RefCell<H> {
        let main_thread = *self.main_thread.get_ref().unwrap();
        &main_thread.handler
    }
}

impl<Inner> LoopHandler for &'static AppInner<Inner>
where
    Inner: AppHandler,
{
    fn input(&mut self, input_event: InputEvent) {
        let ctx = self.get_context();

        if ctx.imgui.borrow_mut().handle_input(&input_event) {
            // If the event was processed by egui, we can continue
            return;
        }

        match input_event {
            InputEvent::KeyboardEvent(ref ke) if ke.key == Key::Named(NamedKey::F12) && ke.state == KeyState::Down => {
                ctx.rdoc_capture_requested.set(true);
            }
            _ => {}
        }

        // Otherwise, pass the event to the inner handler
        self.get_handler().borrow_mut().input(input_event);
    }

    fn event(&mut self, event: UserEvent) {
        self.get_handler().borrow_mut().event(event);
    }

    fn resized(&mut self, width: u32, height: u32) {
        self.get_handler().borrow_mut().resized(width, height)
    }

    fn vsync(&mut self) {
        let ctx = self.get_context();
        let handler = self.get_handler();
        let mut handler = handler.borrow_mut();

        // invoke application vsync handler
        handler.vsync();

        // update the GUI
        ctx.imgui.borrow_mut().run(|imgui_ctx| {
            handler.imgui(imgui_ctx);
        });

        // start frame capture if requested and RenderDoc is available
        if ctx.rdoc_capture_requested.get() {
            ctx.start_renderdoc_capture();
        }

        // render the frame (the application is expected to render the GUI as part of its rendering)
        ctx.platform.render(&mut |render_target| {
            handler.render(render_target);
        });

        // end frame capture
        if ctx.rdoc_capture_requested.get() {
            ctx.rdoc_capture_requested.set(false);
            ctx.end_renderdoc_capture();
        }

        // mark the end of the frame for tracy
        tracy_client::frame_mark();

        // cleanup expired GPU resources
        gpu::Device::global().cleanup();

        // ask for a re-render on the next vsync
        ctx.platform.wake_at_next_vsync();
    }

    fn poll(&mut self) {
        // TODO
    }

    fn close_requested(&mut self) {
        self.get_handler().borrow_mut().close_requested();
    }
}
