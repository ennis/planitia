//! Windows platform backend
use crate::platform::{InitOptions, RenderTargetImage, UserEvent};
use log::error;
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::time::Instant;
use slotmap::SlotMap;
use windows::Win32::Foundation::{HANDLE, HWND};
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Device, ID3D12Fence};
use windows::Win32::Graphics::Dxgi::IDXGIFactory4;
use windows::Win32::System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx};
use windows::core::{Interface, Owned};
use winit::platform::windows::WindowAttributesExtWindows;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

mod compositor_clock;
mod event_loop;
mod graphics;
mod keys;
mod swap_chain;
mod window;

use crate::context::LoopHandler;
use crate::platform::win32::compositor_clock::CompositorClock;
use crate::platform::win32::graphics::GraphicsContext;
use crate::platform::win32::window::Window;
pub use event_loop::wake_event_loop;
use crate::event::EventToken;
use crate::platform::win32::event_loop::ACTIVE_EVENT_LOOP;
use crate::WindowCreateInfo;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to create window: {0}")]
    CreateWindow(#[from] winit::error::OsError),
}

/// Default number of buffers in the swap chain.
///
/// 3 is the minimum. 2 leads to contentions on the present queue.
pub(super) const SWAP_CHAIN_BUFFER_COUNT: u32 = 3;

/// Defines a send+sync wrapper over a windows interface type.
///
/// This signifies that it's OK to call the interface's methods from multiple threads simultaneously:
/// the object itself should synchronize the calls.
///
/// # COM thread safety notes
///
/// Some interfaces are thread-safe, some are not, and for some we don't know due to poor documentation.
/// Additionally, some interfaces should only be called on the thread in which they were created.
///
/// - For thread-safe interfaces: wrap them in a `Send+Sync` newtype
/// - For interfaces bound to a thread: wrap them in `ThreadBound`
/// - For interfaces not bound to a thread but with unsynchronized method calls:
///      wrap them in a `Send` newtype, and if you actually need to call the methods from multiple threads, `Mutex`.
macro_rules! sync_com_ptr_wrapper {
    ($wrapper:ident ( $iface:ident ) ) => {
        #[derive(Clone)]
        pub(crate) struct $wrapper(pub(crate) $iface);
        unsafe impl Sync for $wrapper {} // ok to send &I across threads
        unsafe impl Send for $wrapper {} // ok to send I across threads
        impl ::std::ops::Deref for $wrapper {
            type Target = $iface;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}

// TODO: the wrappers are not necessary anymore since ApplicationBackend is not accessible from
//       threads other than the main thread. We can just use the raw interfaces directly.
sync_com_ptr_wrapper! { D3D12Device(ID3D12Device) }
sync_com_ptr_wrapper! { DXGIFactory4(IDXGIFactory4) }
sync_com_ptr_wrapper! { D3D12CommandQueue(ID3D12CommandQueue) }

struct GpuFenceData {
    fence: ID3D12Fence,
    event: Owned<HANDLE>,
    value: Cell<u64>,
}

/// Some bullshit to get the HWND from winit
fn get_hwnd(handle: RawWindowHandle) -> HWND {
    match handle {
        RawWindowHandle::Win32(win32) => HWND(win32.hwnd.get() as *mut c_void),
        _ => unreachable!("only win32 windows are supported"),
    }
}

#[derive(Debug)]
enum WakeReason {
    VSync,
    Task,
    User(UserEvent),
}

#[derive(Copy, Clone, PartialOrd, Ord, Eq, PartialEq, Debug)]
struct TimerEntry {
    deadline: Instant,
    token: EventToken,
}

slotmap::new_key_type! {
    pub(crate) struct WindowKey;
}

/// Win32 window handle.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Win32WindowHandle {
    key: WindowKey,
}

/// Win32-specific window creation options.
#[derive(Debug, Copy, Clone)]
pub struct Win32WindowCreateInfo {
    // nothing for now
    pub _dummy: PhantomData<()> = PhantomData,
}

impl Default for Win32WindowCreateInfo {
    fn default() -> Self {
        Self { .. }
    }
}


/// Win32 platform state.
#[allow(dead_code)]
pub struct Win32Platform {
    options: InitOptions,
    /// Controls the compositor clock used to trigger VSync events.
    vsync_clock: CompositorClock,
    //window: RefCell<Option<Window>>,
    /// Active timers.
    timers: RefCell<Vec<TimerEntry>>,
    poll_requested: Cell<bool>,
    /// Something has requested application exit.
    quit_requested: Cell<bool>,
    /// Open windows.
    windows: RefCell<SlotMap<WindowKey, Window>>,
}


impl Win32Platform {

    /// Creates a window.
    pub(crate) fn create_window(&self, create_info: &WindowCreateInfo) -> Win32WindowHandle {
        ACTIVE_EVENT_LOOP.with(|event_loop| {
            let window = Window::new(event_loop, create_info.title, create_info.width, create_info.height)
                .expect("failed to create window");
            let mut windows = self.windows.borrow_mut();
            let key = windows.insert(window);
            Win32WindowHandle { key }
        })
    }

    /// Destroys a window.
    pub(crate) fn destroy_window(&self, handle: Win32WindowHandle) {
        let mut windows = self.windows.borrow_mut();
        windows.remove(handle.key);
    }

    /// Finds a window by its winit ID.
    fn find_window_by_id(&self, id: winit::window::WindowId) -> Option<Win32WindowHandle> {
        let windows = self.windows.borrow();
        windows.iter().find_map(|(key, window)| {
            if window.inner.id() == id {
                Some(Win32WindowHandle { key })
            } else {
                None
            }
        })
    }

    /// Releases the platform's internal resources in preparation for application exit.
    pub(crate) fn teardown(&self) {
        self.windows.borrow_mut().clear();
        self.vsync_clock.stop();
    }

    /// Renders all windows that need to be rendered.
    pub(crate) fn render_all(&self, render_callback: &mut dyn FnMut(Win32WindowHandle, RenderTargetImage)) {
        let mut windows = self.windows.borrow_mut();
        for (key, window) in windows.iter_mut() {
            if let Some(image) = window.get_swap_chain_image() {
                render_callback(Win32WindowHandle { key }, RenderTargetImage { image });
                window.present();
            }
        }
    }

    /// Renders a frame to the given window.
    pub(crate) fn render(&self, window: Win32WindowHandle, render_callback: &mut dyn FnMut(RenderTargetImage)) {
        let mut windows = self.windows.borrow_mut();
        if let Some(window) = windows.get_mut(window.key) {
            if let Some(image) = window.get_swap_chain_image() {
                render_callback(RenderTargetImage { image });
                window.present();
            }
        } else {
            error!("render called with invalid window handle: {:?}", window);
        }
    }

    /// Schedules a wakeup of the event loop (see `LoopHandler::vsync`) on the next VSync.
    pub(crate) fn wake_at_next_vsync(&self) {
        self.vsync_clock.trigger();
    }

    /// Registers a timeout to trigger at the given time.
    pub(crate) fn add_timeout(&self, at: Instant, token: EventToken) {
        let mut timers = self.timers.borrow_mut();
        // insert the timer in sorted order
        let entry = TimerEntry { deadline: at, token };
        let pos = timers.binary_search(&entry).unwrap_or_else(|e| e);
        timers.insert(pos, entry);
        debug_assert!(timers.is_sorted());
    }

    /// Enters the event loop, which will run until `quit` is called.
    pub(crate) fn run(&'static self, handler: Box<dyn LoopHandler + '_>) {
        self.run_event_loop(handler);
    }

    /// Requests to quit the application.
    ///
    /// This causes `Platform::run` to return to the caller.
    pub(crate) fn quit(&self) {
        self.quit_requested.set(true);
    }
}

impl Win32Platform {
    pub(crate) fn new(options: &InitOptions) -> Win32Platform {
        unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED).unwrap() };

        // intialize graphics context
        let _ = GraphicsContext::current();
        let vsync_clock = CompositorClock::new();

        Win32Platform {
            vsync_clock,
            options: options.clone(),
            poll_requested: Cell::new(false),
            timers: RefCell::new(vec![]),
            quit_requested: Cell::new(false),
            windows: RefCell::new(SlotMap::with_key()),
        }
    }
}
