//! Windows platform backend
use crate::platform::{EventToken, InitOptions, PlatformHandler, RenderTargetImage, UserEvent};
use log::error;
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::time::Instant;
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
mod key_code;
mod swap_chain;
mod window;

use crate::context::LoopHandler;
use crate::platform::windows::compositor_clock::CompositorClock;
use crate::platform::windows::graphics::GraphicsContext;
use crate::platform::windows::window::Window;
pub use event_loop::wake_event_loop;

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

/// Global platform-specific objects.
#[allow(dead_code)]
pub struct Win32Platform {
    options: InitOptions,
    vsync_clock: CompositorClock,
    window: RefCell<Option<Window>>,
    timers: RefCell<Vec<TimerEntry>>,
    poll_requested: Cell<bool>,
    quit_requested: Cell<bool>,
}

impl PlatformHandler for Win32Platform {
    fn teardown(&self) {
        // release window resources
        self.window.borrow_mut().take();
        self.vsync_clock.stop();
    }

    fn render(&self, render_callback: &mut dyn FnMut(RenderTargetImage)) {
        let window = self.window.borrow();
        let window = window.as_ref().unwrap();
        let render_target = window.get_swap_chain_image();
        render_callback(render_target);
        window.present();
    }

    fn wake_at_next_vsync(&self) {
        self.vsync_clock.trigger();
    }

    fn add_timeout(&self, at: Instant, token: EventToken) {
        let mut timers = self.timers.borrow_mut();
        // insert the timer in sorted order
        let entry = TimerEntry { deadline: at, token };
        let pos = timers.binary_search(&entry).unwrap_or_else(|e| e);
        timers.insert(pos, entry);
        debug_assert!(timers.is_sorted());
    }

    fn run(&'static self, handler: Box<dyn LoopHandler + '_>) {
        self.run_event_loop(handler);
    }

    fn quit(&self) {
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
            window: RefCell::new(None),
            options: options.clone(),
            poll_requested: Cell::new(false),
            timers: RefCell::new(vec![]),
            quit_requested: Cell::new(false),
        }
    }
}
