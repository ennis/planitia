//! Platform-specific implementations of certain types and functions.

use crate::input::InputEvent;
use std::time::Instant;

#[cfg(windows)]
pub mod windows;

#[cfg(windows)]
pub type Platform = windows::Win32Platform;

//----------------------------------------------------------------------------------

/// Image returned by `acquire` that can be rendered to.
#[derive(Clone)]
pub struct RenderTargetImage<'a> {
    pub image: &'a gpu::Image,
    /// Semaphore that should be waited on before rendering to the image.
    pub ready: gpu::SemaphoreWait,
    /// Should be signaled after rendering to the image.
    pub rendering_finished: gpu::SemaphoreSignal,
}

/// A token that uniquely identifies a non-input event.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EventToken(pub u64);

impl EventToken {
    /// Predefined event token for async task wakeups.
    pub const TASK: Self = EventToken(0);
    /// Predefined event token for vsync events.
    pub const VSYNC: Self = EventToken(1);
}

/// Identifies the reason for waking the event loop.
pub enum LoopEvent {
    /// An input event, such as a key press or mouse movement.
    Input(InputEvent),
    /// A user event, emitted when `wake_event_loop` is called.
    Event(EventToken),
    /// Polling
    Poll,
}

/// A request to wake up the event loop at a particular time.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WakeupRequest {
    /// Wake up the event loop on the next vsync.
    VSync,
    /// Wake up the event loop at the specified time.
    At(Instant),
    /// Wake up the event loop immediately after the current event loop iteration.
    Poll,
}

/// Platform initialization options.
#[derive(Debug, Clone, Copy)]
pub struct InitOptions {
    /// The initial size of the window.
    pub width: u32,
    /// The initial height of the window.
    pub height: u32,
    pub window_title: &'static str,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            window_title: "Game",
        }
    }
}

/// Defines methods that are called when the event loop resumes.
#[allow(unused_variables)]
pub trait LoopHandler {
    fn input(&mut self, input_event: InputEvent);
    fn event(&mut self, token: EventToken);
    fn resized(&mut self, width: u32, height: u32);
    fn vsync(&mut self);
    fn poll(&mut self);
    fn close_requested(&mut self);
}

pub trait PlatformHandler {
    /// Releases global resources.
    fn teardown(&self);

    /// Acquires a render target image that can be used for rendering.
    fn render(&self, render_callback: &mut dyn FnMut(RenderTargetImage));

    /// Returns the Vulkan device instance.
    fn get_gpu_device(&self) -> &gpu::RcDevice;

    /// Signals the event loop to wake up on the next vsync.
    fn wake_at_next_vsync(&self);

    /// Registers a timeout.
    fn add_timeout(&self, at: Instant, token: EventToken);

    /// Runs the event loop with the given handler.
    fn run(&'static self, handler: Box<dyn LoopHandler + '_>);

    fn quit(&self);
}
