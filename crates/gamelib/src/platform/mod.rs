//! Platform-specific implementations of certain types and functions.

use crate::input::InputEvent;
use std::any::Any;
use std::path::Path;
use std::time::Instant;

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
pub use windows::wake_event_loop;

#[cfg(windows)]
pub type Platform = windows::Win32Platform;

//----------------------------------------------------------------------------------

/// Image returned by `acquire` that can be rendered to.
#[derive(Clone)]
pub struct RenderTargetImage<'a> {
    pub image: &'a gpu::Image,
}

/// A token that uniquely identifies a non-input event.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EventToken(pub u64);

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

pub type UserEvent = Box<dyn Any + Send>;

/// Defines methods that are called when the event loop resumes.
#[allow(unused_variables)]
pub trait LoopHandler {
    fn input(&mut self, input_event: InputEvent);
    fn event(&mut self, payload: UserEvent);
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

    /// Signals the event loop to wake up on the next vsync.
    fn wake_at_next_vsync(&self);

    /// Registers a timeout.
    fn add_timeout(&self, at: Instant, token: EventToken);

    /// Runs the event loop with the given handler.
    fn run(&'static self, handler: Box<dyn LoopHandler + '_>);

    fn quit(&self);
}
