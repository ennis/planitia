//! Platform-specific implementations of certain types and functions.

use crate::input::InputEvent;
use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;

#[cfg(windows)]
pub mod win32;
use crate::event::UserEvent;
#[cfg(windows)]
pub use win32::wake_event_loop;

#[cfg(windows)]
pub type Platform = win32::Win32Platform;
#[cfg(windows)]
pub type PlatformWindowCreateInfo = win32::Win32WindowCreateInfo;
#[cfg(windows)]
pub type WindowHandle = win32::Win32WindowHandle;
#[cfg(windows)]
pub type InputDeviceId = win32::Win32InputDeviceId;

//----------------------------------------------------------------------------------

/// Image returned by `acquire` that can be rendered to.
#[derive(Copy, Clone)]
pub struct RenderTargetImage<'a> {
    pub image: &'a gpu::Image,
}

/// Defines methods that are called when the event loop resumes.
#[allow(unused_variables)]
pub trait LoopHandler {
    /// Called when the event loop starts running.
    fn started(&mut self);
    /// Called when the event loop receives an input event.
    fn input(&mut self, window: WindowHandle, input_event: InputEvent);
    /// Called when the event loop receives a user event.
    fn event(&mut self, payload: UserEvent);
    /// Called when a window is resized.
    fn resized(&mut self, window: WindowHandle, width: u32, height: u32);
    /// Called on VSync events.
    fn vsync(&mut self);
    /// Called continuously when the event loop is idle.
    fn poll(&mut self);
    /// Called when the user requested to close a window.
    fn close_requested(&mut self, window: WindowHandle);
    /// Called when the application is exiting.
    fn exiting(&mut self);
}
