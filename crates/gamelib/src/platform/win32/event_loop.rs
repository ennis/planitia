//! Handles the winit event loop.

use crate::input::{InputEvent, MouseScrollDelta, PointerButton};
use crate::platform::win32::keys::key_event_to_key_code;
use crate::platform::win32::{Error, TimerEntry, WakeReason, Win32Platform};
use crate::platform::{LoopHandler, UserEvent};
use keyboard_types::KeyboardEvent;
use scoped_tls::scoped_thread_local;
use std::sync::OnceLock;
use std::task::{RawWaker, RawWakerVTable, Waker};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::window::WindowId;

// References to `ActiveEventLoop` are necessary to create windows and do other things;
// they are passed down to us via callbacks in `winit::ApplicationHandler` and winit expects the
// application to pass a `&ActiveEventLoop` down to functions that may create windows.
//
// While understandable from winit's perspective, this approach is too invasive
// and no self-respecting framework would do that, so we pass it around in a thread-local variable instead.
scoped_thread_local! {
    pub(super) static ACTIVE_EVENT_LOOP: ActiveEventLoop
}

fn update_keyboard_modifiers(
    modifiers: &mut keyboard_types::Modifiers,
    key: &keyboard_types::Key,
    state: keyboard_types::KeyState,
) {
    if let keyboard_types::Key::Named(nk) = key {
        match (nk, state) {
            (keyboard_types::NamedKey::Shift, keyboard_types::KeyState::Down) => {
                modifiers.insert(keyboard_types::Modifiers::SHIFT)
            }
            (keyboard_types::NamedKey::Shift, keyboard_types::KeyState::Up) => {
                modifiers.remove(keyboard_types::Modifiers::SHIFT)
            }
            (keyboard_types::NamedKey::Control, keyboard_types::KeyState::Down) => {
                modifiers.insert(keyboard_types::Modifiers::CONTROL)
            }
            (keyboard_types::NamedKey::Control, keyboard_types::KeyState::Up) => {
                modifiers.remove(keyboard_types::Modifiers::CONTROL)
            }
            (keyboard_types::NamedKey::Alt, keyboard_types::KeyState::Down) => {
                modifiers.insert(keyboard_types::Modifiers::ALT)
            }
            (keyboard_types::NamedKey::Alt, keyboard_types::KeyState::Up) => {
                modifiers.remove(keyboard_types::Modifiers::ALT)
            }
            (keyboard_types::NamedKey::Meta, keyboard_types::KeyState::Down) => {
                modifiers.insert(keyboard_types::Modifiers::META)
            }
            (keyboard_types::NamedKey::Meta, keyboard_types::KeyState::Up) => {
                modifiers.remove(keyboard_types::Modifiers::META)
            }
            _ => {}
        }
    }
}

//------------------------
struct WinitAppHandler<'a> {
    this: &'static Win32Platform,
    inner: &'a mut dyn LoopHandler,
    modifiers: keyboard_types::Modifiers,
}

impl<'a> WinitAppHandler<'a> {

    /// Maintains the list of active timers, firing events for all expired timers.
    ///
    /// For each expired timer, this sends an event to the handler with the corresponding token,
    /// and removes it from the list.
    ///
    /// # Return value
    ///
    /// The timer which should be the next to expire, if there's one.
    fn update_timers(&mut self) -> Option<TimerEntry> {
        let now = Instant::now();
        let mut next = None;
        loop {
            next = self.this.timers.borrow_mut().pop();
            if let Some(TimerEntry { deadline, token }) = next {
                if deadline <= now {
                    // TODO: timer callback instead of user event?
                    self.inner.event(UserEvent::Timeout(token));
                } else {
                    // Timer not expired, put it back and break
                    self.this.timers.borrow_mut().push(next.unwrap());
                    break;
                }
            } else {
                // No more timers to handle
                break;
            }
        }
        next
    }
}

impl<'a> ApplicationHandler<WakeReason> for WinitAppHandler<'a> {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        let next_timer = self.update_timers();
        event_loop.set_control_flow(if let Some(TimerEntry { deadline, .. }) = next_timer {
            ControlFlow::WaitUntil(deadline)
        } else {
            ControlFlow::Wait
        });
        if cause == StartCause::Poll {
            // explicit polling requested
            self.inner.poll();
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // `resumed` is only called once on desktop platforms, which are the only ones we care about

        // start the vsync clock to send vblank ticks to the event loop
        self.this.vsync_clock.start();

        // notify the application
        ACTIVE_EVENT_LOOP.set(event_loop, || {
            self.inner.started();
        });
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, reason: WakeReason) {
        match reason {
            WakeReason::VSync => {
                // request redraw on all windows
                let windows = self.this.windows.borrow_mut();
                for window in windows.values() {
                    window.inner.request_redraw();
                }
            }
            WakeReason::Task => {}
            WakeReason::User(event) => {
                ACTIVE_EVENT_LOOP.set(event_loop, || {
                    self.inner.event(event);
                });
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, window_event: WindowEvent) {
        let window_handle = self.this.find_window_by_id(window_id).expect("received event for unknown window");

        // translate winit window event to input event
        let mut event = None;

        match window_event {
            WindowEvent::Resized(size) => {
                // if resizing to zero, ignore; it's invalid to resize a swap chain to zero size
                if size.width == 0 || size.height == 0 {
                    return;
                }
                // resize swapchain
                let mut windows = self.this.windows.borrow_mut();
                let mut window = windows.get_mut(window_handle.key).unwrap();
                window.resize(size.width, size.height);
                self.inner.resized(window_handle, size.width, size.height);
            }
            WindowEvent::CloseRequested => {
                self.inner.close_requested(window_handle);
            }
            WindowEvent::CursorMoved { position, device_id } => {
                let mut windows = self.this.windows.borrow_mut();
                let mut window = windows.get_mut(window_handle.key).unwrap();
                window.input_state.cursor_position.x = position.x as i32;
                window.input_state.cursor_position.y = position.y as i32;
                event = Some(InputEvent::CursorMoved { x: position.x as i32, y: position.y as i32 });
            }
            WindowEvent::MouseInput { state, button, device_id, .. } => {
                let button = match button {
                    winit::event::MouseButton::Left => PointerButton::LEFT,
                    winit::event::MouseButton::Right => PointerButton::RIGHT,
                    winit::event::MouseButton::Middle => PointerButton::MIDDLE,
                    winit::event::MouseButton::Other(n) => PointerButton(n),
                    winit::event::MouseButton::Back => PointerButton::X1,
                    winit::event::MouseButton::Forward => PointerButton::X2,
                };

                let mut windows = self.this.windows.borrow_mut();
                let mut window = windows.get_mut(window_handle.key).unwrap();
                let x = window.input_state.cursor_position.x;
                let y = window.input_state.cursor_position.y;

                match state {
                    winit::event::ElementState::Pressed => {
                        event = Some(InputEvent::PointerDown { button, x, y });
                    }
                    winit::event::ElementState::Released => {
                        event = Some(InputEvent::PointerUp { button, x, y });
                    }
                }
            }
            WindowEvent::KeyboardInput { device_id, event: ke, is_synthetic } => {
                let (key, code) = key_event_to_key_code(&ke);
                let state = if ke.state == winit::event::ElementState::Pressed {
                    keyboard_types::KeyState::Down
                } else {
                    keyboard_types::KeyState::Up
                };

                update_keyboard_modifiers(&mut self.modifiers, &key, state);

                event = Some(InputEvent::KeyboardEvent(KeyboardEvent {
                    state,
                    key,
                    code,
                    location: Default::default(), // TODO
                    modifiers: self.modifiers,
                    repeat: false,
                    is_composing: false,
                }));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => MouseScrollDelta::LineDelta { x, y },
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        MouseScrollDelta::PixelDelta { x: pos.x as f32, y: pos.y as f32 }
                    }
                };

                event = Some(InputEvent::MouseWheel(delta));
            }
            WindowEvent::RedrawRequested => {
                ACTIVE_EVENT_LOOP.set(event_loop, || {
                    self.inner.vsync();
                });
                return;
            }
            _ => {}
        }

        ACTIVE_EVENT_LOOP.set(event_loop, || {
            // propagate input event to the input handler
            if let Some(event) = event {
                self.inner.input(window_handle, event);
            }
        });
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.this.quit_requested.get() {
            self.this.vsync_clock.stop();
            event_loop.exit();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        ACTIVE_EVENT_LOOP.set(_event_loop, || {
            self.inner.exiting();
        });
    }
}

/// Proxy to wake the event loop from other threads.
pub(crate) static EVENT_LOOP_PROXY: OnceLock<EventLoopProxy<WakeReason>> = OnceLock::new();

/// Wakes the event loop with the given user event.
pub fn wake_event_loop(callback: impl FnOnce() + Send + 'static) {
    EVENT_LOOP_PROXY.get().unwrap().send_event(WakeReason::User(UserEvent::Callback(Box::new(callback)))).unwrap()
}

fn main_loop_waker() -> Waker {
    static VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_: *const ()| -> RawWaker { RawWaker::new(std::ptr::null(), &VTABLE) },
        |_: *const ()| EVENT_LOOP_PROXY.get().unwrap().send_event(WakeReason::Task).unwrap(),
        |_: *const ()| EVENT_LOOP_PROXY.get().unwrap().send_event(WakeReason::Task).unwrap(),
        |_: *const ()| {},
    );
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

impl Win32Platform {
    pub(crate) fn run_event_loop(&'static self, mut handler: &mut dyn LoopHandler) {
        let event_loop = winit::event_loop::EventLoop::<WakeReason>::with_user_event().build().unwrap();
        EVENT_LOOP_PROXY.set(event_loop.create_proxy()).expect("main loop already initialized");
        event_loop
            .run_app(&mut WinitAppHandler {
                this: self,
                inner: handler,
                modifiers: Default::default(),
            })
            .unwrap();
    }
}
