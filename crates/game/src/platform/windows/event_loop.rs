use crate::input::{InputEvent, PointerButton};
use crate::platform::windows::key_code::key_event_to_key_code;
use crate::platform::windows::window::Window;
use crate::platform::windows::{Error, TimerEntry, WakeReason, Win32Platform};
use crate::platform::{EventToken, LoopHandler};
use keyboard_types::KeyboardEvent;
use std::sync::OnceLock;
use std::task::{RawWaker, RawWakerVTable, Waker};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::window::WindowId;

//------------------------
struct WinitAppHandler<'a> {
    this: &'static Win32Platform,
    inner: Box<dyn LoopHandler + 'a>,
    cursor_x: u32,
    cursor_y: u32,
    modifiers: keyboard_types::Modifiers,
}

impl<'a> WinitAppHandler<'a> {
    fn init_window(&self, event_loop: &ActiveEventLoop) -> Result<(), Error> {
        if self.this.window.borrow().is_some() {
            return Ok(()); // window already created
        }

        self.this.window.replace(Some(Window::new(
            event_loop,
            self.this.options.window_title,
            self.this.options.width,
            self.this.options.height,
        )?));

        // start sending vblank ticks to the event loop
        self.this.vsync_clock.start();

        Ok(())
    }

    fn update_modifiers(&mut self, key: &keyboard_types::Key, state: keyboard_types::KeyState) {
        // update modifiers
        if let keyboard_types::Key::Named(nk) = key {
            match (nk, state) {
                (keyboard_types::NamedKey::Shift, keyboard_types::KeyState::Down) => {
                    self.modifiers.insert(keyboard_types::Modifiers::SHIFT)
                }
                (keyboard_types::NamedKey::Shift, keyboard_types::KeyState::Up) => {
                    self.modifiers.remove(keyboard_types::Modifiers::SHIFT)
                }
                (keyboard_types::NamedKey::Control, keyboard_types::KeyState::Down) => {
                    self.modifiers.insert(keyboard_types::Modifiers::CONTROL)
                }
                (keyboard_types::NamedKey::Control, keyboard_types::KeyState::Up) => {
                    self.modifiers.remove(keyboard_types::Modifiers::CONTROL)
                }
                (keyboard_types::NamedKey::Alt, keyboard_types::KeyState::Down) => {
                    self.modifiers.insert(keyboard_types::Modifiers::ALT)
                }
                (keyboard_types::NamedKey::Alt, keyboard_types::KeyState::Up) => {
                    self.modifiers.remove(keyboard_types::Modifiers::ALT)
                }
                (keyboard_types::NamedKey::Meta, keyboard_types::KeyState::Down) => {
                    self.modifiers.insert(keyboard_types::Modifiers::META)
                }
                (keyboard_types::NamedKey::Meta, keyboard_types::KeyState::Up) => {
                    self.modifiers.remove(keyboard_types::Modifiers::META)
                }
                _ => {}
            }
        }
    }

    fn update_timers(&mut self) -> Option<TimerEntry> {
        // remove expired deadlines
        let now = Instant::now();
        let mut next = None;
        loop {
            next = self.this.timers.borrow_mut().pop();
            if let Some(TimerEntry { deadline, token }) = next {
                if deadline <= now {
                    self.inner.event(token);
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
            self.inner.poll();
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.init_window(event_loop).unwrap();
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, reason: WakeReason) {
        match reason {
            WakeReason::VSync => {
                //eprintln!("got vsync");
                self.inner.vsync();
            }
            WakeReason::Token(token) => {
                self.inner.event(token);
            }
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _window_id: WindowId, window_event: WindowEvent) {
        // handle window events
        let mut event = None;
        match window_event {
            WindowEvent::Resized(size) => {
                self.inner.resized(size.width, size.height);
            }
            WindowEvent::CloseRequested => {
                self.inner.close_requested();
            }
            WindowEvent::CursorMoved { position, device_id } => {
                self.cursor_x = position.x as u32;
                self.cursor_y = position.y as u32;
                event = Some(InputEvent::CursorMoved {
                    x: self.cursor_x,
                    y: self.cursor_y,
                });
            }
            WindowEvent::MouseInput {
                state,
                button,
                device_id,
                ..
            } => {
                let button = match button {
                    winit::event::MouseButton::Left => PointerButton::LEFT,
                    winit::event::MouseButton::Right => PointerButton::RIGHT,
                    winit::event::MouseButton::Middle => PointerButton::MIDDLE,
                    winit::event::MouseButton::Other(n) => PointerButton(n),
                    winit::event::MouseButton::Back => PointerButton::X1,
                    winit::event::MouseButton::Forward => PointerButton::X2,
                };

                let x = self.cursor_x;
                let y = self.cursor_y;

                match state {
                    winit::event::ElementState::Pressed => {
                        event = Some(InputEvent::PointerDown { button, x, y });
                    }
                    winit::event::ElementState::Released => {
                        event = Some(InputEvent::PointerUp { button, x, y });
                    }
                }
            }
            WindowEvent::KeyboardInput {
                device_id,
                event: ke,
                is_synthetic,
            } => {
                let (key, code) = key_event_to_key_code(&ke);
                let state = if ke.state == winit::event::ElementState::Pressed {
                    keyboard_types::KeyState::Down
                } else {
                    keyboard_types::KeyState::Up
                };
                self.update_modifiers(&key, state);

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
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => {
                        // TODO pass down lines or pixels
                        event = Some(InputEvent::MouseWheel { delta_x: x, delta_y: y });
                    }
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        event = Some(InputEvent::MouseWheel {
                            delta_x: pos.x as f32,
                            delta_y: pos.y as f32,
                        });
                    }
                }

            }
            _ => {}
        }

        if let Some(event) = event {
            // propagate input event to the input handler
            self.inner.input(event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.this.quit_requested.get() {
            self.this.vsync_clock.stop();
            event_loop.exit();
        }
    }
}

/// Proxy to wake the event loop from other threads.
pub(crate) static EVENT_LOOP_PROXY: OnceLock<EventLoopProxy<WakeReason>> = OnceLock::new();

pub fn wake_event_loop(token: EventToken) {
    EVENT_LOOP_PROXY
        .get()
        .unwrap()
        .send_event(WakeReason::Token(token))
        .unwrap()
}

fn main_loop_waker() -> Waker {
    static VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_: *const ()| -> RawWaker { RawWaker::new(std::ptr::null(), &VTABLE) },
        |_: *const ()| {
            wake_event_loop(EventToken::TASK);
        },
        |_: *const ()| {
            wake_event_loop(EventToken::TASK);
        },
        |_: *const ()| {},
    );
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

impl Win32Platform {
    pub(crate) fn run_event_loop(&'static self, mut handler: Box<dyn LoopHandler + '_>) {
        let event_loop = winit::event_loop::EventLoop::<WakeReason>::with_user_event()
            .build()
            .unwrap();
        EVENT_LOOP_PROXY
            .set(event_loop.create_proxy())
            .expect("main loop already initialized");
        event_loop
            .run_app(&mut WinitAppHandler {
                this: self,
                inner: handler,
                cursor_x: 0,
                cursor_y: 0,
                modifiers: Default::default(),
            })
            .unwrap();
    }
}
