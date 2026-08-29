use crate::Device;
use crate::surface::get_hwnd_for_surface;
use ash::vk;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{HWND, POINT};
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VIRTUAL_KEY, VK_CONTROL, VK_DOWN, VK_ESCAPE, VK_LBUTTON, VK_LEFT, VK_MBUTTON, VK_MENU,
    VK_RBUTTON, VK_RETURN, VK_RIGHT, VK_SHIFT, VK_SPACE, VK_TAB, VK_UP,
};
use windows::Win32::UI::WindowsAndMessaging::GetCursorPos;
use crate::overlay::gui::with_imgui_context;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum KeyEventKind {
    Press,
    Repeat,
    Release,
}

const NKEYS: usize = 11;

static KEY_MAP: [(imgui::Key, VIRTUAL_KEY); NKEYS] = [
    (imgui::Key::LeftArrow, VK_LEFT),
    (imgui::Key::RightArrow, VK_RIGHT),
    (imgui::Key::UpArrow, VK_UP),
    (imgui::Key::DownArrow, VK_DOWN),
    (imgui::Key::Enter, VK_RETURN),
    (imgui::Key::Escape, VK_ESCAPE),
    (imgui::Key::Tab, VK_TAB),
    (imgui::Key::Space, VK_SPACE),
    (imgui::Key::LeftCtrl, VK_CONTROL),
    (imgui::Key::LeftShift, VK_SHIFT),
    (imgui::Key::LeftAlt, VK_MENU),
    //(imgui::Key::MouseLeft, VK_LBUTTON),
    //(imgui::Key::MouseRight, VK_RBUTTON),
    //(imgui::Key::MouseMiddle, VK_MBUTTON),
];

static INITIAL_REPEAT_DELAY: Duration = Duration::from_millis(300);
static REPEAT_DELAY: Duration = Duration::from_millis(20);

type KeyboardState = [bool; NKEYS];

pub struct InputState {
    last_key_press: Instant,
    repeat_delay: Duration,
    keyb: KeyboardState,
    events: [Option<KeyEventKind>; NKEYS],
}

impl InputState {
    pub fn new() -> InputState {
        InputState {
            last_key_press: Instant::now(),
            repeat_delay: Default::default(),
            keyb: [false; NKEYS],
            events: [None; NKEYS],
        }
    }

    fn fetch(&mut self, hwnd: HWND) {
        let now = Instant::now();
        let duration_since_last_key_press = now - self.last_key_press;

        let mut keyb = [false; NKEYS];
        for i in 0..KEY_MAP.len() {
            unsafe {
                keyb[i] = GetAsyncKeyState(KEY_MAP[i].1.0 as i32) < 0;
            }

            match (self.keyb[i], keyb[i]) {
                (false, true) => {
                    self.events[i] = Some(KeyEventKind::Press);
                    self.last_key_press = now;
                    self.repeat_delay = INITIAL_REPEAT_DELAY;
                }
                (true, false) => {
                    self.events[i] = Some(KeyEventKind::Release);
                }
                (true, true) if duration_since_last_key_press >= self.repeat_delay => {
                    self.events[i] = Some(KeyEventKind::Repeat);
                    self.last_key_press = now;
                    self.repeat_delay = REPEAT_DELAY;
                }
                _ => {
                    self.events[i] = None;
                }
            }
        }
        self.keyb = keyb;

        // fetch cursor position relative to HWND
        let (cursor_x, cursor_y) = unsafe {
            let mut point = POINT::default();
            GetCursorPos(&mut point).unwrap();
            ScreenToClient(hwnd, &mut point).unwrap();
            eprintln!("Cursor position: ({}, {})", point.x, point.y);
            (point.x, point.y)
        };

        // fetch mouse button states
        let mouse_left;
        let mouse_right;
        let mouse_middle;
        unsafe {
            mouse_left = GetAsyncKeyState(VK_LBUTTON.0 as i32) < 0;
            mouse_right = GetAsyncKeyState(VK_RBUTTON.0 as i32) < 0;
            mouse_middle = GetAsyncKeyState(VK_MBUTTON.0 as i32) < 0;
        }

        with_imgui_context(|ctx| {
            let io = ctx.io_mut();
            io.mouse_pos = [cursor_x as f32, cursor_y as f32];
            io.mouse_down[0] = mouse_left;
            io.mouse_down[1] = mouse_right;
            io.mouse_down[2] = mouse_middle;
            for i in 0..NKEYS {
                if let Some(event) = self.events[i] {
                    match event {
                        KeyEventKind::Press | KeyEventKind::Repeat => {
                            io.add_key_event(KEY_MAP[i].0, true);
                        }
                        KeyEventKind::Release => {
                            io.add_key_event(KEY_MAP[i].0, false);
                        }
                    }
                }
            }
        });
    }
}

impl Device {
    pub fn update_inputs_for_surface(&self, surface: vk::SurfaceKHR) {
        let Some(hwnd) = get_hwnd_for_surface(surface) else {
            eprintln!("Failed to get HWND for surface {:?}", surface);
            return;
        };

        // Update inputs
        let mut inputs = self.input.lock();
        inputs.fetch(hwnd);
    }
}
