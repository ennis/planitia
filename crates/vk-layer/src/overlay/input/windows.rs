use crate::overlay::input::{KeyCode, KeyEventKind};
use std::time::{Duration, Instant};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VIRTUAL_KEY, VK_CONTROL, VK_DOWN, VK_ESCAPE, VK_LEFT, VK_RETURN, VK_RIGHT, VK_UP,
};

const NKEYS: usize = KeyCode::KeyMax as usize;
static KEYS: [VIRTUAL_KEY; NKEYS] = [VK_LEFT, VK_RIGHT, VK_UP, VK_DOWN, VK_CONTROL, VK_RETURN, VK_ESCAPE];

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

    pub fn fetch_inputs(&mut self) {
        let now = Instant::now();
        let duration_since_last_key_press = now - self.last_key_press;

        let mut keyb = [false; NKEYS];
        for i in 0..NKEYS {
            unsafe {
                keyb[i] = GetAsyncKeyState(KEYS[i].0 as i32) < 0;
            }

            // duration since last key press
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
    }

    pub fn pressed(&self, key: KeyCode) -> bool {
        matches!(self.events[key as usize], Some(KeyEventKind::Press) | Some(KeyEventKind::Repeat))
    }

    pub fn released(&self, key: KeyCode) -> bool {
        self.events[key as usize] == Some(KeyEventKind::Release)
    }

    pub fn key_down(&self, key: KeyCode) -> bool {
        self.keyb[key as usize]
    }
}
