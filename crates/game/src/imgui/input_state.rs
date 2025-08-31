//! Glue code for input events.
//!
//! We still use egui for the immediate mode GUI, which means that we need glue to convert
//! keyboard events for consumption by egui.
//! Ultimately this means more chaff code, more bugs, more maintenance...
//!
//! Hopefully egui will switch to using `keyboard_types` directly in the future.
//! We might also move to our own immediate mode GUI in the future.

use crate::input::{InputEvent, PointerButton};
use arboard::Clipboard;
use egui::{OutputCommand, Pos2};
use keyboard_types::{Key, Modifiers, NamedKey};
use log::{error, warn};

pub struct EguiInputState {
    pub cursor_pos: Pos2,
    pub raw: egui::RawInput,
}

impl Default for EguiInputState {
    fn default() -> Self {
        Self {
            cursor_pos: Pos2::new(0.0, 0.0),
            raw: egui::RawInput::default(),
        }
    }
}

impl EguiInputState {
    fn update_modifiers(&mut self, modifiers: Modifiers) {
        self.raw.modifiers = egui::Modifiers {
            alt: modifiers.alt(),
            ctrl: modifiers.ctrl(),
            shift: modifiers.shift(),
            mac_cmd: modifiers.meta(),
            command: modifiers.meta(),
        };
    }

    pub(crate) fn handle_platform_output(&mut self, platform_output: &egui::PlatformOutput) {
        let egui::PlatformOutput {
            commands,
            cursor_icon,
            events,
            mutable_text_under_cursor,
            ime,
            num_completed_passes,
            request_discard_reasons,
            ..
        } = platform_output;

        for cmd in commands.iter() {
            match cmd {
                OutputCommand::CopyText(copied_text) => {}
                OutputCommand::CopyImage(_) => {}
                OutputCommand::OpenUrl(open_url) => {
                    warn!("unimplemented: open URL: {}", open_url.url);
                }
            }
        }

        if let Some(ime) = ime {
            //warn!("unimplemented: IME: {:?}", ime);
        }
    }

    pub fn update(&mut self, ctx: &egui::Context, input_event: &InputEvent) -> bool {
        fn convert_pointer_button(button: PointerButton) -> Option<egui::PointerButton> {
            match button {
                PointerButton::LEFT => Some(egui::PointerButton::Primary),
                PointerButton::MIDDLE => Some(egui::PointerButton::Middle),
                PointerButton::RIGHT => Some(egui::PointerButton::Secondary),
                PointerButton::X1 => Some(egui::PointerButton::Extra1),
                PointerButton::X2 => Some(egui::PointerButton::Extra2),
                _ => None,
            }
        }

        match input_event {
            InputEvent::CursorMoved { x, y } => {
                self.raw
                    .events
                    .push(egui::Event::PointerMoved(egui::pos2(*x as f32, *y as f32)));
                self.cursor_pos = egui::pos2(*x as f32, *y as f32);
                ctx.is_using_pointer()
            }
            InputEvent::PointerDown { button, x, y } => {
                self.raw.events.push(egui::Event::PointerButton {
                    pos: egui::pos2(*x as f32, *y as f32),
                    button: convert_pointer_button(*button).unwrap_or(egui::PointerButton::Primary),
                    pressed: true,
                    modifiers: self.raw.modifiers,
                });
                ctx.wants_pointer_input()
            }
            InputEvent::PointerUp { button, x, y } => {
                self.raw.events.push(egui::Event::PointerButton {
                    pos: egui::pos2(*x as f32, *y as f32),
                    button: convert_pointer_button(*button).unwrap_or(egui::PointerButton::Primary),
                    pressed: false,
                    modifiers: self.raw.modifiers,
                });
                ctx.wants_pointer_input()
            }
            InputEvent::KeyboardEvent(key) => {
                self.update_modifiers(key.modifiers);
                if let Some(event) = key_event_to_egui(key) {
                    self.raw.events.push(event);
                }
                if let Some(event) = key_event_to_egui_text(key) {
                    self.raw.events.push(event);
                }
                ctx.wants_keyboard_input()
            }
            &InputEvent::Resized { width, height } => {
                self.raw.screen_rect = Some(egui::Rect::from_min_size(
                    Pos2::new(0.0, 0.0),
                    egui::vec2(width as f32, height as f32),
                ));
                false
            }
        }
    }
}

fn key_event_to_egui_text(key: &keyboard_types::KeyboardEvent) -> Option<egui::Event> {
    if key.state != keyboard_types::KeyState::Down {
        return None;
    }

    // no text for control/alt keys
    if key.modifiers.ctrl() || key.modifiers.alt() || key.modifiers.meta() {
        return None;
    }

    match &key.key {
        Key::Character(ch) => Some(egui::Event::Text(ch.clone())),
        Key::Named(_) => None, // Named keys do not produce text events
    }
}

fn key_event_to_egui(key: &keyboard_types::KeyboardEvent) -> Option<egui::Event> {
    match &key.key {
        Key::Character(ch) => {
            let ctrl = key.modifiers.ctrl();
            let shift = key.modifiers.shift();
            let alt = key.modifiers.alt();
            let meta = key.modifiers.meta();

            let ctrl_only = ctrl && !shift && !alt && !meta;

            if ctrl_only {
                match ch.as_str() {
                    "c" => return Some(egui::Event::Copy),
                    "x" => return Some(egui::Event::Cut),
                    "v" => {
                        // fetch clipboard content
                        let mut clipboard = arboard::Clipboard::new().unwrap();
                        let text = clipboard.get_text().unwrap_or_default();
                        return Some(egui::Event::Paste(text));
                    }
                    _ => {
                        // If it's a control character, we don't want to generate an event
                        return None;
                    }
                }
            }

            let egui_key = match ch.as_str() {
                "a" | "A" => egui::Key::A,
                "b" | "B" => egui::Key::B,
                "c" | "C" => egui::Key::C,
                "d" | "D" => egui::Key::D,
                "e" | "E" => egui::Key::E,
                "f" | "F" => egui::Key::F,
                "g" | "G" => egui::Key::G,
                "h" | "H" => egui::Key::H,
                "i" | "I" => egui::Key::I,
                "j" | "J" => egui::Key::J,
                "k" | "K" => egui::Key::K,
                "l" | "L" => egui::Key::L,
                "m" | "M" => egui::Key::M,
                "n" | "N" => egui::Key::N,
                "o" | "O" => egui::Key::O,
                "p" | "P" => egui::Key::P,
                "q" | "Q" => egui::Key::Q,
                "r" | "R" => egui::Key::R,
                "s" | "S" => egui::Key::S,
                "t" | "T" => egui::Key::T,
                "u" | "U" => egui::Key::U,
                "v" | "V" => egui::Key::V,
                "w" | "W" => egui::Key::W,
                "x" | "X" => egui::Key::X,
                "y" | "Y" => egui::Key::Y,
                "z" | "Z" => egui::Key::Z,
                "0" => egui::Key::Num0,
                "1" => egui::Key::Num1,
                "2" => egui::Key::Num2,
                "3" => egui::Key::Num3,
                "4" => egui::Key::Num4,
                "5" => egui::Key::Num5,
                "6" => egui::Key::Num6,
                "7" => egui::Key::Num7,
                "8" => egui::Key::Num8,
                ":" => egui::Key::Colon,
                "," => egui::Key::Comma,
                "-" => egui::Key::Minus,
                "." => egui::Key::Period,
                "+" => egui::Key::Plus,
                "=" => egui::Key::Equals,
                ";" => egui::Key::Semicolon,
                "\\" => egui::Key::Backslash,
                "[" => egui::Key::OpenBracket,
                "]" => egui::Key::CloseBracket,
                "`" => egui::Key::Backtick,
                _ => {
                    // If it's not a recognized character, we don't want to generate an event
                    return None;
                }
            };
            Some(egui::Event::Key {
                key: egui_key,
                physical_key: None,
                pressed: key.state == keyboard_types::KeyState::Down,
                repeat: false,
                modifiers: egui::Modifiers {
                    alt,
                    ctrl,
                    shift,
                    mac_cmd: meta,
                    command: meta,
                },
            })
        }

        Key::Named(named_key) => {
            let egui_named_key = match named_key {
                NamedKey::Enter => egui::Key::Enter,
                NamedKey::Tab => egui::Key::Tab,
                NamedKey::ArrowDown => egui::Key::ArrowDown,
                NamedKey::ArrowLeft => egui::Key::ArrowLeft,
                NamedKey::ArrowRight => egui::Key::ArrowRight,
                NamedKey::ArrowUp => egui::Key::ArrowUp,
                NamedKey::End => egui::Key::End,
                NamedKey::Home => egui::Key::Home,
                NamedKey::PageDown => egui::Key::PageDown,
                NamedKey::PageUp => egui::Key::PageUp,
                NamedKey::Backspace => egui::Key::Backspace,
                NamedKey::Copy => egui::Key::Copy,
                NamedKey::Cut => egui::Key::Cut,
                NamedKey::Delete => egui::Key::Delete,
                NamedKey::Insert => egui::Key::Insert,
                NamedKey::Paste => egui::Key::Paste,
                NamedKey::Escape => egui::Key::Escape,
                NamedKey::F1 => egui::Key::F1,
                NamedKey::F2 => egui::Key::F2,
                NamedKey::F3 => egui::Key::F3,
                NamedKey::F4 => egui::Key::F4,
                NamedKey::F5 => egui::Key::F5,
                NamedKey::F6 => egui::Key::F6,
                NamedKey::F7 => egui::Key::F7,
                NamedKey::F8 => egui::Key::F8,
                NamedKey::F9 => egui::Key::F9,
                NamedKey::F10 => egui::Key::F10,
                NamedKey::F11 => egui::Key::F11,
                NamedKey::F12 => egui::Key::F12,
                NamedKey::F13 => egui::Key::F13,
                NamedKey::F14 => egui::Key::F14,
                NamedKey::F15 => egui::Key::F15,
                NamedKey::F16 => egui::Key::F16,
                NamedKey::F17 => egui::Key::F17,
                NamedKey::F18 => egui::Key::F18,
                NamedKey::F19 => egui::Key::F19,
                NamedKey::F20 => egui::Key::F20,
                _ => return None, // Unsupported named key
            };
            Some(egui::Event::Key {
                key: egui_named_key,
                physical_key: None,
                pressed: key.state == keyboard_types::KeyState::Down,
                repeat: false,
                modifiers: egui::Modifiers {
                    alt: key.modifiers.alt(),
                    ctrl: key.modifiers.ctrl(),
                    shift: key.modifiers.shift(),
                    mac_cmd: key.modifiers.meta(),
                    command: key.modifiers.meta(),
                },
            })
        }
    }
}
