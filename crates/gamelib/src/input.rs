pub use keyboard_types::{Key, KeyState, KeyboardEvent, Location, Modifiers, NamedKey};
use std::fmt;
use std::str::FromStr;
use log::error;

/// Represents a pointer button.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PointerButton(pub u16);

impl PointerButton {
    pub const LEFT: PointerButton = PointerButton(0); // Or touch/pen contact
    pub const MIDDLE: PointerButton = PointerButton(1);
    pub const RIGHT: PointerButton = PointerButton(2); // Or pen barrel
    pub const X1: PointerButton = PointerButton(3);
    pub const X2: PointerButton = PointerButton(4);
    pub const ERASER: PointerButton = PointerButton(5);
}

/// The state of the mouse buttons.
// TODO why u no bitflags?
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct PointerButtons(pub u32);

impl PointerButtons {
    pub const ALL: PointerButtons = PointerButtons(0xFFFFFFFF);

    pub fn new() -> PointerButtons {
        PointerButtons(0)
    }

    pub fn with(self, button: PointerButton) -> Self {
        PointerButtons(self.0 | (1u32 << button.0 as u32))
    }

    /// Checks if the specified mouse button is pressed.
    pub fn test(self, button: PointerButton) -> bool {
        self.0 & (1u32 << button.0 as u32) != 0
    }
    pub fn set(&mut self, button: PointerButton) {
        self.0 |= 1u32 << button.0 as u32;
    }
    pub fn reset(&mut self, button: PointerButton) {
        self.0 &= !(1u32 << button.0 as u32);
    }
    pub fn intersects(&self, buttons: PointerButtons) -> bool {
        (self.0 & buttons.0) != 0
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Debug for PointerButtons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{")?;
        if self.test(PointerButton::LEFT) {
            write!(f, "LEFT")?;
        }
        if self.test(PointerButton::RIGHT) {
            write!(f, "RIGHT")?;
        }
        if self.test(PointerButton::MIDDLE) {
            write!(f, "MIDDLE")?;
        }
        if self.test(PointerButton::X1) {
            write!(f, "X1")?;
        }
        if self.test(PointerButton::X2) {
            write!(f, "X2")?;
        }
        write!(f, " +{:04x}", self.0)?;
        write!(f, "}}")?;
        Ok(())
    }
}

impl Default for PointerButtons {
    fn default() -> Self {
        PointerButtons::new()
    }
}

/// Represents an amount of scrolling from pointer wheel input.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MouseScrollDelta {
    /// Scroll amount in lines.
    LineDelta { x: f32, y: f32 },
    /// Scroll amount in pixels.
    PixelDelta { x: f32, y: f32 },
}

/// Represents an input event (mouse, keyboard, window resize).
#[derive(Clone, Debug, PartialEq)]
pub enum InputEvent {
    /// The cursor moved to the specified window coordinates.
    CursorMoved { x: u32, y: u32 },
    /// A pointer button was pressed.
    PointerDown { button: PointerButton, x: u32, y: u32 },
    /// A pointer button was released.
    PointerUp { button: PointerButton, x: u32, y: u32 },
    /// The mouse wheel was scrolled.
    MouseWheel(MouseScrollDelta),
    /// A key was pressed or released.
    KeyboardEvent(KeyboardEvent),
    /// The application window was resized.
    Resized { width: u32, height: u32 },
}

impl InputEvent {
    /// Returns whether the event matches the specified keyboard shortcut.
    ///
    /// Specifically, this looks for key down events (possibly repeated) that match the shortcut.
    pub fn is_shortcut<S>(&self, shortcut: S) -> bool where S: TryInto<Shortcut>, S::Error: fmt::Display {
        let shortcut = match shortcut.try_into() {
            Ok(s) => s,
            Err(err) => {
                panic!("{err}");
            }
        };
        match self {
            InputEvent::KeyboardEvent(ke) if ke.state == KeyState::Down => shortcut.matches(&ke.key, ke.modifiers),
            _ => false,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Represents the non-modifier key in a keyboard shortcut.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShortcutKey {
    /// The key is a character key.
    Character(char),
    /// The key is a named key that does not correspond to a character (e.g., Enter, Escape, function keys, etc.).
    Named(NamedKey),
}

/// Represents a keyboard shortcut, consisting of zero or more modifier keys and a single non-modifier key.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shortcut {
    pub modifiers: Modifiers,
    pub key: ShortcutKey,
}

impl Shortcut {
    /// Checks if the given key and modifiers match this shortcut.
    pub fn matches(&self, key: &Key, modifiers: Modifiers) -> bool {
        if self.modifiers == modifiers {
            match (key, &self.key) {
                (Key::Character(c1), ShortcutKey::Character(c2)) if c1.len() == 1 => {
                    let c1 = c1.chars().next().unwrap();
                    c1.eq_ignore_ascii_case(&c2)
                }
                (Key::Named(nk1), ShortcutKey::Named(nk2)) => nk1 == nk2,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Parses a keyboard shortcut string.
    ///
    /// # Example
    ///
    /// TODO
    pub fn parse(mut keys: &str) -> Result<Shortcut, ParseShortcutError> {
        let mut mods = Modifiers::empty();
        if let Some(rest) = keys.strip_prefix("Ctrl+") {
            keys = rest;
            mods |= Modifiers::CONTROL;
        }
        if let Some(rest) = keys.strip_prefix("Alt+") {
            keys = rest;
            mods |= Modifiers::ALT;
        }
        if let Some(rest) = keys.strip_prefix("Shift+") {
            keys = rest;
            mods |= Modifiers::SHIFT;
        }
        if let Some(rest) = keys.strip_prefix("Meta+") {
            keys = rest;
            mods |= Modifiers::META;
        }
        let key = if keys.len() == 1 {
            ShortcutKey::Character(keys.chars().next().unwrap())
        } else {
            match NamedKey::from_str(keys) {
                Ok(nk) => ShortcutKey::Named(nk),
                Err(_) => return Err(ParseShortcutError::UnrecognizedKey),
            }
        };
        Ok(Shortcut { modifiers: mods, key })
    }
}


#[derive(thiserror::Error, Debug)]
pub enum ParseShortcutError {
    #[error("invalid key in shortcut")]
    UnrecognizedKey,
}


impl FromStr for Shortcut {
    type Err = ParseShortcutError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Shortcut::parse(s)
    }
}

impl TryFrom<&str> for Shortcut {
    type Error = ParseShortcutError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Shortcut::parse(value)
    }
}