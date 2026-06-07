//! Input event types and keyboard shortcut parsing.

pub use keyboard_types::{Key, KeyState, KeyboardEvent, Location, Modifiers, NamedKey};
use log::error;
use std::fmt;
use std::str::FromStr;

/// Identifies a single pointer (mouse / pen / touch) button.
///
/// The inner `u16` is the zero-based button index as reported by the platform.
/// The well-known buttons are available as associated constants.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PointerButton(pub u16);

impl PointerButton {
    /// Primary mouse button, or touch/pen contact.
    pub const LEFT: PointerButton = PointerButton(0);
    /// Middle mouse button (wheel click).
    pub const MIDDLE: PointerButton = PointerButton(1);
    /// Secondary mouse button, or pen barrel button.
    pub const RIGHT: PointerButton = PointerButton(2);
    /// First extra (back) mouse button.
    pub const X1: PointerButton = PointerButton(3);
    /// Second extra (forward) mouse button.
    pub const X2: PointerButton = PointerButton(4);
    /// Pen eraser button.
    pub const ERASER: PointerButton = PointerButton(5);
}

/// Bitset recording which pointer buttons are currently pressed.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct PointerButtons(pub u32);

impl PointerButtons {
    /// A value with every button bit set.
    pub const ALL: PointerButtons = PointerButtons(0xFFFFFFFF);

    /// Returns a new [`PointerButtons`] with no buttons pressed.
    pub fn new() -> PointerButtons {
        PointerButtons(0)
    }

    /// Returns a copy of `self` with `button` added.
    pub fn with(self, button: PointerButton) -> Self {
        PointerButtons(self.0 | (1u32 << button.0 as u32))
    }

    /// Returns `true` if `button` is currently pressed.
    pub fn test(self, button: PointerButton) -> bool {
        self.0 & (1u32 << button.0 as u32) != 0
    }

    /// Marks `button` as pressed.
    pub fn set(&mut self, button: PointerButton) {
        self.0 |= 1u32 << button.0 as u32;
    }

    /// Marks `button` as released.
    pub fn reset(&mut self, button: PointerButton) {
        self.0 &= !(1u32 << button.0 as u32);
    }

    /// Returns `true` if any of the buttons in `buttons` are pressed.
    pub fn intersects(&self, buttons: PointerButtons) -> bool {
        (self.0 & buttons.0) != 0
    }

    /// Returns `true` if no buttons are pressed.
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

/// The amount of scrolling produced by a single mouse-wheel event.
///
/// Platforms may report scroll in either line or pixel units depending on the
/// input device and the OS.  Callers that need a uniform unit should apply
/// their own line-to-pixel conversion factor for [`LineDelta`](Self::LineDelta).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MouseScrollDelta {
    /// Scroll amount in logical lines.
    ///
    /// Positive `y` means scroll up / towards the user; positive `x` means
    /// scroll right.
    LineDelta { x: f32, y: f32 },
    /// Scroll amount in physical pixels.
    ///
    /// Positive `y` means scroll up / towards the user; positive `x` means
    /// scroll right.
    PixelDelta { x: f32, y: f32 },
}

/// Input event.
///
/// The event loop produces these events and routes them through
/// [`AppHandler::input`](crate::app::AppHandler::input).
///
/// # Coordinate system
///
/// Pointer coordinates (`x`, `y`) are in physical (device) pixels relative to
/// the top-left corner of the client area of the window.
#[derive(Clone, Debug, PartialEq)]
pub enum InputEvent {
    /// The pointer moved to the given client-area coordinates (in physical pixels).
    CursorMoved { x: i32, y: i32 },
    /// A pointer button was pressed at the given client-area coordinates.
    PointerDown { button: PointerButton, x: i32, y: i32 },
    /// A pointer button was released at the given client-area coordinates.
    PointerUp { button: PointerButton, x: i32, y: i32 },
    /// The mouse wheel (or trackpad) was scrolled.
    MouseWheel(MouseScrollDelta),
    /// A keyboard key was pressed, held (key-repeat), or released.
    KeyboardEvent(KeyboardEvent),
    /// The window client area was resized to the given dimensions in physical pixels.
    Resized { width: u32, height: u32 },
}

impl InputEvent {
    /// Returns `true` if this event is a key-down (or key-repeat) event that
    /// matches `shortcut`.
    ///
    /// `shortcut` can be any type that converts to [`Shortcut`], most
    /// conveniently a `&str` in the format accepted by [`Shortcut::parse`]
    /// (e.g.: `"Ctrl+S"`, `"Shift+F9"`, `"Escape"`).
    ///
    /// If `shortcut` cannot be parsed, the method returns `false`.
    ///
    /// # Character matching
    ///
    /// Character keys are compared case-insensitively, so `"Ctrl+s"` and `"Ctrl+S"` are equivalent.
    ///
    /// # Examples
    ///
    /// ```
    /// if event.is_shortcut("Ctrl+Z") { self.undo(); }
    /// if event.is_shortcut("F5")     { self.refresh(); }
    /// ```
    pub fn is_shortcut<S>(&self, shortcut: S) -> bool
    where
        S: TryInto<Shortcut>,
        S::Error: fmt::Display,
    {
        let shortcut = match shortcut.try_into() {
            Ok(s) => s,
            Err(err) => {
                error!("could not parse shortcut: {err}");
                return false;
            }
        };
        match self {
            InputEvent::KeyboardEvent(ke) if ke.state == KeyState::Down => shortcut.matches(&ke.key, ke.modifiers),
            _ => false,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// The non-modifier key portion of a [`Shortcut`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShortcutKey {
    /// A printable character key (e.g. `'s'`, `'/'`, `'1'`).
    Character(char),
    /// A named key that does not produce a character (e.g. `F5`, `Escape`, `Enter`, arrow keys).
    Named(NamedKey),
}

/// Represents a keyboard shortcut, consisting of zero or more modifier keys and a single non-modifier key.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shortcut {
    /// Modifiers (e.g. `Modifiers::CONTROL | Modifiers::SHIFT` for `Ctrl+Shift`).
    pub modifiers: Modifiers,
    /// The primary (non-modifier) key (e.g. "A").
    pub key: ShortcutKey,
}

impl Shortcut {
    /// Returns `true` if `key` and `modifiers` satisfy this shortcut.
    ///
    /// - [`ShortcutKey::Character`] keys are matched case-insensitively in the
    ///   ASCII range.  Only single-character strings are
    ///   considered; multi-character sequences are treated as no match.
    /// - [`ShortcutKey::Named`] keys require an exact [`NamedKey`] match.
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

    /// Parses a shortcut from a human-readable string.
    ///
    /// The expected format is:
    ///
    /// ```text
    /// [Modifier+]* Key
    /// ```
    ///
    /// where each `Modifier` is one of `Ctrl`, `Alt`, `Shift`, or `Meta`,
    /// and `Key` is either:
    /// - a single ASCII character (e.g. `S`, `1`, `/`), or
    /// - a named key string recognized by [`NamedKey`] (e.g. `F5`, `Escape`,
    ///   `Enter`, `ArrowUp`).
    ///
    /// Modifiers must appear in the fixed order `Ctrl` → `Alt` → `Shift` → `Meta`.
    /// Alternative orderings (e.g. `"Alt+Ctrl+S"`) are not accepted and
    /// will return an error.
    ///
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

/// Error returned when a shortcut string cannot be parsed.
#[derive(thiserror::Error, Debug)]
pub enum ParseShortcutError {
    #[error("unrecognised key in shortcut string")]
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
