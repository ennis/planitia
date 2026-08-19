#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;

#[derive(Copy,Clone,Debug, Eq,PartialEq)]
pub enum KeyEventKind {
    Press,
    Repeat,
    Release,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KeyCode {
    KeyLeft = 0,
    KeyRight,
    KeyUp,
    KeyDown,
    KeyControl,
    KeyEnter,
    KeyEscape,
    KeyMax,
}
