use math::IVec2;
use crate::app::{with_app_ctx, CURRENT_CTX};
use crate::input::PointerButtons;
use crate::platform::{PlatformWindowCreateInfo, WindowHandle};

/// Window input state.
pub struct WindowInputState {
    /// Currently active key modifiers.
    pub modifiers: keyboard_types::Modifiers,
    /// Last known cursor position in client coordinates.
    pub cursor_position: IVec2,
    /// Currently pressed pointer buttons.
    pub pointer_buttons: PointerButtons,
}

/// Window creation options.
#[derive(Debug, Clone, Copy)]
pub struct WindowCreateInfo<'a> {
    /// Initial client area width.
    pub width: u32 = 800,
    /// Initial client area height.
    pub height: u32 = 600,
    /// Window title.
    pub title: &'a str = "Window",
    /// Platform-specific options.
    pub platform: PlatformWindowCreateInfo = PlatformWindowCreateInfo { .. },
}


pub fn create_window(create_info: &WindowCreateInfo) -> WindowHandle {
    with_app_ctx(|ctx| {
        ctx.platform.create_window(create_info)
    })
}