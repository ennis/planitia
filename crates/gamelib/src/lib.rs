#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]
#![allow(unsafe_op_in_unsafe_fn, reason = "too verbose, and my IDE already highlights unsafe call sites")]

// debug! and error! macros are used frequently enough so that it's convenient to have them available
// without having to import them in every file.
#[macro_use]
extern crate log;

pub mod app;
pub mod asset;
pub mod camera_control;
mod component;
pub mod error;
mod event;
pub mod executor;
pub mod imgui;
pub mod input;
pub mod paint;
pub mod platform;
mod plugin_host;
pub mod render;
mod timer;
mod tweak;
pub mod util;
mod window;
mod world;

pub use app::{
    App, AppHandler, FileDialogOptions, pick_file, print_message, quit, render_imgui, show_file_dialog, unwatch_file,
    watch_file,
};
pub use event::UserEvent;
pub use input::InputEvent;
pub use platform::{WindowHandle, wake_event_loop};
pub use plugin_host::{PluginEvent, PluginResult, PluginState, register_plugin, reload_plugins};
pub use window::{WindowCreateInfo, WindowInputState, create_window};

//--- reexports ---
pub use color;
pub use egui;
pub use gpu;
pub use math;
pub use tracy_client;
pub use tweak::*;

///////////////////////////////////////////////////////////////////

use crate::asset::AssetCache;

/// Registers gamelib's asset directory with the `AssetCache`.
///
/// This should be called at the start of the program.
pub fn register_asset_directory() {
    // in development mode, load from the local gamelib/assets directory
    AssetCache::register_directory(concat!(env!("CARGO_MANIFEST_DIR"), "/../gamelib/assets"));
    // TODO in production mode, bundle with the executable
}

#[doc(hidden)]
pub fn tracy_create_span(location: &'static tracy_client::SpanLocation, callstack_depth: u16) -> tracy_client::Span {
    tracy_client::Client::running().expect("span! without a running Client").span(location, callstack_depth)
}

/// Wrapper for `tracy_client::span!`, to work around `https://github.com/rust-lang/rust/issues/65610`
#[macro_export]
macro_rules! span {
    () => {
        $crate::tracy_create_span($crate::tracy_client::span_location!(), 0)
    };
    ($name: expr) => {
        $crate::span!($name, 0)
    };
    ($name: expr, $callstack_depth: expr) => {{
        let location = $crate::tracy_client::span_location!($name);
        $crate::tracy_create_span(location, $callstack_depth)
    }};
}
