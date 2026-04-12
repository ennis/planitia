#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]
#![allow(
    unsafe_op_in_unsafe_fn,
    reason = "too verbose, and my IDE already highlights unsafe call sites"
)]

pub mod asset;
pub mod camera_control;
mod component;
pub mod context;
mod event;
pub mod executor;
pub mod imgui;
pub mod input;
mod notifier;
pub mod paint;
pub mod platform;
pub mod render;
mod shaders;
mod timer;
mod tweak;
pub mod util;
mod world;

//--- reexports ---
pub use color;
pub use egui;
pub use gpu;
pub use math;
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
