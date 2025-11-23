#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]

pub mod camera_control;
pub mod context;
mod event;
pub mod executor;
pub mod imgui;
pub mod input;
mod notifier;
pub mod paint;
pub mod platform;
mod shaders;
mod timer;
pub mod util;
mod world;
pub mod asset;
pub mod pipeline_cache;

//--- reexports ---
pub use gpu;
pub use egui;
pub use math;
pub use color;

///////////////////////////////////////////////////////////////////

use crate::asset::AssetCache;

pub fn register_asset_directory() {
    // in development mode, load from the local gamelib/assets directory
    AssetCache::register_directory(concat!(env!("CARGO_MANIFEST_DIR"), "/../gamelib/assets"));
    // TODO in production mode, bundle with the executable
}