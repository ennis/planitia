#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]

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
pub mod pipeline_cache;
pub mod platform;
mod render_world;
mod shaders;
mod timer;
pub mod util;
mod world;
mod tweak;

//--- reexports ---
pub use {color, egui, gpu, math};
pub use tweak::*;

///////////////////////////////////////////////////////////////////

use crate::asset::AssetCache;

pub fn register_asset_directory() {
    // in development mode, load from the local gamelib/assets directory
    AssetCache::register_directory(concat!(env!("CARGO_MANIFEST_DIR"), "/../gamelib/assets"));
    // TODO in production mode, bundle with the executable
}
