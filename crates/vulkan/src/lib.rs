//! Vulkan bindings generated from vk.xml.
#![feature(default_field_values)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub mod vk_layer;
pub mod vk_video;
pub mod vk_util;
#[cfg(feature = "format_info")]
pub mod vk_format_info;
mod macros;
mod generated;
mod entry;
mod types;
mod handle;

pub use types::*;
pub use generated::*;
pub use handle::*;


pub const fn vk_api_version_minor(version: u32) -> u32 {
    version >> 12 & 0x3ff
}

pub const fn vk_api_version_major(version: u32) -> u32 {
    version >> 22 & 0x7f
}

pub const fn vk_api_version_patch(version: u32) -> u32 {
    version & 0xfff
}

pub const fn vk_api_version_variant(version: u32) -> u32 {
    version >> 29
}

pub const fn vk_make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}