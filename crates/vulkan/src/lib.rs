//! Vulkan bindings generated from vk.xml.
#![feature(default_field_values)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
pub mod layer;
pub mod video;
mod macros;
mod platform_types;
mod generated;
mod entry;
mod basetypes;
mod handle;
mod wrapper;

pub use platform_types::*;
pub use basetypes::*;
pub use generated::*;
pub use handle::*;

use std::ffi::CStr;

pub trait TaggedStructure {
    const S_TYPE: VkStructureType;
}

#[cold]
pub(crate) fn proc_not_found(procname: &CStr) -> ! {
    panic!("vulkan entry point not found: `{}`", procname.to_string_lossy());
}

#[cold]
#[track_caller]
pub fn panic_vulkan_api_call_failed(result: VkResult) -> ! {
    panic!("Vulkan API call failed: {:?}", result);
}

pub(crate) const unsafe fn zero<T>() -> T {
    unsafe {
        std::mem::zeroed()
    }
}

pub fn vk_api_version_minor(version: u32) -> u32 {
    version >> 12 & 0x3ff
}

pub fn vk_api_version_major(version: u32) -> u32 {
    version >> 22 & 0x7f
}

pub fn vk_api_version_patch(version: u32) -> u32 {
    version & 0xfff
}

pub fn vk_api_version_variant(version: u32) -> u32 {
    version >> 29
}