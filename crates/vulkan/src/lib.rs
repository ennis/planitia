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

pub use platform_types::*;
pub use basetypes::*;
pub use generated::*;
pub use handle::*;

use std::ffi::CStr;

#[cold]
pub(crate) fn proc_not_found(procname: &CStr) -> ! {
    panic!("vulkan entry point not found: `{}`", procname.to_string_lossy());
}
