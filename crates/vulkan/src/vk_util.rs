use std::ffi::CStr;
use crate::{VkResult, VkStructureType};

pub trait VkTaggedStructure {
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
