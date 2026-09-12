pub type VkSampleMask = u32;
pub type VkBool32 = u32;
pub type VkFlags = u32;
pub type VkFlags64 = u64;
pub type VkDeviceSize = u64;
pub type VkDeviceAddress = u64;

/*
#[cold]
#[track_caller]
fn panic_vulkan_api_call_failed(result: VkResult) -> ! {
    panic!("Vulkan API call failed: {:?}", result);
}
*/