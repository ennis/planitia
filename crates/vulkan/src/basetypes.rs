
pub type VkSampleMask = u32;
pub type VkBool32 = u32;
pub type VkFlags = u32;
pub type VkFlags64 = u64;
pub type VkDeviceSize = u64;
pub type VkDeviceAddress = u64;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct VkResult(pub i32);

impl VkResult {
    #[track_caller]
    pub fn check(self) -> VkResult {
        if self.0 < 0 {
            panic_vulkan_api_call_failed(self);
        }
        self
    }
}

#[cold]
#[track_caller]
fn panic_vulkan_api_call_failed(result: VkResult) -> ! {
    panic!("Vulkan API call failed: {:?}", result);
}