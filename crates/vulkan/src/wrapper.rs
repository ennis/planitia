use crate::{
    PFN_vkVoidFunction, VkDevice, VkInstance, Vulkan_1_0_DeviceDispatch, Vulkan_1_0_InstanceDispatch,
    Vulkan_1_1_DeviceDispatch, Vulkan_1_1_InstanceDispatch, Vulkan_1_2_DeviceDispatch, Vulkan_1_3_DeviceDispatch,
    Vulkan_1_3_InstanceDispatch, Vulkan_1_4_DeviceDispatch,
};
use std::ffi::{CStr, c_void};

#[derive(Copy, Clone)]
pub struct VulkanDevice {
    pub device: VkDevice,
    pub fp_1_0: Vulkan_1_0_DeviceDispatch,
    pub fp_1_1: Vulkan_1_1_DeviceDispatch,
    pub fp_1_2: Vulkan_1_2_DeviceDispatch,
    pub fp_1_3: Vulkan_1_3_DeviceDispatch,
    pub fp_1_4: Vulkan_1_4_DeviceDispatch,
}

impl VulkanDevice {
    pub unsafe fn load_with(device: VkDevice, mut load_fn: impl FnMut(&CStr) -> PFN_vkVoidFunction) -> Self {
        unsafe {
            let fp_1_0 = Vulkan_1_0_DeviceDispatch::load_with(&mut load_fn);
            let fp_1_1 = Vulkan_1_1_DeviceDispatch::load_with(&mut load_fn);
            let fp_1_2 = Vulkan_1_2_DeviceDispatch::load_with(&mut load_fn);
            let fp_1_3 = Vulkan_1_3_DeviceDispatch::load_with(&mut load_fn);
            let fp_1_4 = Vulkan_1_4_DeviceDispatch::load_with(&mut load_fn);

            VulkanDevice { device, fp_1_0, fp_1_1, fp_1_2, fp_1_3, fp_1_4 }
        }
    }
}

#[derive(Copy, Clone)]
pub struct VulkanInstance {
    pub instance: VkInstance,
    pub fp_1_0: Vulkan_1_0_InstanceDispatch,
    pub fp_1_1: Vulkan_1_1_InstanceDispatch,
    pub fp_1_3: Vulkan_1_3_InstanceDispatch,
}

impl VulkanInstance {
    pub unsafe fn load_with(instance: VkInstance, mut load_fn: impl FnMut(&CStr) -> PFN_vkVoidFunction) -> Self {
        unsafe {
            let fp_1_0 = Vulkan_1_0_InstanceDispatch::load_with(&mut load_fn);
            let fp_1_1 = Vulkan_1_1_InstanceDispatch::load_with(&mut load_fn);
            let fp_1_3 = Vulkan_1_3_InstanceDispatch::load_with(&mut load_fn);

            VulkanInstance { instance, fp_1_0, fp_1_1, fp_1_3 }
        }
    }
}
