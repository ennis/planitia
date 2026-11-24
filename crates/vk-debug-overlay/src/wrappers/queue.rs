use crate::Device;
use std::slice::from_raw_parts;
use vulkan::*;

impl Device {
    pub unsafe fn hook_get_device_queue(
        &self,
        device: VkDevice,
        queue_family_index: u32,
        queue_index: u32,
        p_queue: *mut VkQueue,
    ) {
        self.GetDeviceQueue(device, queue_family_index, queue_index, p_queue);
    }

}
