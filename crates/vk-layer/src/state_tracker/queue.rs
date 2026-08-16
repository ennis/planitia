use crate::DeviceState;
use ash::vk;

impl DeviceState {
    pub unsafe fn hook_get_device_queue(
        &self,
        device: vk::Device,
        queue_family_index: u32,
        queue_index: u32,
        p_queue: *mut vk::Queue,
    ) {
        (self.fp_v1_0().get_device_queue)(device, queue_family_index, queue_index, p_queue);
    }
}
