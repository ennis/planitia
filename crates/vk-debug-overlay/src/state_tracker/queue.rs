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

    pub unsafe fn hook_queue_submit(
        &self,
        queue: VkQueue,
        submit_count: u32,
        p_submits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> VkResult {
        let mut dbg = self.debugger.lock();
        let submits = from_raw_parts(p_submits, submit_count as usize);
        for submit in submits {
            if submit.commandBufferCount != 0 {
                let command_buffers = from_raw_parts(submit.pCommandBuffers, submit.commandBufferCount as usize);
                dbg.queue_submit(self, queue, command_buffers);
            }
        }
        self.QueueSubmit(queue, submit_count, p_submits, fence)
    }
}
