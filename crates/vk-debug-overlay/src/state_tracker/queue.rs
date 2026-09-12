use crate::{Device, Submission};
use vulkan::*;
use std::mem;
use std::slice::from_raw_parts;

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
        let mut sbs = self.submissions.lock();
        let submits = from_raw_parts(p_submits, submit_count as usize);
        for submit in submits {
            if submit.commandBufferCount != 0 {
                let command_buffers = from_raw_parts(submit.pCommandBuffers, submit.commandBufferCount as usize);
                for (icb, &cmd_buf) in command_buffers.iter().enumerate() {
                    let private_data = self.get_private_data_mut(cmd_buf).unwrap();
                    let mut commands = mem::take(&mut private_data.commands);
                    for (i, cmd) in commands.iter_mut().enumerate() {
                        cmd.idx.sub = sbs.submission_count as u32;
                        cmd.idx.cmd_buf = icb as u32;
                        cmd.idx.cmd = i as u32;
                    }
                    sbs.subs.push(Submission { cmd_buf, commands })
                }
            }
            sbs.submission_count += 1;
        }
        self.QueueSubmit(queue, submit_count, p_submits, fence)
        // TODO: capture resource heaps here
    }
}
