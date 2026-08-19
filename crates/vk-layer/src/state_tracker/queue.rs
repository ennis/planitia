use crate::{DeviceState, Submission};
use ash::vk;
use std::slice::from_raw_parts;
use std::mem;

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

    pub unsafe fn hook_queue_submit(
        &self,
        queue: vk::Queue,
        submit_count: u32,
        p_submits: *const vk::SubmitInfo<'_>,
        fence: vk::Fence,
    ) -> vk::Result
    {

        let mut cmd = self.submissions.lock();
        let submits = from_raw_parts(p_submits, submit_count as usize);
        for submit in submits {
            if submit.command_buffer_count == 0 {
                continue;
            }
            let command_buffers = from_raw_parts(submit.p_command_buffers, submit.command_buffer_count as usize);
            for &cmd_buf in command_buffers {
                let private_data = self.get_private_data_mut(cmd_buf).unwrap();
                cmd.subs.push(Submission { cmd_buf, commands: mem::take(&mut private_data.commands) })
            }
        }

        (self.fp_v1_0().queue_submit)(queue, submit_count, p_submits, fence)
    }
}
