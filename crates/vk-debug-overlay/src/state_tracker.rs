use crate::Device;
use ash::vk;
use ash::vk::Handle;
use std::ffi::CStr;

pub mod buffer;
pub mod command;
pub mod pipeline;
pub mod queue;
pub mod swapchain;
pub mod image;
pub mod memory;

impl Device {
    pub unsafe fn hook_set_debug_utils_object_name(
        &self,
        device: vk::Device,
        p_name_info: *const vk::DebugUtilsObjectNameInfoEXT<'_>,
    ) -> vk::Result {
        let handle = (*p_name_info).object_handle;
        let name = CStr::from_ptr((*p_name_info).p_object_name).to_string_lossy().into_owned();

        //eprintln!("setDebugUtilsObjectName {} {}", handle, name);

        match (*p_name_info).object_type {
            vk::ObjectType::COMMAND_BUFFER => {
                let cmd_buf = vk::CommandBuffer::from_raw(handle);
                self.get_private_data_mut(cmd_buf).unwrap().name = name;
            }
            vk::ObjectType::PIPELINE => {
                let pipeline = vk::Pipeline::from_raw(handle);
                self.get_private_data_mut(pipeline).unwrap().name = name;
            }
            vk::ObjectType::BUFFER => {
                let buffer = vk::Buffer::from_raw(handle);
                self.get_private_data_mut(buffer).unwrap().name = name;
            }
            vk::ObjectType::IMAGE => {
                let image = vk::Image::from_raw(handle);
                self.get_private_data_mut(image).unwrap().name = name;
            }
            vk::ObjectType::QUEUE => {
                // TODO
            }
            _ => {}
        }

        (self.ext_debug_utils.set_debug_utils_object_name_ext)(device, p_name_info)
    }
}
