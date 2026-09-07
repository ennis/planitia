use crate::Device;
use vulkan::*;
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
        device: VkDevice,
        p_name_info: *const VkDebugUtilsObjectNameInfoEXT,
    ) -> VkResult {
        let handle = (*p_name_info).objectHandle;
        let name = CStr::from_ptr((*p_name_info).pObjectName).to_string_lossy().into_owned();

        //eprintln!("setDebugUtilsObjectName {} {}", handle, name);

        match (*p_name_info).objectType {
            VK_OBJECT_TYPE_COMMAND_BUFFER => {
                let cmd_buf = VkCommandBuffer(handle as *mut _);
                self.get_private_data_mut(cmd_buf).unwrap().name = name;
            }
            VK_OBJECT_TYPE_PIPELINE => {
                let pipeline = VkPipeline::from_raw(handle);
                self.get_private_data_mut(pipeline).unwrap().name = name;
            }
            VK_OBJECT_TYPE_BUFFER => {
                let buffer = VkBuffer::from_raw(handle);
                self.get_private_data_mut(buffer).unwrap().name = name;
            }
            VK_OBJECT_TYPE_IMAGE => {
                let image = VkImage::from_raw(handle);
                self.get_private_data_mut(image).unwrap().name = name;
            }
            VK_OBJECT_TYPE_QUEUE => {
                // TODO
            }
            _ => {}
        }

        (self.ext_debug_utils.set_debug_utils_object_name_ext)(device, p_name_info)
    }
}
