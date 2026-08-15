//! Command tracking.
use ash::vk;
use ash::vk::PFN_vkCmdDraw;

use crate::{ layer_fn};
use crate::helper::PrivateData;

#[derive(Copy, Clone)]
pub struct CommandBufferData {
}

impl PrivateData for CommandBufferData {
    type Handle = vk::CommandBuffer;
}

impl CommandBufferData {

}

layer_fn! {
    #[proc(PFN_vkCmdDraw)]
    fn layer_vkCmdDraw(
        command_buffer: vk::CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {

    }
}