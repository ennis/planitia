//! Command tracking.
use crate::helper::{DeviceHelper, PrivateData};
use crate::DeviceState;
use ash::vk;
use vulkan_headers::vulkan::vulkan::{VkBindHeapInfoEXT, VkCommandBuffer, VkDevice, VkHostAddressRangeEXT, VkPushDataInfoEXT, VkResourceDescriptorInfoEXT, VkResult, VkSamplerCreateInfo};

#[derive(Copy, Clone)]
pub struct CommandBufferData {}

impl PrivateData for CommandBufferData {
    type Handle = vk::CommandBuffer;
}

impl CommandBufferData {
    fn new() -> CommandBufferData {
        CommandBufferData {}
    }

    unsafe fn set(device: &DeviceHelper, cmdbuf: vk::CommandBuffer) {
        device.set_private_data(cmdbuf, CommandBufferData::new());
    }

    unsafe fn get<'a>(device: &DeviceHelper, cmdbuf: vk::CommandBuffer) -> &'a mut CommandBufferData {
        device.get_private_data(cmdbuf).unwrap().as_mut()
    }
}

impl DeviceState {
    pub unsafe fn hook_allocate_command_buffers(
        &self,
        device: vk::Device,
        p_allocate_info: *const vk::CommandBufferAllocateInfo,
        p_command_buffers: *mut vk::CommandBuffer,
    ) -> vk::Result {
        (self.fp_v1_0().allocate_command_buffers)(device, p_allocate_info, p_command_buffers)
    }

    pub unsafe fn hook_free_command_buffers(
        &self,
        device: vk::Device,
        command_pool: vk::CommandPool,
        command_buffer_count: u32,
        p_command_buffers: *const vk::CommandBuffer,
    ) {
        (self.fp_v1_0().free_command_buffers)(device, command_pool, command_buffer_count, p_command_buffers)
    }

    pub unsafe fn hook_begin_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        p_begin_info: *const vk::CommandBufferBeginInfo,
    ) -> vk::Result {
        (self.fp_v1_0().begin_command_buffer)(command_buffer, p_begin_info)
    }

    pub unsafe fn hook_end_command_buffer(&self, command_buffer: vk::CommandBuffer) -> vk::Result {
        (self.fp_v1_0().end_command_buffer)(command_buffer)
    }

    pub unsafe fn hook_cmd_push_data_ext(
        &self,
        command_buffer: VkCommandBuffer,
        p_push_data_info: *const VkPushDataInfoEXT,
    ) {
        (self.ext_descriptor_heap.cmd_push_data)(command_buffer, p_push_data_info);
    }

    pub unsafe fn hook_cmd_bind_resource_heap_ext(
        &self,
        commandBuffer: VkCommandBuffer,
        pBindInfo: *const VkBindHeapInfoEXT,
    ) {
        (self.ext_descriptor_heap.cmd_bind_resource_heap)(commandBuffer, pBindInfo);
    }

    pub unsafe fn hook_cmd_bind_sampler_heap_ext(
        &self,
        commandBuffer: VkCommandBuffer,
        pBindInfo: *const VkBindHeapInfoEXT,
    ) {
        (self.ext_descriptor_heap.cmd_bind_sampler_heap)(commandBuffer, pBindInfo);
    }

    pub unsafe fn hook_write_resource_descriptors_ext(
        &self,
        device: VkDevice, resourceCount: u32, pResources: *const VkResourceDescriptorInfoEXT, pDescriptors: *const VkHostAddressRangeEXT
    ) -> VkResult {
        (self.ext_descriptor_heap.write_resource_descriptors)(device, resourceCount, pResources, pDescriptors)
    }

    pub unsafe fn hook_write_sampler_descriptors_ext(
        &self,
        device: VkDevice, samplerCount: u32, pSamplers: *const VkSamplerCreateInfo, pDescriptors: *const VkHostAddressRangeEXT
    ) -> VkResult {
        (self.ext_descriptor_heap.write_sampler_descriptors)(device, samplerCount, pSamplers, pDescriptors)
    }

    pub unsafe fn hook_cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        (self.fp_v1_0().cmd_draw)(command_buffer, vertex_count, instance_count, first_vertex, first_instance)
    }

    pub unsafe fn hook_cmd_draw_indexed(
        &self,
        command_buffer: vk::CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        (self.fp_v1_0().cmd_draw_indexed)(
            command_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )
    }

    pub unsafe fn hook_cmd_draw_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        (self.fp_v1_0().cmd_draw_indirect)(command_buffer, buffer, offset, draw_count, stride)
    }

    pub unsafe fn hook_cmd_dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        (self.fp_v1_0().cmd_dispatch)(command_buffer, group_count_x, group_count_y, group_count_z)
    }
}
