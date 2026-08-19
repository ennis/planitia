//! Command tracking.

use crate::helper::HasPrivateData;
use crate::DeviceState;
use ash::vk;
use ash::vk::Handle;
use std::{ptr, slice};
use vulkan_headers::vulkan::vulkan::{
    VkBindHeapInfoEXT, VkCommandBuffer, VkDevice, VkHostAddressRangeEXT, VkPushDataInfoEXT,
    VkResourceDescriptorInfoEXT, VkResult, VkSamplerCreateInfo,
};
use crate::bump::Alloc;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CmdKind {
    Draw { first_vertex: u32, vertex_count: u32, first_instance: u32, instance_count: u32 },
    DrawIndexed { index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32 },
    DrawIndirect { draw_count: u32, stride: u32 },
    DrawMeshTasks { group_count_x: u32, group_count_y: u32, group_count_z: u32 },
    Dispatch { group_count_x: u32, group_count_y: u32, group_count_z: u32 },
}

// Command key, used to match commands across frames.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CmdKey {
    // Give a high priority to matching command buffer markers
    pub markers: String,
    // Matching commands should have consistent pipelines.
    pub pipeline: vk::Pipeline,
    // Matching commands should have consistent draw or dispatch parameters.
    pub kind: CmdKind,
    // In last resort, compare the command indices.
    pub cmd_idx: usize,
}

pub struct Command {
    // Command index local to command buffer.
    pub id: usize,
    pub cmd_buf: vk::CommandBuffer,
    pub key: CmdKey,
    pub push: Vec<u8>,
    pub readback: Option<Alloc>,
}

pub struct CommandBufferData {
    pub name: String,
    pub commands: Vec<Command>,
    // Push data buffer, holds the last known push data
    pub push: Vec<u8>,
    pub graphics: vk::Pipeline,
    pub compute: vk::Pipeline,
    // Index of current render pass begin in commands array
    pub render_pass_begin: usize,
}

impl CommandBufferData {
    fn new() -> CommandBufferData {
        CommandBufferData {
            name: String::new(),
            commands: vec![],
            push: vec![],
            graphics: Default::default(),
            compute: Default::default(),
            render_pass_begin: 0,
        }
    }

    fn get_pipeline_for_command(&self, kind: &CmdKind) -> vk::Pipeline {
        match kind {
            CmdKind::DrawIndexed { .. }
            | CmdKind::Draw { .. }
            | CmdKind::DrawIndirect { .. }
            | CmdKind::DrawMeshTasks { .. } => self.graphics,
            CmdKind::Dispatch { .. } => self.compute,
        }
    }
}

impl HasPrivateData for vk::CommandBuffer {
    type PrivateData = CommandBufferData;
}

/*
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
}*/

impl DeviceState {
    pub unsafe fn hook_allocate_command_buffers(
        &self,
        device: vk::Device,
        p_allocate_info: *const vk::CommandBufferAllocateInfo,
        p_command_buffers: *mut vk::CommandBuffer,
    ) -> vk::Result {
        let result = (self.fp_v1_0().allocate_command_buffers)(device, p_allocate_info, p_command_buffers);
        if result != vk::Result::SUCCESS {
            return result;
        }
        let command_buffers =
            slice::from_raw_parts(p_command_buffers, (*p_allocate_info).command_buffer_count as usize);
        for cmd_buf in command_buffers {
            self.set_private_data(*cmd_buf, CommandBufferData::new());
        }
        vk::Result::SUCCESS
    }

    pub unsafe fn hook_free_command_buffers(
        &self,
        device: vk::Device,
        command_pool: vk::CommandPool,
        command_buffer_count: u32,
        p_command_buffers: *const vk::CommandBuffer,
    ) {
        let command_buffers = slice::from_raw_parts(p_command_buffers, command_buffer_count as usize);
        for cmd_buf in command_buffers {
            self.take_private_data(*cmd_buf);
        }
        (self.fp_v1_0().free_command_buffers)(device, command_pool, command_buffer_count, p_command_buffers)
    }

    pub unsafe fn hook_begin_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        p_begin_info: *const vk::CommandBufferBeginInfo,
    ) -> vk::Result {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.commands.clear();
        d.push.clear();
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
        let d = self.get_private_data_mut(vk::CommandBuffer::from_raw(command_buffer as u64)).unwrap();
        let offset = (*p_push_data_info).offset as usize;
        let size = (*p_push_data_info).data.size;
        let ptr = (*p_push_data_info).data.address;
        d.push.resize(offset + size, 0xCC);
        ptr::copy_nonoverlapping(ptr as *const u8, d.push.as_mut_ptr().add(offset), size);
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
        device: VkDevice,
        resourceCount: u32,
        pResources: *const VkResourceDescriptorInfoEXT,
        pDescriptors: *const VkHostAddressRangeEXT,
    ) -> VkResult {
        (self.ext_descriptor_heap.write_resource_descriptors)(device, resourceCount, pResources, pDescriptors)
    }

    pub unsafe fn hook_write_sampler_descriptors_ext(
        &self,
        device: VkDevice,
        samplerCount: u32,
        pSamplers: *const VkSamplerCreateInfo,
        pDescriptors: *const VkHostAddressRangeEXT,
    ) -> VkResult {
        (self.ext_descriptor_heap.write_sampler_descriptors)(device, samplerCount, pSamplers, pDescriptors)
    }

    unsafe fn wrap_command<R>(&self, cmd_buf: vk::CommandBuffer, kind: CmdKind, f: impl FnOnce(&Self) -> R) -> R {
        let d = self.get_private_data_mut(cmd_buf).unwrap();
        let id = d.commands.len();
        let pipeline = d.get_pipeline_for_command(&kind);

        let cmd_key = CmdKey { markers: "".to_string(), pipeline, kind: kind.clone(), cmd_idx: id };

        //self.handle_probes_before_cmd(cmd_buf, &cmd_key);
        let r = f(self);
        //self.handle_probes_after_cmd(cmd_buf, &cmd_key);

        d.commands.push(Command { id, cmd_buf, key: cmd_key, push: d.push.clone(), readback: None });
        r
    }

    pub unsafe fn hook_cmd_bind_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        match pipeline_bind_point {
            vk::PipelineBindPoint::GRAPHICS => d.graphics = pipeline,
            vk::PipelineBindPoint::COMPUTE => d.compute = pipeline,
            _ => panic!("invalid bind point"),
        };
        (self.fp_v1_0().cmd_bind_pipeline)(command_buffer, pipeline_bind_point, pipeline)
    }

    pub unsafe fn hook_cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.wrap_command(
            command_buffer,
            CmdKind::Draw { first_vertex, vertex_count, first_instance, instance_count },
            |this| {
                (this.fp_v1_0().cmd_draw)(command_buffer, vertex_count, instance_count, first_vertex, first_instance)
            },
        );
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
        self.wrap_command(
            command_buffer,
            CmdKind::DrawIndexed { first_index, index_count, vertex_offset, first_instance, instance_count },
            |this| {
                (this.fp_v1_0().cmd_draw_indexed)(
                    command_buffer,
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                )
            },
        );
    }

    pub unsafe fn hook_cmd_draw_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        self.wrap_command(command_buffer, CmdKind::DrawIndirect { draw_count, stride }, |this| {
            (this.fp_v1_0().cmd_draw_indirect)(command_buffer, buffer, offset, draw_count, stride)
        });
    }

    pub unsafe fn hook_cmd_dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        self.wrap_command(command_buffer, CmdKind::Dispatch { group_count_x, group_count_y, group_count_z }, |this| {
            (this.fp_v1_0().cmd_dispatch)(command_buffer, group_count_x, group_count_y, group_count_z)
        });
        let d = self.get_private_data_mut(command_buffer).unwrap();
        let n = d.commands.len() - 1;
        self.handle_probes_after_render_pass(command_buffer, &mut d.commands[n..]);
    }


    pub unsafe fn hook_cmd_begin_render_pass(
        &self,
        command_buffer: vk::CommandBuffer,
        p_render_pass_begin: *const vk::RenderPassBeginInfo,
        contents: vk::SubpassContents,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.render_pass_begin = d.commands.len();
        (self.fp_v1_0().cmd_begin_render_pass)(command_buffer, p_render_pass_begin, contents);
    }

    pub unsafe fn hook_cmd_begin_render_pass2(&self, command_buffer: vk::CommandBuffer, p_render_pass_begin: *const vk::RenderPassBeginInfo<'_>, p_subpass_begin_info: *const vk::SubpassBeginInfo<'_>) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.render_pass_begin = d.commands.len();
        (self.fp_v1_2().cmd_begin_render_pass2)(command_buffer, p_render_pass_begin, p_subpass_begin_info);
    }

    pub unsafe fn hook_cmd_begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        p_rendering_info: *const vk::RenderingInfo,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.render_pass_begin = d.commands.len();
        (self.fp_v1_3().cmd_begin_rendering)(command_buffer, p_rendering_info);
    }


    pub unsafe fn hook_cmd_end_render_pass(&self, command_buffer: vk::CommandBuffer) {
        (self.fp_v1_0().cmd_end_render_pass)(command_buffer);
        let d = self.get_private_data_mut(command_buffer).unwrap();
        self.handle_probes_after_render_pass(command_buffer, &mut d.commands[d.render_pass_begin..]);
    }

    pub unsafe fn hook_cmd_end_render_pass2(&self, command_buffer: vk::CommandBuffer, p_subpass_end_info: *const vk::SubpassEndInfo<'_>) {
        (self.fp_v1_2().cmd_end_render_pass2)(command_buffer, p_subpass_end_info);
        let d = self.get_private_data_mut(command_buffer).unwrap();
        self.handle_probes_after_render_pass(command_buffer, &mut d.commands[d.render_pass_begin..]);
    }

    pub unsafe fn hook_cmd_end_rendering(&self, command_buffer: vk::CommandBuffer) {
        (self.fp_v1_3().cmd_end_rendering)(command_buffer);
        let d = self.get_private_data_mut(command_buffer).unwrap();
        self.handle_probes_after_render_pass(command_buffer, &mut d.commands[d.render_pass_begin..]);
    }
}
