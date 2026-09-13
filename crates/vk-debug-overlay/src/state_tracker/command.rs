//! Command tracking.
use crate::Device;
use crate::event::EId;
use crate::helper::HasPrivateData;
use std::ffi::{CStr, CString};
use std::fmt::Formatter;
use std::{fmt, ptr, slice};
use vulkan::*;

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
    pub pipeline: VkPipeline,
    // In last resort, compare the command indices.
    pub cmd_idx: usize,
}

// Identifies a command by submission+command buffer+command index
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct CmdIdx {
    // Submission index (vkQueueSubmit)
    pub sub: u32,
    // Command buffer index (command buffer idx in VkSubmitInfo)
    pub cmd_buf: u32,
    // Command index (index of the command in the command buffer)
    pub cmd: u32,
}

impl fmt::Debug for CmdIdx {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "cmd:{}.{}.{}", self.sub, self.cmd_buf, self.cmd)
    }
}

/// Information about a command recorded in a command buffer (vkCmdSomething).
pub struct Command {
    // Event ID (should be relatively stable across frames).
    pub eid: EId,
    // Index of the command in submission order.
    pub idx: CmdIdx,
    pub cmd_buf: VkCommandBuffer,
    pub key: CmdKey,
    pub push: Vec<u8>,
    // Current descriptor heap ranges
    // NOTE: we don't care about the contents of the range, as they are driver-specific and not
    //       interpretable by tools. We use those ranges as keys
    pub resource_heap: VkDeviceAddressRangeEXT,
    pub sampler_heap: VkDeviceAddressRangeEXT,
}

pub struct CommandBufferData {
    pub name: String,
    pub commands: Vec<Command>,
    // Push data buffer, holds the last known push data
    pub push: Vec<u8>,
    pub graphics: VkPipeline,
    pub compute: VkPipeline,
    // Index of current render pass begin in commands array
    pub render_pass_begin: usize,
    // Color+depth formats of the last started render pass
    pub color_formats: Vec<VkFormat>,
    pub depth_format: VkFormat,
    // Debug region markers
    pub regions: Vec<CString>,
    pub resource_heap: VkDeviceAddressRangeEXT,
    pub sampler_heap: VkDeviceAddressRangeEXT,
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
            color_formats: vec![],
            depth_format: Default::default(),
            regions: vec![],
            resource_heap: VkDeviceAddressRangeKHR {..},
            sampler_heap: VkDeviceAddressRangeKHR {..},
        }
    }

    fn get_pipeline_for_command(&self, kind: &CmdKind) -> VkPipeline {
        match kind {
            CmdKind::DrawIndexed { .. }
            | CmdKind::Draw { .. }
            | CmdKind::DrawIndirect { .. }
            | CmdKind::DrawMeshTasks { .. } => self.graphics,
            CmdKind::Dispatch { .. } => self.compute,
        }
    }
}

impl HasPrivateData for VkCommandBuffer {
    type PrivateData = CommandBufferData;
}

/*
impl CommandBufferData {
    fn new() -> CommandBufferData {
        CommandBufferData {}
    }

    unsafe fn set(device: &DeviceHelper, cmdbuf: VkCommandBuffer) {
        device.set_private_data(cmdbuf, CommandBufferData::new());
    }

    unsafe fn get<'a>(device: &DeviceHelper, cmdbuf: VkCommandBuffer) -> &'a mut CommandBufferData {
        device.get_private_data(cmdbuf).unwrap().as_mut()
    }
}*/

impl Device {
    pub unsafe fn hook_allocate_command_buffers(
        &self,
        device: VkDevice,
        p_allocate_info: *const VkCommandBufferAllocateInfo,
        p_command_buffers: *mut VkCommandBuffer,
    ) -> VkResult {
        let result = self.AllocateCommandBuffers(device, p_allocate_info, p_command_buffers);
        if result < 0 {
            return result;
        }
        let command_buffers = slice::from_raw_parts(p_command_buffers, (*p_allocate_info).commandBufferCount as usize);
        for cmd_buf in command_buffers {
            self.set_private_data(*cmd_buf, CommandBufferData::new());
        }
        result
    }

    pub unsafe fn hook_free_command_buffers(
        &self,
        device: VkDevice,
        command_pool: VkCommandPool,
        command_buffer_count: u32,
        p_command_buffers: *const VkCommandBuffer,
    ) {
        let command_buffers = slice::from_raw_parts(p_command_buffers, command_buffer_count as usize);
        for cmd_buf in command_buffers {
            self.take_private_data(*cmd_buf);
        }
        self.FreeCommandBuffers(device, command_pool, command_buffer_count, p_command_buffers)
    }

    pub unsafe fn hook_cmd_begin_debug_utils_label(
        &self,
        command_buffer: VkCommandBuffer,
        p_label_info: *const VkDebugUtilsLabelEXT,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        let label: CString = CStr::from_ptr((*p_label_info).pLabelName).into();
        d.regions.push(label);
        self.CmdBeginDebugUtilsLabelEXT(command_buffer, p_label_info)
    }

    pub unsafe fn hook_cmd_end_debug_utils_label(&self, command_buffer: VkCommandBuffer) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.regions.pop();
        self.CmdEndDebugUtilsLabelEXT(command_buffer)
    }

    pub unsafe fn hook_begin_command_buffer(
        &self,
        command_buffer: VkCommandBuffer,
        p_begin_info: *const VkCommandBufferBeginInfo,
    ) -> VkResult {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.commands.clear();
        d.push.clear();
        self.BeginCommandBuffer(command_buffer, p_begin_info)
    }

    pub unsafe fn hook_end_command_buffer(&self, command_buffer: VkCommandBuffer) -> VkResult {
        self.EndCommandBuffer(command_buffer)
    }

    pub unsafe fn hook_cmd_push_data_ext(
        &self,
        command_buffer: VkCommandBuffer,
        p_push_data_info: *const VkPushDataInfoEXT,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        let offset = (*p_push_data_info).offset as usize;
        let size = (*p_push_data_info).data.size;
        let ptr = (*p_push_data_info).data.address;
        d.push.resize(offset + size, 0xCC);
        ptr::copy_nonoverlapping(ptr as *const u8, d.push.as_mut_ptr().add(offset), size);
        self.CmdPushDataEXT(command_buffer, p_push_data_info);
    }

    pub unsafe fn hook_cmd_bind_resource_heap_ext(
        &self,
        commandBuffer: VkCommandBuffer,
        pBindInfo: *const VkBindHeapInfoEXT,
    ) {
        let d = self.get_private_data_mut(commandBuffer).unwrap();
        d.resource_heap = (*pBindInfo).heapRange;
        self.CmdBindResourceHeapEXT(commandBuffer, pBindInfo);
    }

    pub unsafe fn hook_cmd_bind_sampler_heap_ext(
        &self,
        commandBuffer: VkCommandBuffer,
        pBindInfo: *const VkBindHeapInfoEXT,
    ) {
        let d = self.get_private_data_mut(commandBuffer).unwrap();
        d.sampler_heap = (*pBindInfo).heapRange;
        self.CmdBindSamplerHeapEXT(commandBuffer, pBindInfo);
    }

    pub unsafe fn hook_write_resource_descriptors_ext(
        &self,
        device: VkDevice,
        resourceCount: u32,
        pResources: *const VkResourceDescriptorInfoEXT,
        pDescriptors: *const VkHostAddressRangeEXT,
    ) -> VkResult {
        let result = self.WriteResourceDescriptorsEXT(device, resourceCount, pResources, pDescriptors);
        if result >= 0 {
            self.debugger.lock().write_resource_descriptors(self, resourceCount, pResources, pDescriptors);
        }
        result
    }

    pub unsafe fn hook_write_sampler_descriptors_ext(
        &self,
        device: VkDevice,
        samplerCount: u32,
        pSamplers: *const VkSamplerCreateInfo,
        pDescriptors: *const VkHostAddressRangeEXT,
    ) -> VkResult {
        let result = self.WriteSamplerDescriptorsEXT(device, samplerCount, pSamplers, pDescriptors);
        if result >= 0 {
            self.debugger.lock().write_sampler_descriptors(self, samplerCount, pSamplers, pDescriptors);
        }
        result
    }

    unsafe fn wrap_command<R>(&self, cmd_buf: VkCommandBuffer, kind: CmdKind, f: impl FnOnce(&Self) -> R) -> R {
        let d = self.get_private_data_mut(cmd_buf).unwrap();
        let id = d.commands.len();
        let pipeline = d.get_pipeline_for_command(&kind);
        let cmd_key = CmdKey { markers: "".to_string(), pipeline, cmd_idx: id };
        let r = f(self);
        let eid = self.get_command_eid(&d.regions, pipeline, &kind);
        d.commands.push(Command {
            eid,
            // Filled in vkQueueSubmit
            idx: CmdIdx::default(),
            cmd_buf,
            key: cmd_key,
            push: d.push.clone(),
            resource_heap: d.resource_heap,
            sampler_heap: d.sampler_heap,
        });
        r
    }

    pub unsafe fn hook_cmd_bind_pipeline(
        &self,
        command_buffer: VkCommandBuffer,
        pipeline_bind_point: VkPipelineBindPoint,
        pipeline: VkPipeline,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        match pipeline_bind_point {
            VK_PIPELINE_BIND_POINT_GRAPHICS => d.graphics = pipeline,
            VK_PIPELINE_BIND_POINT_COMPUTE => d.compute = pipeline,
            _ => panic!("invalid bind point"),
        };
        self.CmdBindPipeline(command_buffer, pipeline_bind_point, pipeline)
    }

    pub unsafe fn hook_cmd_draw(
        &self,
        command_buffer: VkCommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.wrap_command(
            command_buffer,
            CmdKind::Draw { first_vertex, vertex_count, first_instance, instance_count },
            |this| this.CmdDraw(command_buffer, vertex_count, instance_count, first_vertex, first_instance),
        );
    }

    pub unsafe fn hook_cmd_draw_indexed(
        &self,
        command_buffer: VkCommandBuffer,
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
                this.CmdDrawIndexed(
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
        command_buffer: VkCommandBuffer,
        buffer: VkBuffer,
        offset: VkDeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        self.wrap_command(command_buffer, CmdKind::DrawIndirect { draw_count, stride }, |this| {
            this.CmdDrawIndirect(command_buffer, buffer, offset, draw_count, stride)
        });
    }

    pub unsafe fn hook_cmd_dispatch(
        &self,
        command_buffer: VkCommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        self.wrap_command(command_buffer, CmdKind::Dispatch { group_count_x, group_count_y, group_count_z }, |this| {
            this.CmdDispatch(command_buffer, group_count_x, group_count_y, group_count_z)
        });
        let d = self.get_private_data_mut(command_buffer).unwrap();
        let n = d.commands.len() - 1;
        self.after_command(&d.commands[n]);
    }

    pub unsafe fn hook_cmd_begin_render_pass(
        &self,
        command_buffer: VkCommandBuffer,
        p_render_pass_begin: *const VkRenderPassBeginInfo,
        contents: VkSubpassContents,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.render_pass_begin = d.commands.len();
        self.CmdBeginRenderPass(command_buffer, p_render_pass_begin, contents);
    }

    pub unsafe fn hook_cmd_begin_render_pass2(
        &self,
        command_buffer: VkCommandBuffer,
        p_render_pass_begin: *const VkRenderPassBeginInfo,
        p_subpass_begin_info: *const VkSubpassBeginInfo,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.render_pass_begin = d.commands.len();
        self.CmdBeginRenderPass2(command_buffer, p_render_pass_begin, p_subpass_begin_info);
    }

    pub unsafe fn hook_cmd_begin_rendering(
        &self,
        command_buffer: VkCommandBuffer,
        p_rendering_info: *const VkRenderingInfo,
    ) {
        let d = self.get_private_data_mut(command_buffer).unwrap();
        d.render_pass_begin = d.commands.len();

        // record color+depth attachment formats for building command EIDs
        let color_attachments = slice::from_raw_parts(
            (*p_rendering_info).pColorAttachments,
            (*p_rendering_info).colorAttachmentCount as usize,
        );
        d.color_formats = color_attachments
            .iter()
            .map(|a| a.imageView)
            .map(|iv| self.get_private_data_ref(iv).unwrap().format)
            .collect();
        d.depth_format = if !(*p_rendering_info).pDepthAttachment.is_null() {
            self.get_private_data_ref((*(*p_rendering_info).pDepthAttachment).imageView).unwrap().format
        } else {
            VK_FORMAT_UNDEFINED
        };
        self.CmdBeginRendering(command_buffer, p_rendering_info);
    }

    unsafe fn end_rendering_common(&self, cmd_buf: VkCommandBuffer) {
        let d = self.get_private_data_mut(cmd_buf).unwrap();
        for cmd in &d.commands[d.render_pass_begin..] {
            self.after_command(cmd);
        }
    }

    pub unsafe fn hook_cmd_end_render_pass(&self, command_buffer: VkCommandBuffer) {
        self.end_rendering_common(command_buffer);
        self.CmdEndRenderPass(command_buffer);
    }

    pub unsafe fn hook_cmd_end_render_pass2(
        &self,
        command_buffer: VkCommandBuffer,
        p_subpass_end_info: *const VkSubpassEndInfo,
    ) {
        self.end_rendering_common(command_buffer);
        self.CmdEndRenderPass2(command_buffer, p_subpass_end_info);
    }

    pub unsafe fn hook_cmd_end_rendering(&self, command_buffer: VkCommandBuffer) {
        self.end_rendering_common(command_buffer);
        self.CmdEndRendering(command_buffer);
    }
}
