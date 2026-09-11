//! Render command encoders
use crate::{
    Buffer, BufferUntyped, ClearColorValue, ColorAttachment, CommandBuffer, DepthBias, DepthStencilAttachment, Device,
    GraphicsPipeline, PrimitiveTopology, PushDataSource, Rect2D, is_depth_and_stencil_format,
};
use std::ops::Range;
use std::ptr;
use vulkan::*;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct RenderEncoder<'a> {
    parent: &'a mut CommandBuffer,
    render_area: VkRect2D,
}

/// Represents an indirect draw command.
// This must match the layout of `VkDrawIndirectCommand` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Represents an indirect draw command.
// This must match the layout of `VkDrawIndexedIndirectCommand` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}
const _: () = assert!(size_of::<DrawIndexedIndirectCommand>() == size_of::<VkDrawIndexedIndirectCommand>());

impl<'a> RenderEncoder<'a> {
    #[inline]
    pub fn set_depth_bias(&mut self, db: Option<DepthBias>) {
        let device = Device::instance();
        unsafe {
            match db {
                Some(db) => {
                    device.vk.CmdSetDepthBiasEnable(self.parent.cmdbuf, VK_TRUE);
                    device.vk.CmdSetDepthBias(self.parent.cmdbuf, db.constant_factor, db.clamp, db.slope_factor);
                }
                None => {
                    device.vk.CmdSetDepthBiasEnable(self.parent.cmdbuf, VK_FALSE);
                }
            }
        }
    }

    /// Binds a graphics pipeline.
    ///
    /// Calling this function invalidates all descriptor & push constant state set by previous calls
    /// to `push_descriptors`, `bind_descriptor_set`, and `push_constants`.
    #[inline]
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        // SAFETY: TBD, but the pipeline should live at least until the current frame has finished executing
        let device = Device::instance();
        unsafe {
            device.vk.CmdBindPipeline(self.parent.cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
        }
    }

    /// Sets the viewport.
    #[inline]
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        let device = Device::instance();
        unsafe {
            device.vk.CmdSetViewport(
                self.parent.cmdbuf,
                0,
                1,
                &VkViewport { x, y, width, height, minDepth: min_depth, maxDepth: max_depth },
            );
        }
    }

    #[inline]
    pub fn set_viewport_to_render_area(&mut self) {
        self.set_viewport(
            self.render_area.offset.x as f32,
            self.render_area.offset.y as f32,
            self.render_area.extent.width as f32,
            self.render_area.extent.height as f32,
            0.0,
            1.0,
        );
    }

    /// Sets the scissor rectangle.
    #[inline]
    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let device = Device::instance();
        unsafe {
            device.vk.CmdSetScissor(
                self.parent.cmdbuf,
                0,
                1,
                &VkRect2D { offset: VkOffset2D { x, y }, extent: VkExtent2D { width, height } },
            );
        }
    }

    #[inline]
    pub fn set_scissor_to_render_area(&mut self) {
        self.set_scissor(
            self.render_area.offset.x,
            self.render_area.offset.y,
            self.render_area.extent.width,
            self.render_area.extent.height,
        );
    }

    #[inline]
    pub fn clear_color(&mut self, attachment: u32, color: ClearColorValue) {
        self.clear_color_rect(
            attachment,
            color,
            Rect2D::from_xywh(0, 0, self.render_area.extent.width, self.render_area.extent.height),
        );
    }

    #[inline]
    pub fn clear_depth(&mut self, depth: f32) {
        self.clear_depth_rect(
            depth,
            Rect2D::from_xywh(0, 0, self.render_area.extent.width, self.render_area.extent.height),
        );
    }

    #[inline]
    pub fn clear_color_rect(&mut self, attachment: u32, color: ClearColorValue, rect: Rect2D) {
        let device = Device::instance();
        unsafe {
            device.vk.CmdClearAttachments(
                self.parent.cmdbuf,
                1,
                &VkClearAttachment {
                    aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                    colorAttachment: attachment,
                    clearValue: VkClearValue { color: color.into() },
                },
                1,
                &VkClearRect {
                    baseArrayLayer: 0,
                    layerCount: 1,
                    rect: VkRect2D {
                        offset: VkOffset2D { x: rect.min.x, y: rect.min.y },
                        extent: VkExtent2D { width: rect.width(), height: rect.height() },
                    },
                },
            );
        }
    }

    #[inline]
    pub fn clear_depth_rect(&mut self, depth: f32, rect: Rect2D) {
        let device = Device::instance();
        unsafe {
            device.vk.CmdClearAttachments(
                self.parent.cmdbuf,
                1,
                &VkClearAttachment {
                    aspectMask: VK_IMAGE_ASPECT_DEPTH_BIT,
                    colorAttachment: 0,
                    clearValue: VkClearValue { depthStencil: VkClearDepthStencilValue { depth, stencil: 0 } },
                },
                1,
                &VkClearRect {
                    baseArrayLayer: 0,
                    layerCount: 1,
                    rect: VkRect2D {
                        offset: VkOffset2D { x: rect.min.x, y: rect.min.y },
                        extent: VkExtent2D { width: rect.width(), height: rect.height() },
                    },
                },
            );
        }
    }

    /// Equivalent to [`draw(TriangleList, None, 0..6, 0..1, root_params)`](draw).
    ///
    /// To draw a screen-covering quad, use a vertex shader similar to this:
    /// ```
    /// ScreenQuadVSOut shader(uint vertex_id : SV_VertexID) {
    ///     float2 positions[6] = {
    ///         float2(-1.0, -1.0),
    ///         float2( 1.0, -1.0),
    ///         float2(-1.0,  1.0),
    ///         float2(-1.0,  1.0),
    ///         float2( 1.0, -1.0),
    ///         float2( 1.0,  1.0)
    ///     };
    ///     ScreenQuadVSOut o;
    ///     o.pos = float4(positions[vertex_id], 0.0, 1.0);
    ///     return o;
    /// }
    /// ```
    #[inline]
    pub fn draw_screen_quad<'params, T: Copy + 'static>(&mut self, root_params: impl Into<PushDataSource<'params, T>>) {
        self.draw(PrimitiveTopology::TriangleList, None, 0..6, 0..1, root_params);
    }

    /// Draws primitives.
    ///
    /// # Root parameters
    /// `root_params` specifies the uniforms passed to the shaders in [push constants](https://docs.vulkan.org/guide/latest/push_constants.html).
    /// The following types are supported (through implicit conversions to [`PushDataSource`]):
    /// * [`gpu::Ptr<T>`](gpu::Ptr) ([`PushDataSource::Indirect`]): a GPU pointer to an instance of `T`.
    ///   The 64-bit pointer is passed in the push constants (8 bytes).
    ///   The shader should expect a pointer in push constants..
    /// * `&T` ([`PushDataSource::IndirectUpload`]): reference to CPU data. The data is uploaded to a temporary GPU buffer and the
    ///   64-bit GPU pointer to that buffer is passed in the push constants (8 bytes).
    ///   The shader should expect a pointer in push constants.
    /// * [`ImmediatePushData<T>`](ImmediatePushData) ([`PushDataSource::Direct`]): the data is passed directly in the push constants
    ///  (`size_of<T>` bytes).
    ///
    /// # Examples
    ///
    /// TODO
    pub fn draw<'params, T: Copy + 'static>(
        &mut self,
        topology: PrimitiveTopology,
        vertex_buffer: Option<&BufferUntyped>,
        vertices: Range<u32>,
        instances: Range<u32>,
        root_params: impl Into<PushDataSource<'params, T>>,
    ) {
        unsafe {
            self.parent.set_push_data(self.parent.cmdbuf, root_params.into());
            let device = Device::instance();
            if let Some(vertex_buffer) = vertex_buffer {
                let buffers = [vertex_buffer.handle()];
                let offsets = [0];
                device.vk.CmdBindVertexBuffers(self.parent.cmdbuf, 0, 1, buffers.as_ptr(), offsets.as_ptr());
            }
            device.vk.CmdSetPrimitiveTopology(self.parent.cmdbuf, topology.to_vk_primitive_topology());
            device.vk.CmdDraw(
                self.parent.cmdbuf,
                vertices.len() as u32,
                instances.len() as u32,
                vertices.start,
                instances.start,
            );
        }
    }

    pub fn draw_indexed<'params, T: Copy + 'static>(
        &mut self,
        topology: PrimitiveTopology,
        index_buffer: &Buffer<u32>,
        index_range: Range<u32>,
        vertex_buffer: Option<&BufferUntyped>,
        base_vertex: i32,
        instances: Range<u32>,
        root_params: impl Into<PushDataSource<'params, T>>,
    ) {
        unsafe {
            self.parent.set_push_data(self.parent.cmdbuf, root_params.into());

            let device = Device::instance();
            if let Some(vertex_buffer) = vertex_buffer {
                let buffers = [vertex_buffer.handle()];
                let offsets = [0];
                device.vk.CmdBindVertexBuffers(self.parent.cmdbuf, 0, 1, buffers.as_ptr(), offsets.as_ptr());
            }
            device.vk.CmdBindIndexBuffer(self.parent.cmdbuf, index_buffer.handle(), 0, VK_INDEX_TYPE_UINT32);
            device.vk.CmdSetPrimitiveTopology(self.parent.cmdbuf, topology.to_vk_primitive_topology());
            device.vk.CmdDrawIndexed(
                self.parent.cmdbuf,
                index_range.len() as u32,
                instances.len() as u32,
                index_range.start,
                base_vertex,
                instances.start,
            );
        }
    }

    pub fn draw_indirect<'params, T: Copy + 'static>(
        &mut self,
        topology: PrimitiveTopology,
        vertex_buffer: Option<&BufferUntyped>,
        commands: &Buffer<DrawIndirectCommand>,
        draw_range: Range<u32>,
        root_params: impl Into<PushDataSource<'params, T>>,
    ) {
        unsafe {
            self.parent.set_push_data(self.parent.cmdbuf, root_params.into());
            let device = Device::instance();
            if let Some(vertex_buffer) = vertex_buffer {
                let buffers = [vertex_buffer.handle()];
                let offsets = [0];
                device.vk.CmdBindVertexBuffers(self.parent.cmdbuf, 0, 1, buffers.as_ptr(), offsets.as_ptr());
            }
            device.vk.CmdSetPrimitiveTopology(self.parent.cmdbuf, topology.to_vk_primitive_topology());
            device.vk.CmdDrawIndirect(
                self.parent.cmdbuf,
                commands.handle(),
                draw_range.start as u64 * size_of::<DrawIndirectCommand>() as u64,
                draw_range.len() as u32,
                size_of::<DrawIndirectCommand>() as u32,
            );
        }
    }

    pub fn draw_indexed_indirect<'params, T: Copy + 'static>(
        &mut self,
        topology: PrimitiveTopology,
        index_buffer: &Buffer<u32>,
        vertex_buffer: Option<&BufferUntyped>,
        commands: &Buffer<DrawIndexedIndirectCommand>,
        draw_range: Range<u32>,
        root_params: impl Into<PushDataSource<'params, T>>,
    ) {
        unsafe {
            self.parent.set_push_data(self.parent.cmdbuf, root_params.into());
            let device = Device::instance();
            if let Some(vertex_buffer) = vertex_buffer {
                let buffers = [vertex_buffer.handle()];
                let offsets = [0];
                device.vk.CmdBindVertexBuffers(self.parent.cmdbuf, 0, 1, buffers.as_ptr(), offsets.as_ptr());
            }
            device.vk.CmdBindIndexBuffer(self.parent.cmdbuf, index_buffer.handle(), 0, VK_INDEX_TYPE_UINT32);
            device.vk.CmdSetPrimitiveTopology(self.parent.cmdbuf, topology.to_vk_primitive_topology());
            device.vk.CmdDrawIndexedIndirect(
                self.parent.cmdbuf,
                commands.handle(),
                draw_range.start as u64 * size_of::<VkDrawIndexedIndirectCommand>() as u64,
                draw_range.len() as u32,
                size_of::<VkDrawIndexedIndirectCommand>() as u32,
            );
        }
    }

    #[inline]
    pub fn draw_mesh_tasks<'params, T: Copy + 'static>(
        &mut self,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        root_params: impl Into<PushDataSource<'params, T>>,
    ) {
        unsafe {
            let device = Device::instance();
            self.parent.set_push_data(self.parent.cmdbuf, root_params.into());
            device.ext.mesh_shader.CmdDrawMeshTasksEXT(self.parent.cmdbuf, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn finish(self) {
        // Nothing to do. Drop impl does the work (and calls `do_finish`).
    }

    #[inline]
    fn do_finish(&mut self) {
        unsafe {
            let device = Device::instance();
            device.vk.CmdEndRendering(self.parent.cmdbuf);
        }
    }
}

impl<'a> Drop for RenderEncoder<'a> {
    fn drop(&mut self) {
        self.do_finish();
    }
}

impl CommandBuffer {
    /// Starts a rendering pass.
    ///
    /// The render area is set to cover the entire size of the attachments.
    /// The initial viewport and scissor rects are set to cover the entire render area.
    ///
    /// # Arguments
    ///
    /// * `color_attachments` - The attachments to use for the render pass
    /// * `depth_stencil_attachment` - The depth-stencil attachment to use for the render pass.
    pub fn begin_rendering(
        &mut self,
        color_attachments: &[ColorAttachment],
        depth_stencil_attachment: Option<DepthStencilAttachment>,
    ) -> RenderEncoder<'_> {
        // determine render area
        let render_area = {
            // FIXME validate that all attachments have the same size
            // FIXME validate that all images are 2D
            let extent;
            if let Some(color) = color_attachments.first() {
                extent = color.image.size();
            } else if let Some(ref depth) = depth_stencil_attachment {
                extent = depth.image.size();
            } else {
                panic!("render_area must be specified if no attachments are specified");
            }
            VkRect2D {
                offset: VkOffset2D { x: 0, y: 0 },
                extent: VkExtent2D { width: extent.width, height: extent.height },
            }
        };

        // Begin render pass
        let color_attachment_infos: Vec<_> = color_attachments
            .iter()
            .map(|a| {
                VkRenderingAttachmentInfo {
                    imageView: a.image.attachment_view,
                    imageLayout: VK_IMAGE_LAYOUT_GENERAL,
                    resolveMode: VK_RESOLVE_MODE_NONE,
                    loadOp: if a.clear.is_some() { VK_ATTACHMENT_LOAD_OP_CLEAR } else { VK_ATTACHMENT_LOAD_OP_LOAD },
                    storeOp: VK_ATTACHMENT_STORE_OP_STORE,
                    clearValue: VkClearValue { color: a.get_vk_clear_color_value() },
                    // TODO multisampling resolve
                    ..
                }
            })
            .collect();
        let depth_attachment;
        let stencil_attachment;
        let p_depth_attachment;
        let p_stencil_attachment;
        if let Some(ref depth) = depth_stencil_attachment {
            depth_attachment = VkRenderingAttachmentInfo {
                imageView: depth.image.attachment_view,
                imageLayout: VK_IMAGE_LAYOUT_GENERAL,
                resolveMode: VK_RESOLVE_MODE_NONE,
                loadOp: if depth.depth_clear.is_some() {
                    VK_ATTACHMENT_LOAD_OP_CLEAR
                } else {
                    VK_ATTACHMENT_LOAD_OP_LOAD
                },
                storeOp: VK_ATTACHMENT_STORE_OP_STORE,
                clearValue: VkClearValue { depthStencil: depth.get_vk_clear_depth_stencil_value() },
                // TODO multisampling resolve
                ..
            };
            p_depth_attachment = &depth_attachment as *const _;
            if is_depth_and_stencil_format(depth.image.format()) {
                stencil_attachment = VkRenderingAttachmentInfo {
                    imageView: depth.image.attachment_view,
                    imageLayout: VK_IMAGE_LAYOUT_GENERAL,
                    resolveMode: VK_RESOLVE_MODE_NONE,
                    loadOp: if depth.stencil_clear.is_some() {
                        VK_ATTACHMENT_LOAD_OP_CLEAR
                    } else {
                        VK_ATTACHMENT_LOAD_OP_LOAD
                    },
                    storeOp: VK_ATTACHMENT_STORE_OP_STORE,
                    clearValue: VkClearValue { depthStencil: depth.get_vk_clear_depth_stencil_value() },
                    // TODO multisampling resolve
                    ..
                };
                p_stencil_attachment = &stencil_attachment as *const _;
            } else {
                p_stencil_attachment = ptr::null();
            }
        } else {
            p_depth_attachment = ptr::null();
            p_stencil_attachment = ptr::null();
        };

        let rendering_info = VkRenderingInfo {
            flags: 0,
            renderArea: render_area,
            layerCount: 1, // TODO?
            viewMask: 0,
            colorAttachmentCount: color_attachment_infos.len() as u32,
            pColorAttachments: color_attachment_infos.as_ptr(),
            pDepthAttachment: p_depth_attachment,
            pStencilAttachment: p_stencil_attachment,
            ..
        };
        unsafe {
            let device = Device::instance();
            device.vk.CmdBeginRendering(self.cmdbuf, &rendering_info);
        }

        let mut encoder = RenderEncoder { parent: self, render_area };
        encoder.set_viewport(0.0, 0.0, render_area.extent.width as f32, render_area.extent.height as f32, 0.0, 1.0);
        encoder.set_scissor(0, 0, render_area.extent.width, render_area.extent.height);
        encoder
    }
}
