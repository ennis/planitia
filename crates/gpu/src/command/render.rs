//! Render command encoders
use crate::{
    Buffer, BufferUntyped, ClearColorValue, ColorAttachment, CommandBuffer, DepthBias, DepthStencilAttachment, Device,
    GraphicsPipeline, PrimitiveTopology, Ptr, PushDataSource, Rect2D, is_depth_and_stencil_format,
};
use ash::vk;
use std::ops::Range;
use std::ptr;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct RenderEncoder<'a> {
    parent: &'a mut CommandBuffer,
    render_area: vk::Rect2D,
}

/// Represents an indirect draw command.
// This must match the layout of `vk::DrawIndirectCommand` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Represents an indirect draw command.
// This must match the layout of `vk::DrawIndexedIndirectCommand` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}
const _: () = assert!(size_of::<DrawIndexedIndirectCommand>() == size_of::<vk::DrawIndexedIndirectCommand>());

impl<'a> RenderEncoder<'a> {
    /*/// Binds a descriptor set (`vkCmdBindDescriptorSets`).
    ///
    /// # Safety
    ///
    /// The caller is responsible for ensuring that the descriptor set is compatible with the
    /// currently bound pipeline, and that the descriptor set is not destroyed while it is still
    /// in use by the GPU.
    #[deprecated = "use descriptor heaps and push data instead"]
    pub unsafe fn bind_descriptor_set(&mut self, index: u32, set: vk::DescriptorSet) {
        Device::instance().raw.cmd_bind_descriptor_sets(
            self.parent.cmdbuf,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            index,
            &[set],
            &[],
        )
    }*/

    /*
    /// Specifies descriptors for subsequent draw calls with `vkCmdPushDescriptorSetKHR`.
    pub fn push_descriptors(&mut self, set: u32, bindings: &[(u32, Descriptor)]) {
        assert!(
            self.pipeline_layout != vk::PipelineLayout::null(),
            "encoder must have a pipeline bound before binding arguments"
        );

        unsafe {
            self.stream.do_cmd_push_descriptor_set(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                set,
                bindings,
            );
        }
    }*/

    #[inline]
    pub fn set_depth_bias(&mut self, db: Option<DepthBias>) {
        let device = &Device::instance().raw;
        unsafe {
            match db {
                Some(db) => {
                    device.cmd_set_depth_bias_enable(self.parent.cmdbuf, true);
                    device.cmd_set_depth_bias(self.parent.cmdbuf, db.constant_factor, db.clamp, db.slope_factor);
                }
                None => {
                    device.cmd_set_depth_bias_enable(self.parent.cmdbuf, false);
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
        // Note about pipeline compatibility:
        //
        // Calling CmdBindPipeline doesn't really invalidate descriptor sets or push constants,
        // but they are only valid for this pipeline if its layout is "compatible" with the layout
        // used previously.
        // There is a notion of "partial compatibility", in which the first N descriptor set bindings
        // stay valid if the pipeline layouts have the same N first descriptor set layouts.
        // However, partial compatibility requires that layouts have the *same push constants ranges*
        // which is far too restrictive for our use cases
        // (bindless, with pass-specific parameters in push constants).
        //
        // So, don't bother with this insanity and rebind everything between pipeline changes.
        // Hopefully vkCmdBindDescriptorSets is cheap enough. I'm pretty sure it doesn't do much
        // if the sets are already bound
        // (for reference, see https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/vulkan/nvk_cmd_buffer.c?ref_type=heads#L648)

        // SAFETY: TBD
        // TODO strong ref to pipeline
        unsafe {
            Device::instance().raw.cmd_bind_pipeline(
                self.parent.cmdbuf,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
            /*if pipeline.bindless {
                self.stream.bind_bindless_descriptor_sets(
                    self.parent.cmdbuf,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline_layout,
                );
            }
            self.pipeline_layout = pipeline.pipeline_layout;*/
        }
    }

    /// Sets the viewport.
    #[inline]
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        unsafe {
            Device::instance().raw.cmd_set_viewport(
                self.parent.cmdbuf,
                0,
                &[vk::Viewport { x, y, width, height, min_depth, max_depth }],
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
        unsafe {
            Device::instance().raw.cmd_set_scissor(
                self.parent.cmdbuf,
                0,
                &[vk::Rect2D { offset: vk::Offset2D { x, y }, extent: vk::Extent2D { width, height } }],
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
        unsafe {
            Device::instance().raw.cmd_clear_attachments(
                self.parent.cmdbuf,
                &[vk::ClearAttachment {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    color_attachment: attachment,
                    clear_value: vk::ClearValue { color: color.into() },
                }],
                &[vk::ClearRect {
                    base_array_layer: 0,
                    layer_count: 1,
                    rect: vk::Rect2D {
                        offset: vk::Offset2D { x: rect.min.x, y: rect.min.y },
                        extent: vk::Extent2D { width: rect.width(), height: rect.height() },
                    },
                }],
            );
        }
    }

    #[inline]
    pub fn clear_depth_rect(&mut self, depth: f32, rect: Rect2D) {
        unsafe {
            Device::instance().raw.cmd_clear_attachments(
                self.parent.cmdbuf,
                &[vk::ClearAttachment {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    color_attachment: 0,
                    clear_value: vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 } },
                }],
                &[vk::ClearRect {
                    base_array_layer: 0,
                    layer_count: 1,
                    rect: vk::Rect2D {
                        offset: vk::Offset2D { x: rect.min.x, y: rect.min.y },
                        extent: vk::Extent2D { width: rect.width(), height: rect.height() },
                    },
                }],
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
            let device = &Device::instance().raw;
            if let Some(vb) = vertex_buffer {
                device.cmd_bind_vertex_buffers(self.parent.cmdbuf, 0, &[vb.handle()], &[0]);
            }
            device.cmd_set_primitive_topology(self.parent.cmdbuf, topology.to_vk_primitive_topology());
            device.cmd_draw(
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

            let device = &Device::instance().raw;
            if let Some(vb) = vertex_buffer {
                device.cmd_bind_vertex_buffers(self.parent.cmdbuf, 0, &[vb.handle()], &[0]);
            }
            device.cmd_bind_index_buffer(self.parent.cmdbuf, index_buffer.handle(), 0, vk::IndexType::UINT32);
            device.cmd_set_primitive_topology(self.parent.cmdbuf, topology.to_vk_primitive_topology());
            device.cmd_draw_indexed(
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
            let device = &Device::instance().raw;
            if let Some(vb) = vertex_buffer {
                device.cmd_bind_vertex_buffers(self.parent.cmdbuf, 0, &[vb.handle()], &[0]);
            }
            device.cmd_set_primitive_topology(self.parent.cmdbuf, topology.to_vk_primitive_topology());
            device.cmd_draw_indirect(
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
            let device = &Device::instance().raw;
            if let Some(vb) = vertex_buffer {
                device.cmd_bind_vertex_buffers(self.parent.cmdbuf, 0, &[vb.handle()], &[0]);
            }
            device.cmd_bind_index_buffer(self.parent.cmdbuf, index_buffer.handle(), 0, vk::IndexType::UINT32);
            device.cmd_set_primitive_topology(self.parent.cmdbuf, topology.to_vk_primitive_topology());
            device.cmd_draw_indexed_indirect(
                self.parent.cmdbuf,
                commands.handle(),
                draw_range.start as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64,
                draw_range.len() as u32,
                size_of::<vk::DrawIndexedIndirectCommand>() as u32,
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
            self.parent.set_push_data(self.parent.cmdbuf, root_params.into());
            Device::instance().ext.mesh_shader.cmd_draw_mesh_tasks(
                self.parent.cmdbuf,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }

    pub fn finish(self) {
        // Nothing to do. Drop impl does the work (and calls `do_finish`).
    }

    #[inline]
    fn do_finish(&mut self) {
        unsafe {
            Device::instance().raw.cmd_end_rendering(self.parent.cmdbuf);
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
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: extent.width, height: extent.height },
            }
        };

        // Begin render pass
        let color_attachment_infos: Vec<_> = color_attachments
            .iter()
            .map(|a| {
                vk::RenderingAttachmentInfo {
                    image_view: a.image.attachment_view,
                    image_layout: vk::ImageLayout::GENERAL,
                    resolve_mode: vk::ResolveModeFlags::NONE,
                    load_op: if a.clear.is_some() { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::LOAD },
                    store_op: vk::AttachmentStoreOp::STORE,
                    clear_value: vk::ClearValue { color: a.get_vk_clear_color_value() },
                    // TODO multisampling resolve
                    ..Default::default()
                }
            })
            .collect();
        let depth_attachment;
        let stencil_attachment;
        let p_depth_attachment;
        let p_stencil_attachment;
        if let Some(ref depth) = depth_stencil_attachment {
            depth_attachment = vk::RenderingAttachmentInfo {
                image_view: depth.image.attachment_view,
                image_layout: vk::ImageLayout::GENERAL,
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: if depth.depth_clear.is_some() {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::LOAD
                },
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: vk::ClearValue { depth_stencil: depth.get_vk_clear_depth_stencil_value() },
                // TODO multisampling resolve
                ..Default::default()
            };
            p_depth_attachment = &depth_attachment as *const _;
            if is_depth_and_stencil_format(depth.image.format()) {
                stencil_attachment = vk::RenderingAttachmentInfo {
                    image_view: depth.image.attachment_view,
                    image_layout: vk::ImageLayout::GENERAL,
                    resolve_mode: vk::ResolveModeFlags::NONE,
                    load_op: if depth.stencil_clear.is_some() {
                        vk::AttachmentLoadOp::CLEAR
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    },
                    store_op: vk::AttachmentStoreOp::STORE,
                    clear_value: vk::ClearValue { depth_stencil: depth.get_vk_clear_depth_stencil_value() },
                    // TODO multisampling resolve
                    ..Default::default()
                };
                p_stencil_attachment = &stencil_attachment as *const _;
            } else {
                p_stencil_attachment = ptr::null();
            }
        } else {
            p_depth_attachment = ptr::null();
            p_stencil_attachment = ptr::null();
        };

        let rendering_info = vk::RenderingInfo {
            flags: Default::default(),
            render_area,
            layer_count: 1, // TODO?
            view_mask: 0,
            color_attachment_count: color_attachment_infos.len() as u32,
            p_color_attachments: color_attachment_infos.as_ptr(),
            p_depth_attachment,
            p_stencil_attachment,
            ..Default::default()
        };
        unsafe {
            Device::instance().raw.cmd_begin_rendering(self.cmdbuf, &rendering_info);
        }

        let mut encoder = RenderEncoder { parent: self, render_area };
        encoder.set_viewport(0.0, 0.0, render_area.extent.width as f32, render_area.extent.height as f32, 0.0, 1.0);
        encoder.set_scissor(0, 0, render_area.extent.width, render_area.extent.height);
        encoder
    }
}
