use crate::helper::{include_bytes_as_u32, Buffer, Descriptor, GraphicsPipelineHelperCreateInfo, Image};
use crate::{DeviceHelper, Device, Pipeline, FRAMES_IN_FLIGHT};
use ash::vk;
use parking_lot::Mutex;
use std::cell::RefCell;
use std::{array, ptr};
use crate::overlay::gui::with_imgui_context;

#[derive(Default)]
pub struct FrameData {
    pub cmd_buf: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub vtx_buf: Buffer,
    pub idx_buf: Buffer,
}

struct FrameResources {
    frame_index: usize,
    frame_data: [FrameData; FRAMES_IN_FLIGHT],
    tmp_image: Option<Image>,
    last_width: u32,
    last_height: u32,
}

impl FrameResources {
    unsafe fn new(dh: &DeviceHelper) -> FrameResources {
        let command_buffers = dh.allocate_command_buffers_helper(FRAMES_IN_FLIGHT);
        let mut frame_data = array::from_fn(|_| FrameData::default());
        for i in 0..FRAMES_IN_FLIGHT {
            let fence = dh
                .create_fence(
                    &vk::FenceCreateInfo { flags: vk::FenceCreateFlags::SIGNALED, ..Default::default() },
                    None,
                )
                .unwrap();
            let vertex_buffer = dh.create_buffer_helper(
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MAX_VERTICES * size_of::<Vertex>(),
                None,
            );
            let index_buffer = dh.create_buffer_helper(
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MAX_INDICES * size_of::<u16>(),
                None,
            );
            frame_data[i].cmd_buf = command_buffers[i];
            frame_data[i].fence = fence;
            frame_data[i].vtx_buf = vertex_buffer;
            frame_data[i].idx_buf = index_buffer;
        }
        FrameResources { frame_index: 0, frame_data, tmp_image: None, last_width: 0, last_height: 0 }
    }
}

pub const MAX_VERTICES: usize = 1024 * 1024;
pub const MAX_INDICES: usize = 1024 * 1024;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub texcoord: [f32; 2],
    pub color: [u8; 4],
}

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub enum ShuffleChannel {
    R = 0,
    G = 1,
    B = 2,
    A = 3,
    R1 = 4,
    G1 = 5,
    B1 = 6,
    A1 = 7,
    Zero = 8,
    One = 9,
}

pub type Shuffle = [ShuffleChannel; 4];

pub const SHUFFLE_TEX0: Shuffle = [ShuffleChannel::R, ShuffleChannel::G, ShuffleChannel::B, ShuffleChannel::A];
pub const SHUFFLE_TEX1: Shuffle = [ShuffleChannel::R1, ShuffleChannel::G1, ShuffleChannel::B1, ShuffleChannel::A1];
pub const SHUFFLE_TEX1_NOALPHA: Shuffle =
    [ShuffleChannel::R1, ShuffleChannel::G1, ShuffleChannel::B1, ShuffleChannel::One];
pub const SHUFFLE_ONE: Shuffle = [ShuffleChannel::One, ShuffleChannel::One, ShuffleChannel::One, ShuffleChannel::One];

pub type RGBA8 = [u8; 4];

static IDENTITY: [[f32; 4]; 4] =
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

pub const TEXID_FONT: imgui::TextureId = imgui::TextureId::new(0);
pub const TEXID_SWAPCHAIN: imgui::TextureId = imgui::TextureId::new(1);

pub struct RenderData {
    pub width: i32,
    pub height: i32,
    pub image_copy: vk::Image,
    pub image_copy_view: vk::ImageView,
}

impl RenderData {
    pub fn texel2uv(&self, x: i32, y: i32) -> (f32, f32) {
        (x as f32 / self.width as f32, y as f32 / self.height as f32)
    }
}

/// Parameters passed to the font pipeline via push constants.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct PushConstants {
    pub matrix: [[f32; 4]; 4],
    pub screen_size: [i32; 2],
    pub offset: [i32; 2],
    pub shuffle: [u8; 4],
    pub color: [u8; 4],
}

pub enum DrawCommand {
    DrawQuad {
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
        u0: f32,
        v0: f32,
        u1: f32,
        v1: f32,
        color: [u8; 4],
        shuffle: Shuffle,
    },
}

pub struct OverlayResources {
    pub font_texture: Image,
    pub font_sampler: vk::Sampler,
    pub pipeline: Pipeline,
    frame_resources: Mutex<FrameResources>,
}

impl OverlayResources {
    pub fn new(dh: &DeviceHelper) -> OverlayResources {
        let font_texture = with_imgui_context(|ctx| {
            let fonts = ctx.fonts();
            fonts.tex_id = TEXID_FONT;
            let texture = fonts.build_alpha8_texture();
            unsafe {
                dh.create_color_image_from_data(
                    vk::Format::R8_UNORM,
                    texture.width,
                    texture.height,
                    vk::ImageUsageFlags::SAMPLED,
                    texture.data,
                )
            }
        });

        let font_sampler = unsafe {
            dh.create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::NEAREST,
                    min_filter: vk::Filter::NEAREST,
                    ..Default::default()
                },
                None,
            )
            .unwrap()
        };

        let pipeline = unsafe {
            dh.create_graphics_pipeline_helper(&GraphicsPipelineHelperCreateInfo {
                spirv: include_bytes_as_u32!("overlay.spv"),
                vertex_entry: c"overlay_vertex",
                fragment_entry: c"overlay_fragment",
                vertex_attributes: &[
                    vk::VertexInputAttributeDescription {
                        location: 0,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 0,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 1,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 8,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 2,
                        binding: 0,
                        format: vk::Format::R8G8B8A8_UNORM,
                        offset: 16,
                    },
                ],
                bindings: &[
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 2,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                ],
                vertex_stride: size_of::<Vertex>(),
                push_constants_size: size_of::<PushConstants>(),
                color_attachment_format: vk::Format::R8G8B8A8_UNORM,
            })
        };

        let frame_resources = unsafe { FrameResources::new(dh) };
        OverlayResources { font_texture, font_sampler, pipeline, frame_resources: Mutex::new(frame_resources) }
    }

    pub fn render(&self, dd: &Device, fd: &FrameData, rd: &RenderData, ctx: &mut imgui::Context) {
        let draw_data = ctx.render();
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if !(fb_width > 0.0 && fb_height > 0.0) {
            return;
        }

        let width = draw_data.display_size[0];
        let height = draw_data.display_size[1];
        let offset_x = draw_data.display_pos[0] / width;
        let offset_y = draw_data.display_pos[1] / height;

        let matrix = [
            [2.0 / width, 0.0, 0.0, 0.0],
            [0.0, 2.0 / height, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0 - offset_x * 2.0, -1.0 - offset_y * 2.0, 0.0, 1.0],
        ];

        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;

        let mut vertex_offset = 0;
        let mut index_offset = 0;

        unsafe {
            dd.cmd_bind_pipeline(fd.cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            dd.cmd_set_viewport_helper(fd.cmd_buf, 0, 0, rd.width, rd.height);
            dd.cmd_set_scissor_helper(fd.cmd_buf, 0, 0, rd.width, rd.height);
            dd.cmd_bind_vertex_buffers(fd.cmd_buf, 0, &[fd.vtx_buf.buffer], &[0]);
            dd.cmd_bind_index_buffer(fd.cmd_buf, fd.idx_buf.buffer, 0, vk::IndexType::UINT16);
            dd.cmd_push_descriptors_helper(
                fd.cmd_buf,
                self.pipeline.pipeline_layout,
                &[
                    Descriptor::Texture {
                        binding: 0,
                        image_view: self.font_texture.image_view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                    Descriptor::Texture {
                        binding: 1,
                        image_view: rd.image_copy_view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                    Descriptor::Sampler { binding: 2, sampler: self.font_sampler },
                ],
            );

            for draw_list in draw_data.draw_lists() {
                // upload vertex & index data
                let vertex_data = draw_list.transmute_vtx_buffer::<Vertex>();
                let index_data = draw_list.idx_buffer();
                if vertex_offset + vertex_data.len() > MAX_VERTICES || index_offset + index_data.len() > MAX_INDICES {
                    panic!("vertex or index buffer overflow");
                }
                ptr::copy(vertex_data.as_ptr(), (fd.vtx_buf.ptr as *mut Vertex).add(vertex_offset), vertex_data.len());
                ptr::copy(index_data.as_ptr(), (fd.idx_buf.ptr as *mut u16).add(index_offset), index_data.len());

                for cmd in draw_list.commands() {
                    match cmd {
                        imgui::DrawCmd::Elements {
                            count,
                            cmd_params: imgui::DrawCmdParams { clip_rect, texture_id, vtx_offset, idx_offset, .. },
                        } => {
                            let clip_rect = [
                                (clip_rect[0] - clip_off[0]) * clip_scale[0],
                                (clip_rect[1] - clip_off[1]) * clip_scale[1],
                                (clip_rect[2] - clip_off[0]) * clip_scale[0],
                                (clip_rect[3] - clip_off[1]) * clip_scale[1],
                            ];

                            if clip_rect[0] < fb_width
                                && clip_rect[1] < fb_height
                                && clip_rect[2] >= 0.0
                                && clip_rect[3] >= 0.0
                            {
                                let shuffle = match texture_id.id() {
                                    0 => [
                                        ShuffleChannel::One as u8,
                                        ShuffleChannel::One as u8,
                                        ShuffleChannel::One as u8,
                                        ShuffleChannel::R as u8,
                                    ],
                                    1 | _ => [
                                        ShuffleChannel::R1 as u8,
                                        ShuffleChannel::G1 as u8,
                                        ShuffleChannel::B1 as u8,
                                        ShuffleChannel::One as u8,
                                    ],
                                };

                                dd.cmd_set_scissor_helper(
                                    fd.cmd_buf,
                                    clip_rect[0].floor() as i32,
                                    clip_rect[1].floor() as i32,
                                    (clip_rect[2] - clip_rect[0]).floor() as i32,
                                    (clip_rect[3] - clip_rect[1]).floor() as i32,
                                );

                                dd.push_constants_helper(
                                    fd.cmd_buf,
                                    self.pipeline.pipeline_layout,
                                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                    &PushConstants {
                                        matrix,
                                        screen_size: [rd.width, rd.height],
                                        offset: [0, 0],
                                        shuffle,
                                        color: [255, 255, 255, 255],
                                    },
                                );

                                dd.cmd_draw_indexed(
                                    fd.cmd_buf,
                                    count as u32,
                                    1,
                                    index_offset as u32 + idx_offset as u32,
                                    vertex_offset as i32 + vtx_offset as i32,
                                    0,
                                );
                            }
                        }
                        imgui::DrawCmd::ResetRenderState => (),
                        imgui::DrawCmd::RawCallback { .. } => {}
                    }
                }

                vertex_offset += vertex_data.len();
                index_offset += index_data.len();
            }
        }
    }
}

/// Renders the overlay on a swapchain image.
///
/// # Arguments
/// * dd - device data
/// * queue - the queue where the present operation will be submitted
/// * swapchain - swapchain
/// * image_index - index of the swapchain image to render to
pub(crate) unsafe fn render_overlay(
    dd: &Device,
    queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
    wait_semaphores: &[vk::Semaphore],
) -> vk::Result {
    let trk = dd.tracked_objects.lock();
    let sc = trk.swapchains.iter().find(|sc| sc.swapchain == swapchain).expect("unknown swapchain");
    let image = sc.images[image_index as usize];
    let image_view = sc.image_views[image_index as usize];

    // Draw the overlay

    // Record the command buffer
    // Wait for frame data to be ready
    let mut fr = dd.overlay.frame_resources.lock();
    let fr = &mut *fr;

    let frame_index = fr.frame_index;
    let fd = &fr.frame_data[frame_index];

    dd.wait_for_fence_and_reset(fd.fence);
    dd.reset_and_begin_command_buffer(fd.cmd_buf);

    // Copy the contents of the swapchain image to a sampleable texture
    if fr.tmp_image.is_none() || fr.last_width != sc.extent.width || fr.last_height != sc.extent.height {
        if let Some(tmp_image) = fr.tmp_image.take() {
            dd.destroy_image_helper(tmp_image);
        }
        fr.tmp_image = Some(dd.create_color_image_helper(
            sc.format,
            sc.extent.width,
            sc.extent.height,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        ));
        fr.last_width = sc.extent.width;
        fr.last_height = sc.extent.height;
    }

    let image_copy = fr.tmp_image.as_ref().unwrap();

    // The application should have transitioned the image to PRESENT before vkQueuePresent.
    dd.layout_barrier(
        fd.cmd_buf,
        &[
            (image, vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            (image_copy.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
        ],
    );

    dd.cmd_copy_image(
        fd.cmd_buf,
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        image_copy.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            extent: vk::Extent3D { width: sc.extent.width, height: sc.extent.height, depth: 1 },
        }],
    );

    dd.layout_barrier(
        fd.cmd_buf,
        &[
            (image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            (image_copy.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ],
    );

    let attachments = [vk::RenderingAttachmentInfo {
        image_view,
        image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        load_op: vk::AttachmentLoadOp::LOAD,
        store_op: vk::AttachmentStoreOp::STORE,
        ..Default::default()
    }];
    dd.cmd_begin_rendering(
        fd.cmd_buf,
        &vk::RenderingInfo {
            flags: Default::default(),
            render_area: vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: sc.extent },
            layer_count: 1,
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachments: attachments.as_ptr(),
            ..Default::default()
        },
    );

    let rd = RenderData {
        width: sc.extent.width as i32,
        height: sc.extent.height as i32,
        image_copy: image_copy.image,
        image_copy_view: image_copy.image_view,
    };

    dd.render_gui(&rd, fd);

    dd.cmd_end_rendering(fd.cmd_buf);
    // transition back to PRESENT
    dd.layout_barrier(
        fd.cmd_buf,
        &[(image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR)],
    );

    dd.end_command_buffer(fd.cmd_buf).unwrap();

    // Submit the command buffer, with a fence, waiting on the semaphores provided by the application.
    let render_to_present = sc.render_to_present[image_index as usize];
    dd.queue_submit_helper(queue, fd.cmd_buf, wait_semaphores, &[render_to_present], fd.fence).unwrap();
    dd.queue_present_helper(queue, swapchain, image_index, render_to_present).unwrap();

    fr.frame_index = (frame_index + 1) % FRAMES_IN_FLIGHT;

    //eprintln!("[planitia-layer] Rendering overlay on swapchain {:?}, image index {}", swapchain, image_index);
    vk::Result::SUCCESS
}
