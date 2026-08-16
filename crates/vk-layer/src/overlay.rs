use crate::font::{glyph_position, GLYPH_HEIGHT, GLYPH_WIDTH};
use crate::helper::{Buffer, Descriptor, GraphicsPipelineHelperCreateInfo, Image};
use crate::{font, DeviceHelper, DeviceState, Pipeline, TrackedResources, FRAMES_IN_FLIGHT};
use ash::vk;
use ash::vk::Handle;
use std::ffi::{c_void, CStr};
use std::{mem, ptr};

/// Static resources
pub struct StaticResources {
    font_tex: Image,
    font_sampler: vk::Sampler,
    pipeline: Pipeline,
}

#[derive(Copy, Clone, Default)]
pub struct FrameData {
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    vtx_buf: Buffer,
}

pub struct FrameResources {
    frame_index: usize,
    frame_data: [FrameData; FRAMES_IN_FLIGHT],
}

///
#[derive(Default)]
pub struct OverlayResources {
    tmp_image: Option<Image>,
    last_width: u32,
    last_height: u32,
}

//--------------------------------------------------------------------------------------------------

const MAX_VERTICES: usize = 1024 * 1024;

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 2],
    texcoord: [f32; 2],
    color: [u8; 4],
}

const SHUF_R: u8 = 0;
const SHUF_G: u8 = 1;
const SHUF_B: u8 = 2;
const SHUF_A: u8 = 3;
const SHUF_ZERO: u8 = 4;
const SHUF_ONE: u8 = 5;

/// Parameters passed to the font pipeline via push constants.
#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
    screen_size: [i32; 2],
    offset: [i32; 2],
    shuffle: [u8; 4],
    color: [u8; 4],
}

unsafe fn create_font_texture(d: &DeviceHelper) -> Image {
    // Creating a texture in Vulkan is fairly simple.
    // You just have to follow this 25-step process:
    //
    // 1. Create the image
    // 2. Query the image memory requirements
    // 3. Allocate memory for the image
    // 3.1. Find a compatible memory type for the image
    // 4. Bind image memory
    // 5. Create a staging buffer
    // 6. Query the staging buffer memory requirements
    // 7. Allocate staging buffer memory
    // 7.1. Find a compatible memory type for the staging buffer
    // 8. Bind staging buffer memory
    // 9. Map staging buffer memory
    // 10. Copy texture data to buffer
    // 11. Unmap staging buffer memory
    // 12. Allocate command pool
    // 13. Allocate command buffer in command pool
    // 14. Issue CmdPipelineBarrier to transition image to TRANSFER_DST
    // 15. Issue CmdCopyBufferToImage command to copy staging buffer to image
    // 16. Issue CmdPipelineBarrier to transition image to WHATEVER_IS_NEEDED
    // 17. Create a fence
    // 18. Submit command buffer to queue, signalling the fence
    // 19. Wait for fence
    // 20. Delete fence
    // 21. Delete staging buffer
    // 22. Free memory of staging buffer
    // 23. Destroy the command pool

    let width = font::ATLAS_WIDTH as u32;
    let height = font::ATLAS_HEIGHT as u32;

    let image = d.create_color_image_helper(
        vk::Format::R8_UNORM,
        width,
        height,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
    );

    // Staging buffer: host-visible, coherent.
    let staging_buf = d.create_buffer_from_data(vk::BufferUsageFlags::TRANSFER_SRC, font::ATLAS_DATA);

    d.submit_oneshot(|device, upload_cmdbuf| {
        d.layout_barrier(
            upload_cmdbuf,
            &[(image.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)],
        );

        device.cmd_copy_buffer_to_image(
            upload_cmdbuf,
            staging_buf.buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width, height, depth: 1 },
            }],
        );

        d.layout_barrier(
            upload_cmdbuf,
            &[(image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
        );
    });

    // Cleanup transient resources.
    d.device_wait_idle().unwrap();
    d.destroy_buffer_helper(staging_buf);
    image
}

pub(crate) unsafe fn initialize_frame_resources(d: &DeviceHelper) -> FrameResources {
    let command_buffers = d.allocate_command_buffers_helper(FRAMES_IN_FLIGHT);
    let mut frame_data = [FrameData::default(); FRAMES_IN_FLIGHT];
    for i in 0..FRAMES_IN_FLIGHT {
        let fence = d
            .device
            .create_fence(&vk::FenceCreateInfo { flags: vk::FenceCreateFlags::SIGNALED, ..Default::default() }, None)
            .unwrap();
        let vertex_buffer = d.create_buffer_helper(
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MAX_VERTICES * size_of::<Vertex>(),
            None,
        );
        frame_data[i].cmd_buf = command_buffers[i];
        frame_data[i].fence = fence;
        frame_data[i].vtx_buf = vertex_buffer;
    }
    FrameResources { frame_index: 0, frame_data }
}

unsafe fn create_font_sampler(d: &DeviceHelper) -> vk::Sampler {
    d.device
        .create_sampler(
            &vk::SamplerCreateInfo {
                mag_filter: vk::Filter::NEAREST,
                min_filter: vk::Filter::NEAREST,
                ..Default::default()
            },
            None,
        )
        .unwrap()
}

/// Initializes the resources for rendering the overlay.
///
/// # Arguments
///
/// * d - Device dispatch tables
/// * queue_family_index - the family index of the queue on which the swapchain will be presented
pub(crate) unsafe fn initialize_static_resources(d: &DeviceHelper) -> StaticResources {
    let font_tex = create_font_texture(d);
    let font_sampler = create_font_sampler(d);
    let overlay_pipeline = d.create_graphics_pipeline_helper(&GraphicsPipelineHelperCreateInfo {
        spirv: include_bytes!("overlay.spv"),
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
                descriptor_type: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ],
        vertex_stride: size_of::<Vertex>(),
        push_constants_size: size_of::<PushConstants>(),
        color_attachment_format: vk::Format::R8G8B8A8_UNORM,
    });

    StaticResources { font_tex, font_sampler, pipeline: overlay_pipeline }
}

//--------------------------------------------------------------------------------------------------

type RGBA8 = [u8; 4];

struct RenderData {
    width: i32,
    height: i32,
    image_copy: vk::ImageView,
    image_copy_view: vk::ImageView,
}

struct OverlayRenderer<'a> {
    dd: &'a DeviceState,
    cmd_buf: vk::CommandBuffer,
    vtx_buf: *mut Vertex,
    vtx_offset: usize,
    //texts: Vec<OverlayText>,
    cur_fg: RGBA8,
    cur_bg: RGBA8,
    rd: &'a RenderData,
}

impl<'a> OverlayRenderer<'a> {
    fn new(
        dd: &'a DeviceState,
        rd: &'a RenderData,
        cmd_buf: vk::CommandBuffer,
        vtx_buf: &Buffer,
    ) -> OverlayRenderer<'a> {
        OverlayRenderer {
            dd,
            cmd_buf,
            vtx_buf: vtx_buf.ptr as *mut Vertex,
            vtx_offset: 0,
            cur_fg: [255, 255, 255, 255],
            cur_bg: [0, 0, 0, 0],
            rd,
        }
    }

    fn set_text_color(&mut self, color: RGBA8) {
        self.cur_fg = color;
    }

    fn print(&mut self, x: i32, y: i32, text: &str) {
        // Fill vertex buffers and emit draw calls
        let ptr = self.vtx_buf.cast::<Vertex>();
        let mut offset = self.vtx_offset;

        let sw = self.rd.width as f32;
        let sh = self.rd.height as f32;

        let mut cursor_x = x;
        let y = y;
        let first_vertex = self.vtx_offset;

        for ch in text.chars() {
            let (ax, ay) = glyph_position(ch).unwrap_or((0, 0));

            let x0 = cursor_x as f32;
            let x1 = x0 + GLYPH_WIDTH as f32;
            let y0 = y as f32;
            let y1 = y0 + GLYPH_HEIGHT as f32;

            // NDC coords
            let nx0 = 2.0 * x0 / sw - 1.0;
            let nx1 = 2.0 * x1 / sw - 1.0;
            let ny0 = 2.0 * y0 / sh - 1.0;
            let ny1 = 2.0 * y1 / sh - 1.0;

            // Atlas UV coordinates (atlas is 128×128).
            let u0 = ax as f32 / 128.0;
            let u1 = (ax as f32 + GLYPH_WIDTH as f32) / 128.0;
            let v0 = ay as f32 / 128.0;
            let v1 = (ay as f32 + GLYPH_HEIGHT as f32) / 128.0;

            let color = [255u8, 255u8, 255u8, 255u8];

            if self.vtx_offset + 6 > MAX_VERTICES {
                // We've reached the max number of vertices we can draw in a frame.
                eprintln!("[planitia-layer] vertex buffer overflow");
                break;
            }

            unsafe {
                ptr::copy_nonoverlapping(
                    [
                        Vertex { pos: [nx0, ny0], texcoord: [u0, v0], color },
                        Vertex { pos: [nx1, ny0], texcoord: [u1, v0], color },
                        Vertex { pos: [nx1, ny1], texcoord: [u1, v1], color },
                        Vertex { pos: [nx0, ny0], texcoord: [u0, v0], color },
                        Vertex { pos: [nx1, ny1], texcoord: [u1, v1], color },
                        Vertex { pos: [nx0, ny1], texcoord: [u0, v1], color },
                    ]
                    .as_ptr(),
                    ptr.add(self.vtx_offset),
                    6,
                );
            }

            self.vtx_offset += 6;
            cursor_x += GLYPH_WIDTH as i32;
        }

        let vertex_count = self.vtx_offset - first_vertex;

        // draw call
        unsafe {
            self.dd.push_constants_helper(
                self.cmd_buf,
                self.dd.static_resources.pipeline.pipeline_layout,
                &PushConstants {
                    screen_size: [self.rd.width, self.rd.height],
                    offset: [0, 0],
                    shuffle: [SHUF_ONE, SHUF_ONE, SHUF_ONE, SHUF_R],
                    color: [255, 255, 255, 255],
                },
            );

            self.dd.cmd_draw(self.cmd_buf, vertex_count as u32, 1, first_vertex as u32, 0);
        }
    }
}

unsafe fn push_constants_helper<T: Copy + 'static>(
    dd: &DeviceHelper,
    cmdbuf: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    data: &T,
) {
    dd.cmd_push_constants(
        cmdbuf,
        pipeline_layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        std::slice::from_raw_parts(data as *const _ as *const u8, size_of::<T>()),
    );
}

/// Renders the overlay on a swapchain image.
///
/// # Arguments
/// * dd - device data
/// * queue - the queue where the present operation will be submitted
/// * swapchain - swapchain
/// * image_index - index of the swapchain image to render to
pub(crate) unsafe fn render_overlay(
    dd: &DeviceState,
    queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
    wait_semaphores: &[vk::Semaphore],
) -> vk::Result {
    let mut fr = dd.frame_resources.lock().unwrap();
    let mut ovr = dd.overlay_resources.lock().unwrap();
    let mut trk = dd.tracked_resources.lock().unwrap();

    let sc = trk.swapchains.iter().find(|sc| sc.swapchain == swapchain).expect("unknown swapchain");
    let image = sc.images[image_index as usize];
    let image_view = sc.image_views[image_index as usize];

    // Draw the overlay

    // Record the command buffer
    // Wait for frame data to be ready
    let frame_index = fr.frame_index;
    let fd = &fr.frame_data[frame_index];

    dd.wait_for_fence_and_reset(fd.fence);
    dd.reset_and_begin_command_buffer(fd.cmd_buf);

    // Copy the contents of the swapchain image to a sampleable texture
    if ovr.tmp_image.is_none() || ovr.last_width != sc.extent.width || ovr.last_height != sc.extent.height {
        if let Some(tmp_image) = ovr.tmp_image.take() {
            dd.destroy_image_helper(tmp_image);
        }
        ovr.tmp_image = Some(dd.create_color_image_helper(
            sc.format,
            sc.extent.width,
            sc.extent.height,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        ));
        ovr.last_width = sc.extent.width;
        ovr.last_height = sc.extent.height;
    }

    let image_copy = ovr.tmp_image.unwrap();

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
        image_copy: Default::default(),
        image_copy_view: Default::default(),
    };

    dd.cmd_bind_pipeline(fd.cmd_buf, vk::PipelineBindPoint::GRAPHICS, dd.static_resources.pipeline.pipeline);
    dd.cmd_set_viewport_helper(fd.cmd_buf, 0, 0, rd.width, rd.height);
    dd.cmd_set_scissor_helper(fd.cmd_buf, 0, 0, rd.width, rd.height);
    dd.cmd_bind_vertex_buffers(fd.cmd_buf, 0, &[fd.vtx_buf.buffer], &[0]);
    dd.cmd_push_descriptors_helper(
        fd.cmd_buf,
        dd.static_resources.pipeline.pipeline_layout,
        &[
            Descriptor::Texture {
                binding: 0,
                image_view: dd.static_resources.font_tex.image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Descriptor::Sampler { binding: 1, sampler: dd.static_resources.font_sampler },
        ],
    );

    // Draw overlay
    let mut or = OverlayRenderer::new(dd, &rd, fd.cmd_buf, &fd.vtx_buf);
    draw_overlay(&mut or, &*trk);

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

fn draw_overlay(or: &mut OverlayRenderer, trk: &TrackedResources) {
    or.print(10, 10, "Debug layer active");
    let mut y = 10 + GLYPH_HEIGHT as i32;
    for pipeline in trk.pipelines.iter() {
        or.print(10, y, &format!("Pipeline: {:p}", *pipeline));
        y += GLYPH_HEIGHT as i32;
    }
}
