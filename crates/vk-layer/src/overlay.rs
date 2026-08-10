use crate::font::{glyph_position, GLYPH_HEIGHT, GLYPH_WIDTH};
use crate::helper::transition_image_layout;
use crate::{
    font, Buffer, DeviceData, DeviceHelper, FrameData, FrameResources, Image, LayerDeviceInner, StaticResources,
    FRAMES_IN_FLIGHT,
};
use ash::vk;
use ash::vk::Handle;
use std::ffi::{c_void, CStr};
use std::ptr;

const MAX_VERTICES: usize = 1024 * 1024;

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 2],
    texcoord: [f32; 2],
    color: [u8; 4],
}

/// Parameters passed to the font pipeline via push constants.
#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
    screen_size: [i32; 2],
    offset: [i32; 2],
    bg_color: [u8; 4],
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

    let image = d.create_image_helper(
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8_UNORM,
            extent: vk::Extent3D { width, height, depth: 1 },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    // Staging buffer: host-visible, coherent.
    let staging_buf = d.create_buffer_helper(
        &vk::BufferCreateInfo {
            size: font::ATLAS_DATA.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        },
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        Some(font::ATLAS_DATA),
    );

    d.submit_oneshot(|device, upload_cmdbuf| {
        transition_image_layout(
            device,
            upload_cmdbuf,
            image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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

        transition_image_layout(
            device,
            upload_cmdbuf,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    });

    // Cleanup transient resources.
    d.device_wait_idle().unwrap();
    d.destroy_buffer_helper(staging_buf);
    image
}

/// Creates the graphics pipeline for the overlay shader.
///
/// The pipeline uses a mixed interface:
/// - Vertex data (pos/uv/color) is supplied via a traditional vertex buffer at binding 0
///   (stride 20 bytes: `float2 pos | float2 uv | u8x4 color`).
/// - Remaining per-draw parameters are in an 8-byte push constant (`RootParams*`).
/// - The font texture (binding 0) and sampler (binding 1) are bound through descriptor set 0.
///
/// # Arguments
/// * `d` - Device dispatch tables and helpers
/// * `color_attachment_format` - Format of the swapchain image that will be rendered into
///
/// # Returns
/// `(pipeline, pipeline_layout, descriptor_set_layout)`. All three must be destroyed when no
/// longer needed; destroy the pipeline and pipeline layout before the descriptor set layout.
pub(crate) unsafe fn create_overlay_pipeline(
    d: &DeviceHelper,
    color_attachment_format: vk::Format,
) -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout) {
    let device = &d.dispatch.device;

    let spv_bytes = include_bytes!("overlay.spv");
    assert_eq!(spv_bytes.len() % 4, 0, "SPIR-V size must be a multiple of 4");
    let spv_words: Vec<u32> = spv_bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();

    let shader_module = device
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv_words), None)
        .expect("failed to create overlay shader module");

    // Descriptor set layout (set 0):
    //   binding 0 — SAMPLED_IMAGE   — font texture
    //   binding 1 — SAMPLER         — font sampler
    let dsl_bindings = [
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
    ];
    let descriptor_set_layout = device
        .create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo {
                flags: vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
                binding_count: dsl_bindings.len() as u32,
                p_bindings: dsl_bindings.as_ptr(),
                ..Default::default()
            },
            None,
        )
        .expect("failed to create overlay descriptor set layout");

    // Push constant: 8 bytes for the RootParams* device address, visible to both stages.
    let push_constant_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: size_of::<PushConstants>() as u32,
    };
    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            },
            None,
        )
        .expect("failed to create overlay pipeline layout");

    let vertex_entry = CStr::from_bytes_with_nul(b"text_vertex_main\0").unwrap();
    let fragment_entry = CStr::from_bytes_with_nul(b"text_fragment_main\0").unwrap();
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            module: shader_module,
            p_name: vertex_entry.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: shader_module,
            p_name: fragment_entry.as_ptr(),
            ..Default::default()
        },
    ];

    // Vertex layout: [float2 pos | float2 uv | u8x4 color] = 20 bytes/vertex.
    // color uses R8G8B8A8_UNORM so the hardware unpacks it to [0,1] float4.
    let vertex_binding = vk::VertexInputBindingDescription {
        binding: 0,
        stride: 20, // 4+4 + 4+4 + 4
        input_rate: vk::VertexInputRate::VERTEX,
    };
    let vertex_attributes = [
        vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 0 },
        vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 8 },
        vk::VertexInputAttributeDescription { location: 2, binding: 0, format: vk::Format::R8G8B8A8_UNORM, offset: 16 },
    ];
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(&vertex_binding))
        .vertex_attribute_descriptions(&vertex_attributes);
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let viewport_state =
        vk::PipelineViewportStateCreateInfo { viewport_count: 1, scissor_count: 1, ..Default::default() };

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisample_state =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // Standard "source-over" alpha compositing for the overlay.
    let blend_attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::TRUE,
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };
    let color_blend_state =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&blend_attachment));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    // Attach the color attachment format via the dynamic-rendering extension chain.
    let mut rendering_info = vk::PipelineRenderingCreateInfo {
        color_attachment_count: 1,
        p_color_attachment_formats: &color_attachment_format,
        ..Default::default()
    };

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
        p_next: &mut rendering_info as *const _ as *const c_void,
        p_stages: shader_stages.as_ptr(),
        stage_count: shader_stages.len() as u32,
        p_vertex_input_state: &vertex_input_state,
        p_input_assembly_state: &input_assembly_state,
        p_viewport_state: &viewport_state,
        p_rasterization_state: &rasterization_state,
        p_multisample_state: &multisample_state,
        p_color_blend_state: &color_blend_state,
        p_dynamic_state: &dynamic_state,
        layout: pipeline_layout,
        render_pass: vk::RenderPass::null(),
        ..Default::default()
    };

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_create_info), None)
        .expect("failed to create overlay graphics pipeline")[0];

    // The shader module is no longer needed once the pipeline is built.
    device.destroy_shader_module(shader_module, None);

    (pipeline, pipeline_layout, descriptor_set_layout)
}

pub(crate) unsafe fn initialize_frame_resources(d: &DeviceHelper) -> FrameResources {
    let command_buffers = d
        .device
        .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
            command_pool: d.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        })
        .expect("allocate_command_buffers failed");
    let mut frame_data = [FrameData::default(); FRAMES_IN_FLIGHT];
    for i in 0..FRAMES_IN_FLIGHT {
        let fence = d
            .device
            .create_fence(&vk::FenceCreateInfo { flags: vk::FenceCreateFlags::SIGNALED, ..Default::default() }, None)
            .unwrap();
        let vertex_buffer = d.create_buffer_helper(
            &vk::BufferCreateInfo {
                size: MAX_VERTICES as u64 * size_of::<Vertex>() as u64,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
            None,
        );

        frame_data[i].cmdbuf = command_buffers[i];
        frame_data[i].fence = fence;
        frame_data[i].vtxbuf = vertex_buffer;
        let _ = (d.set_device_loader_data)(d.device.handle(), frame_data[i].cmdbuf.as_raw() as *mut c_void);
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
    // FIXME: we don't know the swapchain format at this point.
    let (pipeline, pipeline_layout, descriptor_set_layout) = create_overlay_pipeline(d, vk::Format::R8G8B8A8_UNORM);
    StaticResources { font_tex, font_sampler, pipeline, pipeline_layout, descriptor_set_layout }
}

type RGBA8 = [u8; 4];

struct OverlayText {
    pos: [i32; 2],
    text: String,
    color: RGBA8,
    bg_color: RGBA8,
}

struct OverlayBuilder {
    texts: Vec<OverlayText>,
    cur_fg: RGBA8,
    cur_bg: RGBA8,
}

impl OverlayBuilder {
    fn new() -> OverlayBuilder {
        OverlayBuilder { texts: Vec::new(), cur_fg: [255, 255, 255, 255], cur_bg: [0, 0, 0, 0] }
    }

    fn setfg(&mut self, color: RGBA8) -> &mut Self {
        self.texts.last_mut().unwrap().color = color;
        self
    }

    fn setbg(&mut self, color: RGBA8) -> &mut Self {
        self.texts.last_mut().unwrap().bg_color = color;
        self
    }

    fn print(&mut self, x: i32, y: i32, text: &str) {
        self.texts.push(OverlayText { pos: [x, y], text: text.to_string(), color: self.cur_fg, bg_color: self.cur_bg });
    }
}

fn draw_overlay(dd: &DeviceData, ld: &LayerDeviceInner) -> OverlayBuilder {
    let mut builder = OverlayBuilder::new();
    builder.print(10, 10, "Debug layer active");
    let mut y = 10 + GLYPH_HEIGHT as i32;
    for pipeline in ld.pipelines.iter() {
        builder.print(10, y, &format!("Pipeline: {:?}", pipeline.pipeline));
        y += GLYPH_HEIGHT as i32;
    }
    builder
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

struct RenderData {
    width: i32,
    height: i32,
}

unsafe fn render_overlay_text(
    dd: &DeviceData,
    rd: &RenderData,
    cmdbuf: vk::CommandBuffer,
    vtxbuf: &Buffer,
    overlay: &OverlayBuilder,
) {
    dd.cmd_bind_pipeline(cmdbuf, vk::PipelineBindPoint::GRAPHICS, dd.static_resources.pipeline);
    dd.cmd_set_viewport(
        cmdbuf,
        0,
        &[vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: rd.width as f32,
            height: rd.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
    );
    dd.cmd_set_scissor(
        cmdbuf,
        0,
        &[vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: rd.width as u32, height: rd.height as u32 },
        }],
    );
    dd.cmd_bind_vertex_buffers(cmdbuf, 0, &[vtxbuf.buffer], &[0]);
    (dd.khr_push_descriptors.cmd_push_descriptor_set_khr)(
        cmdbuf,
        vk::PipelineBindPoint::GRAPHICS,
        dd.static_resources.pipeline_layout,
        0,
        2,
        [
            vk::WriteDescriptorSet {
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                p_image_info: &vk::DescriptorImageInfo {
                    image_view: dd.static_resources.font_tex.image_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    sampler: vk::Sampler::null(),
                },
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLER,
                p_image_info: &vk::DescriptorImageInfo {
                    sampler: dd.static_resources.font_sampler,
                    image_view: vk::ImageView::null(),
                    image_layout: vk::ImageLayout::UNDEFINED,
                },
                ..Default::default()
            },
        ]
        .as_ptr(),
    );

    // Fill vertex buffers and emit draw calls
    let ptr = vtxbuf.ptr.cast::<Vertex>();
    let mut offset = 0;

    let sw = rd.width as f32;
    let sh = rd.height as f32;

    for text in overlay.texts.iter() {
        let mut cursor_x = text.pos[0];
        let y = text.pos[1];
        let first_vertex = offset;

        for ch in text.text.chars() {
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

            if offset + 6 > MAX_VERTICES {
                // We've reached the max number of vertices we can draw in a frame.
                break;
            }

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
                ptr.add(offset),
                6,
            );

            offset += 6;
            cursor_x += GLYPH_WIDTH as i32;
        }

        let vertex_count = offset - first_vertex;

        // draw call
        push_constants_helper(
            dd,
            cmdbuf,
            dd.static_resources.pipeline_layout,
            &PushConstants { screen_size: [rd.width, rd.height], offset: [0, 0], bg_color: text.bg_color },
        );

        dd.cmd_draw(cmdbuf, vertex_count as u32, 1, first_vertex as u32, 0);

        if offset + 6 > MAX_VERTICES {
            break;
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
    dd: &DeviceData,
    queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
    wait_semaphores: &[vk::Semaphore],
) -> vk::Result {
    let device = &dd.dispatch.device;

    let mut fr = dd.frame_resources.lock().unwrap();

    let mut inner = dd.inner.lock().unwrap();
    let sc = inner.swapchains.iter().find(|sc| sc.swapchain == swapchain).expect("unknown swapchain");
    let image_view = sc.image_views[image_index as usize];

    // Draw the overlay
    let rd = RenderData { width: sc.extent.width as i32, height: sc.extent.height as i32 };
    let overlay = draw_overlay(dd, &*inner);

    // Record the command buffer
    // Wait for frame data to be ready
    let frame_index = fr.frame_index;
    let fd = &fr.frame_data[frame_index];

    device.wait_for_fences(&[fd.fence], true, u64::MAX).unwrap();
    device.reset_fences(&[fd.fence]).unwrap();

    device.reset_command_buffer(fd.cmdbuf, vk::CommandBufferResetFlags::empty()).unwrap();
    device
        .begin_command_buffer(
            fd.cmdbuf,
            &vk::CommandBufferBeginInfo { flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, ..Default::default() },
        )
        .unwrap();

    // The application should have transitioned the image to PRESENT before vkQueuePresent,
    // so transition it back to COLOR_ATTACHMENT_OPTIMAL so that we can render to it.
    transition_image_layout(
        device,
        fd.cmdbuf,
        sc.images[image_index as usize],
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );
    let attachments = [vk::RenderingAttachmentInfo {
        image_view,
        image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        load_op: vk::AttachmentLoadOp::LOAD,
        store_op: vk::AttachmentStoreOp::STORE,
        ..Default::default()
    }];
    device.cmd_begin_rendering(
        fd.cmdbuf,
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

    render_overlay_text(dd, &rd, fd.cmdbuf, &fd.vtxbuf, &overlay);

    /*device.cmd_clear_attachments(
        fd.cmdbuf,
        &[vk::ClearAttachment {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            color_attachment: 0,
            clear_value: vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 1.0, 0.0, 1.0] } },
        }],
        &[vk::ClearRect {
            rect: vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: sc.extent },
            base_array_layer: 0,
            layer_count: 1,
        }],
    );*/

    device.cmd_end_rendering(fd.cmdbuf);
    // transition back to PRESENT
    transition_image_layout(
        device,
        fd.cmdbuf,
        sc.images[image_index as usize],
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
    );

    device.end_command_buffer(fd.cmdbuf).unwrap();

    // Submit the command buffer, with a fence, waiting on the semaphores provided by the application.

    // It's a sad thing that we have to allocate memory dynamically for something that is probably
    // ignored by the driver, but here we are.
    let wait_dst_stage_mask = vec![vk::PipelineStageFlags::ALL_COMMANDS; wait_semaphores.len()];

    let submit_info = [vk::SubmitInfo {
        wait_semaphore_count: wait_semaphores.len() as u32,
        p_wait_semaphores: wait_semaphores.as_ptr(),
        p_wait_dst_stage_mask: wait_dst_stage_mask.as_ptr(),
        command_buffer_count: 1,
        p_command_buffers: &fd.cmdbuf,
        signal_semaphore_count: 1,
        p_signal_semaphores: &sc.render_to_present[image_index as usize],
        ..Default::default()
    }];

    device.queue_submit(queue, &submit_info, fd.fence).unwrap();

    let render_to_present = sc.render_to_present[image_index as usize];
    fr.frame_index = (frame_index + 1) % FRAMES_IN_FLIGHT;

    // present
    let present_info = vk::PresentInfoKHR {
        wait_semaphore_count: 1,
        p_wait_semaphores: &render_to_present,
        swapchain_count: 1,
        p_swapchains: &swapchain,
        p_image_indices: &image_index,
        ..Default::default()
    };
    let _result = (dd.dispatch.khr_swapchain.queue_present_khr)(queue, &present_info);

    eprintln!("[planitia-layer] Rendering overlay on swapchain {:?}, image index {}", swapchain, image_index);
    vk::Result::SUCCESS
}
