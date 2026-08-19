use std::cell::{OnceCell, RefCell};
use crate::helper::{Descriptor, DeviceHelper, Image};
use crate::overlay::renderer::{FrameData, PushConstants, RenderData, ShuffleChannel, Vertex, MAX_INDICES, MAX_VERTICES, SHUFFLE_TEX0, SHUFFLE_TEX1, SHUFFLE_TEX1_NOALPHA};
use crate::DeviceState;
use ash::vk;
use imgui::internal::RawWrapper;
use imgui::TextureId;
use std::ptr;

// imgui::Context is not thread safe, so don't store it in DeviceState

thread_local! {
    pub static IMGUI: OnceCell<RefCell<ImGuiRenderer>> = OnceCell::new();
}

pub fn with_imgui_context(f: impl FnOnce(&mut imgui::Context)) {
    IMGUI.with(|cell| {
        if let Some(renderer) = cell.get() {
            let mut renderer = renderer.borrow_mut();
            f(&mut renderer.context);
        }
    });
}

pub fn imgui_build(dh: &DeviceHelper, rd: &RenderData, f: impl FnOnce(&mut imgui::Ui)) {
    IMGUI.with(|cell| {
        cell.get_or_init(|| RefCell::new(ImGuiRenderer::new(dh)));
        let mut renderer = cell.get().expect("ImGuiRenderer not initialized").borrow_mut();
        renderer.context.io_mut().display_size = [rd.width as f32, rd.height as f32];
        let mut ui = renderer.context.frame();
        f(&mut ui);
    });
}

pub fn imgui_render(dd: &DeviceState, fd: &FrameData, rd: &RenderData) {
    IMGUI.with(|cell| {
        if let Some(renderer) = cell.get() {
            let mut renderer = renderer.borrow_mut();
            renderer.render(&dd, &fd, &rd);
        }
    });
}

pub struct ImGuiRenderer {
    pub context: imgui::Context,
    pub font_texture: Image,
}

impl ImGuiRenderer {
    pub fn new(dh: &DeviceHelper) -> ImGuiRenderer {
        let mut context = imgui::Context::create();
        let fonts = context.fonts();
        let texture = fonts.build_alpha8_texture();
        let font_texture = unsafe {
            dh.create_color_image_from_data(
                vk::Format::R8_UNORM,
                texture.width,
                texture.height,
                vk::ImageUsageFlags::SAMPLED,
                texture.data,
            )
        };
        fonts.tex_id = TextureId::from(0);
        ImGuiRenderer { context, font_texture }
    }

    pub fn render(&mut self, dd: &DeviceState, fd: &FrameData, rd: &RenderData) {

        let draw_data = self.context.render();

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
            dd.cmd_bind_pipeline(fd.cmd_buf, vk::PipelineBindPoint::GRAPHICS, dd.static_resources.pipeline.pipeline);
            dd.cmd_set_viewport_helper(fd.cmd_buf, 0, 0, rd.width, rd.height);
            dd.cmd_set_scissor_helper(fd.cmd_buf, 0, 0, rd.width, rd.height);
            dd.cmd_bind_vertex_buffers(fd.cmd_buf, 0, &[fd.vtx_buf.buffer], &[0]);
            dd.cmd_bind_index_buffer(fd.cmd_buf, fd.idx_buf.buffer, 0, vk::IndexType::UINT16);
            dd.cmd_push_descriptors_helper(
                fd.cmd_buf,
                dd.static_resources.pipeline.pipeline_layout,
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
                    Descriptor::Sampler { binding: 2, sampler: dd.static_resources.font_sampler },
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
                vertex_offset += vertex_data.len();
                index_offset += index_data.len();

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
                                    0 => SHUFFLE_TEX0,
                                    1 => SHUFFLE_TEX1,
                                    _ => SHUFFLE_TEX1_NOALPHA,
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
                                    dd.static_resources.pipeline.pipeline_layout,
                                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                    &PushConstants {
                                        matrix,
                                        screen_size: [rd.width, rd.height],
                                        offset: [0, 0],
                                        shuffle: [
                                            ShuffleChannel::One as u8,
                                            ShuffleChannel::One as u8,
                                            ShuffleChannel::One as u8,
                                            ShuffleChannel::R as u8,
                                        ],
                                        color: [255, 255, 255, 255],
                                    },
                                );

                                dd.cmd_draw_indexed(
                                    fd.cmd_buf,
                                    count as u32,
                                    1,
                                    idx_offset as u32,
                                    vtx_offset as i32,
                                    0,
                                );
                            }
                        }
                        imgui::DrawCmd::ResetRenderState => (),
                        imgui::DrawCmd::RawCallback { callback, raw_cmd } => callback(draw_list.raw(), raw_cmd),
                    }
                }
            }
        }
    }
}
