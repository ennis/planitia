// reexport all of imgui
pub use imgui_rs::*;

use crate::imgui;
use crate::input::{MouseScrollDelta, PointerButton};
use gpu::PrimitiveTopology::TriangleList;
use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;
use gpu::vulkan::{VK_FILTER_NEAREST, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8_UNORM};

const PERSISTED_SETTINGS_FILE: &str = "egui_settings.ron";

#[gpu::shader_module("assets/gamelib/shaders/imgui.slang#0")]
mod shaders {}

//--------------------------------------------------------------------------------------------------


pub struct ImGuiResources {
    pub font_texture: gpu::Image,
    pub font_sampler: gpu::SamplerHandle,
}

impl ImGuiResources {
    pub fn new() -> ImGuiResources {
        let font_texture = with_imgui_context(|ctx| {
            let fonts = ctx.fonts();
            let texture = unsafe {
                let texture = fonts.build_alpha8_texture();
                gpu::create_image_with_data(
                    &gpu::ImageCreateInfo {
                        format: VK_FORMAT_R8_UNORM,
                        width: texture.width,
                        height: texture.height,
                        usage: gpu::ImageUsage::SAMPLED,
                        ..
                    },
                    gpu::ImageAspect::All,
                    texture.data,
                )
            };
            fonts.tex_id = TextureId::new(texture.texture_handle().to_usize());
            texture
        });
        ImGuiResources {
            font_texture,
            font_sampler: gpu::register_sampler(&gpu::SamplerParams {
                mag_filter: VK_FILTER_NEAREST,
                min_filter: VK_FILTER_NEAREST,
                ..Default::default()
            }),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct RootParams {
    screen_size: [f32; 2],
    tex: gpu::TextureHandle,
    sampler: gpu::SamplerHandle,
    vertices: gpu::Ptr<DrawVert>,
    indices: gpu::Ptr<u16>,
    base_vertex: u32,
}

const _: () = assert!(size_of::<u16>() == size_of::<sys::ImDrawIdx>(), "unexpected size of DrawIdx type");

pub(crate) fn render_imgui(resources: &ImGuiResources, ctx: &mut Context, color_target: &gpu::Image) {
    assert!(color_target.format() == VK_FORMAT_R8G8B8A8_UNORM);
    let draw_data = ctx.render();
    let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
    let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
    if !(fb_width > 0.0 && fb_height > 0.0) {
        return;
    }
    //let width = draw_data.display_size[0];
    //let height = draw_data.display_size[1];
    let clip_off = draw_data.display_pos;
    let clip_scale = draw_data.framebuffer_scale;

    gpu::render(&[gpu::ColorAttachment { image: color_target, clear: None }], None, |encoder| {
        encoder.set_viewport(0.0, 0.0, fb_width, fb_height, 0.0, 1.0);
        encoder.set_scissor(0, 0, fb_width as u32, fb_height as u32);
        encoder.bind_graphics_pipeline(&shaders::imgui);

        for draw_list in draw_data.draw_lists() {
            let vertex_data =  draw_list.vtx_buffer();
            let index_data = draw_list.idx_buffer();
            let vtx_buf = gpu::alloc_temp_slice(vertex_data);
            let idx_buf = gpu::alloc_temp_slice(index_data);

            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements {
                        count,
                        cmd_params: DrawCmdParams { clip_rect, texture_id, vtx_offset, idx_offset, .. },
                    } => {
                        let clip_rect = [
                            (clip_rect[0] - clip_off[0]) * clip_scale[0],
                            (clip_rect[1] - clip_off[1]) * clip_scale[1],
                            (clip_rect[2] - clip_off[0]) * clip_scale[0],
                            (clip_rect[3] - clip_off[1]) * clip_scale[1],
                        ];
                        let texture_handle = gpu::TextureHandle::from_usize(texture_id.id());
                        if clip_rect[0] < fb_width
                            && clip_rect[1] < fb_height
                            && clip_rect[2] >= 0.0
                            && clip_rect[3] >= 0.0
                        {
                            encoder.set_scissor(
                                clip_rect[0].floor() as i32,
                                clip_rect[1].floor() as i32,
                                (clip_rect[2] - clip_rect[0]).floor() as u32,
                                (clip_rect[3] - clip_rect[1]).floor() as u32,
                            );

                            encoder.draw(
                                TriangleList,
                                None,
                                idx_offset as u32..(idx_offset + count) as u32,
                                0..1,
                                &RootParams {
                                    screen_size: [fb_width, fb_height],
                                    base_vertex: vtx_offset as u32,
                                    tex: texture_handle,
                                    sampler: resources.font_sampler,
                                    vertices: vtx_buf,
                                    indices: idx_buf,
                                },
                            );
                        }
                    }
                    DrawCmd::ResetRenderState => {}
                    DrawCmd::RawCallback { .. } => {}
                }
            }
        }
    });
}

pub(crate) fn handle_input_event(ctx: &mut Context, event: &crate::InputEvent) {
    let io = ctx.io_mut();

    fn convert_mouse_button(button: PointerButton) -> Option<MouseButton> {
        match button {
            PointerButton::LEFT => Some(MouseButton::Left),
            PointerButton::RIGHT => Some(MouseButton::Right),
            PointerButton::MIDDLE => Some(MouseButton::Middle),
            _ => None,
        }
    }

    match *event {
        crate::InputEvent::CursorMoved { x, y } => {
            io.add_mouse_pos_event([x as f32, y as f32]);
        }
        crate::InputEvent::PointerDown { x, y, button } => {
            let Some(imgui_button) = convert_mouse_button(button) else {
                return;
            };
            io.add_mouse_button_event(imgui_button, true);
        }
        crate::InputEvent::PointerUp { button, .. } => {
            let Some(imgui_button) = convert_mouse_button(button) else {
                return;
            };
            io.add_mouse_button_event(imgui_button, false);
        }
        crate::InputEvent::MouseWheel(delta) => {
            match delta {
                MouseScrollDelta::LineDelta { x, y } => {
                    io.add_mouse_wheel_event([x, y]);
                }
                MouseScrollDelta::PixelDelta { x, y } => {
                    // TODO pixel scrolling
                    io.add_mouse_wheel_event([x, y]);
                }
            }
        }
        crate::InputEvent::KeyboardEvent(ref ke) => {
            let down = ke.state == crate::input::KeyState::Down;
            match &ke.key {
                crate::input::Key::Character(chars) => {
                    for c in chars.chars() {
                        if down {
                            io.add_input_character(c);
                        }
                    }
                }
                crate::input::Key::Named(named) => {
                    let key = match named {
                        crate::input::NamedKey::Tab => io.add_key_event(Key::Tab, down),
                        crate::input::NamedKey::ArrowLeft => io.add_key_event(Key::LeftArrow, down),
                        crate::input::NamedKey::ArrowRight => io.add_key_event(Key::RightArrow, down),
                        crate::input::NamedKey::ArrowUp => io.add_key_event(Key::UpArrow, down),
                        crate::input::NamedKey::ArrowDown => io.add_key_event(Key::DownArrow, down),
                        crate::input::NamedKey::PageUp => io.add_key_event(Key::PageUp, down),
                        crate::input::NamedKey::PageDown => io.add_key_event(Key::PageDown, down),
                        crate::input::NamedKey::Home => io.add_key_event(Key::Home, down),
                        crate::input::NamedKey::End => io.add_key_event(Key::End, down),
                        crate::input::NamedKey::Insert => io.add_key_event(Key::Insert, down),
                        crate::input::NamedKey::Delete => io.add_key_event(Key::Delete, down),
                        crate::input::NamedKey::Backspace => io.add_key_event(Key::Backspace, down),
                        crate::input::NamedKey::Enter => io.add_key_event(Key::Enter, down),
                        crate::input::NamedKey::Escape => io.add_key_event(Key::Escape, down),
                        crate::input::NamedKey::F1 => io.add_key_event(Key::F1, down),
                        crate::input::NamedKey::F2 => io.add_key_event(Key::F2, down),
                        crate::input::NamedKey::F3 => io.add_key_event(Key::F3, down),
                        crate::input::NamedKey::F4 => io.add_key_event(Key::F4, down),
                        crate::input::NamedKey::F5 => io.add_key_event(Key::F5, down),
                        crate::input::NamedKey::F6 => io.add_key_event(Key::F6, down),
                        crate::input::NamedKey::F7 => io.add_key_event(Key::F7, down),
                        crate::input::NamedKey::F8 => io.add_key_event(Key::F8, down),
                        crate::input::NamedKey::F9 => io.add_key_event(Key::F9, down),
                        crate::input::NamedKey::F10 => io.add_key_event(Key::F10, down),
                        crate::input::NamedKey::F11 => io.add_key_event(Key::F11, down),
                        crate::input::NamedKey::F12 => io.add_key_event(Key::F12, down),
                        crate::input::NamedKey::Control => io.add_key_event(Key::ModCtrl, down),
                        crate::input::NamedKey::Shift => io.add_key_event(Key::ModShift, down),
                        crate::input::NamedKey::Alt => io.add_key_event(Key::ModAlt, down),
                        crate::input::NamedKey::Meta => io.add_key_event(Key::ModSuper, down),
                        crate::input::NamedKey::CapsLock => io.add_key_event(Key::CapsLock, down),
                        crate::input::NamedKey::ScrollLock => io.add_key_event(Key::ScrollLock, down),
                        crate::input::NamedKey::NumLock => io.add_key_event(Key::NumLock, down),
                        crate::input::NamedKey::PrintScreen => io.add_key_event(Key::PrintScreen, down),
                        crate::input::NamedKey::Pause => io.add_key_event(Key::Pause, down),
                        _ => {}
                    };
                }
           };
        }
    }
}

//--------------------------------------------------------------------------------------------------
// IMGUI context
thread_local! {
    pub(crate) static IMGUI: RefCell<Context> = RefCell::new(Context::create());
}

/// Runs the closure with a [`imgui::Context`](imgui::Context).
pub(crate) fn with_imgui_context<R>(f: impl FnOnce(&mut Context) -> R) -> R {
    IMGUI.with(|cell| {
        let mut context = cell.borrow_mut();
        f(&mut context)
    })
}
