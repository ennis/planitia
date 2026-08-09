//! Debugger rendering.
use crate::debugger::Debugger;
use crate::debugger::font::{ATLAS_DATA, ATLAS_HEIGHT, ATLAS_WIDTH, GLYPH_HEIGHT, GLYPH_WIDTH, glyph_position};
use crate::{ColorAttachment, Image, Sampler};
use ash::vk;
use gpu_types::{SamplerHandle, SamplerParams, TextureHandle};

#[gpu_macros::shader_module("src/debugger/debugger.slang#1")]
mod shader {}

pub(super) struct Renderer {
    pub(super) font_tex: Image,
    pub(super) sampler: Sampler,
}

impl Renderer {
    pub(super) fn new() -> Self {
        let font_tex = Image::new_texture_with_data(
            ATLAS_WIDTH as u32,
            ATLAS_HEIGHT as u32,
            crate::Format::R8_UNORM,
            crate::ImageAspect::All,
            ATLAS_DATA,
        );
        let sampler = Sampler::new(SamplerParams {
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ..
        });
        Renderer { font_tex, sampler }
    }
}

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
struct Params {
    screen_size: [i32; 2],
    vertices: crate::Ptr<Vertex>,
    font_tex: TextureHandle,
    sampler: SamplerHandle,
    bg_color: [u8; 4],
}

struct RenderData<'a, 'enc> {
    encoder: &'a mut crate::RenderEncoder<'enc>,
    width: i32,
    height: i32,
}

impl Renderer {
    /// Draws `text` at pixel position `(x, y)`.
    pub fn draw_text(&self, rd: &mut RenderData, x: i32, y: i32, text: &str) {
        let mut verts: Vec<Vertex> = Vec::with_capacity(text.len() * 6);

        let sw = rd.width as f32;
        let sh = rd.height as f32;

        let mut cursor_x = x;
        for ch in text.chars() {
            // Fall back to space for non-printable characters.
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

            verts.push(Vertex { pos: [nx0, ny0], texcoord: [u0, v0], color }); // TL
            verts.push(Vertex { pos: [nx1, ny0], texcoord: [u1, v0], color }); // TR
            verts.push(Vertex { pos: [nx1, ny1], texcoord: [u1, v1], color }); // BR
            verts.push(Vertex { pos: [nx0, ny0], texcoord: [u0, v0], color }); // TL
            verts.push(Vertex { pos: [nx1, ny1], texcoord: [u1, v1], color }); // BR
            verts.push(Vertex { pos: [nx0, ny1], texcoord: [u0, v1], color }); // BL

            cursor_x += 8;
        }

        if verts.is_empty() {
            return;
        }

        let vertex_ptr = rd.encoder.upload_slice(&verts);
        let draw_params = Params {
            screen_size: [rd.width as i32, rd.height as i32],
            font_tex: self.font_tex.texture_handle(),
            sampler: self.sampler.device_handle(),
            vertices: vertex_ptr,
            bg_color: [0, 0, 0, 0],
        };

        rd.encoder.bind_graphics_pipeline(&shader::text);
        rd.encoder.draw(crate::PrimitiveTopology::TriangleList, None, 0..verts.len() as u32, 0..1, &draw_params);
    }

    pub fn fill_rect(&mut self, rd: &mut RenderData, color: [u8; 4], x: i32, y: i32, width: i32, height: i32) {
        if width < 0 || height < 0 {
            return;
        }

        let sw = rd.width as f32;
        let sh = rd.height as f32;

        let x0 = x as f32;
        let x1 = (x + width) as f32;
        let y0 = y as f32;
        let y1 = (y + height) as f32;

        let nx0 = 2.0 * x0 / sw - 1.0;
        let nx1 = 2.0 * x1 / sw - 1.0;
        let ny0 = 2.0 * y0 / sh - 1.0;
        let ny1 = 2.0 * y1 / sh - 1.0;

        let verts = [
            Vertex { pos: [nx0, ny0], texcoord: [0.0, 0.0], color }, // TL
            Vertex { pos: [nx1, ny0], texcoord: [1.0, 0.0], color }, // TR
            Vertex { pos: [nx1, ny1], texcoord: [1.0, 1.0], color }, // BR
            Vertex { pos: [nx0, ny0], texcoord: [0.0, 0.0], color }, // TL
            Vertex { pos: [nx1, ny1], texcoord: [1.0, 1.0], color }, // BR
            Vertex { pos: [nx0, ny1], texcoord: [0.0, 1.0], color }, // BL
        ];

        let vertex_ptr = rd.encoder.upload_slice(&verts);
        let params = Params {
            screen_size: [rd.width, rd.height],
            vertices: vertex_ptr,
            font_tex: self.font_tex.texture_handle(),
            sampler: self.sampler.device_handle(),
            bg_color: [0, 0, 0, 0],
        };

        rd.encoder.bind_graphics_pipeline(&shader::fill);
        rd.encoder.draw(crate::PrimitiveTopology::TriangleList, None, 0..6, 0..1, &params);
    }

    pub fn render(&mut self, target: &Image) {
        let width = target.width() as i32;
        let height = target.height() as i32;
        crate::render(&[ColorAttachment { image: target, clear: None }], None, |encoder| {
            self.render_inner(&mut RenderData { encoder, width, height })
        });
    }

    fn render_inner(&mut self, rd: &mut RenderData) {
        // TODO: Render debugger UI elements here, such as text, overlays, etc.
        self.fill_rect(rd, [0, 0, 0, 128], 20, 20, rd.width - 40, rd.height - 40); // Semi-transparent background
        self.draw_text(rd, 30, 30, "Debugger UI");
    }
}

impl Debugger {
    pub fn render(&mut self, target: &Image) {
        let mut renderer = self.renderer.get_or_insert_with(|| Renderer::new());
        renderer.render(target);
    }
}
