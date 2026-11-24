//! Vector primitives and text rendering.
#![feature(default_field_values)]
mod atlas;
mod lines;
mod text;

use crate::atlas::Atlas;
use crate::text::glyph_cache::GlyphCache;
use color::Srgba8;
use gpu::PrimitiveTopology::TriangleList;
use gpu::vulkan::{VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_R8G8B8A8_UNORM};
use gpu::{root_params, Ptr, BARRIER_TEXTURE};
use math::{u16vec2, uvec2, vec2, vec3, IRect, Mat4, Rect, U16Vec2, UVec2, Vec2, Vec3};
use std::cell::RefCell;
use std::{mem, slice};

pub use text::{DrawGlyphRunOptions, Font, GlyphRun, TextFormat, TextLayout, FormatProperty};

//--------------------------------------------------------------------------------------------------

/// Draws geometric primitives to the specified color and depth targets.
pub fn draw(color_target: &gpu::Image, depth_target: Option<&gpu::Image>, draw_fn: impl FnOnce(&mut Draft)) {
    let mut draft = Draft::new(color_target, None);
    draw_fn(&mut draft);
    draft.render();
}

pub struct Draft<'a> {
    target: &'a gpu::Image,
    depth_target: Option<&'a gpu::Image>,
    clip: Rect,
    transform: Mat4,
    font: Font,
    d: DraftData,
    current_draw: Draw,
}

impl<'a> Draft<'a> {
    fn new(target: &'a gpu::Image, depth_target: Option<&'a gpu::Image>) -> Self {
        // verify that the image format is supported
        // TODO: if not supported, allocate an internal render target and blit to the final target
        assert_eq!(target.format(), VK_FORMAT_R8G8B8A8_UNORM, "unsupported image format: {:?}", target.format());
        if let Some(depth_target) = depth_target {
            assert_eq!(
                depth_target.format(),
                VK_FORMAT_D32_SFLOAT_S8_UINT,
                "unsupported depth image format: {:?}",
                depth_target.format()
            );
        }

        let width = target.width() as f32;
        let height = target.height() as f32;
        let clip = Rect::from_min_max(Vec2::ZERO, vec2(width, height));
        let draft_data = thread_local_resources().tmp_data.take();

        Draft {
            target,
            depth_target,
            clip,
            transform: Mat4::IDENTITY,
            font: Font::default_regular().clone(),
            d: draft_data,
            current_draw: Draw {
                kind: DrawKind::Draw,
                base_vertex: 0,
                start_index: 0,
                index_count: 0,
                fill: Fill::Color(Srgba8::WHITE),
            },
        }
    }

    /// Pushes the current draw state on the stack.
    ///
    /// The draw state includes:
    /// - the current transformation ([`append_transform`](Draft::append_transform))
    /// - the current clip rectangle ([`set_clip_rect`](Draft::set_clip_rect))
    /// - the current font ([`set_font`](Draft::set_font))
    pub fn save(&mut self) {
        self.d.stack.push(State { matrix: self.transform, clip: self.clip, font: self.font.clone() });
    }

    /// Pops and restores draw state from the state stack.
    pub fn restore(&mut self) {
        let State { matrix, clip, font } = self.d.stack.pop().expect("unbalanced calls to save/restore");
        self.transform = matrix;
        self.clip = clip;
        self.font = font;
    }

    /// Sets the current font.
    pub fn set_font(&mut self, font: Font) {
        self.font = font;
    }

    /// Appends to the current transformation matrix.
    ///
    /// The current matrix `M` becomes `M = M * mat`.
    pub fn append_transform(&mut self, mat: &Mat4) {
        self.transform = self.transform * mat;
    }

    /// Sets up a 2D transformation matrix that maps the specified 2D viewport rectangle to the screen.
    pub fn set_viewport_transform(&mut self, viewport: IRect) {
        let w = viewport.width() as f32;
        let h = viewport.height() as f32;
        let xmin = viewport.min.x as f32;
        let ymin = viewport.min.y as f32;
        let mat = Mat4::from_cols_array(&[
            2.0 / w, 0.0, 0.0, 0.0,
            0.0, 2.0 / h, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            - (2.0 * xmin/w + 1.0),
            - (2.0 * ymin/h + 1.0),
            0.0,
            1.0,
        ]);
        self.transform = mat;
    }

    /// Sets the current clip rectangle to the intersection with the specified rectangle.
    ///
    /// The current clip rectangle `R` becomes `R = intersection(R, rect)`
    pub fn set_clip_rect(&mut self, rect: &Rect) {
        self.clip = self.clip.intersect(rect).unwrap_or_default();
    }

    /// Draws a quad.
    ///
    /// The quad vertices are wound as follows:
    /// ```
    /// a --- b
    /// |     |
    /// d-----c
    /// ```
    ///
    /// # Arguments
    /// * a vertex A
    /// * b vertex B
    /// * c vertex C
    /// * d vertex D
    /// * fill how to fill the quad
    pub fn draw_quad(&mut self, a: Vertex, b: Vertex, c: Vertex, d: Vertex, fill: Fill) {
        if fill != self.current_draw.fill {
            self.new_draw(DrawKind::Draw, fill);
        }
        self.append_vertices(&[a, b, c, d], &[0, 1, 2, 0, 2, 3]);
    }

    fn append_quad(&mut self, a: Vertex, b: Vertex, c: Vertex, d: Vertex) {
        self.append_vertices(&[a, b, c, d], &[0, 1, 2, 0, 2, 3]);
    }

    fn append_rect(&mut self, min: Vertex, max: Vertex) {
        self.append_quad(
            min,
            Vertex { pos: vec3(max.pos.x, min.pos.y, 0.0), uv: u16vec2(max.uv.x, min.uv.y), color: min.color },
            max,
            Vertex { pos: vec3(min.pos.x, max.pos.y, 0.0), uv: u16vec2(min.uv.x, max.uv.y), color: min.color },
        );
    }

    fn new_draw(&mut self, kind: DrawKind, fill: Fill) {
        if self.current_draw.kind == kind && self.current_draw.fill == fill {
            return;
        }
        if self.current_draw.index_count > 0 {
            self.d.draws.push(self.current_draw.clone());
        }
        self.current_draw = Draw {
            kind,
            base_vertex: self.d.vertices.len() as u32,
            start_index: self.d.indices.len() as u32,
            index_count: 0,
            fill,
        };
    }

    /// Appends vertices and indices to the current draw.
    fn append_vertices(&mut self, vertices: &[Vertex], indices: &[u32]) {
        let vertex_count = self.d.vertices.len() as u32 - self.current_draw.base_vertex;
        self.current_draw.index_count += indices.len() as u32;
        self.d.vertices.extend_from_slice(vertices);
        self.d.indices.extend(indices.iter().map(|i| i + vertex_count));
    }

    pub fn render(mut self) {
        if self.current_draw.index_count > 0 {
            self.d.draws.push(self.current_draw.clone());
        }
        if self.d.draws.is_empty() {
            return;
        }
        self.render_internal();
    }
}


/// 3D vertices passed to the shaders.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Vertex {
    pub pos: Vec3,
    pub color: Srgba8,
    pub uv: U16Vec2,
}

/// Describes how to fill a shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fill {
    /// Solid color fill.
    Color(Srgba8),
    /// Texture fill.
    Texture { texture: gpu::TextureHandle, sampler: gpu::SamplerHandle },
}

impl Default for Fill {
    fn default() -> Self {
        Fill::Color(Srgba8::WHITE)
    }
}


struct State {
    matrix: Mat4,
    clip: Rect,
    font: Font,
}

/// Represents a single draw call.
#[derive(Default, Clone)]
struct Draw {
    kind: DrawKind,
    base_vertex: u32,
    start_index: u32,
    index_count: u32,
    fill: Fill,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
enum DrawKind {
    #[default]
    Draw,
    DrawLines,
}

#[derive(Default)]
struct DraftData {
    stack: Vec<State>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    draws: Vec<Draw>,
}

#[gpu::shader_module("shaders/draft.slang#33")]
mod shaders {}

impl<'a> Draft<'a> {
    fn render_internal(mut self) {
        let screen_size = vec2(self.target.width() as f32, self.target.height() as f32);
        // Upload vertex/index data to the GPU.
        let vertices = gpu::alloc_temp_slice(&self.d.vertices[..]);
        let indices = gpu::alloc_temp_slice(&self.d.indices[..]);
        // Upload atlas data to the GPU
        let mut texture_atlas = thread_local_resources().texture_atlas.borrow_mut();
        texture_atlas.update_image();
        gpu::barrier(BARRIER_TEXTURE);  // make atlas data visible to subsequent draws
        let sampler = gpu::register_sampler(&gpu::SamplerParams::default());
        gpu::render(
            &[gpu::ColorAttachment { image: self.target, clear: None }],
            self.depth_target.map(|image| gpu::DepthStencilAttachment {
                image,
                depth_clear: None,
                stencil_clear: None,
            }),
            |encoder| {
                for draw in &self.d.draws {
                    match draw.kind {
                        DrawKind::Draw => {
                            let (texture, sampler) = match draw.fill {
                                Fill::Color(_) => (texture_atlas.texture_handle(), sampler),
                                Fill::Texture { texture, sampler } => (texture, sampler),
                            };
                            encoder.bind_graphics_pipeline(&*shaders::draft);
                            encoder.draw(
                                TriangleList,
                                None,
                                0..draw.index_count,
                                0..1,
                                root_params! {
                                    matrix: Mat4 = self.transform,
                                    vertices: Ptr<Vertex> = vertices.offset(draw.base_vertex as usize),
                                    indices: Ptr<u32> = indices.offset(draw.start_index as usize),
                                    screen_size: Vec2 = screen_size,
                                    texture: gpu::TextureHandle = texture,
                                    sampler: gpu::SamplerHandle = sampler
                                },
                            );
                        }
                        DrawKind::DrawLines => {
                            // TODO: not implemented
                        }
                    }
                }
            },
        );
    }
}

//--------------------------------------------------------------------------------------------------

/// Converts a texel coordinate into u16 normalized UV coordinates.
///
/// Equivalent to `pos / texture_size * 65535`.
pub(crate) fn texel_to_normalized_texcoord(pos: Vec2, texture_size: UVec2) -> U16Vec2 {
    u16vec2(((pos.x / texture_size.x as f32) * 65535.0) as u16, ((pos.y / texture_size.y as f32) * 65535.0) as u16)
}

pub(crate) unsafe fn slice_to_u8<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        let len = size_of::<T>() * slice.len();
        slice::from_raw_parts_mut(slice.as_ptr() as *mut u8, len)
    }
}

//--------------------------------------------------------------------------------------------------

/// Returns a reference to this thread's [`ThreadLocalResources`].
pub(crate) fn thread_local_resources() -> &'static ThreadLocalResources {
    RESOURCES.with(|global| *global)
}

/// Thread-local resources & caches shared between [`Draft`] instances.
pub(crate) struct ThreadLocalResources {
    pub(crate) glyph_cache: RefCell<GlyphCache>,
    pub(crate) texture_atlas: RefCell<Atlas>,
    // Texture coordinate of a white pixel in the atlas,
    // used for drawing solid colors without additional logic in the shader.
    pub(crate) white_pixel_uv: U16Vec2,
    pub(crate) white_pixel_uv_f: Vec2,
    // Reusable draft data, to avoid reallocating internal vectors every time.
    pub(crate) tmp_data: RefCell<DraftData>,
}

thread_local! {
    static RESOURCES: &'static ThreadLocalResources = init_thread_local_resources();
}

fn init_thread_local_resources() -> &'static ThreadLocalResources {
    let (texture_atlas, white_pixel_uv) = init_atlas();
    Box::leak(Box::new(ThreadLocalResources {
        glyph_cache: RefCell::new(GlyphCache::new()),
        texture_atlas: RefCell::new(texture_atlas),
        white_pixel_uv,
        white_pixel_uv_f: Vec2::ZERO,
        tmp_data: RefCell::new(DraftData::default()),
    }))
}

/// Initializes the paint texture atlas.
///
/// Returns the atlas and the UV coordinate of a white pixel in the atlas.
fn init_atlas() -> (Atlas, U16Vec2) {
    let mut atlas = Atlas::new(1024, 1024);
    let rect = atlas.try_write(1, 1, &[Srgba8::WHITE], 1, 1).unwrap();
    let pos = texel_to_normalized_texcoord(
        vec2(rect.min.x as f32 + 0.5, rect.min.y as f32 + 0.5),
        uvec2(atlas.width, atlas.height),
    );
    (atlas, pos)
}
