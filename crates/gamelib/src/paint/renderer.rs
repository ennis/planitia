//! Renderer backend.
mod build;
mod dump_svg;

use crate::asset::AssetError;
use crate::error::{Exc, ExcResult, ResultExt};
use crate::paint::fill::Fill;
use crate::paint::renderer::build::build_gpu_scene;
use crate::paint::scene::GroupOptions;
use crate::paint::{GradientExtendMode, GradientIntegralSegment, GradientRampData, Painter, PathVerb};
use crate::{gpu_span, static_assets};
use crate::util::env_flag;
use color::Srgba8;
use gpu::PrimitiveTopology::TriangleList;
use gpu::{ClearColorValue, CommandBuffer, BarrierFlags, Ptr, root_params};
use math::{IVec2, Mat3, Rect, U8Vec4, UVec2, Vec2, Vec4, ivec2, uvec2};
use std::ops::Range;
use std::sync::Once;

/// Size of a raster tile in pixels. The rasterization compute shader processes the scene in tiles of this size.
///
/// NOTE: This should be kept in sync with `paint.slang`
const TILE_SIZE: u32 = 16;

// At the moment the shader uses wave ops to compute a prefix sum along the tile rows, which assumes that the tile size doesn't
// exceed the wave size.
const _: () = assert!(TILE_SIZE <= 32, "TILE_SIZE must not exceed the minimum shader subgroup size (32)");

static_assets! {
    static RASTERIZE_TILES: gpu::ComputePipeline = "/gamelib/shaders/paint.sharc#rasterize_tiles";
    static COPY_TO_SCREEN: gpu::GraphicsPipeline = "/gamelib/shaders/paint.sharc#copy_to_screen";
}

#[derive(thiserror::Error, Debug, Copy, Clone)]
#[error("Scene render error")]
pub struct RenderSceneError;

#[derive(Clone)]
pub(super) enum DrawCommand {
    SetTransform(Mat3),
    Clear(Srgba8),
    BeginGroup(GroupOptions),
    FillPath { verb_range: Range<usize>, base_vertex: usize, fill: usize },
    EndGroup,
}

#[derive(Default)]
pub(super) struct Scene {
    pub(super) path_verbs: Vec<PathVerb>,
    pub(super) path_points: Vec<Vec2>,
    fills: Vec<FillData>,
    pub(super) clear_color: Srgba8,
    pub(super) draw_commands: Vec<DrawCommand>,
    pub(super) gradient_integral_segments: Vec<GradientIntegralSegment>,
    pub(super) gradient_ramps: Vec<GradientRampData>,
    pub(super) paths_bbox: Rect,
}

impl Scene {
    pub(super) fn new(clear_color: Srgba8) -> Self {
        Self { clear_color, ..Default::default() }
    }

    pub(super) fn register_fill(&mut self, fill: &Fill, local_to_device_transform: &Mat3) -> usize {
        let index = self.fills.len();
        let fill_data: FillData = match fill {
            Fill::Solid(color) => {
                let tag = make_fill_tag(FillType::Solid, color.is_opaque());
                SolidFill { tag, color: *color }.into()
            }
            Fill::Texture(tex) => {
                let tag = make_fill_tag(FillType::Texture, false);
                let device_to_local = local_to_device_transform.inverse();
                let device_to_uv = tex.local_to_uv * device_to_local;
                TextureFill {
                    tag,
                    texture: tex.texture,
                    sampler: tex.sampler,
                    device_to_tex_coords: device_to_uv,
                    color: tex.color,
                }
                .into()
            }
            Fill::LinearGradient(g) => {
                let ramp = &self.gradient_ramps[g.ramp];
                let tag = make_fill_tag(FillType::LinearGradient, ramp.opaque);
                let start = local_to_device_transform.transform_point2(g.start);
                let end = local_to_device_transform.transform_point2(g.end);
                LinearGradientFill {
                    tag,
                    start,
                    end,
                    seg_range: uvec2(ramp.segments.start as u32, ramp.segments.end as u32),
                    extend_mode: g.extend_mode,
                    integral: ramp.integral,
                }
                .into()
            }
        };
        self.fills.push(fill_data);
        index
    }
}

//--------------------------------------------------------------------------------
// GPU types

/// Represents a covering of a tile by one path.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct TileCover {
    /// Packed ID+tile coordinates.
    ///
    /// See `pack_tile_cover_id`.
    id: u64,
    /// Tile-level winding number (winding number at the top-left point of the tile).
    winding: i32,
    /// Index into the array of fill descriptions.
    fill: u32,
    /// Offset into the array of clipped segments (`RasterTilesParams::segments`).
    seg_offset: u32,
    /// Number of clipped segments.
    seg_count: u32,
}

impl TileCover {
    /// Returns the coordinates of the tile (in tiles, not pixels).
    ///
    /// E.g. for a tile covering pixels (32..48, 16..32), this would return (2, 1) if the tile size is 16.
    fn tile_coords(&self) -> IVec2 {
        let x = (self.id >> 32 & 0xFFFF) as i16;
        let y = (self.id >> 48 & 0xFFFF) as i16;
        ivec2(x as i32, y as i32)
    }

    /// Returns the path ID.
    fn path(&self) -> u32 {
        (self.id & 0xFFFFFFFF) as u32
    }
}

/// Packs tile coordinates and path ID into a single u64 for efficient sorting.
///
/// The format is: `(MSB) <y:16> <x:16> <path:32> (LSB)`
fn pack_tile_cover_id(x: i32, y: i32, path: u32) -> u64 {
    ((y as u64) << 48) | ((x as u64) << 32) | (path as u64)
}

/// Per-tile draw data.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct TileDrawData {
    /// Offset into the `TileCover` array (`RasterTilesParams::covers`).
    cover_offset: u32,
    /// Number of `TileCover`s.
    cover_count: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum FillType {
    Solid = 0,
    Texture = 1,
    LinearGradient = 2,
}

fn make_fill_tag(ty: FillType, opaque: bool) -> u32 {
    (ty as u32) | if opaque { 0x80000000 } else { 0 }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct SolidFill {
    tag: u32,
    color: Srgba8,
}

impl SolidFill {
    fn to_fill_data(&self) -> FillData {
        FillData { solid_fill: *self }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct TextureFill {
    tag: u32,
    texture: gpu::TextureHandle,
    sampler: gpu::SamplerHandle,
    device_to_tex_coords: Mat3,
    color: Srgba8,
}

impl TextureFill {
    fn to_fill_data(&self) -> FillData {
        FillData { texture_fill: *self }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct LinearGradientFill {
    tag: u32,
    extend_mode: GradientExtendMode,
    start: Vec2,
    end: Vec2,
    seg_range: UVec2,
    integral: Vec4,
}

impl LinearGradientFill {
    fn to_fill_data(&self) -> FillData {
        FillData { linear_gradient_fill: *self }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
union FillData {
    tag: u32,
    data: [u32; 16],
    texture_fill: TextureFill,
    solid_fill: SolidFill,
    linear_gradient_fill: LinearGradientFill,
}

impl FillData {
    fn is_opaque(&self) -> bool {
        let tag = unsafe { self.tag };
        (tag & 0x80000000) != 0
    }
}

impl From<SolidFill> for FillData {
    fn from(solid_fill: SolidFill) -> Self {
        FillData { solid_fill }
    }
}

impl From<TextureFill> for FillData {
    fn from(texture_fill: TextureFill) -> Self {
        FillData { texture_fill }
    }
}

impl From<LinearGradientFill> for FillData {
    fn from(linear_gradient_fill: LinearGradientFill) -> Self {
        FillData { linear_gradient_fill }
    }
}

const _: () = assert!(size_of::<FillData>() == 64, "FillData must be exactly 64 bytes");

#[derive(Copy, Clone)]
#[repr(C)]
struct SceneData {
    segments: Ptr<U8Vec4>,
    covers: Ptr<TileCover>,
    cover_count: u32,
    tiles: Ptr<TileDrawData>,
    fills: Ptr<FillData>,
    gradient_integral_segments: Ptr<GradientIntegralSegment>,
}

/// Shader parameters for the `rasterize_tiles` pass.
#[derive(Copy, Clone)]
#[repr(C)]
struct RasterizeTilesParams {
    scene_data: SceneData,
    output: gpu::StorageImageHandle,
    start_cover_list: u32,
}

//--------------------------------------------------------------------------------

enum Command {
    Clear(Srgba8),
    RasterizeTiles { start: u32, count: u32 },
}

struct GpuSceneData {
    clip_segs: Vec<U8Vec4>,
    covers: Vec<TileCover>,
    tiles: Vec<TileDrawData>,
    commands: Vec<Command>,
}

pub(super) fn render_scene(
    painter: &mut Painter,
    render_target: &gpu::Image,
    scene: &Scene,
) -> ExcResult<(), RenderSceneError> {

    let _span = gpu_span!("render_scene");

    let width = render_target.width();
    let height = render_target.height();
    // Ensure shaders are loaded.
    let rasterize_tiles = RASTERIZE_TILES.read().raise(RenderSceneError)?;
    let copy_to_screen = COPY_TO_SCREEN.read().raise(RenderSceneError)?;

    // Resize and clear internal render target.
    painter.render_target.setup(width, height);
    let clear_color = scene.clear_color.to_linear_array();
    gpu::clear_image(painter.render_target.image(), ClearColorValue::Float(clear_color));

    if !scene.draw_commands.is_empty() {
        // Build the GPU data structures of the scene for rendering.
        let prep_scene = build_gpu_scene(scene, width, height);

        if env_flag("PAINTER_DUMP_SCENE") {
            // Debug: dump scene data to SVG.
            static DEBUG_DUMP_SCENE: Once = Once::new();
            DEBUG_DUMP_SCENE.call_once(|| {
                if let Err(e) = prep_scene.write_svg("debug_scene.svg") {
                    eprintln!("Failed to write debug SVG: {e}");
                }
            });
        }

        if prep_scene.tiles.is_empty() {
            // Nothing to draw (everything was culled).
            return Ok(());
        }

        // Upload scene data to GPU.
        let scene_data = {
            let segments = gpu::alloc_temp_slice(&prep_scene.clip_segs[..]);
            let covers = gpu::alloc_temp_slice(&prep_scene.covers[..]);
            let cover_count = prep_scene.covers.len() as u32;
            let cover_lists = gpu::alloc_temp_slice(&prep_scene.tiles[..]);
            let fills = gpu::alloc_temp_slice(&scene.fills[..]);
            let gradient_integral_segments = gpu::alloc_temp_slice(&scene.gradient_integral_segments);
            SceneData { segments, covers, cover_count, tiles: cover_lists, gradient_integral_segments, fills }
        };

        // Execute draw commands.
        for command in prep_scene.commands.iter() {
            match command {
                Command::Clear(color) => {
                    gpu::clear_image(painter.render_target.image(), ClearColorValue::Float(color.to_linear_array()));
                }
                Command::RasterizeTiles { start, count } => {
                    let _span = gpu_span!("rasterize_tiles");
                    let output = painter.render_target.storage_handle();
                    let raster_tiles_params = RasterizeTilesParams { scene_data, output, start_cover_list: *start };
                    gpu::barrier(BarrierFlags::STORAGE);
                    gpu::dispatch(&*rasterize_tiles, *count, 1, 1, &raster_tiles_params);
                }
            }
        }
    }

    // Copy internal render target to provided render target.
    gpu::barrier(BarrierFlags::TEXTURE);

    {
        let _span = gpu_span!("copy_to_screen");
        gpu::render(&[gpu::ColorAttachment { image: &render_target, clear: None }], None, |encoder| {
            encoder.bind_graphics_pipeline(&*copy_to_screen);
            encoder.draw(
                TriangleList,
                None,
                0..6,
                0..1,
                root_params! {
                    texture: gpu::StorageImageHandle = painter.render_target.storage_handle(),
                    sampler: gpu::SamplerHandle = painter.sampler.device_handle()
                },
            );
        });
    }

    Ok(())
}
