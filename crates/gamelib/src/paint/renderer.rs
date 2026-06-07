//! Renderer backend.
mod dump_svg;
mod prepare;

use crate::asset::AssetLoadError;
use crate::paint::fill::Fill;
use crate::paint::renderer::prepare::prepare_scene;
use crate::paint::scene::GroupOptions;
use crate::paint::{
    GradientExtendMode, GradientIntegralSegment, GradientRampData, PaintRenderParams, Painter, PathVerb,
};
use crate::static_assets;
use crate::util::env_flag;
use color::Srgba8;
use gpu::PrimitiveTopology::TriangleList;
use gpu::{ClearColorValue, CommandBuffer, InvalidateFlags, Ptr, root_params};
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

/// Shader parameters for the `raster_tiles` pass.
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

/// Prepared scene data to be uploaded to the GPU.
struct PreparedSceneData {
    /// Path segments clipped to tiles.
    clip_segs: Vec<U8Vec4>,
    covers: Vec<TileCover>,
    tiles: Vec<TileDrawData>,
    commands: Vec<Command>,
}

pub(super) fn render_scene(
    cb: &mut CommandBuffer,
    painter: &mut Painter,
    params: &PaintRenderParams,
    scene: &Scene,
) -> Result<(), AssetLoadError> {

    // load shaders
    let rasterize_tiles = RASTERIZE_TILES.read()?;
    let copy_to_screen = COPY_TO_SCREEN.read()?;

    // clear coverage target
    let width = params.color_target.width();
    let height = params.color_target.height();
    painter.coverage_target.setup(width, height);
    cb.clear_image(painter.coverage_target.image(), ClearColorValue::Float([0.0; 4]));

    if !scene.draw_commands.is_empty() {

        // prepare the scene for rendering
        let prep_scene = prepare_scene(scene, uvec2(params.color_target.width(), params.color_target.height()));

        if env_flag("PAINTER_DUMP_SCENE") {
            // dump prepared scene to SVG for debugging
            static DEBUG_DUMP_SCENE: Once = Once::new();
            DEBUG_DUMP_SCENE.call_once(|| {
                if let Err(e) = prep_scene.write_svg("debug_scene.svg") {
                    eprintln!("Failed to write debug SVG: {e}");
                }
            });
        }

        if prep_scene.tiles.is_empty() {
            // nothing to draw (everything was culled)
            return Ok(());
        }

        // upload scene data to GPU
        let scene_data = {
            let segments = cb.upload_slice(&prep_scene.clip_segs[..]);
            let covers = cb.upload_slice(&prep_scene.covers[..]);
            let cover_count = prep_scene.covers.len() as u32;
            let cover_lists = cb.upload_slice(&prep_scene.tiles[..]);
            let fills = cb.upload_slice(&scene.fills[..]);
            let gradient_integral_segments = cb.upload_slice(&scene.gradient_integral_segments);
            SceneData { segments, covers, cover_count, tiles: cover_lists, gradient_integral_segments, fills }
        };

        // execute draw commands
        for cmd in prep_scene.commands.iter() {
            match cmd {
                Command::Clear(color) => {
                    cb.clear_image(painter.coverage_target.image(), ClearColorValue::Float(color.to_linear_array()));
                }
                Command::RasterizeTiles { start, count } => {
                    let output = painter.coverage_target.storage_handle();
                    let raster_tiles_params = RasterizeTilesParams { scene_data, output, start_cover_list: *start };
                    cb.barrier(InvalidateFlags::STORAGE);
                    cb.bind_compute_pipeline(&*rasterize_tiles);
                    cb.dispatch(*count, 1, 1, &raster_tiles_params);
                }
            }
        }
    }

    // Copy render target to screen
    cb.barrier(InvalidateFlags::TEXTURE);

    let clear_color = scene.clear_color.to_linear_array();
    let clear_color = [clear_color[0] as f64, clear_color[1] as f64, clear_color[2] as f64, clear_color[3] as f64];
    let mut encoder = cb.begin_rendering(
        &[gpu::ColorAttachment { image: &params.color_target, clear: Some(clear_color) }],
        params.depth_target.as_ref().map(|d| gpu::DepthStencilAttachment {
            image: d,
            depth_clear: None,
            stencil_clear: None,
        }),
    );
    encoder.bind_graphics_pipeline(&*copy_to_screen);
    encoder.draw(
        TriangleList,
        None,
        0..6,
        0..1,
        root_params! {
            texture: gpu::StorageImageHandle = painter.coverage_target.storage_handle(),
            sampler: gpu::SamplerHandle = painter.sampler.device_handle()
        },
    );
    encoder.finish();

    Ok(())
}
