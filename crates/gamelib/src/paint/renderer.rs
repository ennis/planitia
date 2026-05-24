//! Renderer backend.
mod dump_svg;
mod prepare;

use crate::asset::AssetLoadError;
use crate::paint::fill::Fill;
use crate::paint::renderer::prepare::prepare_scene;
use crate::paint::{
    GradientExtendMode, GradientRampData, GradientIntegralSegment, PaintRenderParams, Painter, PathVerb,
};
use crate::static_assets;
use color::Srgba8;
use gpu::PrimitiveTopology::TriangleList;
use gpu::{ClearColorValue, CommandBuffer, InvalidateFlags, Ptr, root_params};
use math::{IVec2, Mat3, Rect, U8Vec4, Vec2, ivec2, uvec2, UVec2, Vec4};
use std::ops::Range;
use std::sync::Once;

//--------------------------------------------------------------------------------------------------

/// Size of a raster tile in pixels. The rasterization compute shader processes the scene in tiles of this size.
///
/// NOTE: This should be kept in sync with `paint.slang`
const RASTER_TILE_SIZE: u32 = 16;

// At the moment the shader uses wave ops to compute a prefix sum along the tile rows, which assumes that the tile size doesn't
// exceed the wave size.
const _: () = assert!(RASTER_TILE_SIZE <= 32, "RASTER_TILE_SIZE must not exceed the minimum shader subgroup size (32)");

static_assets! {
    static RASTER_LINES: gpu::ComputePipeline = "/gamelib/shaders/gamelib_shaders.sharc#paint/raster_lines";
    static RASTER_TILES: gpu::ComputePipeline = "/gamelib/shaders/gamelib_shaders.sharc#paint/raster_tiles";
    static COPY_TO_SCREEN: gpu::GraphicsPipeline = "/gamelib/shaders/gamelib_shaders.sharc#paint/copy_to_screen";
}

//--------------------------------------------------------------------------------
// Scene

#[derive(Clone)]
pub(super) enum DrawCommand {
    SetTransform(Mat3),
    FillPath { verb_range: Range<usize>, base_vertex: usize, fill: usize },
}

#[derive(Default)]
pub(super) struct Scene {
    // List of contours (closed segments with the same
    //pub(super) contours: Vec<Contour>,
    pub(super) path_verbs: Vec<PathVerb>,
    pub(super) path_points: Vec<Vec2>,
    pub(super) fills: Vec<Fill>,
    pub(super) draw_commands: Vec<DrawCommand>,
    pub(super) gradient_integral_segments: Vec<GradientIntegralSegment>,
    pub(super) gradient_ramps: Vec<GradientRampData>,

    // List of paths (ranges into `contours`)
    //pub(super) paths: Vec<Range<usize>>, // range of contours
    pub(super) paths_bbox: Rect,
}

impl Scene {
    pub(super) fn is_fill_opaque(&self, fill_index: usize) -> bool {
        match self.fills[fill_index] {
            Fill::Solid(color) => color.is_opaque(),
            Fill::Texture { .. } => false,
            Fill::LinearGradient(g) => self.gradient_ramps[g.ramp].opaque,
        }
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

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct TileCoverList {
    /// Offset into the `TileCover` array (`RasterTilesParams::covers`).
    offset: u32,
    /// Number of `TileCover`s.
    count: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum FillType {
    Solid = 0,
    Texture = 1,
    LinearGradient = 2,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct SolidFill {
    tag: u32 = FillType::Solid as u32,
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
    tag: u32 = FillType::Texture as u32,
    texture: gpu::TextureHandle,
    sampler: gpu::SamplerHandle,
    local_to_uv: Mat3,
}

impl TextureFill {
    fn to_fill_data(&self) -> FillData {
        FillData { texture_fill: *self }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct LinearGradientFill {
    tag: u32 = FillType::LinearGradient as u32,
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

/// Shader parameters for the `raster_tiles` pass.
#[derive(Copy, Clone)]
#[repr(C)]
struct RasterTilesParams {
    segments: Ptr<U8Vec4>,
    covers: Ptr<TileCover>,
    cover_count: u32,
    cover_lists: Ptr<TileCoverList>,
    cover_list_count: u32,
    fills: Ptr<FillData>,
    gradient_integral_segments: Ptr<GradientIntegralSegment>,
    output: gpu::StorageImageHandle,
}

//--------------------------------------------------------------------------------

/// Prepared scene data to be uploaded to the GPU.
struct PreparedSceneData {
    /// Path segments clipped to tiles.
    clip_segs: Vec<U8Vec4>,
    tile_covers: Vec<TileCover>,
    tile_cover_lists: Vec<TileCoverList>,
    fills: Vec<FillData>,
}

static DEBUG_DUMP_SCENE: Once = Once::new();

pub(super) fn render_scene(
    cb: &mut CommandBuffer,
    painter: &mut Painter,
    params: &PaintRenderParams,
    scene: &Scene,
) -> Result<(), AssetLoadError> {
    if scene.draw_commands.is_empty() {
        return Ok(());
    }

    let viewport_size = uvec2(params.color_target.width(), params.color_target.height());
    let prep_scene = prepare_scene(scene, viewport_size);

    DEBUG_DUMP_SCENE.call_once(|| {
        //prep_scene.dump();
        if let Err(e) = prep_scene.write_svg("debug_scene.svg") {
            eprintln!("Failed to write debug SVG: {e}");
        }
    });

    if prep_scene.tile_cover_lists.is_empty() {
        // nothing to draw (everything was culled)
        return Ok(());
    }

    //prep_scene.dump();
    //eprintln!("tiles={:?}", prep_scene.tile_segs);

    let width = params.color_target.width();
    let height = params.color_target.height();
    painter.coverage_target.setup(width, height);

    let raster_tiles_params = RasterTilesParams {
        segments: cb.upload_slice(&prep_scene.clip_segs[..]),
        covers: cb.upload_slice(&prep_scene.tile_covers[..]),
        cover_count: prep_scene.tile_covers.len() as u32,
        cover_lists: cb.upload_slice(&prep_scene.tile_cover_lists[..]),
        cover_list_count: prep_scene.tile_cover_lists.len() as u32,
        fills: cb.upload_slice(&prep_scene.fills[..]),
        gradient_integral_segments: cb.upload_slice(&scene.gradient_integral_segments),
        output: painter.coverage_target.storage_handle(),
    };

    // Clear render target
    cb.clear_image(painter.coverage_target.image(), ClearColorValue::Float([0.0, 0.0, 0.0, 0.0]));

    // Draw tiles
    cb.bind_compute_pipeline(&*RASTER_TILES.read()?);
    cb.dispatch(prep_scene.tile_cover_lists.len() as u32, 1, 1, &raster_tiles_params);

    cb.barrier(InvalidateFlags::TEXTURE);

    // Copy render target to screen
    let mut encoder = cb.begin_rendering(
        &[gpu::ColorAttachment { image: &params.color_target, clear: None }],
        params.depth_target.as_ref().map(|d| gpu::DepthStencilAttachment {
            image: d,
            depth_clear: None,
            stencil_clear: None,
        }),
    );
    encoder.bind_graphics_pipeline(&*COPY_TO_SCREEN.read()?);
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
