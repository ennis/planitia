use crate::asset::AssetLoadError;
use crate::paint::fill::Fill;
use crate::paint::flatten::Contour;
use crate::paint::tessellation::Mesh;
use crate::paint::{PaintRenderParams, PaintVertex, Painter};
use crate::static_assets;
use color::Srgba8;
use gpu::PrimitiveTopology::TriangleList;
use gpu::{CommandBuffer, InvalidateFlags, Ptr, PushDataSource, root_params};
use math::{Mat3, Rect, Vec2, Vec3, uvec2, vec2, vec3};
use std::ops::Range;

pub(super) struct DrawOp {
    pub(super) mesh: Mesh,
    pub(super) clip: Rect,
    pub(super) device_to_local: Mat3,
    pub(super) fill: Fill,
}

pub(super) struct RenderScene {
    pub(super) vertices: Vec<PaintVertex>,
    pub(super) indices: Vec<u32>,
    pub(super) ops: Vec<DrawOp>,
}

impl Default for RenderScene {
    fn default() -> Self {
        Self { vertices: vec![], indices: vec![], ops: vec![] }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct PaintRootParams {
    device_to_uv_transform: Mat3,
    screen_size: [f32; 2],
    line_width: f32,
    color: Srgba8,
    texture: gpu::TextureHandle = gpu::TextureHandle::INVALID,
    sampler: gpu::SamplerHandle = gpu::SamplerHandle::INVALID,
}

pub(super) fn render_scene(
    cmd: &mut CommandBuffer,
    painter: &mut Painter,
    params: &PaintRenderParams,
    scene: &RenderScene,
) {
    assert_eq!(params.color_target.format(), painter.color_format, "mismatched color target format");
    assert_eq!(
        params.depth_target.as_ref().map(|d| d.format()),
        painter.depth_format,
        "mismatched depth target format"
    );

    // prepare texture atlas
    let _atlas = painter.texture_atlas.prepare_texture(cmd);

    // setup encoder
    let mut encoder = cmd.begin_rendering(
        &[gpu::ColorAttachment { image: &params.color_target, clear: None }],
        params.depth_target.as_ref().map(|d| gpu::DepthStencilAttachment {
            image: d,
            depth_clear: None,
            stencil_clear: None,
        }),
    );

    let width = params.color_target.width();
    let height = params.color_target.height();
    encoder.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    encoder.set_scissor(0, 0, width, height);
    encoder.bind_graphics_pipeline(&painter.pipelines.paint);

    for prim in scene.ops.iter() {
        if prim.clip.is_null() {
            continue;
        }

        let texture;
        let device_to_uv_transform;
        let mut color = Srgba8::WHITE;

        match prim.fill {
            Fill::Solid(solid_color) => {
                texture = painter.texture_atlas.texture_handle();
                device_to_uv_transform = Mat3::from_cols(
                    Vec3::ZERO,
                    Vec3::ZERO,
                    vec3(painter.white_pixel_uv_f.x, painter.white_pixel_uv_f.y, 1.0),
                );
                color = solid_color;
            }
            Fill::Texture { texture: tex, uv_transform } => {
                texture = tex;
                device_to_uv_transform = prim.device_to_local * uv_transform;
            }
        };

        let root_params = encoder.upload(&PaintRootParams {
            screen_size: [width as f32, height as f32],
            line_width: 1.0,
            device_to_uv_transform,
            texture,
            sampler: painter.sampler.device_handle(),
            color,
        });
        draw_mesh(&mut encoder, params, &prim.mesh, prim.clip, root_params);
    }
    encoder.finish();
}

fn set_scissor(encoder: &mut gpu::RenderEncoder, params: &PaintRenderParams, clip: Rect) {
    let width = params.color_target.width();
    let height = params.color_target.height();

    // Transform clip rect to physical pixels
    let pixels_per_point = 1.0;
    let clip_min_x = ((pixels_per_point * clip.min.x).round() as i32).clamp(0, width as i32);
    let clip_min_y = ((pixels_per_point * clip.min.y).round() as i32).clamp(0, height as i32);
    let clip_max_x = ((pixels_per_point * clip.max.x).round() as i32).clamp(clip_min_x, width as i32);
    let clip_max_y = ((pixels_per_point * clip.max.y).round() as i32).clamp(clip_min_y, height as i32);

    encoder.set_scissor(clip_min_x, clip_min_y, clip_max_x as u32, clip_max_y as u32);
}

fn draw_mesh(
    encoder: &mut gpu::RenderEncoder,
    params: &PaintRenderParams,
    mesh: &Mesh,
    clip: Rect,
    root_params: Ptr<PaintRootParams>,
) {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return;
    }
    let vertex_buffer = gpu::Buffer::from_slice(&mesh.vertices);
    let index_buffer = gpu::Buffer::from_slice(&mesh.indices);
    set_scissor(encoder, params, clip);
    encoder.draw_indexed(
        TriangleList,
        &index_buffer,
        0..mesh.indices.len() as u32,
        Some(vertex_buffer.as_bytes()),
        0,
        0..1,
        PushDataSource::Indirect(root_params),
    );
}

//--------------------------------------------------------------------------------------------------

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PathSegKind {
    Line,
    Quad,
    Cubic,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PathSeg {
    kind: PathSegKind,
    start_vertex: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PathBufGpu {
    segments: Ptr<PathSeg>,
    vertices: Ptr<Vec2>,
}

//--------------------------------------------------------------------------------------------------

#[derive(Default)]
pub(super) struct RenderScene2 {
    pub(super) vertices: Vec<Vec2>,
    pub(super) contours: Vec<Contour>,
    // indexed by path
    pub(super) fills: Vec<Fill>,
    pub(super) paths_bbox: Rect,
}

static_assets! {
    static RASTER_LINES: gpu::ComputePipeline = "/gamelib/shaders/gamelib_shaders.sharc#paint/raster_lines";
    static RASTER_TILES: gpu::ComputePipeline = "/gamelib/shaders/gamelib_shaders.sharc#paint/raster_tiles";
    static COPY_TO_SCREEN: gpu::GraphicsPipeline = "/gamelib/shaders/gamelib_shaders.sharc#paint/copy_to_screen";
}

#[derive(Copy, Clone)]
#[repr(C)]
struct RasterLinesParams {
    vertices: Ptr<Vec2>,
    start_vertex: u32,
    vertex_count: u32,
    output: gpu::StorageImageHandle,
}

const RASTER_LINES_WORKGROUP_SIZE: u32 = 64;
const RASTER_TILE_SIZE: u32 = 16;
/*
// vertices of segments
float2* vertices;
// lists of segments for each tile
uint* segments;
TileBin* tile_bins;
uint tile_bin_count;
uint* bin_offsets;
uint tile_count;
RWTexture2D<float4>.Handle output;*/

#[derive(Copy, Clone)]
#[repr(C)]
struct RasterTilesParams {
    segments: Ptr<TileSeg>,
    tile_bins: Ptr<TileBin>,
    tile_bin_count: u32,
    bin_offsets: Ptr<u32>,
    tile_count: u32,
    output: gpu::StorageImageHandle,
}

pub(super) fn render_scene_2(
    cb: &mut CommandBuffer,
    painter: &mut Painter,
    params: &PaintRenderParams,
    scene: &RenderScene2,
) -> Result<(), AssetLoadError> {
    if scene.vertices.len() < 1 {
        return Ok(());
    }

    let prep_scene = prepare_scene(scene);
    //prep_scene.dump();
    //eprintln!("tiles={:?}", prep_scene.tile_segs);

    let width = params.color_target.width();
    let height = params.color_target.height();
    painter.coverage_target.setup(width, height);

    //let vertices = cb.upload_slice(&scene.vertices[..]);
    let segs = cb.upload_slice(&prep_scene.tile_segs[..]);
    let bins = cb.upload_slice(&prep_scene.bins[..]);
    let bin_offsets = cb.upload_slice(&prep_scene.tile_offsets[..]);

    //let line_count = scene.vertices.len() - 1;

    cb.bind_compute_pipeline(&*RASTER_TILES.read()?);
    cb.dispatch(
        prep_scene.tile_offsets.len() as u32,
        1,
        1,
        &RasterTilesParams {
            segments: segs,
            tile_bins: bins,
            tile_bin_count: prep_scene.bins.len() as u32,
            bin_offsets,
            tile_count: prep_scene.tile_offsets.len() as u32,
            output: painter.coverage_target.storage_handle(),
        },
    );

    cb.barrier(InvalidateFlags::TEXTURE);

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

    //cb.bind_compute_pipeline(&*RASTER_LINES.read()?);
    //eprintln!("dispatching raster lines with {} vertices ({} lines)", scene.vertices.len(), line_count);
    //cb.dispatch(
    //    (line_count as u32).div_ceil(RASTER_LINES_WORKGROUP_SIZE),
    //    1,
    //    1,
    //    &RasterLinesParams {
    //        vertices,
    //        start_vertex: 0,
    //        vertex_count: line_count as u32,
    //        output: painter.coverage_target.storage_handle(),
    //    },
    //);

    Ok(())
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Tile {
    x: u32,
    y: u32,
    a: Vec2, // TODO this could have a reduced precision
    b: Vec2,
    winding: i32,
    path: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct TileSeg {
    a: Vec2,
    b: Vec2,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct TileBin {
    coords: math::UVec2,
    path: u32,
    winding: i32,
    seg_offset: u32,
    seg_count: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct TileCommandHeader {
    x: u32,
    y: u32,
    cmd_offset: u32,
    cmd_count: u32,
}

struct PreparedSceneData {
    /// Path segments clipped to tiles.
    tile_segs: Vec<TileSeg>,
    bins: Vec<TileBin>,
    tile_offsets: Vec<u32>,
}

impl PreparedSceneData {
    fn dump(&self) {
        eprintln!("bins:");
        let mut wg_index = 0;
        for (i, b) in self.bins.iter().enumerate() {
            if wg_index + 1 < self.tile_offsets.len() && i as u32 >= self.tile_offsets[wg_index + 1] {
                wg_index += 1;
            }
            eprintln!(
                "  [{}] tile=({},{}), path={}, winding={}, seg_offset={}, seg_count={} ({:?})",
                wg_index,
                b.coords.x,
                b.coords.y,
                b.path,
                b.winding,
                b.seg_offset,
                b.seg_count,
                &self.tile_segs[b.seg_offset as usize..(b.seg_offset + b.seg_count) as usize]
            );
        }
        eprintln!("bin_offsets: {:?}", self.tile_offsets);
    }
}

macro_rules! prof_scope {
    ($name:literal) => {
        let _prof_scope = gamelib::util::ProfilerScope::new(std::panic::Location::caller(), $name);
    };
}

fn prepare_scene(scene: &RenderScene2) -> PreparedSceneData {
    let _span = tracy_client::span!("prepare_scene");

    const TILE: i32 = RASTER_TILE_SIZE as i32;

    // 0. allocate a 2D tile map big enough for the whole drawing

    let mut bins: Vec<TileBin> = vec![];
    let mut tile_segs: Vec<TileSeg> = vec![];

    // emit tiles for each line in the contours
    //
    // the tiles will be sorted by path, since contours are sorted by path

    for c in scene.contours.iter() {
        let mut tiles: Vec<Tile> = vec![];
        let range = c.start as usize..c.end as usize;

        {
            let _span = tracy_client::span!("contour");
            for (i_segment, [p, q]) in scene.vertices[range].array_windows().enumerate() {
                // coarse conservative rasterization of line segment pq
                let d = q - p;
                let dxi = d.x.signum() as i32;
                let dyi = d.y.signum() as i32;

                let p_t_x = p.x as i32 / TILE;
                let q_t_x = q.x as i32 / TILE;
                let p_t_y = p.y as i32 / TILE;
                let q_t_y = q.y as i32 / TILE;

                let mut row_t_x = p_t_x; // current row start tile x coord
                let mut row_loc_x = p.x - (row_t_x * TILE) as f32; // current row start x coord in tile coord space

                let mut t_y = p_t_y;
                while t_y != q_t_y + dyi {
                    let y_1 = ((t_y + dyi.max(0)) * TILE) as f32;
                    // solve intersection of the segment with the bottom or top edge of the tile row
                    let t = (y_1 - p.y) / d.y;

                    // x coord of tile that contains the intersection
                    let isect = if t_y != q_t_y { p.x + t * d.x } else { q.x };
                    let isc_t_x = isect as i32 / TILE;
                    let isc_loc_x = isect - (isc_t_x * TILE) as f32;
                    let mut t_x = row_t_x;
                    while t_x != isc_t_x + dxi {
                        // compute winding number delta given segment entry and exits
                        let top_in = if t_x == row_t_x && t_y != p_t_y && dyi > 0 { -1 } else { 0 };
                        let top_out = if t_x == isc_t_x && t_y != q_t_y && dyi < 0 { 1 } else { 0 };
                        //let right_in = if t_x != row_t_x && dxi < 0 { -1 } else { 0 };
                        //let left_out = if t_x != isc_t_x && dxi < 0 { 1 } else { 0 };
                        let delta_cov = top_in + top_out /* + right_in + left_out*/;

                        //eprintln!("segment {} (path {}) intersects tile ({}, {}) isc_t_x={}, row_t_x={}, t_y={}, p_t_y={}, q_t_y={}, d=({:.1}, {:.1}), dxi={}/dyi={}",
                        //          i_segment, c.path, t_x, t_y, isc_t_x, row_t_x, t_y, p_t_y, q_t_y, d.x, d.y, dxi, dyi
                        //);

                        // clip pq to tile

                        let p_clip_x = if t_x == row_t_x { row_loc_x } else { ((-dxi).max(0) * TILE) as f32 };
                        let q_clip_x = if t_x == isc_t_x { isc_loc_x } else { (dxi.max(0) * TILE) as f32 };
                        // Local Y of the segment's entry point into this tile row:
                        // - first row: the start point p itself
                        // - going down: segment entered from the top edge (y=0)
                        // - going up:   segment entered from the bottom edge (y=TILE)
                        let row_entry_y_local = if t_y == p_t_y {
                            p.y - (t_y * TILE) as f32
                        } else if dyi > 0 {
                            0.0
                        } else {
                            TILE as f32
                        };
                        let p_clip_y;
                        let q_clip_y;
                        if isc_t_x == row_t_x {
                            p_clip_y = (p.y - (t_y * TILE) as f32).max(0.0).min(TILE as f32);
                            q_clip_y = (q.y - (t_y * TILE) as f32).max(0.0).min(TILE as f32);
                        } else {
                            p_clip_y = row_entry_y_local + d.y / d.x * (p_clip_x + ((t_x - row_t_x) * TILE) as f32 - row_loc_x);
                            q_clip_y = row_entry_y_local + d.y / d.x * (q_clip_x + ((t_x - row_t_x) * TILE) as f32 - row_loc_x);
                        }
                        //eprintln!("  tile ({}, {}), delta_cov={}, p_clip=({:.1}, {:.1}), q_clip=({:.1}, {:.1}) row_loc_x={row_loc_x}", t_x, t_y, delta_cov, p_clip_x, p_clip_y, q_clip_x, q_clip_y);
//
                        //debug_assert!(p_clip_x >= 0.0 && p_clip_x <= TILE as f32, "p_clip_x={}", p_clip_x);
                        //debug_assert!(q_clip_x >= 0.0 && q_clip_x <= TILE as f32, "q_clip_x={}", q_clip_x);
                        //debug_assert!(p_clip_y >= 0.0 && p_clip_y <= TILE as f32, "p_clip_y={}", p_clip_y);
                        //debug_assert!(q_clip_y >= 0.0 && q_clip_y <= TILE as f32, "q_clip_y={}", q_clip_y);

                        tiles.push(Tile {
                            x: t_x as u32,
                            y: t_y as u32,
                            a: vec2(p_clip_x, p_clip_y),
                            b: vec2(q_clip_x, q_clip_y),
                            winding: delta_cov,
                            path: c.path,
                        });
                        t_x += dxi;
                    }

                    row_t_x = isc_t_x;
                    row_loc_x = isc_loc_x;
                    t_y += dyi;
                }
            }
        }

        {
            let _span = tracy_client::span!("scanline_sort");
            // sort newly created tiles of the current contour in scanline order
            tiles.sort_by(|a, b| a.y.cmp(&b.y).then(a.x.cmp(&b.x)));
        }

        // merge into bins

        {
            let _span = tracy_client::span!("bin_merge");
            let mut x = tiles[0].x;
            let mut y = tiles[0].y;
            let mut i = 0;
            let mut winding = 0; // winding for the tile to the right
            let mut cmd = TileBin { coords: uvec2(x, y), path: tiles[0].path, winding: 0, seg_offset: 0, seg_count: 0 };

            while i < tiles.len() {
                let tile = tiles[i];

                if tile.y == y && tile.x != x {
                    // end current bin
                    bins.push(cmd);
                    cmd.winding = winding;
                    cmd.seg_count = 0;
                    if winding != 0 {
                        // emit empty bins for skipped tiles if we are inside the shape
                        for xi in x + 1..tile.x {
                            cmd.coords.x = xi;
                            cmd.coords.y = tile.y;
                            bins.push(cmd);
                        }
                    }
                    cmd.coords.x = tile.x;
                    cmd.seg_offset = tile_segs.len() as u32;
                } else if tile.y != y {
                    // new row
                    // end current bin
                    bins.push(cmd);
                    cmd.coords.x = tile.x;
                    cmd.coords.y = tile.y;
                    cmd.winding = 0;
                    cmd.seg_offset = tile_segs.len() as u32;
                    cmd.seg_count = 0;
                }

                tile_segs.push(TileSeg { a: tile.a, b: tile.b });
                x = tile.x;
                y = tile.y;
                i += 1;
                cmd.seg_count += 1;
                winding += tile.winding;
            }

            if cmd.seg_count > 0 {
                bins.push(cmd);
            }
        }
    }

    // sort bins by row, column, path
    {
        let _span = tracy_client::span!("bin_sort");
        bins.sort_by(|a, b| a.coords.y.cmp(&b.coords.y).then(a.coords.x.cmp(&b.coords.x)).then(a.path.cmp(&b.path)));
    }

    let mut tile_starts: Vec<u32> = vec![];

    {
        let _span = tracy_client::span!("tile_starts");
        let mut x = u32::MAX;
        let mut y = u32::MAX;
        for (i, b) in bins.iter().enumerate() {
            if b.coords.y != y || b.coords.x != x {
                x = b.coords.x;
                y = b.coords.y;
                tile_starts.push(i as u32);
            }
        }
    }

    PreparedSceneData { tile_segs, bins, tile_offsets: tile_starts }
}
