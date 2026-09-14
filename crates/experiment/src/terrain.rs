use gamelib::render::RenderTarget;
use gpu::{ImageUsage, BARRIER_STORAGE, BARRIER_TEXTURE};
use gpu::vulkan::VK_FORMAT_R8G8B8A8_UNORM;
use image::ImageReader;
use math::{vec2, vec4, Camera, Mat4, Rect, U8Vec4, UVec2, Vec2, Vec3, Vec4, uvec2};
use std::mem;
use std::ops::{Add, Div, Mul, Sub};
use std::path::Path;
use std::range::Range;

mod stack;

pub use stack::{MatLayer, downsample_material_stacks};

#[gpu::shader_module("shaders/terrain.slang#14")]
mod terrain_shader {}

pub const TERRAIN_FEATURE_COUNT: usize = 8;
pub type MatVecU8 = [u8; TERRAIN_FEATURE_COUNT];

/// 8-dimensional feature vector representing the material in a terrain cell.
#[derive(Copy, Clone, Debug)]
pub struct MatVec {
    pub a: Vec4,
    pub b: Vec4,
}

impl MatVec {
    pub fn lerp(self, other: MatVec, t: f32) -> MatVec {
        MatVec { a: self.a.lerp(other.a, t), b: self.b.lerp(other.b, t) }
    }

    /// Packs the vector to 8x8-bit components.
    pub fn pack(self) -> MatVecU8 {
        let [a, b, c, d] = self.a.as_u8vec4().to_array();
        let [e, f, g, h] = self.b.as_u8vec4().to_array();
        [a, b, c, d, e, f, g, h]
    }

    pub fn unpack(packed: MatVecU8) -> Self {
        Self {
            a: U8Vec4::from_array([packed[0], packed[1], packed[2], packed[3]]).as_vec4(),
            b: U8Vec4::from_array([packed[4], packed[5], packed[6], packed[7]]).as_vec4(),
        }
    }
}

impl Add<MatVec> for MatVec {
    type Output = MatVec;
    fn add(self, rhs: MatVec) -> Self::Output {
        MatVec { a: self.a + rhs.a, b: self.b + rhs.b }
    }
}

impl Sub<MatVec> for MatVec {
    type Output = MatVec;
    fn sub(self, rhs: MatVec) -> Self::Output {
        MatVec { a: self.a - rhs.a, b: self.b - rhs.b }
    }
}

impl Mul<f32> for MatVec {
    type Output = MatVec;
    fn mul(self, rhs: f32) -> Self::Output {
        MatVec { a: self.a * rhs, b: self.b * rhs }
    }
}

impl Mul<MatVec> for f32 {
    type Output = MatVec;
    fn mul(self, rhs: MatVec) -> Self::Output {
        MatVec { a: self * rhs.a, b: self * rhs.b }
    }
}

impl Div<f32> for MatVec {
    type Output = MatVec;
    fn div(self, rhs: f32) -> Self::Output {
        MatVec { a: self.a / rhs, b: self.b / rhs }
    }
}

//--------------------------------------------------------------------------------------------------

const TERRAIN_TILE_SIZE: u32 = 16;
// Terrain data should be the same in-memory and on the GPU.
//
// Issue: the number of terrain slices vary between columns, and thus between tiles.

#[derive(Default)]
pub struct Terrain {
    world_bounds: Rect,
    width: u32,
    height: u32,
    width_tile: u32,
    height_tile: u32,
    layers: gpu::Buffer<MatLayer>,
    tiles: gpu::Buffer<Tile>,
    lods: gpu::Buffer<Lod>,
    render_target: RenderTarget,
}


// shader-interface
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Lod {
    /// Dimensions in tiles.
    dim: UVec2,
    /// Offset into the tile buffer.
    base_tile: u32,
}

// shader-interface
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct MatStack {
    /// Offset in layers.
    offset: u16,
    /// Number of layers.
    count: u16,
}

impl MatStack {
    pub fn new(offset: u16, count: u16) -> Self {
        MatStack { offset, count }
    }
}

// shader-interface
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Tile {
    /// Offset of the tile in the data buffer.
    base_layer: u32,
    /// Min height.
    min_height: u16,
    /// Max height.
    max_height: u16,
    /// Pointers to slices.
    stacks: [MatStack; TERRAIN_TILE_SIZE as usize * TERRAIN_TILE_SIZE as usize],
}

impl Default for Tile {
    fn default() -> Self {
        Tile {
            base_layer: 0,
            min_height: 0,
            max_height: 0,
            stacks: [MatStack::new(0, 0); TERRAIN_TILE_SIZE as usize * TERRAIN_TILE_SIZE as usize],
        }
    }
}

fn write_std_terrain_stack(height: u16, out: &mut Vec<MatLayer>) -> Range<u32> {
    let column = &[
        MatLayer {
            low: 0,
            high: height.saturating_sub(10),
            value: MatVec { a: Vec4::new(0.5, 0.5, 0.5, 1.0), b: Vec4::ZERO }.pack(),
        },
        MatLayer {
            low: height.saturating_sub(10),
            high: height.saturating_sub(1),
            value: MatVec { a: Vec4::new(0.3, 0.2, 0.1, 1.0), b: Vec4::ZERO }.pack(),
        },
        MatLayer {
            low: height.saturating_sub(1),
            high: height,
            value: MatVec { a: Vec4::new(0.1, 0.8, 0.1, 1.0), b: Vec4::ZERO }.pack(),
        },
        MatLayer { low: height, high: u16::MAX, value: MatVec { a: Vec4::ZERO, b: Vec4::ZERO }.pack() },
    ];
    out.extend_from_slice(column);
    let start = out.len() - column.len();
    let end = out.len();
    Range { start: start as u32, end: end as u32 }
}

/// Loads a terrain from a heightmap image.
///
/// Since the heightmap only encodes height data, each height sample is turned into a "standard" terrain column composed of:
/// - a base of stone from 0 to `height - 10`
/// - a layer of dirt from `height - 10` to `height - 1`
/// - a layer of grass at `height`
/// - air above `height`
pub fn load_terrain_from_heightmap<P: AsRef<Path>>(
    heightmap_image_file: P,
    units_per_pixel: f32,
    origin_alignment: Vec2,
) -> anyhow::Result<Terrain> {
    // load heightmap image to float array
    let (heightmap, width, height) = {
        let reader = ImageReader::open(heightmap_image_file)?;
        let image = reader.decode()?;
        (image.to_luma32f(), image.width(), image.height())
    };
    // size in tiles
    let tile_x_count = width.div_ceil(TERRAIN_TILE_SIZE);
    let tile_y_count = height.div_ceil(TERRAIN_TILE_SIZE);
    // shorthand for tile size
    const T: u32 = TERRAIN_TILE_SIZE;

    // tile vector
    let mut tiles = gpu::Buffer::<Tile>::uninit((tile_x_count * tile_y_count) as usize);
    let tiles_ptr = tiles.as_mut_ptr();
    let mut layers = vec![];
    for ty in 0..tile_y_count {
        for tx in 0..tile_x_count {
            let tile_index = (ty * tile_x_count + tx) as usize;
            let mut tile = Tile::default();
            let mut min_height = f32::MAX;
            let mut max_height = f32::MIN;
            tile.base_layer = layers.len() as u32;
            for ly in 0..T {
                for lx in 0..T {
                    let gx = tx * T + lx;
                    let gy = ty * T + ly;
                    if gx < width && gy < height {
                        let height_sample = heightmap.get_pixel(gx, gy)[0];
                        let height_u16 = (height_sample * u16::MAX as f32) as u16;
                        min_height = min_height.min(height_sample);
                        max_height = max_height.max(height_sample);
                        let r = write_std_terrain_stack(height_u16, &mut layers);
                        let offset = (r.start - tile.base_layer) as u16;
                        let count = (r.end - r.start) as u16;
                        tile.stacks[(ly * T + lx) as usize] = MatStack::new(offset, count);
                    }
                }
            }
            tile.min_height = (min_height * u16::MAX as f32) as u16;
            tile.max_height = (max_height * u16::MAX as f32) as u16;
            unsafe {
                tiles_ptr.add(tile_index).write(tile);
            }
        }
    }

    // compute lods (TODO)
    let lods = {
        let mut offset = 0;
        let mut w = tile_x_count;
        let mut h = tile_y_count;
        let mut lods = vec![];
        loop {
            lods.push(Lod { base_tile: offset, dim: uvec2(w, h) });
            if w == 1 && h == 1 {
                break;
            }
            offset += w * h;
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        lods
    };

    //for lod in lods.iter() {
    //eprintln!("LOD: offset={}, width={}, height={}", lod.offset, lod.width, lod.height);
    //}

    // LOD structure:
    // - we may not be able to hold the whole terrain in memory
    // - must decide the granularity of transferred blocks
    // - the blocks should roughly represent the same screen-space area after projection
    //
    // Divide the terrain into separate textures for LOD 0, 1, etc.
    // Blocks are 16x16 tiles into these textures.

    // upload layers & lods
    let layers = gpu::Buffer::from_slice(&layers);
    let lods = gpu::Buffer::from_slice(&lods);
    let scaled_width = width as f32 * units_per_pixel;
    let scaled_height = height as f32 * units_per_pixel;
    let origin = vec2(-origin_alignment.x * scaled_width, -origin_alignment.y * scaled_height);
    let world_bounds = Rect::from_origin_size(origin, vec2(scaled_width, scaled_height));
    Ok(Terrain {
        world_bounds,
        width,
        height,
        width_tile: tile_x_count,
        height_tile: tile_y_count,
        layers,
        tiles,
        lods,
        render_target: RenderTarget::new(
            VK_FORMAT_R8G8B8A8_UNORM,
            ImageUsage::STORAGE | ImageUsage::SAMPLED | ImageUsage::COLOR_ATTACHMENT,
        )
    })
}

// shader-interface
#[derive(Copy, Clone)]
#[repr(C)]
struct TerrainDesc {
    world_bounds: Vec4, // x1,y1,x2,y2
    lod_count: u32,
    tile_count: u32,
    tiles: gpu::Ptr<Tile>,
    lods: gpu::Ptr<Lod>,
    layers: gpu::Ptr<MatLayer>,
}

// shader-interface
#[derive(Copy, Clone)]
#[repr(C)]
struct TerrainParams {
    /*
    Terrain* terrain;
    RWTexture2D<float4>.Handle output;
    int screen_width;
    int screen_height;
    float4x4 view_matrix;
    float4x4 inv_proj_matrix;*/
    terrain: TerrainDesc,
    output: gpu::StorageImageHandle,
    screen_width: u32,
    screen_height: u32,
    view_matrix: Mat4,
    inv_view_matrix: Mat4,
    proj_matrix: Mat4,
    inv_proj_matrix: Mat4,
    eye_pos: Vec3,
}


const RENDER_TILE_SIZE: u32 = 16;


impl Terrain {
    pub fn render(&mut self, camera: &Camera, image: &gpu::Image) {
        if self.tiles.len() == 0 {
            return;
        }
        self.render_target.setup(image.width(), image.height());
        let world_bounds =
            vec4(self.world_bounds.min.x, self.world_bounds.min.y, self.world_bounds.max.x, self.world_bounds.max.y);
        let params = TerrainParams {
            terrain: TerrainDesc {
                world_bounds,
                tile_count: self.tiles.len() as u32,
                tiles: self.tiles.ptr(),
                lod_count: self.lods.len() as u32,
                lods: self.lods.ptr(),
                layers: self.layers.ptr(),
            },
            output: self.render_target.storage_handle(),
            screen_width: image.width(),
            screen_height: image.height(),
            view_matrix: camera.view,
            inv_view_matrix: camera.view_inverse,
            proj_matrix: camera.projection,
            inv_proj_matrix: camera.projection_inverse,
            eye_pos: camera.eye().as_vec3(),
        };
        let tile_x_cnt = image.width().div_ceil(RENDER_TILE_SIZE);
        let tile_y_cnt = image.height().div_ceil(RENDER_TILE_SIZE);
        gpu::dispatch(&terrain_shader::main, tile_x_cnt, tile_y_cnt, 1, &params);
        gpu::barrier(BARRIER_TEXTURE);
        gpu::blit_full_image_top_mip_level(self.render_target.image(), image);
        gpu::wait_idle();
    }
}

