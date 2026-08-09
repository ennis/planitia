use math::{U8Vec4, Vec4};
use std::ops::{Add, Div, Mul, Sub};
use std::path::Path;
use std::range::Range;
use image::ImageReader;

mod column;

pub use column::{TerrSlice, downsample_columns};

pub const TERRAIN_FEATURE_COUNT: usize = 8;
pub type PackedTerrVec = [u8;TERRAIN_FEATURE_COUNT];

/// 8-dimensional feature vector representing the contents of a terrain cell.
#[derive(Copy, Clone, Debug)]
pub struct TerrVec {
    pub a: Vec4,
    pub b: Vec4,
}

impl TerrVec {
    pub fn lerp(self, other: TerrVec, t: f32) -> TerrVec {
        TerrVec { a: self.a.lerp(other.a, t), b: self.b.lerp(other.b, t) }
    }

    /// Packs the vector to 8x8-bit components.
    pub fn pack(self) -> PackedTerrVec {
        let [a, b, c, d] = self.a.as_u8vec4().to_array();
        let [e, f, g, h] = self.b.as_u8vec4().to_array();
        [a, b, c, d, e, f, g, h]
    }

    pub fn unpack(packed: PackedTerrVec) -> Self {
        Self {
            a: U8Vec4::from_array([packed[0], packed[1], packed[2], packed[3]]).as_vec4(),
            b: U8Vec4::from_array([packed[4], packed[5], packed[6], packed[7]]).as_vec4(),
        }
    }
}

impl Add<TerrVec> for TerrVec {
    type Output = TerrVec;

    fn add(self, rhs: TerrVec) -> Self::Output {
        TerrVec { a: self.a + rhs.a, b: self.b + rhs.b }
    }
}

impl Sub<TerrVec> for TerrVec {
    type Output = TerrVec;

    fn sub(self, rhs: TerrVec) -> Self::Output {
        TerrVec { a: self.a - rhs.a, b: self.b - rhs.b }
    }
}

impl Mul<f32> for TerrVec {
    type Output = TerrVec;

    fn mul(self, rhs: f32) -> Self::Output {
        TerrVec { a: self.a * rhs, b: self.b * rhs }
    }
}

impl Mul<TerrVec> for f32 {
    type Output = TerrVec;

    fn mul(self, rhs: TerrVec) -> Self::Output {
        TerrVec { a: self * rhs.a, b: self * rhs.b }
    }
}

impl Div<f32> for TerrVec {
    type Output = TerrVec;

    fn div(self, rhs: f32) -> Self::Output {
        TerrVec { a: self.a / rhs, b: self.b / rhs }
    }
}

//--------------------------------------------------------------------------------------------------

const TERRAIN_TILE_SIZE: u32 = 16;


// Terrain data should be the same in-memory and on the GPU.
//
// Issue: the number of terrain slices vary between columns, and thus between tiles.


pub struct Terrain {
    width: u32,
    height: u32,
    width_tile: u32,
    height_tile: u32,
    //lod_to_slice_offset:
    slices: Vec<TerrSlice>,
    /// Terrain tiles.
    tiles: Vec<TerrainTile>,
    lods: Vec<TerrainLod>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TerrainLod {
    /// Offset into the tile buffer.
    offset: usize,
    /// Width in tiles.
    width: u32,
    /// Height in tiles.
    height: u32,
}

#[derive(Clone)]
#[repr(C)]
pub struct TerrainTile {
    /// Offset of the tile in the data buffer.
    offset: u64,
    /// Min height.
    min_height: f32,
    /// Max height.
    max_height: f32,
    /// Pointers to slices.
    slices: [Range<u32>; TERRAIN_TILE_SIZE as usize * TERRAIN_TILE_SIZE as usize],
}


fn write_std_terrain_column(height: u16, out: &mut Vec<TerrSlice>) -> Range<u32> {
    let column = &[
        TerrSlice { low: 0, high: height.saturating_sub(10), value: TerrVec { a: Vec4::new(0.5, 0.5, 0.5, 1.0), b: Vec4::ZERO }.pack() },
        TerrSlice { low: height.saturating_sub(10), high: height.saturating_sub(1), value: TerrVec { a: Vec4::new(0.3, 0.2, 0.1, 1.0), b: Vec4::ZERO }.pack() },
        TerrSlice { low: height.saturating_sub(1), high: height, value: TerrVec { a: Vec4::new(0.1, 0.8, 0.1, 1.0), b: Vec4::ZERO }.pack() },
        TerrSlice { low: height, high: u16::MAX, value: TerrVec { a: Vec4::ZERO, b: Vec4::ZERO }.pack() },
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
pub fn load_terrain_from_heightmap<P: AsRef<Path>>(heightmap_image_file: P) -> anyhow::Result<Terrain> {

    // load heightmap image to float array
    let (heightmap, width, height) = {
        let reader = ImageReader::open(heightmap_image_file)?;
        let image = reader.decode()?;
        (image.to_luma32f(), image.width(), image.height())
    };


    // size in tiles
    let w_tile = width.div_ceil(TERRAIN_TILE_SIZE);
    let h_tile = height.div_ceil(TERRAIN_TILE_SIZE);
    // shorthand for tile size
    const T: u32 = TERRAIN_TILE_SIZE;

    // tile vector
    let mut tiles = vec![TerrainTile {
        offset: 0,
        min_height: 0.0,
        max_height: 0.0,
        slices: [Range::from(0..0); (T * T) as usize],
    }; (w_tile * h_tile) as usize];

    let mut slices = vec![];

    for ty in 0..h_tile {
        for tx in 0..w_tile {
            let tile_index = (ty * w_tile + tx) as usize;
            let tile = &mut tiles[tile_index];

            let mut min_height = f32::MAX;
            let mut max_height = f32::MIN;

            for ly in 0..T {
                for lx in 0..T {
                    let gx = tx * T + lx;
                    let gy = ty * T + ly;

                    if gx < width && gy < height {
                        let height_sample = heightmap.get_pixel(gx, gy)[0];
                        let height_u16 = (height_sample * u16::MAX as f32) as u16;

                        min_height = min_height.min(height_sample);
                        max_height = max_height.max(height_sample);

                        let slice_range = write_std_terrain_column(height_u16, &mut slices);
                        tile.slices[(ly * T + lx) as usize] = slice_range;
                    }
                }
            }

            tile.min_height = min_height;
            tile.max_height = max_height;
        }
    }

    // compute lods (TODO)
    let lods = {
        let mut offset = 0;

        let mut w = w_tile;
        let mut h = h_tile;
        let mut lods = vec![];

        loop {
            lods.push(TerrainLod {
                offset,
                width: w,
                height: h,
            });

            if w == 1 && h == 1 {
                break;
            }

            offset += w as usize * h as usize;
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        lods
    };

    for lod in lods.iter() {
        eprintln!("LOD: offset={}, width={}, height={}", lod.offset, lod.width, lod.height);
    }


    let mut slices = Vec::new();

    Ok(Terrain {
        width,
        height,
        width_tile: w_tile,
        height_tile: h_tile,
        slices,
        tiles,
        lods
    })
}
