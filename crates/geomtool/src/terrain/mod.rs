//! Terrain mesh generation from heightmaps.
mod triangulation;

use crate::Config;
use crate::terrain::triangulation::{TriangulationOptions, tessellate_heightmap};
use image::ImageReader;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use color_print::cprintln;

pub struct TerrainConfig {
    pub error_threshold: Option<f32>,
    pub triangle_count_target: Option<usize>,
}

pub fn generate_terrain_meshes(heightmap_file: impl AsRef<Path>, terrain_cfg: &TerrainConfig, global_cfg: &Config) {
    generate_terrain_meshes_inner(heightmap_file.as_ref(), terrain_cfg, global_cfg).unwrap();
}

fn generate_terrain_meshes_inner(
    heightmap_file: &Path,
    terrain_cfg: &TerrainConfig,
    global_cfg: &Config,
) -> anyhow::Result<()> {
    // load heightmap image to float array
    let (heightmap, width, height) = {
        let reader = ImageReader::open(heightmap_file)?;
        let image = reader.decode()?;
        (image.to_luma32f(), image.width(), image.height())
    };

    if !global_cfg.quiet {
        cprintln!("Generating terrain mesh from heightmap: {heightmap_file:?}, size: {width}×{height}");
    }

    let bar = if global_cfg.quiet {
        ProgressBar::hidden()
    } else {
        if let Some(triangle_count) = terrain_cfg.triangle_count_target {
            ProgressBar::new(triangle_count as u64)
        } else {
            ProgressBar::new_spinner()
        }
    };

    let mut progress_cb = |triangle_count, error| {
        //bar.println(format!("Current error: {error}"));
        bar.set_position(triangle_count as u64);
    };

    tessellate_heightmap(
        &heightmap,
        &TriangulationOptions {
            error_threshold: terrain_cfg.error_threshold,
            triangle_count_target: terrain_cfg.triangle_count_target,
        },
        &mut progress_cb,
    );

    Ok(())
}
