//! Geometry processing tool.
//!
//! Handles the generation of terrain meshes.

use clap::{Parser, Subcommand};
use crate::terrain::{generate_terrain_meshes, TerrainConfig};

mod terrain;

#[derive(Parser, Debug)]
struct Args {
    /// Path to build manifest.
    manifest_path: Option<String>,
    /// Don't print logs to stdout.
    #[clap(short, long)]
    quiet: bool,
    /// Print cargo dependency directives.
    #[clap(long)]
    emit_cargo_deps: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a terrain mesh file from a heightmap image.
    Terrain {
        /// Path to heightmap image.
        heightmap: String,
        /// Error threshold for the tessellation of the base LOD level.
        #[clap(long)]
        error_threshold: Option<f32>,
        /// Target number of triangles for the base LOD level.
        #[clap(long)]
        triangle_count: Option<usize>,
    },
}

/// Config options common between all tools.
pub struct Config {
    pub quiet: bool,
}


fn main() {
    let args = Args::parse();

    let global_cfg = Config { quiet: args.quiet };

    match args.command {
        Commands::Terrain { heightmap, error_threshold, triangle_count } => {
            let terrain_cfg = TerrainConfig { error_threshold, triangle_count_target: triangle_count };
            generate_terrain_meshes(heightmap, &terrain_cfg, &global_cfg);
        }
    }
}
