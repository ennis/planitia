//! Geometry processing tool.
//!
//! Handles the generation of terrain meshes.
#![feature(default_field_values)]

use std::sync::OnceLock;
use clap::{Parser, Subcommand};
use crate::terrain::{generate_terrain_meshes, TerrainConfig};

mod terrain;
mod houdini;
mod coating;

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
    /// Generate coating meshes from geometry exported from Houdini.
    Coating {
        /// Path to Houdini .geo/.bgeo file.
        geo_file: String,
        /// Path to output file.
        #[clap(short='o', long)]
        output_file: Option<String>,
    },
    /// Dump information about a geometry file.
    Info {
        /// Path to geometry file.
        geo_file: String,
    }
}

/// Config options common between all tools.
#[derive(Debug)]
pub struct Config {
    pub quiet: bool,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

fn cfg() -> &'static Config {
    CONFIG.get().expect("config not initialized")
}


fn main() {
    let args = Args::parse();

    let _ = CONFIG.set(Config { quiet: args.quiet });

    match args.command {
        Commands::Terrain { heightmap, error_threshold, triangle_count } => {
            let terrain_cfg = TerrainConfig { error_threshold, triangle_count_target: triangle_count };
            generate_terrain_meshes(heightmap, &terrain_cfg);
        }
        Commands::Coating { geo_file, output_file } => {
            let coating_cfg = coating::CoatingConfig { output_file };
            coating::create_coating_mesh(geo_file, &coating_cfg);
        },
        Commands::Info { geo_file } => {
            let geo = houdini::Geo::load(&geo_file).expect("failed to load geometry file");
            houdini::print_geometry_summary(&geo);
        }
    }
}
