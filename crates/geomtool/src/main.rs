//! Geometry processing tool.
//!
//! Handles the generation of terrain meshes.
#![feature(default_field_values)]

use crate::terrain::{TerrainConfig, generate_terrain_meshes};
use clap::{Parser, Subcommand};
use color_print::{cprint, cprintln};
use hgeo::Geo;
use std::sync::OnceLock;

mod convert;
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
    /// Convert geometry exported from Houdini.
    Convert {
        /// Path to Houdini .geo/.bgeo file.
        geo_file: String,
        /// Path to output file.
        #[clap(short = 'o', long)]
        output_file: Option<String>,
    },
    /// Dump information about a geometry file.
    Info {
        /// Path to geometry file.
        geo_file: String,
    },
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

/// Prints a color-formatted summary of the geometry to stdout.
fn print_geometry_summary(geo: &Geo) {
    cprintln!("<bold>Geometry Summary</>:");
    cprintln!("  Points:     <b>{}</>", geo.point_count);
    cprintln!("  Primitives: <y>{}</>", geo.primitive_count);
    cprintln!("  Vertices:   <m>{}</>", geo.vertex_count);

    let polygon_count = geo.polygons().count();
    let bezier_count = geo.beziers().count();
    cprint!("  Primitive types:");
    if polygon_count > 0 {
        cprint!(" polygons (incl. polylines) ({polygon_count})");
    }
    if bezier_count > 0 {
        cprint!(" beziers ({bezier_count})");
    }
    cprintln!();

    cprintln!("  Point Attributes:");
    for attr in &geo.point_attributes {
        cprintln!(
            "    <b><bold>{:10}</></>   <i>{} × {:?}</>",
            attr.name,
            attr.size,
            attr.storage_kind()
        );
    }
    cprintln!("  Vertex Attributes:");
    for attr in &geo.vertex_attributes {
        cprintln!(
            "    <m><bold>{:10}</></>   <i>{} × {:?}</>",
            attr.name,
            attr.size,
            attr.storage_kind()
        );
    }
    cprintln!("  Primitive Attributes:");
    for attr in &geo.primitive_attributes {
        cprintln!(
            "    <y><bold>{:10}</></>   <i>{} × {:?}</>",
            attr.name,
            attr.size,
            attr.storage_kind()
        );
    }
    cprintln!("  Point Groups:");
    for (name, group) in geo.point_groups() {
        cprintln!("    <b><bold>{}</></> (<i>{}</>)", name, group.count());
    }
    cprintln!("  Primitive Groups:");
    for (name, group) in geo.primitive_groups() {
        cprintln!("    <y><bold>{}</></> (<i>{}</>)", name, group.count());
    }
}

fn main() {
    let args = Args::parse();

    let _ = CONFIG.set(Config { quiet: args.quiet });

    match args.command {
        Commands::Terrain {
            heightmap,
            error_threshold,
            triangle_count,
        } => {
            let terrain_cfg = TerrainConfig {
                error_threshold,
                triangle_count_target: triangle_count,
            };
            generate_terrain_meshes(heightmap, &terrain_cfg);
        }
        Commands::Convert { geo_file, output_file } => {
            let cfg = convert::ConvertConfig { output_file };
            convert::convert_geo_file(geo_file, &cfg);
        }
        Commands::Info { geo_file } => {
            let geo = Geo::load(&geo_file).expect("failed to load geometry file");
            print_geometry_summary(&geo);
        }
    }
}
