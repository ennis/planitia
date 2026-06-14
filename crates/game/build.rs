#![feature(default_field_values)]
use std::path::PathBuf;

static SHADERS: &[&str] = &["assets/shaders/*.slang"];

fn main() {
    // build shaders
    if let Err(err) = shadertool::build(
        SHADERS.iter().cloned(),
        &shadertool::BuildOptions {
            emit_cargo_deps: true,
            emit_debug_information: true,
            emit_spirv_binaries: true,
            include_paths: vec![PathBuf::from("../gamelib/assets/gamelib/shaders")],
            output_directory: None,
        },
        &shadertool::LogOptions { quiet: true, .. },
    ) {
        err.print_cargo_error();
    }
}
