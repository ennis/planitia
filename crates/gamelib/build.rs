#![feature(default_field_values)]

static SHADERS: &[&str] = &["assets/gamelib/shaders/egui.slang", "assets/gamelib/shaders/paint.slang"];

fn main() {
    // build shaders
    if let Err(err) = shadertool::build(
        SHADERS.iter().cloned(),
        &shadertool::BuildOptions {
            emit_cargo_deps: true,
            emit_debug_information: false,
            emit_spirv_binaries: false,
            include_paths: vec![],
            output_directory: None,
        },
        &shadertool::LogOptions { quiet: true, .. },
    ) {
        err.print_cargo_error();
    }
}
