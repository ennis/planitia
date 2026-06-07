#![feature(default_field_values)]

fn main() {
    // build shaders
    if let Err(err) = shadertool::build(
        "assets/gamelib/shaders/shaders.toml",
        &shadertool::BuildOptions {
            quiet: true,
            emit_cargo_deps: true,
            emit_debug_information: false,
            emit_spirv_binaries: false,
            ..
        },
    ) {
        err.print_cargo_error();
    }
}
