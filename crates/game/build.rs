use std::env;
use std::path::PathBuf;


fn main() {
    //let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // build shaders
    if let Err(err) = shadertool::build_pipeline(
        "assets/shaders/pipelines.json",
        &shadertool::BuildOptions {
            quiet: true,
            emit_cargo_deps: true,
        },
    ) {
        err.print_cargo_error();
    }
}
