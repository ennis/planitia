fn main() {
    //let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    eprintln!(
        "HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!HELLO!"
    );

    // build shaders
    if let Err(err) = shadertool::build_pipeline(
        "assets/shaders/shaders.toml",
        &shadertool::BuildOptions {
            quiet: true,
            emit_cargo_deps: true,
            emit_debug_information: true,
            emit_spirv_binaries: true,
        },
    ) {
        err.print_cargo_error();
    }
}
