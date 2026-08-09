#![feature(default_field_values)]
use std::path::PathBuf;

static SHADERS: &[&str] = &["assets/shaders/*.slang"];

/// Copy Rust stdlib DLLs from the active toolchain's `lib/rustlib/<target>/lib/` directory into
/// the cargo output directory (i.e. next to `game.exe`) so the executable can
/// be run without the Rust toolchain on PATH.
fn copy_rust_stdlib_dlls() {
    // OUT_DIR = target/<profile>/build/<crate>/<hash>/out  (newer Cargo)
    //        or target/<profile>/build/<crate>-<hash>/out  (older Cargo)
    // Find the "build" ancestor and step up one more to reach target/<profile>/.
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(&out_dir);
    let target_dir = out_path
        .ancestors()
        .find(|p| p.file_name().map(|n| n == "build") == Some(true))
        .and_then(|p| p.parent())
        .expect("could not find target/<profile>/ from OUT_DIR")
        .to_path_buf();

    // Use the same rustc that cargo is using.
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let output = std::process::Command::new(&rustc)
        .args(["--print", "sysroot"])
        .output()
        .expect("failed to run `rustc --print sysroot`");
    let sysroot = std::str::from_utf8(&output.stdout).expect("non-UTF8 sysroot").trim().to_string();

    // DLLs are in lib/rustlib/<target>/lib/, not bin/
    let target = std::env::var("TARGET").unwrap_or_default();
    let lib_dir = PathBuf::from(&sysroot).join("lib").join("rustlib").join(&target).join("lib");
    let entries = match std::fs::read_dir(&lib_dir) {
        Ok(e) => e,
        Err(err) => {
            println!("cargo:warning=Could not read {}: {}", lib_dir.display(), err);
            return;
        }
    };

    for entry in entries.flatten() {
        let src = entry.path();
        let is_dll = src.extension().and_then(|e| e.to_str()) == Some("dll");
        let name = src.file_name().unwrap_or_default().to_string_lossy();
        // Copy only the Rust runtime DLLs (std-*.dll), not rustc_driver etc.
        if is_dll && name.starts_with("std-") {
            let dest = target_dir.join(src.file_name().unwrap());
            if let Err(err) = std::fs::copy(&src, &dest) {
                println!("cargo:warning=Failed to copy {} → {}: {}", src.display(), dest.display(), err);
            }
        }
    }
}



fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        copy_rust_stdlib_dlls();
    }

    // build shaders
    if let Err(err) = shadertool::build(
        SHADERS.iter().cloned(),
        &shadertool::BuildOptions {
            emit_cargo_deps: true,
            emit_debug_information: true,
            emit_spirv_binaries: true,
            emit_reflection: false,
            include_paths: vec![PathBuf::from("../gamelib/assets/gamelib/shaders")],
            output_directory: None,
        },
        &shadertool::LogOptions { quiet: true, .. },
    ) {
        err.print_cargo_error();
    }
}
