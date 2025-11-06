use crate::library::{convert_spirv_u8_to_u32, get_push_constants_size};
use crate::session::create_session;
use crate::SHADER_PROFILE;
use heck::ToShoutySnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote, TokenStreamExt};
use slang::Downcast;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::{fs, io};

/// Loads all slang shader modules in a directory.
fn load_shader_modules_in_directory(
    session: &slang::Session,
    shaders_directory: &Path,
) -> Result<Vec<slang::Module>, io::Error> {
    // load all modules in the search paths
    let mut modules = Vec::new();
    for entry in fs::read_dir(shaders_directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let Some(ext) = path.extension() else { continue };
            if ext == OsStr::new("slang") {
                let path_str = path.to_str().unwrap();
                // re-run the build script if the slang file changes
                println!("cargo:rerun-if-changed={path_str}");
                // load the module
                match session.load_module(path_str) {
                    Ok(module) => {
                        modules.push(module);
                    }
                    Err(err) => {
                        // output compilation errors
                        for line in err.to_string().lines() {
                            println!("cargo::error={line}");
                        }
                        panic!("failed to load module: {err}");
                    }
                };
            }
        }
    }
    Ok(modules)
}

pub(crate) fn convert_slang_stage(stage: slang::Stage) -> proc_macro2::Ident {
    let s = match stage {
        slang::Stage::Compute => "Compute",
        slang::Stage::Vertex => "Vertex",
        slang::Stage::Fragment => "Fragment",
        slang::Stage::Geometry => "Geometry",
        slang::Stage::Domain => "TessControl",
        slang::Stage::Hull => "TessEvaluation",
        slang::Stage::Mesh => "Mesh",
        slang::Stage::Amplification => "Task",
        _ => panic!("unsupported shader stage: {:?}", stage),
    };
    format_ident!("{s}")
}

/// Compiles all shaders in the given directory and embeds the SPIR-V code in a rust module.
pub fn compile_and_embed_shaders(
    shaders_directory: &str,
    include_search_paths: &[String],
    _output_directory: &Path,
    bindings_output: &mut dyn io::Write,
) {
    let session = create_session(SHADER_PROFILE, include_search_paths, &[]);
    let modules = load_shader_modules_in_directory(&session, Path::new(shaders_directory)).unwrap();

    // now compile all entry points, and generate bindings
    let mut bindings = TokenStream::new();

    //let mut ctx = bindgen::Ctx::new();

    for module in modules.iter() {
        let mut components = Vec::new();
        components.push(module.downcast().clone());
        let entry_point_count = module.entry_point_count();
        for i in 0..entry_point_count {
            let entry_point = module.entry_point_by_index(i).unwrap();
            components.push(entry_point.downcast().clone());
        }
        let program = session.create_composite_component_type(&components).unwrap();
        let program = program.link().unwrap();
        let reflection = program.layout(0).expect("failed to get reflection");

        //ctx.generate_interface(&reflection);

        for i in 0..entry_point_count {
            let ep = module.entry_point_by_index(i).unwrap();
            let module_file_path = PathBuf::from(module.file_path()).canonicalize().unwrap();
            let entry_point_name = ep.function_reflection().name();
            let code = match program.entry_point_code(i as i64, 0) {
                Ok(code) => code,
                Err(err) => {
                    // output compilation errors
                    for line in err.to_string().lines() {
                        println!("cargo::error={line}");
                    }
                    panic!("failed to get entry point code for {}", module_file_path.display());
                }
            };

            // write SPIR-V code of entry point to output directory
            //let module_file_stem = module_file_path
            //    .file_stem()
            //    .unwrap_or(module_file_path.file_name().unwrap())
            //    .to_str()
            //    .expect("invalid unicode file name");
            //let output_file_name = format!("{module_file_stem}-{entry_point_name}.spv");
            //fs::write(&output_directory.join(&output_file_name), code.as_slice()).unwrap();

            // FIXME: include_bytes will produce a `[u8]` slice but with no alignment guarantees
            //        so we can't cast it to a `[u32]` slice. So in the meantime we write the
            //        SPIR-V code directly to the file as a `[u32]` literal.
            // Convert code to `Vec<u32>`
            let code_u32 = convert_spirv_u8_to_u32(code.as_slice());
            let code_u32_slice = code_u32.as_slice();

            // generate shader info
            let rust_entry_point_name = format_ident!("{}", entry_point_name.to_shouty_snake_case());
            let refl_ep = reflection.entry_point_by_index(i).unwrap();
            let push_constant_buffer_size = get_push_constants_size(&refl_ep);
            let stage = convert_slang_stage(refl_ep.stage());
            let [workgroup_size_x, workgroup_size_y, workgroup_size_z] = refl_ep.compute_thread_group_size();
            let workgroup_size_x = workgroup_size_x as u32;
            let workgroup_size_y = workgroup_size_y as u32;
            let workgroup_size_z = workgroup_size_z as u32;
            let module_file_path_str = module_file_path.to_str().unwrap();

            bindings.append_all(quote! {
                pub const #rust_entry_point_name: ::gpu::ShaderEntryPoint<'static> = ::gpu::ShaderEntryPoint {
                    stage: gpu::ShaderStage::#stage,
                    entry_point: #entry_point_name,
                    source_path: Some(#module_file_path_str),
                    code: &[#(#code_u32_slice),*] ,//Cow::Borrowed(include_bytes!(concat!(env!("OUT_DIR"), "/", #output_file_name))),
                    push_constants_size: #push_constant_buffer_size,
                    workgroup_size: [#workgroup_size_x, #workgroup_size_y, #workgroup_size_z],
                };
            });
        }
    }

    // Write bindings to file
    bindings_output.write_all(bindings.to_string().as_bytes()).unwrap();
}
