use crate::{
    Arena, BuildManifest, BuildOptions, EntryPoint, Error, GraphicsState, Module, ModuleDependency, Pipeline,
    get_file_mtime, get_log_options, reflection,
};
use anyhow::{Context, anyhow};
use color_print::cprintln;
use sharc::gpu_types::vk;
use slang::DebugInfoLevel;
use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::{fs, slice};

fn get_slang_global_session() -> slang::GlobalSession {
    thread_local! {
        static SESSION: OnceCell<slang::GlobalSession> = OnceCell::new();
    }

    SESSION.with(|s| s.get_or_init(|| slang::GlobalSession::new().expect("Failed to create Slang session")).clone())
}

////////////////////////////////////////////////////////////////////////////////////////////////////

///// Wrapper around `slang::Error` that is `Send + Sync` and can therefore be stored in an
///// `anyhow::Error`.  Internally the error message is captured as a `String`.
//#[derive(thiserror::Error, Debug)]
//#[error("{0}")]
//struct SlangError(String);

impl From<slang::Error> for Error {
    fn from(err: slang::Error) -> Self {
        Error::CompilerErrors(err.to_string())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

fn slang_stage_to_stage_flags(stage: slang::Stage) -> vk::ShaderStageFlags {
    match stage {
        slang::Stage::Vertex => vk::ShaderStageFlags::VERTEX,
        slang::Stage::Hull => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        slang::Stage::Domain => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        slang::Stage::Geometry => vk::ShaderStageFlags::GEOMETRY,
        slang::Stage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        slang::Stage::Compute => vk::ShaderStageFlags::COMPUTE,
        slang::Stage::Mesh => vk::ShaderStageFlags::MESH_EXT,
        slang::Stage::Amplification => vk::ShaderStageFlags::TASK_EXT,
        _ => panic!("unsupported shader stage: {:?}", stage),
    }
}

fn convert_spirv_u8_to_u32(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "invalid SPIR-V code length");
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            u32::from_ne_bytes(bytes)
        })
        .collect::<Vec<u32>>()
}

/// Returns the total size (in bytes) of push constants declared by the given entry point.
fn get_push_constants_size(entry_point: &slang::reflection::EntryPoint) -> usize {
    // Push constants are entry-point function parameters in the `Uniform` category.
    // (There is also a `PushConstantBuffer` category, but Slang does not appear to use it.)
    let mut size = 0;
    for p in entry_point.parameters() {
        if p.category().unwrap() == slang::ParameterCategory::Uniform {
            size += p.type_layout().unwrap().size(slang::ParameterCategory::Uniform);
        }
    }
    size
}

/// Creates a Slang compilation session configured according to the manifest and build options.
pub(crate) fn create_slang_session(
    include_paths: &[PathBuf],
    manifest: &BuildManifest,
    options: &BuildOptions,
) -> anyhow::Result<slang::Session> {
    let global_session = get_slang_global_session();

    // Debug information can be requested either in the manifest or via build options.
    let emit_debug_information = manifest.compiler.debug | options.emit_debug_information;

    let mut search_paths_cstr: Vec<CString> =
        include_paths.iter().map(|p| CString::new(&*p.to_string_lossy()).unwrap()).collect();

    // Include paths specified in the manifest
    for path in manifest.include_paths.iter() {
        // Include paths in the manifest are absolute or relative to the manifest path.
        let path = PathBuf::from(path);
        let path = if path.is_absolute() {
            path
        } else {
            manifest.manifest_path.parent().unwrap_or_else(|| Path::new(".")).join(path)
        };
        eprintln!("add include path: {}", path.display());
        search_paths_cstr.push(CString::new(&*path.to_string_lossy()).unwrap());
    }

    let search_path_ptrs: Vec<_> = search_paths_cstr.iter().map(|p| p.as_ptr()).collect();

    let profile = global_session.find_profile(&manifest.compiler.profile);
    let mut compiler_options = slang::CompilerOptions::default()
        .glsl_force_scalar_layout(true)
        .matrix_layout_column(true)
        .optimization(slang::OptimizationLevel::Default)
        .vulkan_use_entry_point_name(true)
        .debug_information(if emit_debug_information { DebugInfoLevel::Maximal } else { DebugInfoLevel::None })
        .profile(profile);

    for (k, v) in manifest.compiler.defines.iter() {
        compiler_options = compiler_options.macro_define(k, v);
    }

    let target_desc = slang::TargetDesc::default().format(slang::CompileTarget::Spirv).options(&compiler_options);
    let targets = [target_desc];
    let session_desc =
        slang::SessionDesc::default().targets(&targets).search_paths(&search_path_ptrs).options(&compiler_options);

    match global_session.create_session(&session_desc) {
        Some(session) => Ok(session),
        None => Err(anyhow!("failed to create Slang session")),
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl BuildManifest {
    /// Resolves the graphics state for a given pipeline name, using the manifest overrides if present.
    fn resolve_graphics_state(&self, pipeline_name: &str) -> Result<GraphicsState, Error> {
        let mut state = self.default.clone();
        // Per-pass overrides are specified as `[pass.pipeline_name]` in TOML.
        if let Some(pass) = self.pass.get(pipeline_name) {
            state.apply_overrides(&pass.raw)?;
        }
        Ok(state)
    }
}

/// Loads a Slang source file, compiles all its shader entry points, and returns the resulting
/// [`Module`] together with reflection data.
pub(crate) fn compile_slang_module<'a>(
    arena: &'a Arena,
    session: &slang::Session,
    file: &Path,
    source: &str,
    manifest: &BuildManifest,
    options: &BuildOptions,
) -> Result<Module<'a>, Error> {
    let (canonical_path, file_mtime) = get_file_mtime(file)?;

    //if get_log_options().verbosity >= 2 {
    //    cprintln!("load_slang_module: {}", file.display());
    //}

    // FIXME: Slang modules are normally declared with `module module_name;`; for now we
    //        fall back to the file stem.
    let module_name =
        file.file_stem().ok_or(anyhow!("invalid shader file name: {}", file.display()))?.to_string_lossy().to_string();
    let module = session.load_module_from_source_string(&module_name, &file.to_string_lossy(), source)?;

    // collect module dependencies
    let mut dependencies = Vec::with_capacity(module.dependency_file_count() as usize);
    for i in 0..module.dependency_file_count() {
        let dep_path = module.dependency_file_path(i);
        dependencies.push(ModuleDependency {
            path: PathBuf::from(dep_path),
            mtime: 0, // Placeholder, replace with actual modification time if available
        });
    }

    let entry_point_count = module.entry_point_count();
    if entry_point_count == 0 {
        // Bail out if there are no entry points in the module.
        // This is not an error; some modules are meant to be used as libraries and don't contain
        // entry points. While it could be possible to emit a SPIR-V "library"
        // with no entry points, it currently crashes the Slang compiler.
        return Ok(Module {
            _session: session.clone(),
            module: module.clone(),
            program: module.clone().into(),
            name: module_name,
            file_path: canonical_path,
            file_mtime,
            spirv: vec![],
            entry_points: vec![],
            pipelines: vec![],
            dependencies,
            refl_struct_types: Default::default(),
            refl_global_params: vec![],
        });
    }

    // Build a composite component (module + every entry point) and link it.
    let mut components = vec![module.clone().into()];
    for i in 0..entry_point_count {
        components.push(module.entry_point_by_index(i).unwrap().into());
    }
    let composite = session.create_composite_component_type(&components)?;
    let program = composite.link()?;
    let program_reflection = program.layout(0).expect("failed to get reflection");

    // retrieve SPIR-V blob of all entry points
    let spirv = {
        let blob = program.target_code(0)?;
        convert_spirv_u8_to_u32(blob.as_slice())
    };

    // Dump SPIR-V to disk if requested.
    if options.emit_spirv_binaries {
        let spirv_dump_path = match options.output_directory {
            Some(ref dir) => dir.join(file.with_added_extension("spv")),
            None => file.with_added_extension("spv"),
        };

        if let Some(parent) = spirv_dump_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }

        if !get_log_options().quiet {
            cprintln!("<g,bold>Dumping</> {}", spirv_dump_path.display());
        }

        fs::write(&spirv_dump_path, unsafe { slice::from_raw_parts(spirv.as_ptr() as *const u8, spirv.len() * 4) })
            .context(format!("dumping SPIR-V at {}", spirv_dump_path.display()))?;
    }

    // Extract per-entry-point metadata & reflection.
    let mut type_collector = reflection::TypeCollector::new(arena);
    let mut entry_points = Vec::with_capacity(entry_point_count as usize);
    for i in 0..entry_point_count {
        let ep = program_reflection.entry_point_by_index(i).unwrap();
        let push_constants_size = get_push_constants_size(ep);
        let workgroup_size = {
            let s = ep.compute_thread_group_size();
            [s[0] as u32, s[1] as u32, s[2] as u32]
        };

        //let params = {
        //    let mut collector = CollectedReflectionData::new();
        //    collector.reflect_entry_point(ep);
        //    collector.params
        //};

        // Determine the pipeline name: prefer the `[pipeline("...")]` attribute, then fall back to
        // stripping well-known stage suffixes from the entry point name.
        let mut pipeline_name = None;

        // `[pipeline("...")]` attribute
        //
        // NOTE: Using an attribute for pass grouping isn't as practical as I'd like,
        //       since the attribute must be defined in shader code before use.
        //       This means that shaders must include a support module, which introduces an
        //       annoying dependency on some external file, or define the `pass` attribute themselves,
        //       which is useless boilerplate.
        //       Ideally, there should be a way to force the inclusion of a module,
        //       or add a custom prelude, or define an attribute on the command line, but AFAIK
        //       no such mechanism exists in slang currently.
        for attr in module.entry_point_by_index(i).unwrap().function_reflection().user_attributes() {
            if attr.name().unwrap() == "pipeline" {
                pipeline_name = attr.argument_value_string(0).map(String::from);
            }
        }

        // If there's no pipeline attribute, try to strip a stage suffix from the entry point name, and use that as the pass name.
        // E.g.: `pass_name_vertex` and `pass_name_fragment` will be grouped into the same pass `pass_name`.
        if pipeline_name.is_none() {
            const STAGE_SUFFIXES: &[&str] = &[
                "_vertex",
                "_fragment",
                "_compute",
                "_mesh",
                "_amplification",
                "_hull",
                "_domain",
                "_geometry",
                "_vertex_main",
                "_fragment_main",
                "_compute_main",
                "_mesh_main",
                "_amplification_main",
                "_hull_main",
                "_domain_main",
                "_geometry_main",
            ];
            let mut name = ep.name().unwrap();
            for suffix in STAGE_SUFFIXES {
                if let Some(stripped) = name.strip_suffix(suffix) {
                    name = stripped;
                    break;
                }
            }
            pipeline_name = Some(name.to_string());
        }

        //if get_log_options().verbosity >= 2 {
        //    cprintln!("entry point: {}/{}", file.display(), ep.name().unwrap());
        //}

        let mut param_collector = reflection::ParamCollector::new(&mut type_collector);
        param_collector.reflect_entry_point(ep);

        entry_points.push(EntryPoint {
            name: ep.name().unwrap().to_string(),
            stage: slang_stage_to_stage_flags(ep.stage()),
            push_constants_size,
            pipeline_name,
            workgroup_size,
            refl_params: param_collector.access_chains,
        });
    }

    debug_assert_eq!(entry_points.len(), entry_point_count as usize);

    // Collect pipelines.
    // Group entry points by pipeline name.
    let mut pipelines_by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, ep) in entry_points.iter().enumerate() {
        if let Some(ref pipeline_name) = ep.pipeline_name {
            pipelines_by_name.entry(pipeline_name).or_default().push(i);
        }
    }

    let mut pipelines: Vec<Pipeline> = Vec::new();
    for (pipeline_name, stages) in pipelines_by_name {
        // Resolve graphics state for the pipeline.
        let graphics_state = manifest.resolve_graphics_state(pipeline_name)?;

        // TODO: sanity checks:
        //       * don't mix graphics and compute stages
        //       * have only one entry point of each type

        // infer push constants size and workgroup size from entry points
        let mut push_constants_size = 0;
        let mut workgroup_size = [1u32; 3];
        let mut stage_flags = vk::ShaderStageFlags::default();
        //let mut all_params = vec![];

        // Collect parameter reflection information:
        // * for global variables (reflect_global_params)
        // * for parameters passed as arguments to the entry point function (reflect_entry_point)
        for stage in stages.iter() {
            let ep = &entry_points[*stage];
            push_constants_size = push_constants_size.max(ep.push_constants_size);
            workgroup_size = ep.workgroup_size;
            stage_flags |= ep.stage;
        }

        pipelines.push(Pipeline {
            name: pipeline_name.to_string(),
            stages,
            workgroup_size,
            graphics_state,
            push_constants_size,
        })
    }

    let mut param_collector = reflection::ParamCollector::new(&mut type_collector);
    param_collector.reflect_global_params(program_reflection);
    
    let refl_global_params = param_collector.access_chains;
    let refl_struct_types = type_collector.struct_types;

    Ok(Module {
        _session: session.clone(),
        module,
        program,
        name: module_name,
        file_path: canonical_path,
        file_mtime,
        pipelines,
        spirv,
        entry_points,
        dependencies,
        refl_struct_types,
        refl_global_params,
    })
}
