use crate::{BuildManifest, BuildOptions, GraphicsState, Pass};
use anyhow::{Context, anyhow, bail};
use color_print::{ceprintln, cprintln};
use log::warn;
use shader_archive::archive::{ArchiveWriter, Offset};
use shader_archive::gpu::vk;
use shader_archive::zstring::ZString64;
use shader_archive::{FileDependency, PipelineEntryData, RootParamInfo, RootParamLayout, ShaderData};
use slang::reflection::TypeLayout;
use slang::{DebugInfoLevel, Downcast};
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::{env, fs, slice};

type PipelineArchiveWriter = ArchiveWriter<shader_archive::PipelineArchiveData>;

fn make_file_dependency(path: &Path, archive: &mut PipelineArchiveWriter) -> anyhow::Result<FileDependency> {
    let canonical_path = path.canonicalize()?;
    let modified_time = fs::metadata(&canonical_path)?.modified()?;
    let mtime = match modified_time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => {
            warn!("invalid mtime for {} (before UNIX_EPOCH)", canonical_path.display());
            0
        }
    };
    let path_offset = archive.write_str(canonical_path.to_string_lossy().as_ref());
    Ok(FileDependency {
        path: path_offset,
        mtime,
    })
}

fn get_file_mtime(path: &Path) -> anyhow::Result<(PathBuf, u64)> {
    let canonical_path = path.canonicalize()?;
    let metadata = fs::metadata(path)?;
    let modified_time = metadata.modified()?;
    let mtime = match modified_time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => {
            warn!("invalid mtime for {} (before UNIX_EPOCH)", canonical_path.display());
            0
        }
    };
    Ok((canonical_path, mtime))
}

fn get_slang_global_session() -> slang::GlobalSession {
    thread_local! {
        static SESSION: OnceCell<slang::GlobalSession> = OnceCell::new();
    }

    SESSION.with(|s| {
        s.get_or_init(|| slang::GlobalSession::new().expect("Failed to create Slang session"))
            .clone()
    })
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wrapper around `slang::Error` that is Send+Sync and thus can be stashed into a `anyhow::Error`.
///
/// Internally we just store the string representation of the error.
#[derive(thiserror::Error, Debug)]
#[error("{0}")]
struct SlangError(String);

impl From<slang::Error> for SlangError {
    fn from(err: slang::Error) -> Self {
        SlangError(err.to_string())
    }
}

/*
/// Bundles compilation errors from multiple jobs into a single error.
#[derive(Debug)]
struct BuildErrors(Vec<BuildError>);

impl Display for BuildErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for err in &self.0 {
            Display::fmt(err, f)?;
            writeln!(f)?; // separate errors by a newline
        }
        Ok(())
    }
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Extracts the root parameters of the shader entry point.
///
/// # Example
///
/// The root parameters are the data that is passed by pointer to the entry point function.
/// For example, in the following shader code:
///
/// ```slang
/// struct RootParams {
///     uniform float4 someValue;
///     uniform float4 anotherValue;
/// };
///
/// [shader("compute")]
/// void main_0(uniform RootParams* params) {
///     // ...
/// }
///
/// ```
///
/// The root parameters are the fields of the `RootParams` struct: `someValue` and `anotherValue`.
///
/// If the shader takes a pointer to a non-struct type instead:
///```slang
/// [shader("compute")]
/// void main_1(uniform float4* param) {
///     // ...
/// }
///```
/// In this case, there's one root parameter, `param` of type `float4`.
///
/// # Return value
///
/// An iterator over the fields of the root parameters struct. If the shader interface
/// doesn't match the pattern described above, returns an empty iterator.
fn get_root_param_info(entry_point: &slang::reflection::EntryPoint) -> Vec<RootParamInfo> {
    #[rustfmt::skip]
    fn convert_root_param_ty(ty: &slang::reflection::Type) -> vk::Format {
        let mut format = vk::Format::UNDEFINED;

        match ty.kind() {
            slang::TypeKind::Scalar => {
                match ty.scalar_type() {
                    slang::ScalarType::Bool => { format = vk::Format::R8_UINT; }
                    slang::ScalarType::Int8 => { format = vk::Format::R8_SINT; }
                    slang::ScalarType::Int16 => { format = vk::Format::R16_SINT; }
                    slang::ScalarType::Int32 => { format = vk::Format::R32_SINT; }
                    slang::ScalarType::Int64 => { format = vk::Format::R64_SINT; }
                    slang::ScalarType::Uint8 => { format = vk::Format::R8_UINT; }
                    slang::ScalarType::Uint16 => { format = vk::Format::R16_UINT; }
                    slang::ScalarType::Uint32 => { format = vk::Format::R32_UINT; }
                    slang::ScalarType::Uint64 => { format = vk::Format::R64_UINT; }
                    slang::ScalarType::Float16 => { format = vk::Format::R16_SFLOAT; }
                    slang::ScalarType::Float32 => { format = vk::Format::R32_SFLOAT; }
                    slang::ScalarType::Float64 => { format = vk::Format::R64_SFLOAT; }
                    _ => {}
                }
            }
            slang::TypeKind::Vector => {
                let element_ty = ty.element_type().scalar_type();
                let element_count = ty.element_count();
                match (element_ty, element_count) {
                    (slang::ScalarType::Float32, 2) => { format = vk::Format::R32G32_SFLOAT; }
                    (slang::ScalarType::Float32, 3) => { format = vk::Format::R32G32B32_SFLOAT; }
                    (slang::ScalarType::Float32, 4) => { format = vk::Format::R32G32B32A32_SFLOAT; }
                    (slang::ScalarType::Int32, 2) => { format = vk::Format::R32G32_SINT; }
                    (slang::ScalarType::Int32, 3) => { format = vk::Format::R32G32B32_SINT; }
                    (slang::ScalarType::Int32, 4) => { format = vk::Format::R32G32B32A32_SINT; }
                    (slang::ScalarType::Uint32, 2) => { format = vk::Format::R32G32_UINT; }
                    (slang::ScalarType::Uint32, 3) => { format = vk::Format::R32G32B32_UINT; }
                    (slang::ScalarType::Uint32, 4) => { format = vk::Format::R32G32B32A32_UINT; }
                    (slang::ScalarType::Uint8, 2) => { format = vk::Format::R8G8_UINT; }
                    (slang::ScalarType::Uint8, 3) => { format = vk::Format::R8G8B8_UINT; }
                    (slang::ScalarType::Uint8, 4) => { format = vk::Format::R8G8B8A8_UINT; }
                    (slang::ScalarType::Int8, 2) => { format = vk::Format::R8G8_SINT; }
                    (slang::ScalarType::Int8, 3) => { format = vk::Format::R8G8B8_SINT; }
                    (slang::ScalarType::Int8, 4) => { format = vk::Format::R8G8B8A8_SINT; }
                    _ => {}
                }
            }
            _ => {}
        }

        format
    }

    fn find_render_world_binding_attr(v: &slang::reflection::Variable) -> Option<String> {
        for attr in v.user_attributes() {
            if attr.name() == "render_world" {
                return attr.argument_value_string(0).map(String::from);
            }
        }
        None
    }

    // push constant are entry point function parameters with kind uniform
    //let mut size = 0;
    // the `params` in `void main(uniform RootParams* params)`
    let mut root_params_arg_name: Option<&str> = None;
    // layout for the `RootParams` struct
    let mut root_params_ty_layout: Option<&TypeLayout> = None;

    for p in entry_point.parameters() {
        if p.category() == slang::ParameterCategory::Uniform && p.ty().kind() == slang::TypeKind::Pointer {
            root_params_ty_layout = Some(p.type_layout().element_type_layout());
            root_params_arg_name = p.variable().name();
            break;
        }
    }

    let Some(root_params_ty_layout) = root_params_ty_layout else {
        // no root parameter pointer found
        return vec![];
    };

    //eprintln!("root_params_ty: {:?}", root_params_ty.name());
    //eprintln!("root_params_ty_layout: {:?}", root_params_ty_layout.name());
    let mut infos = Vec::new();
    if root_params_ty_layout.kind() == slang::TypeKind::Struct {
        for field in root_params_ty_layout.fields() {
            let render_world_binding = find_render_world_binding_attr(field.variable()).unwrap_or_default();
            let offset = field.offset(field.category());
            let size = field.type_layout().size(field.category());
            let format = convert_root_param_ty(field.ty());
            infos.push(RootParamInfo {
                name: field.variable().name().map(String::from).unwrap_or_default().into(),
                render_world_binding: render_world_binding.into(),
                offset: offset as u32,
                size: size as u32,
                format,
            });
        }
    } else {
        infos.push(RootParamInfo {
            name: root_params_arg_name.unwrap_or("").into(),
            render_world_binding: "".into(),
            offset: 0,
            size: root_params_ty_layout.size(slang::ParameterCategory::Uniform) as u32,
            format: convert_root_param_ty(root_params_ty_layout.ty()),
        })
    }

    infos
}

/*
fn find_user_attribute_by_name<'a>(
    decl: &'a slang::reflection::Decl,
    name: &str,
) -> Option<&'a slang::reflection::UserAttribute> {
    for attr in decl.user_attributes() {
        if attr.name() == name {
            return Some(attr);
        }
    }
    None
}*/

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

/// Returns the size of push constants used by the specified entry point.
fn get_push_constants_size(entry_point: &slang::reflection::EntryPoint) -> usize {
    // push constant are entry point function parameters with kind uniform
    let mut size = 0;
    for p in entry_point.parameters() {
        // There's a PushConstantBuffer category, but it doesn't seem to be used
        if p.category() == slang::ParameterCategory::Uniform {
            size += p.type_layout().size(slang::ParameterCategory::Uniform);
        }
    }
    size
}

fn link_entry_point(
    session: &slang::Session,
    module: &slang::Module,
    index: u32,
) -> anyhow::Result<slang::ComponentType> {
    let entry_point = module.entry_point_by_index(index).unwrap();
    let program = session
        .create_composite_component_type(&[module.downcast().clone(), entry_point.downcast().clone()])
        .map_err(SlangError::from)?;
    let program = program.link().map_err(SlangError::from)?;
    Ok(program)
}

/// Represents a shader entry point
struct EntryPoint {
    _module: slang::Module,
    file: PathBuf,
    file_mtime: u64,
    name: String,
    pass: Option<String>,
    stage: vk::ShaderStageFlags,
    push_constants_size: usize,
    root_params: Vec<RootParamInfo>,
    work_group_size: [u32; 3],
    spirv: Vec<u32>,
}

impl BuildManifest {
    fn load_module_entry_point(
        &self,
        session: &slang::Session,
        module: &slang::Module,
        file: &Path,
        file_mtime: u64,
        index: u32,
        _options: &BuildOptions,
    ) -> anyhow::Result<EntryPoint> {
        let mut pass = None;
        {
            // `[pass("...")]` attribute
            for attr in module
                .entry_point_by_index(index)
                .unwrap()
                .function_reflection()
                .user_attributes()
            {
                if attr.name() == "pass" {
                    pass = attr.argument_value_string(0).map(String::from);
                }
            }
        }

        // compile entry point
        let program = link_entry_point(&session, &module, index)?;

        // retrieve SPIR-V blob
        let blob = {
            let blob = program.entry_point_code(0, 0).map_err(SlangError::from)?;
            convert_spirv_u8_to_u32(blob.as_slice())
        };

        // extract data from reflection: root parameters, push constant size, work group size
        let reflection = program.layout(0).expect("failed to get reflection");
        let entry_point = reflection.entry_point_by_index(0).unwrap();
        let push_constants_size = get_push_constants_size(&entry_point);
        let work_group_size = {
            let s = entry_point.compute_thread_group_size();
            [s[0] as u32, s[1] as u32, s[2] as u32]
        };
        let root_params = get_root_param_info(&entry_point);

        Ok(EntryPoint {
            _module: module.clone(),
            name: entry_point.name().to_string(),
            file: file.to_path_buf(),
            file_mtime,
            stage: slang_stage_to_stage_flags(entry_point.stage()),
            push_constants_size,
            root_params,
            pass,
            work_group_size,
            spirv: blob,
        })
    }

    fn create_slang_session(&self, include_paths: &[String]) -> slang::Session {
        let global_session = get_slang_global_session();

        let mut search_paths_cstr = vec![];
        for path in include_paths.iter() {
            search_paths_cstr.push(CString::new(&**path).unwrap());
        }
        let search_path_ptrs = search_paths_cstr.iter().map(|p| p.as_ptr()).collect::<Vec<_>>();

        let profile = global_session.find_profile(&self.compiler.profile);
        let mut compiler_options = slang::CompilerOptions::default()
            .glsl_force_scalar_layout(true)
            .matrix_layout_column(true)
            .optimization(slang::OptimizationLevel::Default)
            .vulkan_use_entry_point_name(true)
            .debug_information(if self.compiler.debug {
                DebugInfoLevel::Maximal
            } else {
                DebugInfoLevel::None
            })
            .profile(profile);

        for (k, v) in self.compiler.defines.iter() {
            compiler_options = compiler_options.macro_define(k, v);
        }

        let target_desc = slang::TargetDesc::default()
            .format(slang::CompileTarget::Spirv)
            .options(&compiler_options);
        let targets = [target_desc];

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_path_ptrs)
            .options(&compiler_options);

        let session = global_session
            .create_session(&session_desc)
            .expect("failed to create session");
        session
    }

    /// Loads a shader module from a file and extracts its entry points.
    fn load_module(
        &self,
        file: &Path,
        include_paths: &[String],
        options: &BuildOptions,
        entry_points: &mut Vec<EntryPoint>,
    ) -> anyhow::Result<()> {
        let session = self.create_slang_session(include_paths);
        let (_canonical_path, file_mtime) = get_file_mtime(file)?;

        // load slang module file
        let module = session.load_module(&file.to_string_lossy()).map_err(SlangError::from)?;

        //let linked_module = module.downcast().link().map_err(SlangError::from)?;
        //let layout = linked_module.layout(0).expect("failed to get layout");

        // compile all entry points
        let mut errors = String::new();
        for i in 0..module.entry_point_count() {
            match self.load_module_entry_point(&session, &module, file, file_mtime, i, options) {
                Ok(entry_point) => {
                    entry_points.push(entry_point);
                }
                Err(err) => {
                    errors.push_str(&err.to_string());
                    errors.push_str("\n");
                }
            }
        }

        if !errors.is_empty() {
            bail!(errors)
        }

        Ok(())
    }

    fn write_pipeline(
        &self,
        archive: &mut PipelineArchiveWriter,
        pipeline_name: &str,
        pass: Option<&Pass>,
        gs: &GraphicsState,
        entry_points: &[&EntryPoint],
        _options: &BuildOptions,
    ) -> anyhow::Result<PipelineEntryData> {
        let mut push_constants_size = 0;
        let mut workgroup_size = [1u32; 3];
        let mut shaders = vec![];
        let mut stage_flags = vk::ShaderStageFlags::default();
        let mut root_params = vec![];
        let mut source_dependencies = BTreeSet::new();

        for &entry_point in entry_points {
            push_constants_size = push_constants_size.max(entry_point.push_constants_size);
            workgroup_size = entry_point.work_group_size;
            stage_flags |= entry_point.stage;
            // TODO check that root parameters agree for all stages
            if !entry_point.root_params.is_empty() {
                root_params = entry_point.root_params.clone();
            }

            source_dependencies.insert((entry_point.file.canonicalize().unwrap(), entry_point.file_mtime));

            let code_offset = archive.write_slice(entry_point.spirv.as_slice());
            shaders.push(ShaderData {
                stage: entry_point.stage,
                entry_point: entry_point.name.as_str().into(),
                spirv: code_offset,
            });
        }

        let pipeline_kind = if stage_flags.contains(vk::ShaderStageFlags::COMPUTE) {
            shader_archive::PipelineKind::Compute(shader_archive::ComputePipelineData {
                push_constants_size: push_constants_size as u16,
                compute_shader: shaders[0],
                workgroup_size,
            })
        } else {
            // TODO check sanity of entry point stages
            let color_targets = {
                let mut color_targets = Vec::new();
                for ct in gs.color_targets.iter() {
                    color_targets.push(archive.write(ct));
                }
                archive.write_slice(color_targets.as_slice())
            };

            // attachments
            let mut color_attachments = Offset::INVALID;
            let mut depth_stencil_attachment = None;
            if let Some(pass) = pass {
                {
                    let mut attachments = Vec::with_capacity(pass.color_attachments.len());
                    for ca in pass.color_attachments.iter() {
                        attachments.push(shader_archive::ColorAttachment {
                            resource_name: ca.resource.as_ref().map(|s| s.as_str().into()).unwrap_or_default(),
                            clear_color: ca.clear_color,
                        });
                    }
                    color_attachments = archive.write_slice(&attachments[..])
                }

                if let Some(dsa) = &pass.depth_stencil_attachment {
                    depth_stencil_attachment = Some(shader_archive::DepthStencilAttachment {
                        resource_name: dsa.resource.as_ref().map(|s| s.as_str().into()).unwrap_or_default(),
                        clear_depth: dsa.clear_depth,
                        clear_stencil: dsa.clear_stencil,
                    })
                }

                // check that we have the correct number of attachments
                if pass.color_attachments.len() != gs.color_targets.len() {
                    bail!(
                        "pipeline `{}` has {} color attachments, but {} color blend targets",
                        pipeline_name,
                        pass.color_attachments.len(),
                        gs.color_targets.len()
                    );
                }
            };

            let shaders = archive.write_slice(&shaders[..]);
            shader_archive::PipelineKind::Graphics(shader_archive::GraphicsPipelineData {
                push_constants_size: push_constants_size as u16,
                shaders,
                rasterization: gs.rasterizer,
                depth_stencil: gs.depth_stencil,
                color_targets,
                color_attachments,
                depth_stencil_attachment,
            })
        };

        // TODO find a way to get paths to module dependencies as well
        let sources = {
            let sources = source_dependencies
                .into_iter()
                .map(|(path, mtime)| shader_archive::FileDependency {
                    path: archive.write_str(&*path.to_string_lossy()),
                    mtime,
                })
                .collect::<Vec<_>>();
            archive.write_slice(&sources)
        };

        let root_params = if !root_params.is_empty() {
            archive.write_slice(&root_params[..])
        } else {
            Offset::INVALID
        };

        Ok(PipelineEntryData {
            name: ZString64::new(&pipeline_name),
            kind: pipeline_kind,
            root_params: RootParamLayout {
                byte_size: 0xABCDEF12, // FIXME
                parameters: root_params,
            },
            sources,
        })
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    /// Resolves relative paths relative to the manifest directory, or
    /// returns the given path if absolute.
    fn resolve_path(&self, path: &str) -> PathBuf {
        self.manifest_path.parent().unwrap().join(path)
    }

    /// Resolves glob patterns to a list of file paths.
    fn resolve_glob_file_paths(&self, patterns: &[String]) -> anyhow::Result<Vec<PathBuf>> {
        // relative paths in the manifest are relative to the manifest directory, so set
        // the current directory to that for glob resolution
        let prev_current_dir = env::current_dir()?.canonicalize()?;
        if let Some(manifest_dir) = self.manifest_path.parent() {
            env::set_current_dir(manifest_dir)?;
        }

        let mut paths = Vec::new();
        for pattern in patterns {
            let entries = glob::glob(pattern)
                .map_err(|err| anyhow!("failed to parse glob pattern '{}': {}", pattern, err.to_string()))?;
            for entry in entries {
                match entry {
                    Ok(path) => {
                        // make path relative to original current dir
                        let canonical_path = path.canonicalize()?;
                        //eprintln!(
                        //    "path={}, canonical={}, cwd={}",
                        //    path.display(),
                        //    canonical_path.display(),
                        //    prev_current_dir.display()
                        //);
                        let relative_path = canonical_path.strip_prefix(&prev_current_dir).unwrap().to_path_buf();
                        paths.push(relative_path);
                    }
                    Err(err) => {
                        warn!("failed to resolve glob entry: {}", err.to_string());
                    }
                }
            }
        }

        env::set_current_dir(prev_current_dir)?;
        Ok(paths)
    }

    /// Scans shader source files for shader definitions and collects a list of shader pipelines and entry points to compile.
    pub(crate) fn build(&self, options: &BuildOptions) -> anyhow::Result<()> {
        // resolve paths
        let files = self.resolve_glob_file_paths(&self.input_files)?;
        let output_file = self.resolve_path(&self.output_file);
        let spirv_dump_path = output_file.parent().unwrap().join("spirv");
        let include_paths: Vec<String> = self
            .include_paths
            .iter()
            .map(|p| self.resolve_path(p).to_string_lossy().into_owned())
            .collect();

        if options.emit_cargo_deps {
            println!("cargo:rerun-if-changed={}", self.manifest_path.display());
        }

        let mut got_errors = false;
        // load all slang modules and compile all entry points
        let mut entry_points = Vec::new();
        for file in files {
            let absolute_file_path = file.canonicalize()?;

            if !options.quiet {
                cprintln!("<g,bold>Compiling</> {}", file.display());
            }
            if options.emit_cargo_deps {
                println!("cargo:rerun-if-changed={}", absolute_file_path.display());
            }

            match self.load_module(&file, &include_paths, options, &mut entry_points) {
                Ok(_) => {}
                Err(err) => {
                    if options.emit_cargo_deps {
                        // use cargo::error when running in a build script, otherwise absolutely
                        // nothing is reported to the user even through stderr, unless running
                        // `cargo -vv`
                        println!("cargo::error={}", err);
                    } else {
                        ceprintln!("<r,bold>error</>: {err}");
                    }
                    got_errors = true;
                }
            }
        }

        // dump SPIR-V files if requested
        if options.emit_spirv_binaries {
            if !spirv_dump_path.exists() {
                fs::create_dir(&spirv_dump_path)?;
            }

            for entry_point in entry_points.iter() {
                let name = &entry_point.name;
                let stage = match entry_point.stage {
                    vk::ShaderStageFlags::VERTEX => "vert",
                    vk::ShaderStageFlags::FRAGMENT => "frag",
                    vk::ShaderStageFlags::COMPUTE => "comp",
                    vk::ShaderStageFlags::MESH_EXT => "mesh",
                    vk::ShaderStageFlags::TASK_EXT => "task",
                    _ => "unknown",
                };
                let output_path = spirv_dump_path.join(format!("{name}.{stage}.spv"));
                if !options.quiet {
                    cprintln!("<g,bold>Dumping</> {}", output_path.display());
                }
                fs::write(output_path, unsafe {
                    slice::from_raw_parts(entry_point.spirv.as_ptr() as *const u8, entry_point.spirv.len() * 4)
                })
                .with_context(|| format!("dumping SPIR-V for entry point {}", entry_point.name))?;
            }
        }

        // exit if there were errors
        if got_errors {
            bail!("errors occurred during shader compilation");
        }

        // collect pipelines
        let pipelines = {
            let mut pipelines: BTreeMap<&str, Vec<&EntryPoint>> = BTreeMap::new();
            for entry_point in entry_points.iter() {
                if let Some(ref pass) = entry_point.pass {
                    pipelines.entry(pass).or_default().push(entry_point);
                }
            }
            pipelines
        };

        if !options.quiet {
            let pipeline_list = pipelines.keys().cloned().collect::<Vec<_>>().join(&",");
            cprintln!(
                "<g,bold>Compiled</> {} entry points, {} pipelines \n\t<dim>{}</>",
                entry_points.len(),
                pipelines.len(),
                pipeline_list
            );
        }

        if pipelines.is_empty() {
            cprintln!(
                "<y,bold>warning</>: no pipelines found in the input files (possibly missing `pass(\"...\")` attributes?)"
            );
        }

        // write archive
        if !options.quiet {
            cprintln!("<g,bold>Writing</> {}", output_file.display());
        }

        let mut archive = ArchiveWriter::new();
        let pipelines_offset = {
            let mut entries = Vec::new();
            for (&pipeline_name, entry_points) in pipelines.iter() {
                let mut state = self.default.clone();
                let pass = if let Some(pass) = self.pass.get(pipeline_name) {
                    state.apply_overrides(&pass.raw)?;
                    Some(pass)
                } else {
                    None
                };
                entries.push(self.write_pipeline(&mut archive, pipeline_name, pass, &state, entry_points, options)?);
            }
            archive.write_slice(&entries[..])
        };

        let manifest_file = make_file_dependency(&self.manifest_path, &mut archive)?;

        let _ = archive.write_root(&shader_archive::PipelineArchiveData {
            manifest_file,
            pipelines: pipelines_offset,
            render_targets: Offset::INVALID,
        });
        archive.write_to_file(&output_file).context("writing output")?;

        Ok(())
    }
}
