use crate::reflection::CollectedReflectionData;
use crate::{BuildManifest, BuildOptions, GraphicsState, Pass, get_file_mtime};
use anyhow::{Context, anyhow, bail};
use color_print::{ceprintln, cprintln};
use log::warn;
use sharc::archive::{ArchiveWriter, Offset};
use sharc::gpu::{ImageUsage, is_depth_format, vk};
use sharc::zstring::ZString64;
use sharc::{FileDependency, RootParamInfo, RootParamLayout, Shader, reflection};
use slang::DebugInfoLevel;
use slang::reflection::TypeLayout;
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::{env, fs, slice};

type ShaderArchiveWriter = ArchiveWriter<sharc::ShaderArchiveRoot>;

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

fn find_user_attribute<'a>(
    decl: &'a slang::reflection::Variable,
    name: &str,
) -> Option<&'a slang::reflection::UserAttribute> {
    for attr in decl.user_attributes() {
        if attr.name().unwrap() == name {
            return Some(attr);
        }
    }
    None
}

fn get_user_attribute_string<'a>(var: &'a slang::reflection::Variable, name: &str, index: u32) -> Option<&'a str> {
    find_user_attribute(var, name).and_then(|attr| attr.argument_value_string(index))
}

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
                let element_ty = ty.element_type().unwrap().scalar_type();
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

    // push constant are entry point function parameters with kind uniform
    //let mut size = 0;
    // the `params` in `void main(uniform RootParams* params)`
    let mut root_params_arg_name: Option<&str> = None;
    // layout for the `RootParams` struct
    let mut root_params_ty_layout: Option<&TypeLayout> = None;

    for p in entry_point.parameters() {
        if p.category() == Some(slang::ParameterCategory::Uniform) && p.ty().unwrap().kind() == slang::TypeKind::Pointer
        {
            root_params_ty_layout = Some(p.type_layout().unwrap().element_type_layout().unwrap());
            root_params_arg_name = p.variable().unwrap().name();
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
            let render_world_binding =
                get_user_attribute_string(field.variable().unwrap(), "RenderWorld", 0).unwrap_or_default();
            let offset = field.offset(field.category().unwrap());
            let size = field.type_layout().unwrap().size(field.category().unwrap());
            let format = convert_root_param_ty(field.ty().unwrap());
            infos.push(RootParamInfo {
                name: field
                    .variable()
                    .unwrap()
                    .name()
                    .map(String::from)
                    .unwrap_or_default()
                    .into(),
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
            format: convert_root_param_ty(root_params_ty_layout.ty().unwrap()),
        })
    }

    infos
}

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
        if p.category().unwrap() == slang::ParameterCategory::Uniform {
            size += p.type_layout().unwrap().size(slang::ParameterCategory::Uniform);
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
        .create_composite_component_type(&[module.clone().into(), entry_point.into()])
        .map_err(SlangError::from)?;
    let program = program.link().map_err(SlangError::from)?;
    Ok(program)
}

/// Represents a shader module
struct Module {
    name: String,
    module: slang::Module,
    file_path: PathBuf,
    file_mtime: u64,
    spirv: Vec<u32>,
    // spirv_archive: Offset<[u32]>,
    program: slang::ComponentType,
    reflection: Vec<reflection::Param>,
    entry_points: Vec<EntryPoint>,
}

/// Represents a shader entry point
struct EntryPoint {
    name: String,
    pass: Option<String>,
    stage: vk::ShaderStageFlags,
    push_constants_size: usize,
    work_group_size: [u32; 3],
}

/*fn load_module_entry_point(
    session: &slang::Session,
    module: &slang::Module,
    file: &Path,
    file_mtime: u64,
    index: u32,
    options: &BuildOptions,
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
    if options.verbosity >= 2 {
        cprintln!(
            "    load_module_entry_point: {}",
            module.entry_point_by_index(index).unwrap().function_reflection().name()
        );
    }
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
        program,
        push_constants_size,
        root_params,
        pass,
        work_group_size,
        spirv: blob,
    })
}*/

#[derive(Default)]
struct Stats {
    entry_point_count: usize,
    pass_count: usize,
    pass_names: BTreeSet<String>,
}

impl BuildManifest {
    /// Resolves relative paths relative to the manifest directory, or
    /// returns the given path if absolute.
    fn resolve_path(&self, path: &str) -> PathBuf {
        self.manifest_path.parent().unwrap().join(path)
    }

    /// Resolves glob patterns to a list of file paths.
    fn resolve_glob_file_paths(&self, patterns: &[String]) -> anyhow::Result<Vec<PathBuf>> {
        // relative paths in the manifest are relative to the manifest directory, so set
        // the current directory to that for glob resolution
        let prev_current_dir = env::current_dir()?;
        let prev_current_dir_canonical = prev_current_dir.canonicalize()?;
        if let Some(manifest_dir) = self.manifest_path.parent() {
            if !manifest_dir.as_os_str().is_empty() {
                env::set_current_dir(manifest_dir).with_context(|| {
                    format!(
                        "failed to set current directory `{}` for glob resolution",
                        manifest_dir.display()
                    )
                })?;
            }
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
                        let relative_path = canonical_path
                            .strip_prefix(&prev_current_dir_canonical)
                            .unwrap()
                            .to_path_buf();
                        paths.push(relative_path);
                    }
                    Err(err) => {
                        warn!("failed to resolve glob entry: {}", err.to_string());
                    }
                }
            }
        }

        // NOTE: be careful not to set canonicalized paths as current directories: on Windows they
        //       start with the extended-length prefix ("\\?\") and this confuses tools down the line.
        //       Previously we were setting canonicalized paths here, which caused issues with SPIR-V
        //       debug information generation (extended-length paths ended up in the SPIR-V debug info
        //       and some tools, like nvidia nsight, don't handle them properly).
        env::set_current_dir(prev_current_dir)?;
        Ok(paths)
    }

    fn create_slang_session(&self, include_paths: &[String], options: &BuildOptions) -> slang::Session {
        let global_session = get_slang_global_session();

        // debug info can be requested either in the manifest or via build options
        let emit_debug_information = self.compiler.debug | options.emit_debug_information;

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
            .debug_information(if emit_debug_information {
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
    fn load_slang_module(
        &self,
        archive: &mut ShaderArchiveWriter,
        session: &slang::Session,
        file: &Path,
        options: &BuildOptions,
    ) -> anyhow::Result<Module> {
        //let session = self.create_slang_session(include_paths, options);
        let (canonical_path, file_mtime) = get_file_mtime(file)?;

        if options.verbosity >= 2 {
            cprintln!("load_slang_module: {}", file.display());
        }
        let module = session.load_module(&file.to_string_lossy()).map_err(SlangError::from)?;

        // FIXME slang modules are normally declared by `module module_name;`
        let module_name = file
            .file_stem()
            .ok_or(anyhow!("invalid shader file name: {}", file.display()))?
            .to_string_lossy()
            .to_string();

        let entry_point_count = module.entry_point_count();
        if entry_point_count == 0 {
            // Bail out if there are no entry points in the module.
            //
            // This is not an error; some modules are meant to be used as libraries and don't contain
            // entry points. While it could be possible to emit a SPIR-V "library"
            // with no entry points, it currently crashes the Slang compiler.
            return Ok(Module {
                module: module.clone(),
                name: module_name,
                file_path: canonical_path,
                file_mtime,
                spirv: vec![],
                // FIXME: it would be better if we didn't return anything at all here
                program: module.clone().into(),
                reflection: vec![],
                entry_points: vec![],
            });
        }

        let mut components = vec![module.clone().into()];
        for i in 0..entry_point_count {
            let entry_point = module.entry_point_by_index(i).unwrap();
            components.push(entry_point.into());
        }

        // module + all entry points composite
        let composite = session
            .create_composite_component_type(&components)
            .map_err(SlangError::from)?;

        // linked program
        let program = composite.link().map_err(SlangError::from)?;

        // reflection v2
        let reflection = {
            let mut collector = CollectedReflectionData::new(archive, options);
            collector.reflect_shader(program.layout(0).map_err(SlangError::from)?);
            collector.params
        };

        // retrieve SPIR-V blob
        let blob = {
            let blob = program.target_code(0).map_err(SlangError::from)?;
            convert_spirv_u8_to_u32(blob.as_slice())
        };

        let spirv_archive = archive.write_slice(blob.as_slice());

        let mut entry_points = Vec::new();
        for i in 0..module.entry_point_count() {
            let reflection = program.layout(0).expect("failed to get reflection");
            let entry_point = reflection.entry_point_by_index(i).unwrap();
            let push_constants_size = get_push_constants_size(&entry_point);
            let work_group_size = {
                let s = entry_point.compute_thread_group_size();
                [s[0] as u32, s[1] as u32, s[2] as u32]
            };
            //let root_params = get_root_param_info(&entry_point);

            let mut pass = None;

            // `[pass("...")]` attribute
            for attr in module
                .entry_point_by_index(i)
                .unwrap()
                .function_reflection()
                .user_attributes()
            {
                if attr.name().unwrap() == "pass" {
                    pass = attr.argument_value_string(0).map(String::from);
                }
            }

            if options.verbosity >= 2 {
                cprintln!("entry point: {}/{}", file.display(), entry_point.name().unwrap());
            }

            entry_points.push(EntryPoint {
                name: entry_point.name().unwrap().to_string(),
                stage: slang_stage_to_stage_flags(entry_point.stage()),
                push_constants_size,
                pass,
                work_group_size,
            });
        }

        Ok(Module {
            module,
            name: module_name,
            file_path: canonical_path,
            file_mtime,
            spirv: blob,
            program,
            reflection,
            entry_points,
        })
    }

    fn write_pass(
        &self,
        archive: &mut ShaderArchiveWriter,
        pipeline_name: &str,
        pass: Option<&Pass>,
        gs: &GraphicsState,
        entry_points: &[&EntryPoint],
        _options: &BuildOptions,
    ) -> anyhow::Result<sharc::Pass> {
        let mut push_constants_size = 0;
        let mut workgroup_size = [1u32; 3];
        let mut shaders = vec![];
        let mut stage_flags = vk::ShaderStageFlags::default();
        //let mut root_params = vec![];

        for &entry_point in entry_points {
            // collect pass information from entry points:
            // - max push constant size across entry points
            // - workgroup size for compute shaders
            // - stages in the pipeline
            push_constants_size = push_constants_size.max(entry_point.push_constants_size);
            workgroup_size = entry_point.work_group_size;
            stage_flags |= entry_point.stage;

            // TODO check that root parameters agree for all stages
            //if !entry_point.root_params.is_empty() {
            //    root_params = entry_point.root_params.clone();
            //}

            shaders.push(Shader {
                stage: entry_point.stage,
                entry_point: entry_point.name.as_str().into(),
            });
        }

        let pipeline_kind = if stage_flags.contains(vk::ShaderStageFlags::COMPUTE) {
            sharc::PipelineKind::Compute(sharc::ComputePipeline {
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
                        attachments.push(sharc::ColorAttachment {
                            resource_name: ca.resource.as_ref().map(|s| s.as_str().into()).unwrap_or_default(),
                            clear_color: ca.clear_color,
                        });
                    }
                    color_attachments = archive.write_slice(&attachments[..])
                }

                if let Some(dsa) = &pass.depth_stencil_attachment {
                    depth_stencil_attachment = Some(sharc::DepthStencilAttachment {
                        resource_name: dsa.resource.as_ref().map(|s| s.as_str().into()).unwrap_or_default(),
                        clear_depth: dsa.clear_depth,
                        clear_stencil: dsa.clear_stencil,
                    })
                }

                // check that we have the correct number of attachments
                if pass.color_attachments.len() != gs.color_targets.len() {
                    warn!(
                        "pipeline `{}` has {} color attachments, but {} color blend targets",
                        pipeline_name,
                        pass.color_attachments.len(),
                        gs.color_targets.len()
                    );
                }
            };

            let shaders = archive.write_slice(&shaders[..]);
            sharc::PipelineKind::Graphics(sharc::GraphicsPipeline {
                push_constants_size: push_constants_size as u16,
                shaders,
                rasterization: gs.rasterizer,
                depth_stencil: gs.depth_stencil,
                color_targets,
                color_attachments,
                depth_stencil_attachment,
            })
        };

        /*let root_params = if !root_params.is_empty() {
            archive.write_slice(&root_params[..])
        } else {
            Offset::INVALID
        };*/

        Ok(sharc::Pass {
            name: ZString64::new(&pipeline_name),
            kind: pipeline_kind,
            root_params: RootParamLayout {
                byte_size: 0xABCDEF12, // FIXME
                parameters: Offset::INVALID,
            },
            signature: Offset::INVALID, // TODO
        })
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    /// Builds the list of resources.
    fn write_image_resources(&self, archive: &mut ShaderArchiveWriter) -> Offset<[sharc::ImageResourceDesc]> {
        let mut images = Vec::with_capacity(self.resources.len());
        for (name, desc) in self.resources.iter() {
            let size = match (desc.width, desc.height) {
                (Some(w), Some(h)) => sharc::ImageResourceSize::Fixed { width: w, height: h },
                _ => {
                    // TODO what to do if only one dimension is specified?
                    //      for now, treat as render target size
                    sharc::ImageResourceSize::RenderTarget
                }
            };

            images.push(sharc::ImageResourceDesc {
                name: name.as_str().into(),
                format: desc.format,
                // If the usage is specified explicitly, use that. Otherwise,
                // we assume that most images are going to be used as render target attachments
                // and sampled in shaders.
                // This is maybe pessimistic, but it's complicated and error-prone to infer that
                // from shader reflection alone.
                usage: if let Some(usage) = desc.usage {
                    usage
                } else {
                    (if is_depth_format(desc.format) {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    } else {
                        ImageUsage::COLOR_ATTACHMENT
                    }) | ImageUsage::SAMPLED
                        | ImageUsage::STORAGE
                },
                size,
            });
        }
        archive.write_slice(&images[..])
    }

    fn write_module(
        &self,
        archive: &mut ShaderArchiveWriter,
        module: &Module,
        options: &BuildOptions,
        stats: &mut Stats,
    ) -> anyhow::Result<sharc::Module> {
        // collect passes
        let mut pipelines: BTreeMap<&str, Vec<&EntryPoint>> = BTreeMap::new();
        for entry_point in module.entry_points.iter() {
            if let Some(ref pass) = entry_point.pass {
                pipelines.entry(pass).or_default().push(entry_point);
            }
        }

        let pipelines_offset = {
            let mut entries = Vec::new();
            for (&pipeline_name, entry_points) in pipelines.iter() {
                let mut state = self.default.clone();

                // pipeline overrides are specified as `[pass.module_name.pipeline_name]` in TOML
                let pass = if let Some(module) = self.pass.get(&module.name) {
                    if let Some(pass) = module.get(pipeline_name) {
                        state.apply_overrides(&pass.raw)?;
                        Some(pass)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // record pass name for warning about unused overrides
                stats.pass_names.insert(format!("{}.{}", module.name, pipeline_name));

                entries.push(self.write_pass(archive, pipeline_name, pass, &state, entry_points, options)?);
            }
            archive.write_slice(&entries[..])
        };

        if !options.quiet {
            let pipeline_list = pipelines.keys().cloned().collect::<Vec<_>>().join(&",");
            let entry_point_count = module.entry_points.len();
            cprintln!(
                "<g,bold>Compiled</> {} entry points, {} pipelines \n\t<dim>{}</>",
                entry_point_count,
                pipelines.len(),
                pipeline_list
            );
        }

        stats.pass_count += pipelines.len();
        stats.entry_point_count += module.entry_points.len();

        let spirv = archive.write_slice(&module.spirv[..]);
        let name = archive.write_str(&module.name);
        let params = archive.write_slice(&module.reflection[..]);
        let path = archive.write_str(&module.file_path.to_string_lossy());

        let module = sharc::Module {
            name,
            spirv,
            passes: pipelines_offset,
            file: FileDependency {
                path,
                mtime: module.file_mtime,
            },
            params,
        };
        Ok(module)
    }

    /// Scans shader source files for shader definitions and collects a list of shader pipelines and entry points to compile.
    pub(crate) fn build(&self, options: &BuildOptions) -> anyhow::Result<()> {
        // resolve paths
        let files = self
            .resolve_glob_file_paths(&self.input_files)
            .context("error resolving input files")?;
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

        let compiler_session = self.create_slang_session(&include_paths, options);

        let mut got_errors = false;
        // load all slang modules and compile all entry points

        let mut archive = ArchiveWriter::new();
        let mut stats = Stats::default();
        let mut modules = Vec::new();

        // for each module
        for file in files {
            let absolute_file_path = file.canonicalize()?;

            if !options.quiet {
                cprintln!("<g,bold>Compiling</> {}", file.display());
            }
            if options.emit_cargo_deps {
                println!("cargo:rerun-if-changed={}", absolute_file_path.display());
            }

            match self.load_slang_module(&mut archive, &compiler_session, &file, options) {
                Ok(module) => {
                    // if there are no entry points, skip writing the module
                    if module.entry_points.is_empty() {
                        if options.verbosity >= 2 {
                            cprintln!("<cyan>note</>: `{}` has no entry points, skipping", file.display());
                        }
                        continue;
                    }

                    // dump SPIR-V binaries if requested
                    if options.emit_spirv_binaries {
                        if !spirv_dump_path.exists() {
                            fs::create_dir(&spirv_dump_path)?;
                        }

                        let mod_name = &module.name;
                        let output_path = spirv_dump_path.join(format!("{mod_name}.spv"));
                        if !options.quiet {
                            cprintln!("<g,bold>Dumping</> {}", output_path.display());
                        }
                        fs::write(&output_path, unsafe {
                            slice::from_raw_parts(module.spirv.as_ptr() as *const u8, module.spirv.len() * 4)
                        })
                        .context(format!("dumping SPIR-V at {}", output_path.display()))?;
                    }

                    // write module to archive
                    modules.push(self.write_module(&mut archive, &module, options, &mut stats)?);
                }
                Err(err) => {
                    if options.emit_cargo_deps {
                        // use cargo::error when running in a build script, otherwise absolutely
                        // nothing is reported to the user even through stderr, unless running
                        // `cargo -vv`
                        for line in err.to_string().lines() {
                            println!("cargo::error={}", line);
                        }
                    } else {
                        ceprintln!("<r,bold>error</>: {err}");
                    }
                    got_errors = true;
                }
            }
        }

        // dump SPIR-V files if requested

        // exit if there were errors
        if got_errors {
            bail!("errors occurred during shader compilation");
        }

        if stats.pass_count == 0 {
            cprintln!(
                "<y,bold>warning</>: no pipelines found in the input files (possibly missing `pass(\"...\")` attributes?)"
            );
        }

        // check if there were some pass overrides that didn't match anything
        for (module_name, passes) in self.pass.iter() {
            for (pass_name, _) in passes.iter() {
                let name = format!("{module_name}.{pass_name}");
                if !stats.pass_names.contains(&name) {
                    cprintln!("<y,bold>warning</>: override `{}` did not match any pass", name);
                }
            }
        }

        if !options.quiet {
            cprintln!("<g,bold>Writing</> {}", output_file.display());
        }

        // emit resource entries
        let images = self.write_image_resources(&mut archive);

        let manifest_path = archive.write_str(&self.canonical_manifest_path.to_string_lossy());
        let modules = archive.write_slice(&modules[..]);

        // write archive root and dump to file
        let _ = archive.write_root(&sharc::ShaderArchiveRoot {
            manifest: FileDependency {
                path: manifest_path,
                mtime: self.mtime,
            },
            modules,
            images,
        });

        archive.write_to_file(&output_file).context("writing output")?;

        Ok(())
    }
}
