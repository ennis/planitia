use crate::reflection::CollectedReflectionData;
use crate::{BuildManifest, BuildOptions, GraphicsState, Pass, get_file_mtime, get_log_options};
use anyhow::{Context, anyhow, bail};
use color_print::{ceprintln, cprintln};
use log::warn;
use sharc::archive::{ArchiveWriter, Offset};
use sharc::gpu::{ImageUsage, is_depth_format, vk};
use sharc::zstring::ZString64;
use sharc::{FileDependency, RootParamLayout, Shader, reflection};
use slang::DebugInfoLevel;
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::{env, fs, slice};

type ShaderArchiveWriter = ArchiveWriter<sharc::ShaderArchiveRoot>;

fn get_slang_global_session() -> slang::GlobalSession {
    thread_local! {
        static SESSION: OnceCell<slang::GlobalSession> = OnceCell::new();
    }

    SESSION.with(|s| s.get_or_init(|| slang::GlobalSession::new().expect("Failed to create Slang session")).clone())
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wrapper around `slang::Error` that is `Send + Sync` and can therefore be stored in an
/// `anyhow::Error`.  Internally the error message is captured as a `String`.
#[derive(thiserror::Error, Debug)]
#[error("{0}")]
struct SlangError(String);

impl From<slang::Error> for SlangError {
    fn from(err: slang::Error) -> Self {
        SlangError(err.to_string())
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

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A compiled Slang shader module with all its entry points.
struct Module {
    name: String,
    _module: slang::Module,
    _program: slang::ComponentType,
    file_path: PathBuf,
    file_mtime: u64,
    spirv: Vec<u32>,
    reflection: Vec<reflection::Param>,
    entry_points: Vec<EntryPoint>,
}

/// A single shader entry point extracted from a [`Module`].
struct EntryPoint {
    name: String,
    params: Vec<reflection::Param>,
    /// The pass this entry point belongs to, either from a `[pass("...")]` attribute or inferred
    /// from the entry point name by stripping the stage suffix.
    pass: Option<String>,
    stage: vk::ShaderStageFlags,
    push_constants_size: usize,
    work_group_size: [u32; 3],
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Accumulated build statistics.
#[derive(Default)]
struct Stats {
    entry_point_count: usize,
    pass_count: usize,
    /// Fully-qualified pass names (`module.pass`) seen during the build, used to detect
    /// manifest override entries that do not match any compiled pass.
    pass_names: BTreeSet<String>,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl BuildManifest {
    // Resolves a path relative to the manifest directory (no-op for absolute paths).
    //fn resolve_path(&self, path: &str) -> PathBuf {
    //    self.manifest_path.parent().unwrap().join(path)
    //}

    /// Creates a Slang compilation session configured according to the manifest and build options.
    fn create_slang_session(&self, include_paths: &[PathBuf], options: &BuildOptions) -> anyhow::Result<slang::Session> {
        let global_session = get_slang_global_session();

        // Debug information can be requested either in the manifest or via build options.
        let emit_debug_information = self.compiler.debug | options.emit_debug_information;

        let search_paths_cstr: Vec<CString> = include_paths.iter().map(|p| CString::new(&*p.to_string_lossy()).unwrap()).collect();
        let search_path_ptrs: Vec<_> = search_paths_cstr.iter().map(|p| p.as_ptr()).collect();

        let profile = global_session.find_profile(&self.compiler.profile);
        let mut compiler_options = slang::CompilerOptions::default()
            .glsl_force_scalar_layout(true)
            .matrix_layout_column(true)
            .optimization(slang::OptimizationLevel::Default)
            .vulkan_use_entry_point_name(true)
            .debug_information(if emit_debug_information { DebugInfoLevel::Maximal } else { DebugInfoLevel::None })
            .profile(profile);

        for (k, v) in self.compiler.defines.iter() {
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

    /// Loads a Slang source file, compiles all its shader entry points, and returns the resulting
    /// [`Module`] together with reflection data.
    fn load_slang_module(
        &self,
        archive: &mut ShaderArchiveWriter,
        session: &slang::Session,
        file: &Path,
        source: &str,
        options: &BuildOptions,
    ) -> anyhow::Result<Module> {
        let (canonical_path, file_mtime) = get_file_mtime(file)?;

        if get_log_options().verbosity >= 2 {
            cprintln!("load_slang_module: {}", file.display());
        }

        // FIXME: Slang modules are normally declared with `module module_name;`; for now we
        //        fall back to the file stem.
        let module_name = file
            .file_stem()
            .ok_or(anyhow!("invalid shader file name: {}", file.display()))?
            .to_string_lossy()
            .to_string();
        let module = session
            .load_module_from_source_string(&module_name, &file.to_string_lossy(), source)
            .map_err(SlangError::from)?;

        let entry_point_count = module.entry_point_count();
        if entry_point_count == 0 {
            // Bail out if there are no entry points in the module.
            // This is not an error; some modules are meant to be used as libraries and don't contain
            // entry points. While it could be possible to emit a SPIR-V "library"
            // with no entry points, it currently crashes the Slang compiler.
            return Ok(Module {
                _module: module.clone(),
                _program: module.clone().into(),
                name: module_name,
                file_path: canonical_path,
                file_mtime,
                spirv: vec![],
                reflection: vec![],
                entry_points: vec![],
            });
        }

        // Build a composite component (module + every entry point) and link it.
        let mut components = vec![module.clone().into()];
        for i in 0..entry_point_count {
            components.push(module.entry_point_by_index(i).unwrap().into());
        }
        let composite = session.create_composite_component_type(&components).map_err(SlangError::from)?;
        let program = composite.link().map_err(SlangError::from)?;

        // Collect reflection data.
        let reflection = {
            let mut collector = CollectedReflectionData::new(archive, options);
            collector.reflect_shader(program.layout(0).map_err(SlangError::from)?);
            collector.params
        };

        // retrieve SPIR-V blob of all entry points
        let spirv = {
            let blob = program.target_code(0).map_err(SlangError::from)?;
            convert_spirv_u8_to_u32(blob.as_slice())
        };

        // Extract per-entry-point metadata.
        let mut entry_points = Vec::new();
        for i in 0..entry_point_count {
            let layout = program.layout(0).expect("failed to get reflection");
            let ep = layout.entry_point_by_index(i).unwrap();
            let push_constants_size = get_push_constants_size(ep);
            let work_group_size = {
                let s = ep.compute_thread_group_size();
                [s[0] as u32, s[1] as u32, s[2] as u32]
            };

            let params = {
                let mut collector = CollectedReflectionData::new(archive, options);
                collector.reflect_entry_point(ep);
                collector.params
            };

            // Determine the pass name: prefer the `[pass("...")]` attribute, then fall back to
            // stripping well-known stage suffixes from the entry point name.
            let mut pass = None;

            // `[pass("...")]` attribute
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
                if attr.name().unwrap() == "pass" {
                    pass = attr.argument_value_string(0).map(String::from);
                }
            }

            // If there's no pass attribute, try to strip a stage suffix from the entry point name, and use that as the pass name.
            // E.g.: `pass_name_vertex` and `pass_name_fragment` will be grouped into the same pass `pass_name`.
            if pass.is_none() {
                const STAGE_SUFFIXES: &[&str] =
                    &["_vertex", "_fragment", "_compute", "_mesh", "_amplification", "_hull", "_domain", "_geometry"];
                let mut name = ep.name().unwrap();
                for suffix in STAGE_SUFFIXES {
                    if let Some(stripped) = name.strip_suffix(suffix) {
                        name = stripped;
                        break;
                    }
                }
                pass = Some(name.to_string());
            }

            if get_log_options().verbosity >= 2 {
                cprintln!("entry point: {}/{}", file.display(), ep.name().unwrap());
            }

            entry_points.push(EntryPoint {
                name: ep.name().unwrap().to_string(),
                stage: slang_stage_to_stage_flags(ep.stage()),
                push_constants_size,
                pass,
                work_group_size,
                params,
            });
        }

        Ok(Module {
            _module: module,
            _program: program,
            name: module_name,
            file_path: canonical_path,
            file_mtime,
            spirv,
            reflection,
            entry_points,
        })
    }

    /// Writes a single pass (graphics or compute pipeline) into the archive.
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
        let mut all_params = vec![];

        for &ep in entry_points {
            push_constants_size = push_constants_size.max(ep.push_constants_size);
            workgroup_size = ep.work_group_size;
            stage_flags |= ep.stage;
            all_params.extend_from_slice(&ep.params);
            shaders.push(Shader { stage: ep.stage, entry_point: ep.name.as_str().into() });
        }

        let pipeline_kind = if stage_flags.contains(vk::ShaderStageFlags::COMPUTE) {
            sharc::PipelineKind::Compute(sharc::ComputePipeline {
                push_constants_size: push_constants_size as u16,
                compute_shader: shaders[0],
                workgroup_size,
            })
        } else {
            let color_targets = {
                let offsets: Vec<_> = gs.color_targets.iter().map(|ct| archive.write(ct)).collect();
                archive.write_slice(&offsets)
            };

            let mut color_attachments = Offset::INVALID;
            let mut depth_stencil_attachment = None;
            if let Some(pass) = pass {
                let attachments: Vec<_> = pass
                    .color_attachments
                    .iter()
                    .map(|ca| sharc::ColorAttachment {
                        resource_name: ca.resource.as_ref().map(|s| s.as_str().into()).unwrap_or_default(),
                        clear_color: ca.clear_color,
                    })
                    .collect();
                color_attachments = archive.write_slice(&attachments);

                if let Some(dsa) = &pass.depth_stencil_attachment {
                    depth_stencil_attachment = Some(sharc::DepthStencilAttachment {
                        resource_name: dsa.resource.as_ref().map(|s| s.as_str().into()).unwrap_or_default(),
                        clear_depth: dsa.clear_depth,
                        clear_stencil: dsa.clear_stencil,
                    });
                }

                // Check that we have the same number of color attachments
                // as blend targets.
                // This is not a requirement, but it might indicate a mistake if the counts
                // don't match.
                if pass.color_attachments.len() != gs.color_targets.len() {
                    warn!(
                        "pipeline `{}` has {} color attachments, but {} color blend targets",
                        pipeline_name,
                        pass.color_attachments.len(),
                        gs.color_targets.len()
                    );
                }
            }

            let shaders = archive.write_slice(&shaders);
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

        let refl_params = archive.write_slice(&all_params);
        let refl_resources = archive.write_slice(&[]);
        let signature = archive.write(&sharc::reflection::Signature {
            params: refl_params,
            resources: refl_resources,
        });

        Ok(sharc::Pass {
            name: ZString64::new(pipeline_name),
            kind: pipeline_kind,
            root_params: RootParamLayout {
                byte_size: 0xABCDEF12, // FIXME: populate from reflection
                parameters: Offset::INVALID,
            },
            signature
        })
    }

    /// Writes the image resource descriptors declared in the manifest into the archive.
    fn write_image_resources(&self, archive: &mut ShaderArchiveWriter) -> Offset<[sharc::ImageResourceDesc]> {
        // TODO Figure out what we want to do with resources.
        //      I'm not sure if we should put this in shadertool.
        /*let images: Vec<_> = self
            .resources
            .iter()
            .map(|(name, desc)| {
                let size = match (desc.width, desc.height) {
                    (Some(w), Some(h)) => sharc::ImageResourceSize::Fixed { width: w, height: h },
                    // TODO: handle the case where only one dimension is specified.
                    _ => sharc::ImageResourceSize::RenderTarget,
                };
                // If the usage is not specified explicitly, assume the image will be used as a
                // render-target attachment and sampled in shaders.  This is conservative, but
                // it is difficult to infer the correct usage from reflection alone.
                let usage = desc.usage.unwrap_or_else(|| {
                    let base = if is_depth_format(desc.format) {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    } else {
                        ImageUsage::COLOR_ATTACHMENT
                    };
                    base | ImageUsage::SAMPLED | ImageUsage::STORAGE
                });
                sharc::ImageResourceDesc { name: name.as_str().into(), format: desc.format, usage, size }
            })
            .collect();*/
        archive.write_slice(&[])
    }

    /// Writes a compiled module into the archive.
    fn write_module(
        &self,
        archive: &mut ShaderArchiveWriter,
        module: &Module,
        options: &BuildOptions,
        stats: &mut Stats,
    ) -> anyhow::Result<sharc::Module> {
        // Group entry points by pass name.
        let mut pipelines: BTreeMap<&str, Vec<&EntryPoint>> = BTreeMap::new();
        for ep in module.entry_points.iter() {
            if let Some(ref pass) = ep.pass {
                pipelines.entry(pass).or_default().push(ep);
            }
        }

        let pipelines_offset = {
            let mut entries = Vec::new();
            for (&pipeline_name, entry_points) in pipelines.iter() {
                let mut state = self.default.clone();

                // Per-pass overrides are specified as `[pass.pipeline_name]` in TOML.
                let pass = if let Some(pass) = self.pass.get(pipeline_name) {
                    state.apply_overrides(&pass.raw)?;
                    Some(pass)
                } else {
                    None
                };

                // record pass name for warning about unused overrides
                stats.pass_names.insert(pipeline_name.to_string());
                entries.push(self.write_pass(archive, pipeline_name, pass, &state, entry_points, options)?);
            }
            archive.write_slice(&entries)
        };

        if !get_log_options().quiet {
            let pipeline_list = pipelines.keys().cloned().collect::<Vec<_>>().join(", ");
            cprintln!(
                "<g,bold>Compiled</> {} entry points, {} pipelines \n\t<dim>{}</>",
                module.entry_points.len(),
                pipelines.len(),
                pipeline_list
            );
        }

        stats.pass_count += pipelines.len();
        stats.entry_point_count += module.entry_points.len();

        let spirv = archive.write_slice(&module.spirv);
        let name = archive.write_str(&module.name);
        let params = archive.write_slice(&module.reflection);
        let path = archive.write_str(&module.file_path.to_string_lossy());
        let include_paths = {
            let mut include_paths = vec![];
            for p in options.include_paths.iter() {
                let Ok(canonical) = p.canonicalize() else {
                    bail!("failed to canonicalize include path: {}", p.display());
                };
                include_paths.push(archive.write_str(&*canonical.to_string_lossy()));
            }
            archive.write_slice(&include_paths)
        };

        Ok(sharc::Module {
            name,
            spirv,
            passes: pipelines_offset,
            file: FileDependency { path, mtime: module.file_mtime },
            params,
            debug_info: options.emit_debug_information,
            include_paths
        })
    }

    /// Compiles all shader source files listed in the manifest and writes one `.sharc` archive per
    /// input file.
    pub(crate) fn build(&self, file: &Path, source_text: &str, options: &BuildOptions) -> anyhow::Result<()> {
        let quiet = get_log_options().quiet;
        let verbosity = get_log_options().verbosity;

        if options.emit_cargo_deps {
            // Emit cargo dependency information.
            let absolute_file_path = file.canonicalize()?;
            println!("cargo:rerun-if-changed={}", absolute_file_path.display());
            if !self.manifest_path.as_os_str().is_empty() {
                println!("cargo:rerun-if-changed={}", self.manifest_path.display());
            }
        }

        let mut got_errors = false;
        let mut stats = Stats::default();
        let mut archive = ArchiveWriter::new();
        let mut modules = Vec::new();

        // Determine output file paths.
        let output_file = match options.output_directory {
            Some(ref dir) => dir.join(file.file_name().unwrap()).with_extension("sharc"),
            None => file.with_extension("sharc"),
        };
        let spirv_dump_path = output_file.parent().unwrap().join("spirv");

        if !quiet {
            cprintln!("<g,bold>Compiling</> {}", file.display());
        }
        if verbosity >= 2 {
            cprintln!("output file: {}", output_file.display());
            if options.emit_spirv_binaries {
                cprintln!("SPIR-V dump path: {}", spirv_dump_path.display());
            }
        }

        // Create slang compiler session.
        let compiler_session = self.create_slang_session(&options.include_paths, options)?;

        'compile: {
            match self.load_slang_module(&mut archive, &compiler_session, &file, source_text, options) {
                Ok(module) => {
                    if module.entry_points.is_empty() {
                        if verbosity >= 2 {
                            cprintln!("<cyan>note</>: `{}` has no entry points, skipping", file.display());
                        }
                        break 'compile;
                    }

                    if options.emit_spirv_binaries {
                        if !spirv_dump_path.exists() {
                            fs::create_dir(&spirv_dump_path)?;
                        }
                        let spv_out_path = spirv_dump_path.join(format!("{}.spv", module.name));
                        if !quiet {
                            cprintln!("<g,bold>Dumping</> {}", spv_out_path.display());
                        }
                        fs::write(&spv_out_path, unsafe {
                            slice::from_raw_parts(module.spirv.as_ptr() as *const u8, module.spirv.len() * 4)
                        })
                        .context(format!("dumping SPIR-V at {}", spv_out_path.display()))?;
                    }

                    let written = self.write_module(&mut archive, &module, options, &mut stats)?;
                    modules.push(written);
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
                    // Don't write the archive if there were compile errors.
                    break 'compile;
                }
            }

            if !quiet {
                cprintln!("<g,bold>Writing</> {}", output_file.display());
            }

            let images = self.write_image_resources(&mut archive);
            let manifest_path = archive.write_str(&self.canonical_manifest_path.to_string_lossy());
            let modules = archive.write_slice(&modules);
            let _ = archive.write_root(&sharc::ShaderArchiveRoot {
                manifest: FileDependency { path: manifest_path, mtime: self.mtime },
                modules,
                images,
            });
            archive.write_to_file(&output_file).context("writing output")?;
        }

        // Warn if no passes were found across all input files.
        if stats.pass_count == 0 {
            cprintln!(
                "<y,bold>warning</>: no pipelines found in the input files \
                 (possibly missing `[pass(\"...\")]` attributes?)"
            );
        }

        // Warn about manifest override entries that did not match any compiled pass.
        for (pass_name, _) in self.pass.iter() {
            if !stats.pass_names.contains(pass_name) {
                cprintln!("<y,bold>warning</>: override `{}` did not match any pass", pass_name);
            }
        }

        if got_errors {
            bail!("errors occurred during shader compilation");
        }

        Ok(())
    }
}
