//! sharc archive output
//!
//! TODO some code is duplicated from build.rs, refactor
//! TODO with the new hot-reload system based on reloading rust DLLs, shader archives may become
//!      less useful, so consider phasing them out

use crate::build::{compile_slang_module, create_slang_session};
use crate::{get_log_options, Arena, BuildManifest, BuildOptions, EntryPoint, Error, GraphicsState, Module, Pass};
use anyhow::{Context, bail, anyhow};
use color_print::{ceprintln, cprintln};
use log::warn;
use sharc::archive::{ArchiveWriter, Offset};
use sharc::gpu_types::vk;
use sharc::zstring::ZString64;
use sharc::{FileDependency, RootParamLayout, Shader};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

type ShaderArchiveWriter = ArchiveWriter<sharc::ShaderArchiveRoot>;

/// Accumulated build statistics.
#[derive(Default)]
struct Stats {
    entry_point_count: usize,
    pass_count: usize,
    /// Fully-qualified pass names (`module.pass`) seen during the build, used to detect
    /// manifest override entries that do not match any compiled pass.
    pass_names: BTreeSet<String>,
}

/// Writes a single pass (graphics or compute pipeline) into the archive.
fn write_pass(
    archive: &mut ShaderArchiveWriter,
    pipeline_name: &str,
    pass: Option<&Pass>,
    gs: &GraphicsState,
    entry_points: &[&EntryPoint],
    _manifest: &BuildManifest,
    _options: &BuildOptions,
) -> anyhow::Result<sharc::Pass> {
    let mut push_constants_size = 0;
    let mut workgroup_size = [1u32; 3];
    let mut shaders = vec![];
    let mut stage_flags = vk::ShaderStageFlags::default();
    //let mut all_params = vec![];

    for &ep in entry_points {
        push_constants_size = push_constants_size.max(ep.push_constants_size);
        workgroup_size = ep.workgroup_size;
        stage_flags |= ep.stage;
        //all_params.extend_from_slice(&ep.params);
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

    //let refl_params = archive.write_slice(&all_params);
    //let refl_resources = archive.write_slice(&[]);
    let signature =
        archive.write(&sharc::reflection::Signature { params: Offset::INVALID, resources: Offset::INVALID });

    Ok(sharc::Pass {
        name: ZString64::new(pipeline_name),
        kind: pipeline_kind,
        root_params: RootParamLayout {
            byte_size: 0xABCDEF12, // FIXME: populate from reflection
            parameters: Offset::INVALID,
        },
        signature,
    })
}

/// Writes a compiled module into the archive.
fn write_module(
    archive: &mut ShaderArchiveWriter,
    module: &Module,
    manifest: &BuildManifest,
    options: &BuildOptions,
    stats: &mut Stats,
) -> anyhow::Result<sharc::Module> {
    // Group entry points by pass name.
    let mut pipelines: BTreeMap<&str, Vec<&EntryPoint>> = BTreeMap::new();
    for ep in module.entry_points.iter() {
        if let Some(ref pass) = ep.pipeline_name {
            pipelines.entry(pass).or_default().push(ep);
        }
    }

    let pipelines_offset = {
        let mut entries = Vec::new();
        for (&pipeline_name, entry_points) in pipelines.iter() {
            let mut state = manifest.default.clone();

            // Per-pass overrides are specified as `[pass.pipeline_name]` in TOML.
            let pass = if let Some(pass) = manifest.pass.get(pipeline_name) {
                state.apply_overrides(&pass.raw)?;
                Some(pass)
            } else {
                None
            };

            // record pass name for warning about unused overrides
            stats.pass_names.insert(pipeline_name.to_string());
            entries.push(write_pass(archive, pipeline_name, pass, &state, entry_points, manifest, options)?);
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
    //let params = archive.write_slice(&module.reflection);
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
        params: Offset::INVALID,
        debug_info: options.emit_debug_information,
        include_paths,
    })
}

/// Compiles the given slang shader module and writes the resulting archive to disk.
pub(crate) fn build_and_write_archive(
    file: &Path,
    source_text: &str,
    manifest: &BuildManifest,
    options: &BuildOptions,
) -> Result<(), Error> {
    let arena = Arena::new();

    let quiet = get_log_options().quiet;
    let verbosity = get_log_options().verbosity;

    if options.emit_cargo_deps {
        // Emit cargo dependency information.
        let absolute_file_path = file.canonicalize()?;
        println!("cargo:rerun-if-changed={}", absolute_file_path.display());
        if !manifest.manifest_path.as_os_str().is_empty() {
            println!("cargo:rerun-if-changed={}", manifest.manifest_path.display());
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

    if !quiet {
        cprintln!("<g,bold>Compiling</> {}", file.display());
    }
    if verbosity >= 2 {
        cprintln!("output file: {}", output_file.display());
    }

    // Create slang compiler session.
    let compiler_session = create_slang_session(&options.include_paths, manifest, options)?;

    'compile: {
        match compile_slang_module(&arena, &compiler_session, &file, source_text, manifest, options) {
            Ok(module) => {
                if module.entry_points.is_empty() {
                    if verbosity >= 2 {
                        cprintln!("<cyan>note</>: `{}` has no entry points, skipping", file.display());
                    }
                    break 'compile;
                }

                if options.emit_reflection {
                    //let reflection = module.generate_reflection();
                    //eprintln!("{reflection}");
                }

                let written = write_module(&mut archive, &module, manifest, options, &mut stats)?;
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

        let manifest_path = archive.write_str(&manifest.canonical_manifest_path.to_string_lossy());
        let modules = archive.write_slice(&modules);
        let _ = archive.write_root(&sharc::ShaderArchiveRoot {
            manifest: FileDependency { path: manifest_path, mtime: manifest.mtime },
            modules,
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
    for (pass_name, _) in manifest.pass.iter() {
        if !stats.pass_names.contains(pass_name) {
            cprintln!("<y,bold>warning</>: override `{}` did not match any pass", pass_name);
        }
    }

    if got_errors {
        return Err(anyhow!("errors occurred during shader compilation").into());
    }

    Ok(())
}
