use crate::BuildOptions;
use crate::manifest::{BuildManifest, Configuration, Error, Input, PipelineType, Variant};
use anyhow::anyhow;
use color_print::{ceprintln, cprintln};
use log::warn;
use shader_archive::archive::{ArchiveWriter, Offset};
use shader_archive::gpu::ShaderStage;
use shader_archive::zstring::{ZString32, ZString64};
use shader_archive::{FileDependency, PIPELINE_ARCHIVE_MAGIC, PipelineEntryData, ShaderData};
use shader_bridge::{ShaderLibrary, ShaderLibraryLoadOptions};
use std::fmt::{Debug, Display, Pointer};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::{env, fmt};

fn make_file_dependency(path: &Path, archive: &mut ArchiveWriter) -> anyhow::Result<FileDependency> {
    let canonical_path = path.canonicalize()?;
    let modified_time = std::fs::metadata(&canonical_path)?.modified()?;
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

/*
fn add_variant_permutations<'a>(variants: &'a [Variant], out_permutations: &mut Vec<Vec<&'a Variant>>) {
    let mut sets = BTreeMap::new();
    for variant in variants {
        if !sets.contains_key(variant.tag.set()) {
            sets.insert(variant.tag.set().to_string(), vec![variant]);
        } else {
            let entry = sets.get_mut(variant.tag.set()).unwrap();
            entry.push(variant);
        }
    }

    eprintln!("variant sets: {:?}", sets);

    let mut tmp = vec![];
    for (_tag_set, tags) in &sets {
        tmp.clear();
        tmp.extend(out_permutations.drain(..));
        out_permutations.clear();
        for perm in tmp.drain(..) {
            if tags.len() == 1 && tags[0].tag.value().is_none() {
                out_permutations.push(perm.clone());
                {
                    let mut p = perm;
                    p.push(tags[0]);
                    out_permutations.push(p);
                }
            } else {
                for tag in tags {
                    let mut p = perm.clone();
                    p.push(tag);
                    out_permutations.push(p);
                }
            }
        }
    }
}

fn print_permutation(permutation: &[&Variant]) {
    let mut parts = vec![];
    for v in permutation {
        if let Some(value) = v.tag.value() {
            parts.push(format!("{}={}", v.tag.set(), value));
        } else {
            parts.push(format!("{}", v.tag.set()));
        }
    }
    info!("  variant: {}", parts.join(", "));
}*/

#[derive(thiserror::Error, Debug)]
enum BuildError {
    #[error("{0}")]
    CompilationError(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

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
}

/*
/// Formats a unique identifier for a given input and variant permutation.
///
/// The format is: `<file_path>:<name>?<tag1>=<value1>&<tag2>&<tag3>=<value3>...`
fn format_variant_identifier(input: &Input, permutation: &[&Variant]) -> String {
    let mut name = format!("{}:{}", input.file_path, input.name);
    let mut first = true;
    for v in permutation {
        if first {
            name.push('?');
            first = false;
        } else {
            name.push('&');
        }
        if let Some(value) = v.tag.value() {
            name.push_str(&format!("{}={}", v.tag.set(), value));
        } else {
            name.push_str(v.tag.set());
        }
    }

    name
}*/

impl BuildManifest {
    pub(crate) fn build_all(&self, options: &BuildOptions) -> anyhow::Result<()> {
        let output_path = self.resolve_path(&self.output);

        // compile and write archive
        let mut archive = ArchiveWriter::new();
        self.compile_to_archive(&mut archive, options)?;
        archive.write_to_file(&output_path)?;
        Ok(())
    }

    fn compile_to_archive(&self, archive: &mut ArchiveWriter, options: &BuildOptions) -> anyhow::Result<()> {
        // write the header first since it contains the signature
        // TODO: the signature should be a feature of ArchiveWriter
        let header = archive.write(shader_archive::PipelineArchiveData {
            magic: PIPELINE_ARCHIVE_MAGIC,
            version: 1,
            manifest_file: FileDependency::default(),
            entries: Offset::INVALID,
        });

        archive[header].manifest_file = make_file_dependency(&self.manifest_path, archive)?;

        if options.emit_cargo_deps {
            println!("cargo:rerun-if-changed={}", self.manifest_path.display());
        }

        let mut build_errors = BuildErrors(Vec::new());
        let mut entries = Vec::new();

        for input in &self.inputs {
            let mut config = self.base_configuration.clone();
            if let Some(ref overrides) = input.overrides {
                config.apply_overrides(overrides)?;
            }

            if !options.quiet {
                cprintln!("<g,bold>Compiling</> {} (<i>{}</>)", input.file_path, input.name);
            }

            match self.compile_input(archive, input, &config, options) {
                Ok(entry) => {
                    entries.push(entry);
                }
                Err(err) => {
                    ceprintln!("<r,bold>Error(s)</>: {:#}", err);
                    build_errors.0.push(err);
                    eprintln!();
                }
            }
        }

        if !build_errors.0.is_empty() {
            Err(anyhow!(build_errors))
        } else {
            let entries_offset = archive.write_slice(entries.as_slice());
            archive[header].entries = entries_offset;
            Ok(())
        }
    }

    /// Resolves relative paths relative to the manifest directory, or
    /// returns the given path if absolute
    fn resolve_path(&self, path: &str) -> PathBuf {
        let manifest_dir = if let Some(parent) = self.manifest_path.parent() {
            parent.to_path_buf()
        } else {
            env::current_dir().unwrap()
        };

        let input_path = Path::new(path);
        if input_path.is_absolute() {
            input_path.to_path_buf()
        } else {
            manifest_dir.join(input_path)
        }
    }

    fn compile_input(
        &self,
        archive: &mut ArchiveWriter,
        input: &Input,
        config: &Configuration,
        options: &BuildOptions,
    ) -> Result<PipelineEntryData, BuildError> {
        let resolved_shader_path = self.resolve_path(&input.file_path);

        let lib = {
            let r = ShaderLibrary::with_options(
                &resolved_shader_path,
                ShaderLibraryLoadOptions {
                    module_search_paths: self.module_search_paths.clone(),
                    macro_definitions: vec![],
                    shader_profile: None,
                },
            );
            match r {
                Ok(lib) => lib,
                Err(err) => {
                    // convert to string error as the Error type returned by ShaderLibrary is not Send+Sync
                    return Err(BuildError::CompilationError(err.to_string()));
                }
            }
        };

        if options.emit_cargo_deps {
            println!("cargo:rerun-if-changed={}", resolved_shader_path.display());
        }

        let mut push_constants_size = 0;
        let pipeline_kind;

        match self.type_ {
            PipelineType::Graphics => {
                let mut vertex_shader_offset = Offset::INVALID;
                let mut fragment_shader_offset = Offset::INVALID;
                let mut mesh_shader_offset = Offset::INVALID;
                let mut task_shader_offset = Offset::INVALID;
                for (ty, entry_point_name) in [
                    (ShaderStage::Vertex, input.vertex_entry_point.as_str()),
                    (ShaderStage::Fragment, input.fragment_entry_point.as_str()),
                    (ShaderStage::Mesh, input.mesh_entry_point.as_str()),
                    (ShaderStage::Task, input.task_entry_point.as_str()),
                ] {
                    if entry_point_name.is_empty() {
                        continue;
                    }

                    let entry_point_info = lib
                        .get_compiled_entry_point(entry_point_name)
                        .map_err(|err| BuildError::CompilationError(err.to_string()))?;
                    push_constants_size = push_constants_size.max(entry_point_info.push_constants_size);
                    let offset = {
                        let spirv = archive.write_slice(entry_point_info.spirv.as_slice());
                        archive.write(ShaderData {
                            entry_point: ZString32::new(entry_point_name),
                            spirv,
                        })
                    };
                    match ty {
                        ShaderStage::Vertex => vertex_shader_offset = offset,
                        ShaderStage::Fragment => fragment_shader_offset = offset,
                        ShaderStage::Mesh => mesh_shader_offset = offset,
                        ShaderStage::Task => task_shader_offset = offset,
                        _ => {}
                    }
                }

                // check that not both mesh and vertex shaders are used
                if mesh_shader_offset.is_valid() && vertex_shader_offset.is_valid() {
                    return Err(anyhow!("cannot use both mesh and vertex shaders in the same pipeline").into());
                }

                let vertex_or_mesh_shaders = if mesh_shader_offset.is_valid() {
                    // mesh shading mode (mesh + optional task shader)
                    shader_archive::GraphicsPipelineShaders::Mesh {
                        mesh: mesh_shader_offset,
                        task: task_shader_offset,
                    }
                } else {
                    // primitive shading mode (vertex shader)
                    shader_archive::GraphicsPipelineShaders::Primitive {
                        vertex: vertex_shader_offset,
                    }
                };

                let mut color_targets = Vec::new();
                for ct in config.color_targets.iter() {
                    color_targets.push(archive.write(*ct));
                }
                let color_targets = archive.write_slice(color_targets.as_slice());
                pipeline_kind = shader_archive::PipelineKind::Graphics(shader_archive::GraphicsPipelineData {
                    push_constants_size: push_constants_size as u16,
                    vertex_or_mesh_shaders,
                    rasterization: config.rasterization_state,
                    depth_stencil: config.depth_stencil_state,
                    color_targets,
                    fragment_shader: fragment_shader_offset,
                });
            }
            PipelineType::Compute => {
                let entry_point_name = &input.compute_entry_point;
                let entry_point_info = lib
                    .get_compiled_entry_point(entry_point_name)
                    .map_err(|err| BuildError::CompilationError(err.to_string()))?;
                push_constants_size = entry_point_info.push_constants_size;
                let compute_shader_offset = {
                    let spirv = archive.write_slice(entry_point_info.spirv.as_slice());
                    archive.write(ShaderData {
                        entry_point: ZString32::new(entry_point_name),
                        spirv,
                    })
                };
                pipeline_kind = shader_archive::PipelineKind::Compute(shader_archive::ComputePipelineData {
                    push_constants_size: push_constants_size as u16,
                    compute_shader: compute_shader_offset,
                    workgroup_size: entry_point_info.work_group_size,
                });
            }
        }

        // TODO find a way to get paths to module dependencies as well
        let source_file = make_file_dependency(&resolved_shader_path, archive)?;
        let sources = archive.write_slice(&[source_file]);

        Ok(PipelineEntryData {
            name: ZString64::new(&input.name),
            kind: pipeline_kind,
            sources,
        })
    }
}
