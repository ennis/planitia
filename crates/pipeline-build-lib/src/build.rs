use crate::manifest::{BuildManifest, Configuration, Error, Input, PipelineType, Variant};
use crate::{BuildOptions, Tag};
use anyhow::anyhow;
use color_print::{ceprintln, cprintln};
use log::info;
use pipeline_archive::archive::{ArchiveWriter, Offset};
use pipeline_archive::gpu::ShaderStage;
use pipeline_archive::zstring::{ZString16, ZString32};
use pipeline_archive::{PIPELINE_ARCHIVE_MAGIC, PipelineEntryData};
use shader_bridge::ShaderLibrary;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::{env, fmt};

struct CompilationJob<'a> {
    input: &'a Input,
    configuration: Configuration,
}

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
            for tag in tags {
                let mut p = perm.clone();
                p.push(tag);
                out_permutations.push(p);
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
}

/// Bundles compilation errors from multiple jobs into a single error.
#[derive(Debug)]
struct CompilationErrors(Vec<anyhow::Error>);

impl fmt::Display for CompilationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Compilation failed with the following errors:")?;
        for err in &self.0 {
            writeln!(f, "{err}")?;
        }
        Ok(())
    }
}

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
}

impl BuildManifest {
    pub(crate) fn build_all(&self, options: &BuildOptions) -> anyhow::Result<()> {
        let output_path = self.resolve_path(&self.output);

        // compile and write archive
        let mut archive = ArchiveWriter::new();
        self.compile_to_archive(&mut archive, options)?;
        archive.write_to_file(&output_path)?;
        Ok(())
    }

    pub(crate) fn compile_to_archive(&self, archive: &mut ArchiveWriter, options: &BuildOptions) -> anyhow::Result<()> {
        let mut base_permutations = vec![vec![]];
        add_variant_permutations(&self.variants[..], &mut base_permutations);

        eprintln!("variants: {:?}", self.variants.len());
        eprintln!("base permutations: {}", base_permutations.len());

        // header
        let header = archive.write(pipeline_archive::PipelineArchiveData {
            magic: PIPELINE_ARCHIVE_MAGIC,
            version: 1,
            entries: Offset::INVALID,
        });

        let mut compilation_errors = CompilationErrors(Vec::new());
        let mut entries = Vec::new();

        for input in &self.inputs {
            // add input-specific variants
            let mut permutations = base_permutations.clone();
            add_variant_permutations(&input.variants[..], &mut permutations);

            // resolve configuration for each permutation and compile them
            for p in &permutations {
                let mut config = self.base.clone();
                for variant in p {
                    config.apply_overrides(&variant.overrides)?;
                }

                let variant_id = format_variant_identifier(input, p);
                if !options.quiet {
                    cprintln!("<g,bold>Compiling</> {} - <i>{}</>", input.file_path, variant_id);
                }

                match self.compile_variant(archive, input, &config, p, options) {
                    Ok(entry) => {
                        entries.push(entry);
                    }
                    Err(err) => {
                        ceprintln!("<r,bold>Error(s)</>: {:#}", err);
                        compilation_errors
                            .0
                            .push(err.context(format!("compiling variant {variant_id}")));
                        eprintln!();
                    }
                }
            }
        }

        if !compilation_errors.0.is_empty() {
            Err(anyhow!(compilation_errors))
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

    fn compile_variant(
        &self,
        archive: &mut ArchiveWriter,
        input: &Input,
        config: &Configuration,
        permutation: &[&Variant],
        _options: &BuildOptions,
    ) -> anyhow::Result<PipelineEntryData> {

        let lib = match ShaderLibrary::new(self.resolve_path(&input.file_path)) {
            Ok(lib) => lib,
            Err(err) => {
                // convert to string error as the Error type returned by ShaderLibrary is not Send+Sync
                let cause = anyhow!("{err}");
                return Err(cause.context(format!("Error loading {}", input.file_path)));
            }
        };

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
                        .map_err(|err| Error::CompilationError(err.to_string()))?;
                    push_constants_size = push_constants_size.max(entry_point_info.push_constants_size);
                    let offset = archive.write_slice(entry_point_info.spirv.as_slice());
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
                    return Err(anyhow!("cannot use both mesh and vertex shaders in the same pipeline",));
                }

                let vertex_or_mesh_shaders = if mesh_shader_offset.is_valid() {
                    // mesh shading mode (mesh + optional task shader)
                    pipeline_archive::GraphicsPipelineShaders::Mesh {
                        mesh_shader: mesh_shader_offset,
                        task_shader: task_shader_offset,
                    }
                } else {
                    // primitive shading mode (vertex shader)
                    pipeline_archive::GraphicsPipelineShaders::Primitive {
                        vertex_shader: vertex_shader_offset,
                    }
                };

                let mut color_targets = Vec::new();
                for ct in config.color_targets.iter() {
                    color_targets.push(archive.write(*ct));
                }
                let color_targets = archive.write_slice(color_targets.as_slice());
                pipeline_kind = pipeline_archive::PipelineKind::Graphics(pipeline_archive::GraphicsPipelineData {
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
                    .map_err(|err| Error::CompilationError(err.to_string()))?;
                push_constants_size = entry_point_info.push_constants_size;
                let offset = archive.write_slice(entry_point_info.spirv.as_slice());
                let compute_shader_offset = offset;
                pipeline_kind = pipeline_archive::PipelineKind::Compute(pipeline_archive::ComputePipelineData {
                    compute_shader: compute_shader_offset,
                    workgroup_size: entry_point_info.work_group_size,
                });
            }
        }

        let tags = {
            let mut tag_offsets = Vec::new();
            for variant in permutation.iter() {
                tag_offsets.push(archive.write(ZString16::new(variant.tag.value().unwrap_or(""))));
            }
            let tag_slice = archive.write_slice(tag_offsets.as_slice());
            tag_slice
        };

        Ok(pipeline_archive::PipelineEntryData {
            name: ZString32::new(&input.name),
            tags,
            push_constants_size: push_constants_size as u16,
            kind: pipeline_kind,
        })
    }
}
