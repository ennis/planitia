use crate::asset::{AssetCache, DefaultLoader, Dependencies, FileMetadata, Handle, LoadResult, Provider, VfsPath};
use gpu::{PreRasterizationShaders, ShaderEntryPoint, vk};
use log::{debug, warn};
use shader_archive::{ShaderArchive, ShaderData};
use std::sync::MutexGuard;
use std::time::SystemTime;
use std::{fs, io};
use utils::archive::Offset;

#[derive(thiserror::Error, Debug)]
pub enum PipelineCreateError {
    #[error("failed to load pipeline archive: {0}")]
    ArchiveLoadError(String),
    #[error("pipeline not found: {0}")]
    PipelineNotFound(String),
    #[error("failed to create graphics pipeline: {0}")]
    GraphicsPipelineCreationError(#[from] gpu::Error),
}

#[derive(Clone)]
pub struct ArchiveLoadOptions {
    /// Watches the archive file for changes and reloads it automatically.
    pub hot_reload: bool,
}

fn unix_mtime(last_modified: SystemTime) -> u64 {
    if last_modified > SystemTime::now() {
        warn!("last modification time is in the future: {:?}", last_modified);
    }

    match last_modified.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => {
            warn!("invalid modification time (before UNIX_EPOCH)");
            0
        }
    }
}

/// Loads a pipeline archive file or retrieves it from the asset cache.
pub fn load_pipeline_archive(path: impl AsRef<VfsPath>) -> Handle<ShaderArchive> {
    let cache = AssetCache::instance();
    cache.load(path.as_ref(), |path, metadata, provider, deps| {
        // add dependencies on the manifest and source files
        let data = provider.load(path)?;
        let a = ShaderArchive::from_bytes(&*data).unwrap();

        // hot reloading support
        #[cfg(feature = "hot_reload")]
        {
            fn should_rebuild_archive(archive: &ShaderArchive) -> bool {
                fn inner(archive: &ShaderArchive) -> io::Result<bool> {
                    let manifest_path = &archive[archive.manifest_file().path];
                    let manifest_mtime = unix_mtime(fs::metadata(manifest_path)?.modified()?);

                    if manifest_mtime > archive.manifest_file().mtime {
                        debug!(
                            "shader manifest modified: {} (last:{:?}, archive:{:?})",
                            manifest_path,
                            manifest_mtime,
                            archive.manifest_file().mtime
                        );
                        return Ok(true);
                    }

                    for source in archive.source_files() {
                        let path = &archive[source.path];
                        let source_metadata = fs::metadata(path)?;
                        if unix_mtime(source_metadata.modified()?) > source.mtime {
                            debug!(
                                "shader archive dependency modified: {} (last:{:?}, archive:{:?})",
                                path,
                                source_metadata.modified()?,
                                source.mtime
                            );
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }

                inner(archive).unwrap_or(false)
            }
            if should_rebuild_archive(&a) {
                shadertool::build_pipeline(
                    &a[a.manifest_file().path],
                    &shadertool::BuildOptions {
                        quiet: false,
                        emit_cargo_deps: false,
                        emit_debug_information: true, // TODO
                        emit_spirv_binaries: true,
                    },
                )?;
            }

            for source in a.source_files() {
                deps.add_local_file(&a[source.path]);
            }
            deps.add_local_file(&a[a.manifest_file().path]);
        }

        Ok(a)
    })
}

fn get_shader_entry_point<'a>(
    stage: gpu::ShaderStage,
    archive: &'a ShaderArchive,
    shader: &'a ShaderData,
) -> ShaderEntryPoint<'a> {
    let spirv = &archive[shader.spirv];
    ShaderEntryPoint {
        stage,
        code: spirv,
        entry_point: shader.entry_point.as_str(),
        push_constants_size: 0, // ignored by gpu anyway
        source_path: None,
        workgroup_size: [0, 0, 0], // ignored by gpu anyway
    }
}

pub(crate) fn create_graphics_pipeline_from_archive(
    archive: &ShaderArchive,
    entry: &shader_archive::GraphicsPipelineData,
) -> Result<gpu::GraphicsPipeline, PipelineCreateError> {
    let color_targets: Vec<_> = {
        let color_targets = &archive[entry.color_targets];
        color_targets
            .iter()
            .map(|&c| {
                let c = &archive[c];
                gpu::ColorTargetState {
                    format: c.format,
                    blend_equation: Some(gpu::ColorBlendEquation {
                        src_color_blend_factor: c.blend.src_color_blend_factor,
                        dst_color_blend_factor: c.blend.dst_color_blend_factor,
                        color_blend_op: c.blend.color_blend_op,
                        src_alpha_blend_factor: c.blend.src_alpha_blend_factor,
                        dst_alpha_blend_factor: c.blend.dst_alpha_blend_factor,
                        alpha_blend_op: c.blend.alpha_blend_op,
                    }),
                    color_write_mask: vk::ColorComponentFlags::RGBA,
                }
            })
            .collect()
    };

    let depth_stencil = if entry.depth_stencil.enable {
        Some(gpu::DepthStencilState {
            format: entry.depth_stencil.format,
            depth_write_enable: entry.depth_stencil.depth_write_enable,
            depth_compare_op: entry.depth_stencil.depth_compare_op,
            stencil_state: Default::default(),
        })
    } else {
        None
    };

    let rasterization = gpu::RasterizationState {
        polygon_mode: entry.rasterization.polygon_mode,
        cull_mode: entry.rasterization.cull_mode,
        ..Default::default()
    };

    let mut vertex_shader = None;
    let mut fragment_shader = None;
    let mut mesh_shader = None;
    let mut task_shader = None;
    let mut stage_flags = vk::ShaderStageFlags::empty();

    for shader in &archive[entry.shaders] {
        match shader.stage {
            vk::ShaderStageFlags::VERTEX => {
                vertex_shader = Some(get_shader_entry_point(gpu::ShaderStage::Vertex, &archive, shader));
                stage_flags |= vk::ShaderStageFlags::VERTEX;
            }
            vk::ShaderStageFlags::FRAGMENT => {
                fragment_shader = Some(get_shader_entry_point(gpu::ShaderStage::Fragment, &archive, shader));
                stage_flags |= vk::ShaderStageFlags::FRAGMENT;
            }
            vk::ShaderStageFlags::MESH_EXT => {
                mesh_shader = Some(get_shader_entry_point(gpu::ShaderStage::Mesh, &archive, shader));
                stage_flags |= vk::ShaderStageFlags::MESH_EXT;
            }
            vk::ShaderStageFlags::TASK_EXT => {
                task_shader = Some(get_shader_entry_point(gpu::ShaderStage::Task, &archive, shader));
                stage_flags |= vk::ShaderStageFlags::TASK_EXT;
            }
            _ => {
                panic!("unsupported shader stage in graphics pipeline: {:?}", shader.stage);
            }
        }
    }

    let pre_rasterization_shaders = if stage_flags.contains(vk::ShaderStageFlags::MESH_EXT) {
        PreRasterizationShaders::MeshShading {
            task: task_shader,
            mesh: mesh_shader.expect("mesh shader missing in graphics pipeline"),
        }
    } else {
        PreRasterizationShaders::PrimitiveShading {
            vertex: vertex_shader.expect("vertex shader missing in graphics pipeline"),
        }
    };

    let gpci = gpu::GraphicsPipelineCreateInfo {
        set_layouts: &[],
        push_constants_size: entry.push_constants_size as usize,
        vertex_input: Default::default(),
        pre_rasterization_shaders,
        rasterization,
        depth_stencil,
        fragment: gpu::FragmentState {
            shader: fragment_shader.unwrap(),
            multisample: Default::default(),
            color_targets: &color_targets[..],
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        },
    };
    let pipeline = gpu::GraphicsPipeline::new(gpci)?;
    Ok(pipeline)
}

pub(crate) fn create_compute_pipeline_from_archive(
    archive: &ShaderArchive,
    entry: &shader_archive::ComputePipelineData,
) -> Result<gpu::ComputePipeline, PipelineCreateError> {
    let shader = get_shader_entry_point(gpu::ShaderStage::Compute, &archive, &entry.compute_shader);
    let cpci = gpu::ComputePipelineCreateInfo {
        set_layouts: &[],
        push_constants_size: entry.push_constants_size as usize,
        shader,
    };
    let pipeline = gpu::ComputePipeline::new(cpci)?;
    Ok(pipeline)
}

pub fn load_graphics_pipeline(
    path: &VfsPath,
    _metadata: &FileMetadata,
    _provider: &dyn Provider,
    _dependencies: &mut Dependencies,
) -> LoadResult<gpu::GraphicsPipeline> {
    let archive_file = path.path_without_fragment();
    let name = path.fragment().expect("pipeline name missing in path");
    let archive_handle = load_pipeline_archive(archive_file);

    let archive = archive_handle.read()?;
    let entry = archive
        .find_graphics_pipeline(name)
        .ok_or_else(|| PipelineCreateError::PipelineNotFound(name.to_string()))?;

    Ok(create_graphics_pipeline_from_archive(&*archive, entry)?)
}

pub fn load_compute_pipeline(
    path: &VfsPath,
    _metadata: &FileMetadata,
    _provider: &dyn Provider,
    _dependencies: &mut Dependencies,
) -> LoadResult<gpu::ComputePipeline> {
    let archive_file = path.path_without_fragment();
    let name = path.fragment().expect("pipeline name missing in path");
    let archive_handle = load_pipeline_archive(archive_file);

    let archive = archive_handle.read()?;
    let entry = archive
        .find_compute_pipeline(name)
        .ok_or_else(|| PipelineCreateError::PipelineNotFound(name.to_string()))?;
    Ok(create_compute_pipeline_from_archive(&*archive, entry)?)
}

/// Loads a graphics pipeline object from the specified archive file and pipeline name.
pub fn get_graphics_pipeline(path: impl AsRef<VfsPath>) -> Handle<gpu::GraphicsPipeline> {
    let path = path.as_ref();
    AssetCache::instance().load(path, load_graphics_pipeline)
}

pub fn get_compute_pipeline(path: impl AsRef<VfsPath>) -> Handle<gpu::ComputePipeline> {
    let path = path.as_ref();
    AssetCache::instance().load(path, load_compute_pipeline)
}

impl DefaultLoader for gpu::GraphicsPipeline {
    fn load(
        path: &VfsPath,
        metadata: &FileMetadata,
        provider: &dyn Provider,
        dependencies: &mut Dependencies,
    ) -> LoadResult<Self> {
        load_graphics_pipeline(path, metadata, provider, dependencies)
    }
}

impl DefaultLoader for gpu::ComputePipeline {
    fn load(
        path: &VfsPath,
        metadata: &FileMetadata,
        provider: &dyn Provider,
        dependencies: &mut Dependencies,
    ) -> LoadResult<Self> {
        load_compute_pipeline(path, metadata, provider, dependencies)
    }
}
