use crate::asset::{AssetCache, Dependencies, Handle, VfsPath};
use gpu::{PreRasterizationShaders, ShaderEntryPoint};
use pipeline_archive::{GraphicsPipelineShaders, PipelineArchive, ShaderData};
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

/// Loads a pipeline archive file or retrieves it from the asset cache.
pub fn load_pipeline_archive(path: impl AsRef<VfsPath>) -> Handle<PipelineArchive> {
    let cache = AssetCache::instance();
    cache.load_and_insert(path.as_ref(), |data, _| PipelineArchive::from_bytes(data).unwrap())
}

fn get_shader_entry_point(
    stage: gpu::ShaderStage,
    archive: &PipelineArchive,
    shader: Offset<ShaderData>,
) -> ShaderEntryPoint<'_> {
    let shader = &archive[shader];
    let spirv = &archive[shader.spirv];
    ShaderEntryPoint {
        stage,
        code: spirv,
        entry_point: shader.entry_point.as_str(),
        push_constants_size: 0,
        source_path: None,
        workgroup_size: [0, 0, 0],
    }
}

fn create_graphics_pipeline_from_archive(
    archive: &PipelineArchive,
    name: &str,
) -> Result<gpu::GraphicsPipeline, PipelineCreateError> {
    let entry = archive
        .find_graphics_pipeline(name)
        .ok_or_else(|| PipelineCreateError::PipelineNotFound(name.to_string()))?;
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
                    color_write_mask: Default::default(),
                }
            })
            .collect()
    };

    let depth_stencil = if entry.depth_stencil.depth_test_enable {
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
        ..Default::default()
    };

    let gpci = gpu::GraphicsPipelineCreateInfo {
        set_layouts: &[],
        push_constants_size: entry.push_constants_size as usize,
        vertex_input: Default::default(),
        pre_rasterization_shaders: match entry.vertex_or_mesh_shaders {
            GraphicsPipelineShaders::Primitive { vertex } => PreRasterizationShaders::PrimitiveShading {
                vertex: get_shader_entry_point(gpu::ShaderStage::Vertex, &archive, vertex),
            },
            GraphicsPipelineShaders::Mesh { task, mesh } => PreRasterizationShaders::MeshShading {
                task: if task.is_valid() {
                    Some(get_shader_entry_point(gpu::ShaderStage::Task, &archive, task))
                } else {
                    None
                },
                mesh: get_shader_entry_point(gpu::ShaderStage::Mesh, &archive, mesh),
            },
        },
        rasterization,
        depth_stencil,
        fragment: gpu::FragmentState {
            shader: get_shader_entry_point(gpu::ShaderStage::Fragment, &archive, entry.fragment_shader),
            multisample: Default::default(),
            color_targets: &color_targets[..],
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        },
    };
    let pipeline = gpu::GraphicsPipeline::new(gpci)?;
    Ok(pipeline)
}

fn create_compute_pipeline_from_archive(
    archive: &PipelineArchive,
    name: &str,
) -> Result<gpu::ComputePipeline, PipelineCreateError> {
    let entry = archive
        .find_compute_pipeline(name)
        .ok_or_else(|| PipelineCreateError::PipelineNotFound(name.to_string()))?;
    let shader = get_shader_entry_point(gpu::ShaderStage::Compute, &archive, entry.compute_shader);
    let cpci = gpu::ComputePipelineCreateInfo {
        set_layouts: &[],
        push_constants_size: entry.push_constants_size as usize,
        shader,
    };
    let pipeline = gpu::ComputePipeline::new(cpci)?;
    Ok(pipeline)
}

/// Loads a graphics pipeline object from the specified archive file and pipeline name.
pub fn get_graphics_pipeline(path: impl AsRef<VfsPath>) -> Handle<gpu::GraphicsPipeline> {
    fn load(path: &VfsPath, dependencies: &mut Dependencies) -> gpu::GraphicsPipeline {
        let archive_file = path.path_without_fragment();
        let name = path.fragment().expect("pipeline name missing in path");
        let archive = load_pipeline_archive(archive_file);
        dependencies.add(&archive);
        // TODO: handle errors
        create_graphics_pipeline_from_archive(&archive, name).unwrap()
    }

    let path = path.as_ref();
    AssetCache::instance().insert_with_dependencies(path, load)
}
