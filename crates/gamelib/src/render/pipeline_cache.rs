use crate::asset::{AssetCache, DefaultLoader, Dependencies, FileMetadata, Handle, LoadResult, Provider, VfsPath};
use gpu::{PreRasterizationShaders, ShaderEntryPoint, vk};
use log::{debug, warn};
use sharc::{ShaderArchive, Shader};
use std::sync::MutexGuard;
use std::time::SystemTime;
use std::{fs, io};
use std::ops::Deref;
use utils::archive::Offset;
use crate::render::load_shader_archive;
use crate::render::reflection::GraphicsPipelineReflection;

#[derive(thiserror::Error, Debug)]
pub enum PipelineCreateError {
    #[error("failed to load pipeline archive: {0}")]
    ArchiveLoadError(String),
    #[error("pipeline not found: {0}")]
    PipelineNotFound(String),
    #[error("failed to create graphics pipeline: {0}")]
    GraphicsPipelineCreationError(#[from] gpu::Error),
}

fn get_shader_entry_point<'a>(
    stage: gpu::ShaderStage,
    archive: &'a ShaderArchive,
    shader: &'a Shader,
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

fn create_graphics_pipeline_from_archive(
    archive: &ShaderArchive,
    entry: &sharc::GraphicsPipeline,
) -> Result<gpu::GraphicsPipeline, PipelineCreateError> {
    let color_targets: Vec<_> = {
        let color_targets = &archive[entry.color_targets];
        color_targets
            .iter()
            .map(|&c| {
                let c = &archive[c];
                gpu::ColorTargetState {
                    format: c.format,
                    blend_equation: c.blend.map(|blend| gpu::ColorBlendEquation {
                        src_color_blend_factor: blend.src_color_blend_factor,
                        dst_color_blend_factor: blend.dst_color_blend_factor,
                        color_blend_op: blend.color_blend_op,
                        src_alpha_blend_factor: blend.src_alpha_blend_factor,
                        dst_alpha_blend_factor: blend.dst_alpha_blend_factor,
                        alpha_blend_op: blend.alpha_blend_op,
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

fn create_compute_pipeline_from_archive(
    archive: &ShaderArchive,
    entry: &sharc::ComputePipeline,
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

fn load_graphics_pipeline(
    path: &VfsPath,
    _metadata: &FileMetadata,
    _provider: &dyn Provider,
    _dependencies: &mut Dependencies,
) -> LoadResult<gpu::GraphicsPipeline> {
    let archive_file = path.path_without_fragment();
    let name = path.fragment().expect("pipeline name missing in path");
    let archive_handle = load_shader_archive(archive_file);

    debug!(
        "loading pipeline `{}` (graphics) from `{}`",
        name,
        archive_file.as_str()
    );

    let archive = archive_handle.read()?;
    let entry = archive
        .find_graphics_pipeline(name)
        .ok_or_else(|| PipelineCreateError::PipelineNotFound(name.to_string()))?;

    Ok(create_graphics_pipeline_from_archive(&*archive, entry)?)
}

fn load_compute_pipeline(
    path: &VfsPath,
    _metadata: &FileMetadata,
    _provider: &dyn Provider,
    _dependencies: &mut Dependencies,
) -> LoadResult<gpu::ComputePipeline> {
    let archive_file = path.path_without_fragment();
    let name = path.fragment().expect("pipeline name missing in path");
    let archive_handle = load_shader_archive(archive_file);

    debug!(
        "loading pipeline `{}` (compute) from `{}`",
        name,
        archive_file.as_str()
    );

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


//--------------------------------------------------------------------------------------------------


/// Represents a graphics pipeline with reflection information.
pub struct GraphicsPipeline {
    pub compiled: gpu::GraphicsPipeline,
    reflection_alloc: bumpalo::Bump,
    reflection: *const GraphicsPipelineReflection<'static>,
}

// SAFETY: GraphicsPipeline is normally non-Sync because of bumpalo::Bump and the raw pointer,
//         but we don't touch the bump arena after creation, so everything is effectively immutable
//         and can be safely shared between threads.
unsafe impl Send for GraphicsPipeline {}
unsafe impl Sync for GraphicsPipeline {}

impl GraphicsPipeline {
    pub fn reflection(&self) -> &GraphicsPipelineReflection<'_> {
        // SAFETY: the reflection data is allocated in a bump arena owned by this struct,
        //         so it will remain valid as long as the struct is alive.
        unsafe { &*self.reflection }
    }
}


impl DefaultLoader for GraphicsPipeline {
    fn load(
        path: &VfsPath,
        metadata: &FileMetadata,
        provider: &dyn Provider,
        dependencies: &mut Dependencies,
    ) -> LoadResult<Self>
    {
        let archive_file = path.path_without_fragment();
        let name = path.fragment().expect("pipeline name missing in path");
        let archive_handle = load_shader_archive(archive_file);
        debug!(
            "loading pipeline `{}` (graphics) from `{}`",
            name,
            archive_file.as_str()
        );

        let archive = archive_handle.read()?;
        let pass = archive
            .find_graphics_pipeline(name)
            .ok_or_else(|| PipelineCreateError::PipelineNotFound(name.to_string()))?;
        let pipeline = create_graphics_pipeline_from_archive(&*archive, pass)?;

        // --- extract reflection data into memory ---
        let mut alloc = bumpalo::Bump::new();

        // color output formats²
        let color_formats;
        {
            let mut f = Vec::with_capacity(archive[pass.color_targets].len());
            for color_target in &archive[pass.color_targets] {
                let c = &archive[*color_target];
                f.push(c.format);
            }
            color_formats = alloc.alloc_slice_copy(&f);
        }

        let reflection = alloc.alloc(GraphicsPipelineReflection {
            color_formats,
        }) as *const _ as *const GraphicsPipelineReflection<'static>;

        Ok(GraphicsPipeline {
            compiled: pipeline,
            reflection_alloc: alloc,
            reflection,
        })
    }
}