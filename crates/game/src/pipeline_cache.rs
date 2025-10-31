use crate::asset::{AssetCache, Dependencies, Handle, VfsPath, VfsPathBuf};
use crate::context::get_gpu_device;
use gpu::{PreRasterizationShaders, ShaderEntryPoint, vk};
use pipeline_archive::{GraphicsPipelineShaders, PipelineArchive, PipelineArchiveData, PipelineKind, ShaderData};
use std::collections::{BTreeMap, HashMap};
use threadbound::ThreadBound;
use utils::archive::Offset;

#[derive(thiserror::Error, Debug)]
pub enum PipelineCacheError {
    #[error("failed to load pipeline archive: {0}")]
    ArchiveLoadError(String),
    #[error("pipeline not found: {0}")]
    PipelineNotFound(String),
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
) -> ShaderEntryPoint {
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

pub struct PipelineCache {
    // VfsPath -> Pipeline
    graphics: HashMap<VfsPathBuf, gpu::GraphicsPipeline>,
    compute: HashMap<VfsPathBuf, gpu::ComputePipeline>,
}

impl PipelineCache {
    pub fn get_graphics_pipeline(&self, path: impl AsRef<VfsPath>) -> Handle<gpu::GraphicsPipeline> {

        fn load(path: &VfsPath, dependencies: &mut Dependencies) -> gpu::GraphicsPipeline {

            let archive_file = path.path_without_fragment();
            let name = path.fragment().expect("pipeline name missing in path");
            let archive = load_pipeline_archive(archive_file);
            dependencies.add(&archive);

            let entry = archive.find_by_name(name).unwrap();
            let device = get_gpu_device();
            let graphics_info = entry.kind.as_graphics().unwrap();

            let color_targets: Vec<_> = {
                let color_targets = &archive[graphics_info.color_targets];
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

            let gpci = gpu::GraphicsPipelineCreateInfo {
                set_layouts: &[],
                push_constants_size: entry.push_constants_size as usize,
                vertex_input: Default::default(),
                pre_rasterization_shaders: match graphics_info.vertex_or_mesh_shaders {
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
                rasterization: Default::default(),
                depth_stencil: None,
                fragment: gpu::FragmentState {
                    shader: get_shader_entry_point(gpu::ShaderStage::Fragment, &archive, graphics_info.fragment_shader),
                    multisample: Default::default(),
                    color_targets: &color_targets[..],
                    blend_constants: [0.0, 0.0, 0.0, 0.0],
                },
            };
            let pipeline = device.create_graphics_pipeline(gpci).unwrap();
            pipeline
        }

        let path = path.as_ref();
        AssetCache::instance().insert_with_dependencies(path, load)
    }
}

/*

//----------------------------------------------------------------------------------------

let pipeline = pipeline_cache.get_graphics_pipeline("archive.parc:my_graphics_pipeline")?;
// draw something

//----------------------------------------------------------------------------------------
// Alternative:

struct App {
    ...
    pipeline: GraphicsPipeline,
    ...
}

impl App {
    fn new() {
        let archive = PipelineArchive::load("archive.parc").unwrap();
        let pipeline = archive.get_graphics_pipeline(gpu::device(), "my_graphics_pipeline").unwrap();
        Ok(Self {
            ...
            pipeline: pipeline.clone(),
            ...
        })
    }

    fn render(&mut self) {
        ...
        // draw something
        ...
    }
}

//----------------------------------------------------------------------------------------

The first approach needs only one line of code at use site.
The second approach is more explicit but has the additional boilerplate of loading the file
manually, adding a new state field, modifying the constructor, etc. And this will need to be done
for each pipeline. If the pipeline is used in multiple places, it will also need to be passed around
to avoid duplication.


What are the benefits of a VFS?
- abstract the location of resources: on the file system, in an archive, stored statically in the executable.
    - if this was not abstracted, may need to duplicate constructors for each resource location type.
    - e.g. `PipelineCache::load_from_file`, `PipelineCache::load_from_data`, ....
- support for overlays and overrides


 */
