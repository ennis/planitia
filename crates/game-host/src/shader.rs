use crate::project::ShaderOptions;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use shadertool::PipelineType;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Serialize, Deserialize)]
pub struct GraphicsPipelineConfig {
    pub depth_test: bool,
    pub depth_write: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GraphicsPipeline {
    pub vertex_shader_entry: String,
    pub fragment_shader_entry: String,
    pub workgroup_size: [u32; 3],
    pub rasterization: gpu::RasterizationState,
    pub depth_stencil: gpu::DepthStencilState,
    pub color_targets: Vec<gpu::ColorTargetState>,
    #[serde(skip)]
    pub stale: bool,
    #[serde(skip)]
    pub pipeline: Option<gpu::GraphicsPipeline>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ComputePipeline {
    pub entry_point: String,
    pub workgroup_size: [u32; 3],
    #[serde(skip)]
    pub stale: bool,
    #[serde(skip)]
    pub pipeline: Option<gpu::ComputePipeline>,
}

#[derive(Serialize, Deserialize)]
pub enum Pipeline {
    Primitive(GraphicsPipeline),
    Mesh(GraphicsPipeline),
    Compute(ComputePipeline),
}

#[derive(Default)]
pub enum CompilationState {
    #[default]
    NotCompiled,
    Compiled {
        spirv: Vec<u32>,
        reflect: spirv_reflect::Module,
    },
    CompiledWithErrors,
}

#[derive(Serialize, Deserialize)]
pub struct EntryPoint {
    pub name: String,
    #[serde(skip)]
    pub stale: bool,
}

#[derive(Serialize, Deserialize)]
pub struct Shader {
    pub path: PathBuf,
    pub options: ShaderOptions,
    #[serde(skip)]
    pub compilation_state: CompilationState,
    pub entry_points: BTreeMap<String, EntryPoint>,
    pub pipelines: BTreeMap<String, Pipeline>,
}

impl Shader {
    pub fn new(path: PathBuf, options: ShaderOptions) -> Self {
        Self {
            path,
            options,
            compilation_state: CompilationState::NotCompiled,
            entry_points: BTreeMap::new(),
            pipelines: BTreeMap::new(),
        }
    }

    pub fn reflection(&self) -> Result<&spirv_reflect::Module> {
        match self.compilation_state {
            CompilationState::Compiled { ref reflect, .. } => Ok(reflect),
            _ => Err(anyhow::anyhow!("Shader is not compiled")),
        }
    }

    pub fn compile(&mut self) -> Result<()> {
        let arena = shadertool::Arena::new();
        let module = shadertool::compile(
            &self.path,
            &arena,
            &shadertool::BuildOptions {
                emit_cargo_deps: false,
                emit_debug_information: false,
                emit_spirv_binaries: false,
                emit_reflection: false,
                include_paths: vec![],
                output_directory: None,
            },
        )?;
        self.update_pipelines(&module);
        let reflect = spirv_reflect::Module::parse(&module.spirv)?;
        self.compilation_state = CompilationState::Compiled { spirv: module.spirv, reflect };
        Ok(())
    }

    fn update_pipelines(&mut self, module: &shadertool::Module) {
        for ep in self.entry_points.values_mut() {
            ep.stale = true;
        }
        for pipe in self.pipelines.values_mut() {
            match pipe {
                Pipeline::Primitive(gpipe) => {
                    gpipe.stale = true;
                }
                Pipeline::Compute(cpipe) => {
                    cpipe.stale = true;
                }
                Pipeline::Mesh(gpipe) => {
                    gpipe.stale = true;
                }
            }
        }
        for entry_point in module.entry_points.iter() {
            match self.entry_points.get_mut(&entry_point.name) {
                Some(ep) => {
                    ep.stale = false;
                }
                None => {
                    self.entry_points
                        .insert(entry_point.name.clone(), EntryPoint { name: entry_point.name.clone(), stale: false });
                }
            }
        }
        for pipe in module.pipelines.iter() {
            match pipe.type_ {
                PipelineType::Primitive { vertex, fragment } => match self.pipelines.get_mut(&pipe.name) {
                    Some(Pipeline::Primitive(gpipe)) => {
                        gpipe.vertex_shader_entry = module.entry_points[vertex].name.clone();
                        gpipe.fragment_shader_entry = module.entry_points[fragment].name.clone();
                        gpipe.workgroup_size = pipe.workgroup_size;
                        gpipe.stale = false;
                    }
                    _ => {
                        let gpipe = GraphicsPipeline {
                            vertex_shader_entry: module.entry_points[vertex].name.clone(),
                            fragment_shader_entry: module.entry_points[fragment].name.clone(),
                            workgroup_size: pipe.workgroup_size,
                            rasterization: Default::default(),
                            depth_stencil: Default::default(),
                            color_targets: vec![],
                            stale: false,
                            pipeline: None,
                        };
                        self.pipelines.insert(pipe.name.clone(), Pipeline::Primitive(gpipe));
                    }
                },
                PipelineType::Compute { compute } => match self.pipelines.get_mut(&pipe.name) {
                    Some(Pipeline::Compute(cpipe)) => {
                        cpipe.entry_point = module.entry_points[compute].name.clone();
                        cpipe.workgroup_size = pipe.workgroup_size;
                        cpipe.stale = false;
                    }
                    _ => {
                        self.pipelines.insert(
                            pipe.name.clone(),
                            Pipeline::Compute(ComputePipeline {
                                entry_point: module.entry_points[compute].name.clone(),
                                workgroup_size: pipe.workgroup_size,
                                stale: false,
                                pipeline: None,
                            }),
                        );
                    }
                },
                PipelineType::Mesh { .. } => {
                    // TODO
                }
            }
        }

        // Remove stale pipelines & entry points
        self.entry_points.retain(|_, ep| !ep.stale);
        self.pipelines.retain(|_, pipe| match pipe {
            Pipeline::Primitive(gpipe) => !gpipe.stale,
            Pipeline::Compute(cpipe) => !cpipe.stale,
            Pipeline::Mesh(gpipe) => !gpipe.stale,
        });
    }
}
