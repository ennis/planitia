use std::collections::HashMap;
use std::path::Path;
use gpu::GraphicsPipeline;
use shader_archive::PipelineArchive;

/// Holds data used by the rendering system: GPU resources (render targets, shader pipelines, etc.)
pub struct RenderWorld {
    graphics_pipelines: HashMap<String, GraphicsPipeline>,
}

/*
impl RenderWorld {
    pub fn new(shader_archive: impl AsRef<Path>) -> RenderWorld {
        let archive = PipelineArchive::load(shader_archive).unwrap();
        let header = archive.header().unwrap();

        // Load all pipelines

    }
}*/