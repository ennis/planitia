//! In-memory shader reflection information.

#[derive(Debug, Copy, Clone)]
pub struct GraphicsPipelineReflection<'a> {
    pub color_formats: &'a [gpu::Format],
}

#[derive(Debug, Copy, Clone)]
pub struct ComputePipelineReflection {
    workgroup_size: [u32; 3],
}
