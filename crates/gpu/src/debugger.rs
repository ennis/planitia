use crate::{ComputePipeline, GraphicsPipeline, Image};
use crate::debugger::renderer::Renderer;
use ash::vk;
use std::sync::{LazyLock, Mutex, MutexGuard};
use gpu_types::reflection::ShaderReflection;
use gpu_types::ShaderStage;

mod font;
mod renderer;
mod ui;

//--------------------------------------------------------------------------------------------------

pub struct Debugger {
    renderer: Option<Renderer>,
    stage_reflection: [ShaderReflection; SHADER_STAGE_COUNT],
}

static INSTANCE: LazyLock<Mutex<Debugger>> = LazyLock::new(|| Mutex::new(Debugger::new()));

impl Debugger {
    pub fn new() -> Debugger {
        Debugger { renderer: None, stage_reflection: [ShaderReflection::default(); SHADER_STAGE_COUNT] }
    }

    pub fn instance() -> MutexGuard<'static, Debugger> {
        INSTANCE.lock().unwrap()
    }

    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        //pipeline.

    }

    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {

    }
}

//--------------------------------------------------------------------------------------------------

const SHADER_STAGE_COUNT: usize = 8;

fn shader_stage_index(shader_stage: ShaderStage) -> usize {
    match shader_stage {
        ShaderStage::Vertex => 0,
        ShaderStage::Mesh => 1,
        ShaderStage::Task => 2,
        ShaderStage::TessControl => 3,
        ShaderStage::TessEvaluation => 4,
        ShaderStage::Geometry => 5,
        ShaderStage::Fragment => 6,
        ShaderStage::Compute => 7,
    }
}

pub(crate) fn bind_graphics_pipeline(pipeline: &GraphicsPipeline) {
   // Debugger::instance().bind_graphics_pipeline(pipeline)
}

pub(crate) fn bind_compute_pipeline(pipeline: &ComputePipeline) {
  //  Debugger::instance().bind_compute_pipeline(pipeline)
}

pub(crate) fn render_debugger(target: &Image) {
    Debugger::instance().render(target);
}


/*
enum CommandType {
    Draw,
    Compute,
}

///
struct CapturedBuffer {
    /// Captured buffer data.
    data: Vec<u8>,
}

/// Represents a captured GPU command (draw call or compute dispatch).
struct Command {
    command_type: CommandType,
    buffers: Vec<CapturedBuffer>,
}

fn capture_graphics_command_arguments(pipeline: &GraphicsPipeline, push_data: &[u8]) {}

fn capture_command_arguments(shader_reflection: &ShaderReflection, push_data: &[u8]) {
    for param in shader_reflection.params {
        // the goal here is to collect pointers to all buffers referenced in push data, transitively.
        // the problem is that we don't know the bounds of the
    }
}

fn debug_graphics_pre_command_hook(pipeline: &GraphicsPipeline, push_data: *mut u8, push_data_size: usize) {
    // Here we have an opportunity to capture the state of buffers before they are sent to the GPU,
    // - or to automatically insert/replace data
    // - or to replace the graphics pipeline entirely
}
*/
