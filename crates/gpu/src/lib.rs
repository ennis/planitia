//! A streamlined GPU API layer (for desktop GPUs) on top of Vulkan.
//!
//! The goal is to reduce the amount of Vulkan boilerplate needed to get data on the GPU and dispatch
//! commands. Stuff like device creation, command pools, descriptor set, layout, whatever, etc. are abstracted away.
//!
//! It makes use of somewhat recent Vulkan extensions. However, these should be available
//! on reasonably recent desktop GPUs. Mobile GPUs are explicitly not a target of this library:
//! if an extension exists that makes the API simpler, but is only available
//! on desktop GPUs, the extension will be used unconditionally, even if it makes the library
//! unusable on mobile.
//!
//! # Requirements
//!
//! Vulkan 1.4 is required. In addition, the following extensions are required:
//! - TODO
//!
//! # Safety
//!
//! Safety is not a primary goal. Notably, memory safety on the GPU domain is not guaranteed
//! as this would compromise the ergonomics too much.
//!
//! # No resource usage tracking, manual synchronization
//!
//! Modern GPU APIs have pushed the responsibility of emitting synchronization commands to the user,
//! for better or worse.
//! Manually managing synchronization in a large application is typically seen as unmanageable,
//! which led to the rise of higher-level APIs that automatically track resource usage, or "render graph"-
//! style APIs that force the client to declare all passes up-front.
//! These have a non-negligible cost, either in terms of performance (for resource tracking),
//! or cognitive overhead (for render graphs). All of this to emit barriers with a degree of precision
//! that is encouraged by the Vulkan API, but is often completely ignored by the underlying driver
//! (see https://www.sebastianaaltonen.com/blog/no-graphics-api#barriers:~:text=Barriers%20and%20fences)
//!
//! This API layer intentionally refrains from doing any kind of automatic resource usage tracking.
//! This means that barriers between commands need to be specified by the user, although they have been greatly
//! simplified to a handful of invalidation flags, and are not specified per-resource anymore.

#![feature(default_field_values)]
#![allow(unsafe_op_in_unsafe_fn, reason = "too verbose, and my IDE already highlights unsafe call sites")]
#![expect(unused, reason = "noisy")]

extern crate self as gpu;

mod buffer;
mod command;
mod command_pool;
mod device;
mod image;
mod instance;
pub mod platform;
mod surface;
mod swapchain;
mod temp;
pub mod util;
mod query_pool;

use gpu_types::reflection::ShaderReflection;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Reexports

pub use ash::{self, vk};
pub use gpu_allocator::MemoryLocation;
pub use gpu_types::*;

pub use buffer::*;
pub use command::*;
pub use device::*;
pub use image::*;
pub use instance::*;
pub use surface::*;
pub use swapchain::*;
pub use temp::{alloc_temp, alloc_temp_slice};
pub use query_pool::*;

// proc-macros
pub use gpu_macros::{Vertex, shader_module};

pub mod prelude {
    pub use crate::{
        Buffer, BufferUsage, ClearColorValue, ColorBlendEquation, ColorTargetState, CommandBuffer, DepthStencilState,
        Format, FragmentState, GraphicsPipeline, GraphicsPipelineCreateInfo, Image, ImageCreateInfo, ImageType,
        ImageUsage, MemoryLocation, Point2D, PreRasterizationShaders, RasterizationState, Rect2D, RenderEncoder,
        SamplerParams, ShaderCode, ShaderEntryPoint, ShaderSource, Size2D, StencilState, Vertex,
        VertexBufferLayoutDescription, VertexInputAttributeDescription, VertexInputState, vk,
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to create device")]
    DeviceCreationFailed(#[from] DeviceCreateError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Trait implemented by wrappers of Vulkan API objects.
pub trait VulkanObject {
    /// The Vulkan API handle type associated with the object.
    type Handle: vk::Handle;
    /// Returns the Vulkan API handle of the object.
    fn handle(&self) -> Self::Handle;
}

/// Standard subgroup size.
pub const SUBGROUP_SIZE: u32 = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

pub type FrameIndex = u64;

/// Represents a graphics pipeline.
#[derive(Clone)]
pub struct GraphicsPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) stage_reflection: Vec<(ShaderStage, ShaderReflection)>,
}

impl GraphicsPipeline {
    /// Creates a new graphics pipeline.
    pub fn new(create_info: GraphicsPipelineCreateInfo) -> Result<Self, Error> {
        Device::instance().create_graphics_pipeline(create_info)
    }

    /// Returns the `VkPipeline` handle of this pipeline object.
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        let pipeline = self.pipeline;
        unsafe {
            Device::instance().delete_after_current_frame(move |device| {
                device.raw.destroy_pipeline(pipeline, None);
            })
        }
    }
}

impl VulkanObject for GraphicsPipeline {
    type Handle = vk::Pipeline;
    fn handle(&self) -> Self::Handle {
        self.pipeline
    }
}

/// Compute pipelines.
#[derive(Clone)]
pub struct ComputePipeline {
    pub(crate) pipeline: vk::Pipeline,
    //pub(crate) pipeline_layout: vk::PipelineLayout,
    //_descriptor_set_layouts: Vec<DescriptorSetLayout>,
    /// See `GraphicsPipeline::bindless` for details.
    //pub(crate) bindless: bool,
    pub(crate) reflection: ShaderReflection,
}

impl ComputePipeline {
    /// Creates a new compute pipeline.
    pub fn new(create_info: ComputePipelineCreateInfo) -> Result<Self, Error> {
        Device::instance().create_compute_pipeline(create_info)
    }

    /// Returns the Vulkan pipeline handle.
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        let pipeline = self.pipeline;
        //let pipeline_layout = self.pipeline_layout;

        unsafe {
            // Wait until the current submission has completed execution since it may be using
            // the pipeline.
            Device::instance().delete_after_current_frame(move |device| {
                device.raw.destroy_pipeline(pipeline, None);
                //device.raw.destroy_pipeline_layout(pipeline_layout, None);
            })
        }
    }
}

impl VulkanObject for ComputePipeline {
    type Handle = vk::Pipeline;
    fn handle(&self) -> Self::Handle {
        self.pipeline
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct DescriptorSetLayout {
    last_submission_index: Option<Arc<AtomicU64>>,
    pub handle: vk::DescriptorSetLayout,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        if let Some(last_submission_index) = Arc::into_inner(self.last_submission_index.take().unwrap()) {
            let handle = self.handle;
            Device::instance().call_later(last_submission_index.load(Ordering::Relaxed), move |device| unsafe {
                device.raw.destroy_descriptor_set_layout(handle, None);
            });
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Describes an image buffer that is used as the source or destination of an image transfer operation.
#[derive(Copy, Clone, Debug)]
pub struct ImageCopyBuffer<'a> {
    pub buffer: &'a BufferUntyped,
    pub layout: ImageDataLayout,
}

/// Describes part of an image subresource, for transfer operations.
#[derive(Copy, Clone, Debug)]
pub struct ImageCopyView<'a> {
    pub image: &'a Image,
    pub mip_level: u32 = 0,
    pub origin: Offset3D = Offset3D::ZERO,
    pub aspect: ImageAspect = ImageAspect::All,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

///// Description of one argument in an argument block.
//pub enum Descriptor<'a> {
//    SampledImage { image: &'a Image, layout: vk::ImageLayout },
//    StorageImage { image: &'a Image, layout: vk::ImageLayout },
//    UniformBuffer { buffer: &'a BufferUntyped, offset: u64, size: u64 },
//    StorageBuffer { buffer: &'a BufferUntyped, offset: u64, size: u64 },
//    Sampler { sampler: Sampler },
//}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct BufferRange<'a, T> {
    pub buffer: &'a Buffer<T>,
    /// Offset into the buffer in bytes. Should be a multiple of `size_of::<T>()`.
    pub byte_offset: u64,
    /// Size of the slice in bytes. Should be a multiple of `size_of::<T>()`.
    pub byte_size: u64,
}

// #26925 clone impl
impl<T> Clone for BufferRange<'_, T> {
    fn clone(&self) -> Self {
        Self { buffer: self.buffer, byte_offset: self.byte_offset, byte_size: self.byte_size }
    }
}

impl<'a, T> std::fmt::Debug for BufferRange<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferRange")
            .field("buffer", &self.buffer)
            .field("byte_offset", &self.byte_offset)
            .field("byte_size", &self.byte_size)
            .finish()
    }
}

impl<'a, T: Copy + 'static> BufferRange<'a, T> {
    pub fn len(&self) -> usize {
        (self.byte_size / size_of::<T>() as u64) as usize
    }

    pub fn as_bytes(&self) -> BufferRange<'a, u8> {
        BufferRange {
            buffer: unsafe { self.buffer.as_cast::<u8>() },
            byte_offset: self.byte_offset,
            byte_size: self.byte_size,
        }
    }
}

pub type BufferRangeUntyped<'a> = BufferRange<'a, u8>;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Specifies a color attachment.
#[derive(Clone)]
pub struct ColorAttachment<'a> {
    pub image: &'a Image,
    /// The color to clear the attachment before rendering. If `None`, existing data is kept.
    // f64 because it can represent all i32 and u32 values exactly.
    pub clear: Option<[f64; 4]> = None,
}

impl ColorAttachment<'_> {
    pub(crate) fn get_vk_clear_color_value(&self) -> vk::ClearColorValue {
        if let Some(clear_value) = self.clear {
            match format_numeric_type(self.image.format()) {
                FormatNumericType::UInt => vk::ClearColorValue {
                    uint32: [
                        clear_value[0] as u32,
                        clear_value[1] as u32,
                        clear_value[2] as u32,
                        clear_value[3] as u32,
                    ],
                },
                FormatNumericType::SInt => vk::ClearColorValue {
                    int32: [clear_value[0] as i32, clear_value[1] as i32, clear_value[2] as i32, clear_value[3] as i32],
                },
                FormatNumericType::Float => vk::ClearColorValue {
                    float32: [
                        clear_value[0] as f32,
                        clear_value[1] as f32,
                        clear_value[2] as f32,
                        clear_value[3] as f32,
                    ],
                },
            }
        } else {
            vk::ClearColorValue::default()
        }
    }
}

/// Specifies a depth-stencil attachment.
#[derive(Clone)]
pub struct DepthStencilAttachment<'a> {
    pub image: &'a Image,
    pub depth_clear: Option<f64> = None,
    pub stencil_clear: Option<u32> = None,
}

impl DepthStencilAttachment<'_> {
    pub(crate) fn get_vk_clear_depth_stencil_value(&self) -> vk::ClearDepthStencilValue {
        vk::ClearDepthStencilValue {
            depth: self.depth_clear.unwrap_or(0.0) as f32,
            stencil: self.stencil_clear.unwrap_or(0),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Specifies the code of a shader.
#[derive(Debug, Clone, Copy)]
pub enum ShaderCode<'a> {
    /// Compile the shader from the specified source.
    Source(ShaderSource<'a>),
    /// Create the shader from the specified SPIR-V binary.
    Spirv(&'a [u32]),
}

/// Specifies the shaders of a graphics pipeline.
#[derive(Copy, Clone, Debug)]
pub enum PreRasterizationShaders<'a> {
    /// Shaders of the primitive shading pipeline (the classic vertex, tessellation, geometry and fragment shaders).
    ///
    /// NOTE: tessellation & geometry pipelines are unlikely to be used anytime soon,
    ///       so we don't bother with them (this reduces the maintenance burden).
    PrimitiveShading {
        vertex: ShaderEntryPoint<'a>,
        //tess_control: Option<ShaderDescriptor<'a>>,
        //tess_evaluation: Option<ShaderDescriptor<'a>>,
        //geometry: Option<ShaderDescriptor<'a>>,
    },
    /// Shaders of the mesh shading pipeline (the new mesh and task shaders).
    MeshShading { task: Option<ShaderEntryPoint<'a>>, mesh: ShaderEntryPoint<'a> },
}

#[derive(Copy, Clone, Debug)]
pub struct GraphicsPipelineCreateInfo<'a> {
    /// If left empty, use the universal descriptor set layout.
    pub set_layouts: &'a [DescriptorSetLayout] = &[],
    // None of the relevant drivers on desktop seem to care about precise push constant ranges,
    // so we just store the total size of push constants.
    // FIXME: this is redundant with the information in ShaderDescriptors
    pub push_constants_size: usize = 0,
    pub vertex_input: VertexInputState<'a> = VertexInputState { .. },
    pub pre_rasterization_shaders: PreRasterizationShaders<'a>,
    pub rasterization: RasterizationState,
    pub depth_stencil: Option<DepthStencilState> = None,
    pub fragment: FragmentState<'a>,
}

#[derive(Copy, Clone, Debug)]
pub struct ComputePipelineCreateInfo<'a> {
    /// If left empty, use the universal descriptor set layout.
    pub set_layouts: &'a [DescriptorSetLayout] = &[],
    /// FIXME: this is redundant with the information in `compute_shader`
    pub push_constants_size: usize = 0,
    /// Compute shader.
    pub shader: ShaderEntryPoint<'a>,
}

/// Represents the range of GPU addresses associated with a buffer.
#[derive(Copy, Clone)]
pub(crate) struct BufferAddressRange {
    pub(crate) buffer: vk::Buffer,
    pub(crate) base: vk::DeviceAddress,
    pub(crate) size: usize,
}

// Implementation detail of the VertexInput macro
#[doc(hidden)]
pub const fn append_attributes<const N: usize>(
    head: &'static [VertexInputAttributeDescription],
    binding: u32,
    base_location: u32,
    tail: &'static [VertexAttributeDescription],
) -> [VertexInputAttributeDescription; N] {
    const NULL_ATTR: VertexInputAttributeDescription =
        VertexInputAttributeDescription { location: 0, binding: 0, format: Format::UNDEFINED, offset: 0 };
    let mut result = [NULL_ATTR; N];
    let mut i = 0;
    while i < head.len() {
        result[i] = head[i];
        i += 1;
    }
    while i < N {
        let j = i - head.len();
        result[i] = VertexInputAttributeDescription {
            location: base_location + j as u32,
            binding,
            format: tail[j].format,
            offset: tail[j].offset,
        };
        i += 1;
    }

    result
}

// Implementation detail of shader_module
#[doc(hidden)]
#[macro_export]
macro_rules! include_bytes_as_u32 {
    // https://docs.rs/resb/latest/src/resb/binary.rs.html#25-44
    ($path:literal) => {
        const {
            #[repr(align(4))]
            pub struct AlignedAs<Bytes: ?Sized> {
                pub bytes: Bytes,
            }

            const B: &[u8] = &AlignedAs { bytes: *include_bytes!($path) }.bytes;
            // SAFETY: B is statically borrowed, 4-aligned, and the length is within
            // the static slice (truncated to a multiple of four).
            unsafe { core::slice::from_raw_parts(B.as_ptr() as *const u32, B.len() / size_of::<u32>()) }
        }
    };
}

// Implementation detail of shader_module
#[doc(hidden)]
#[macro_export]
macro_rules! bytes_as_u32 {
    ($bytes:literal) => {
        const {
            #[repr(align(4))]
            pub struct AlignedAs<Bytes: ?Sized> {
                pub bytes: Bytes,
            }

            const B: &[u8] = &AlignedAs { bytes: *$bytes }.bytes;
            // SAFETY: B is statically borrowed, 4-aligned, and the length is within
            // the static slice (truncated to a multiple of four).
            unsafe { core::slice::from_raw_parts(B.as_ptr() as *const u32, B.len() / size_of::<u32>()) }
        }
    };
}
