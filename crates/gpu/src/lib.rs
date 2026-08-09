#![feature(default_field_values)]
#![allow(unsafe_op_in_unsafe_fn, reason = "too verbose, and my IDE already highlights unsafe call sites")]
#![expect(unused, reason = "noisy")]

extern crate self as gpu;

mod buffer;
mod command;
mod debugger;
mod device;
mod image;
mod instance;
pub mod platform;
mod surface;
mod swapchain;
pub mod util;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use gpu_types::reflection::{ShaderReflection};

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

// proc-macros
pub use gpu_macros::{Vertex, shader_module};

pub mod prelude {
    pub use crate::{
        Buffer, BufferUsage, ClearColorValue, ColorBlendEquation, ColorTargetState, CommandBuffer, DepthStencilState,
        Format, FragmentState, GraphicsPipeline, GraphicsPipelineCreateInfo, Image, ImageCreateInfo, ImageType,
        ImageUsage, MemoryLocation, Point2D, PreRasterizationShaders, RasterizationState, Rect2D, RenderEncoder,
        Sampler, SamplerParams, ShaderCode, ShaderEntryPoint, ShaderSource, Size2D, StencilState, Vertex,
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


/// Represents a graphics pipeline.
#[derive(Clone)]
pub struct GraphicsPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    // Push descriptors require live VkDescriptorSetLayouts
    _descriptor_set_layouts: Vec<DescriptorSetLayout>,
    /// Whether this pipeline uses the standard bindless descriptor set.
    ///
    /// The layout of the bindless descriptor set is as follows:
    /// - set 0, binding 0: array of sampler descriptors
    /// - set 0, binding 1: array of combined image sampler descriptors (unused)
    /// - set 0, binding 2: array of storage image descriptors
    ///
    /// The descriptor arrays are kept up-to-date automatically as resources are created and destroyed.
    pub(crate) bindless: bool,
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
        let pipeline_layout = self.pipeline_layout;
        unsafe {
            Device::instance().delete_after_current_submission(move |device| {
                device.raw.destroy_pipeline(pipeline, None);
                device.raw.destroy_pipeline_layout(pipeline_layout, None);
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
    pub(crate) pipeline_layout: vk::PipelineLayout,
    _descriptor_set_layouts: Vec<DescriptorSetLayout>,
    /// See `GraphicsPipeline::bindless` for details.
    pub(crate) bindless: bool,
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
        let pipeline_layout = self.pipeline_layout;

        unsafe {
            // Wait until the current submission has completed execution since it may be using
            // the pipeline.
            Device::instance().delete_after_current_submission(move |device| {
                device.raw.destroy_pipeline(pipeline, None);
                device.raw.destroy_pipeline_layout(pipeline_layout, None);
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

/// Represents a sampler object.
#[derive(Clone, Debug)]
pub struct Sampler {
    descriptor_index: SamplerDescriptorIndex,
    sampler: vk::Sampler,
}

impl Sampler {
    /// Creates a new sampler object.
    ///
    /// Sampler objects are cached by creation parameters, so this function may return
    /// the same underlying `VkSampler`, given the same [`SamplerParams`].
    pub fn new(create_info: SamplerParams) -> Self {
        Device::instance().create_sampler(&create_info)
    }

    /// Returns this sampler as a [`Descriptor`].
    pub fn descriptor(&self) -> Descriptor<'_> {
        Descriptor::Sampler { sampler: self.clone() }
    }

    /// Returns the sampler handle for use in shader parameters.
    pub fn device_handle(&self) -> SamplerHandle {
        SamplerHandle::new(self.descriptor_index.index())
    }
}

impl VulkanObject for Sampler {
    type Handle = vk::Sampler;

    fn handle(&self) -> Self::Handle {
        self.sampler
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates command buffers in a `vk::CommandPool` and allows re-use of freed command buffers.
#[derive(Debug)]
struct CommandPool {
    queue_family: u32,
    command_pool: vk::CommandPool,
    free: Vec<vk::CommandBuffer>,
    used: Vec<vk::CommandBuffer>,
}

impl CommandPool {
    unsafe fn new(device: &ash::Device, queue_family_index: u32) -> CommandPool {
        // create a new one
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            queue_family_index,
            ..Default::default()
        };
        let command_pool = device.create_command_pool(&create_info, None).expect("failed to create a command pool");

        CommandPool { queue_family: queue_family_index, command_pool, free: vec![], used: vec![] }
    }

    fn alloc(&mut self, device: &ash::Device) -> vk::CommandBuffer {
        let cb = self.free.pop().unwrap_or_else(|| unsafe {
            let allocate_info = vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let buffers = device.allocate_command_buffers(&allocate_info).expect("failed to allocate command buffers");
            buffers[0]
        });
        self.used.push(cb);
        cb
    }

    unsafe fn reset(&mut self, device: &ash::Device) {
        device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty()).unwrap();
        self.free.append(&mut self.used)
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

/// Description of one argument in an argument block.
pub enum Descriptor<'a> {
    SampledImage { image: &'a Image, layout: vk::ImageLayout },
    StorageImage { image: &'a Image, layout: vk::ImageLayout },
    UniformBuffer { buffer: &'a BufferUntyped, offset: u64, size: u64 },
    StorageBuffer { buffer: &'a BufferUntyped, offset: u64, size: u64 },
    Sampler { sampler: Sampler },
}

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

    pub fn storage_descriptor(&self) -> Descriptor<'_> {
        Descriptor::StorageBuffer { buffer: self.buffer.as_bytes(), offset: self.byte_offset, size: self.byte_size }
    }

    pub fn uniform_descriptor(&self) -> Descriptor<'_> {
        Descriptor::UniformBuffer { buffer: self.buffer.as_bytes(), offset: self.byte_offset, size: self.byte_size }
    }

    pub fn as_bytes(&self) -> BufferRange<'a, u8> {
        BufferRange {
            buffer: unsafe { self.buffer.as_cast::<u8>() },
            byte_offset: self.byte_offset,
            byte_size: self.byte_size,
        }
    }

    /*pub fn slice(&self, range: impl RangeBounds<usize>) -> BufferRange<'a, [T]> {
        let elem_size = mem::size_of::<T>();
        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(start) => *start,
            Bound::Excluded(start) => *start + 1,
        };
        let end = match range.end_bound() {
            Bound::Unbounded => self.len(),
            Bound::Excluded(end) => *end,
            Bound::Included(end) => *end + 1,
        };
        let start = (start * elem_size) as u64;
        let end = (end * elem_size) as u64;

        BufferRange {
            untyped: BufferRangeAny {
                buffer: self.untyped.buffer,
                offset: self.untyped.offset + start,
                size: end - start,
            },
            _phantom: PhantomData,
        }
    }*/
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
