#![feature(default_field_values)]
mod buffer;
mod command;
mod device;
mod image;
mod instance;
pub mod platform;
mod surface;
mod swapchain;
mod types;
pub mod util;

use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// --- reexports ---

pub use ash::{self, vk};
pub use gpu_allocator::MemoryLocation;

pub use buffer::*;
pub use command::*;
pub use device::*;
pub use image::*;
pub use instance::*;
pub use surface::*;
pub use swapchain::*;
pub use types::*;

// proc-macros
pub use gpu_macros::Vertex;

pub mod prelude {
    pub use crate::{
        vk, Buffer, BufferUsage, ClearColorValue, ColorBlendEquation, ColorTargetState, CommandStream,
        DepthStencilState, Format, FragmentState, GraphicsPipeline, GraphicsPipelineCreateInfo, Image, ImageCreateInfo,
        ImageType, ImageUsage, MemoryLocation, PipelineBindPoint, PipelineLayoutDescriptor, Point2D,
        PreRasterizationShaders, RasterizationState, Rect2D, RenderEncoder, Sampler, SamplerCreateInfo, ShaderCode,
        ShaderEntryPoint, ShaderSource, Size2D, StencilState, Vertex, VertexBufferDescriptor,
        VertexBufferLayoutDescription, VertexInputAttributeDescription, VertexInputState,
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Standard subgroup size.
pub const SUBGROUP_SIZE: u32 = 32;

/*
/// Device address of a GPU buffer.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PtrUntyped {
    pub raw: vk::DeviceAddress,
}*/

/// Device address of a GPU buffer containing elements of type `T` its associated type.
///
/// The type should be `T: Copy` for a buffer containing a single element of type T,
/// or `[T] where T: Copy` for slices of elements of type T.
#[repr(transparent)]
pub struct Ptr<T: ?Sized + 'static> {
    pub raw: vk::DeviceAddress,
    pub _phantom: PhantomData<T>,
}

impl<T: ?Sized + 'static> Ptr<T> {
    /// Null (invalid) device address.
    pub const NULL: Self = Ptr {
        raw: 0,
        _phantom: PhantomData,
    };
}

impl<T: 'static> Ptr<[T]> {
    pub fn offset(self, offset: usize) -> Self {
        Ptr {
            raw: self.raw + (offset * size_of::<T>()) as u64,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized + 'static> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        Ptr {
            raw: self.raw,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized + 'static> Copy for Ptr<T> {}

/// Bindless handle to an image.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct TextureHandle {
    /// Index of the image in the image descriptor array.
    pub index: u32,
    /// For compatibility with slang.
    _unused: u32,
}

impl TextureHandle {
    pub const INVALID: Self = TextureHandle {
        index: u32::MAX,
        _unused: 0,
    };
}

/// Bindless handle to a storage image.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct StorageImageHandle {
    /// Index of the image in the image descriptor array.
    pub index: u32,
    /// For compatibility with slang.
    _unused: u32,
}

impl StorageImageHandle {
    pub const INVALID: Self = StorageImageHandle {
        index: u32::MAX,
        _unused: 0,
    };
}

/// Bindless handle to a sampler.
#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct SamplerHandle {
    /// Index of the image in the sampler descriptor array.
    pub index: u32,
    /// For compatibility with slang.
    _unused: u32,
}

impl SamplerHandle {
    pub const INVALID: Self = SamplerHandle {
        index: u32::MAX,
        _unused: 0,
    };
}

/// Graphics pipelines.
///
/// TODO Drop impl
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
    /// The descriptor arrays are kept up-to-date automatically as resourcess are created and destroyed.
    pub(crate) bindless: bool,
}

impl GraphicsPipeline {
    /// Creates a new graphics pipeline.
    pub fn new(create_info: GraphicsPipelineCreateInfo) -> Result<Self, Error> {
        Device::global().create_graphics_pipeline(create_info)
    }

    /// Returns the Vulkan pipeline handle.
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        let pipeline = self.pipeline;
        let pipeline_layout = self.pipeline_layout;
        unsafe {
            Device::global().delete_after_current_submission(move |device| {
                device.raw.destroy_pipeline(pipeline, None);
                device.raw.destroy_pipeline_layout(pipeline_layout, None);
            })
        }
    }
}

/// Compute pipelines.
///
/// TODO Drop impl
#[derive(Clone)]
pub struct ComputePipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    _descriptor_set_layouts: Vec<DescriptorSetLayout>,
    /// See `GraphicsPipeline::bindless` for details.
    pub(crate) bindless: bool,
}

impl ComputePipeline {
    /// Creates a new compute pipeline.
    pub fn new(create_info: ComputePipelineCreateInfo) -> Result<Self, Error> {
        Device::global().create_compute_pipeline(create_info)
    }

    /// Returns the Vulkan pipeline handle.
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

/// Samplers
#[derive(Clone, Debug)]
pub struct Sampler {
    descriptor_index: SamplerDescriptorIndex,
    sampler: vk::Sampler,
}

impl Sampler {
    /// Creates a new sampler object.
    pub fn new(create_info: SamplerCreateInfo) -> Self {
        Device::global().create_sampler(&create_info)
    }

    /// Returns the Vulkan sampler handle.
    pub fn handle(&self) -> vk::Sampler {
        self.sampler
    }

    /// Returns this sampler as a descriptor.
    pub fn descriptor(&self) -> Descriptor<'_> {
        Descriptor::Sampler { sampler: self.clone() }
    }

    /// Returns the bindless sampler handle.
    pub fn device_handle(&self) -> SamplerHandle {
        SamplerHandle {
            index: self.descriptor_index.index(),
            _unused: 0,
        }
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
        let command_pool = device
            .create_command_pool(&create_info, None)
            .expect("failed to create a command pool");

        CommandPool {
            queue_family: queue_family_index,
            command_pool,
            free: vec![],
            used: vec![],
        }
    }

    fn alloc(&mut self, device: &ash::Device) -> vk::CommandBuffer {
        let cb = self.free.pop().unwrap_or_else(|| unsafe {
            let allocate_info = vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let buffers = device
                .allocate_command_buffers(&allocate_info)
                .expect("failed to allocate command buffers");
            buffers[0]
        });
        self.used.push(cb);
        cb
    }

    unsafe fn reset(&mut self, device: &ash::Device) {
        device
            .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap();
        self.free.append(&mut self.used)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub trait TrackedResource {
    /// Returns the internal tracking ID of the image.
    fn id(&self) -> ResourceId;
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
            Device::global().call_later(last_submission_index.load(Ordering::Relaxed), move |device| unsafe {
                device.raw.destroy_descriptor_set_layout(handle, None);
            });
        }
    }
}

/*
/// Self-contained descriptor set.
pub struct DescriptorSet {
    device: Device,
    last_submission_index: Option<Arc<AtomicU64>>,
    pool: vk::DescriptorPool,
    handle: vk::DescriptorSet,
}

impl DescriptorSet {
    pub fn handle(&self) -> vk::DescriptorSet {
        self.handle
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn set_name(&self, label: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.device.set_object_name(self.handle, label);
        }
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        if let Some(last_submission_index) = Arc::into_inner(self.last_submission_index.take().unwrap()) {
            let device = self.device.clone();
            let pool = self.pool;
            self.device
                .call_later(last_submission_index.load(Ordering::Relaxed), move || unsafe {
                    device.destroy_descriptor_pool(pool, None);
                });
        }
    }
}*/

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
    SampledImage {
        image: &'a Image,
        layout: vk::ImageLayout,
    },
    StorageImage {
        image: &'a Image,
        layout: vk::ImageLayout,
    },
    UniformBuffer {
        buffer: &'a BufferUntyped,
        offset: u64,
        size: u64,
    },
    StorageBuffer {
        buffer: &'a BufferUntyped,
        offset: u64,
        size: u64,
    },
    Sampler {
        sampler: Sampler,
    },
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
/// Typed buffers.
#[repr(transparent)]
pub struct Buffer<T: ?Sized> {
    pub untyped: BufferUntyped,
    _marker: PhantomData<T>,
}*/
/*
impl<T: ?Sized> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            untyped: self.untyped.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> GpuResource for Buffer<T> {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.untyped.set_last_submission_index(submission_index);
    }
}*/

/*
#[derive(Clone, Debug)]
pub struct BufferRangeUntyped {
    pub buffer: BufferUntyped,
    pub offset: u64,
    pub size: u64,
}

impl BufferRangeUntyped {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer.handle
    }

    pub fn size(&self) -> u64 {
        self.size
    }

}*/

pub struct BufferRange<'a, T> {
    pub buffer: &'a Buffer<[T]>,
    /// Offset into the buffer in bytes. Should be a multiple of `size_of::<T>()`.
    pub byte_offset: u64,
    /// Size of the slice in bytes. Should be a multiple of `size_of::<T>()`.
    pub byte_size: u64,
}

// #26925 clone impl
impl<T> Clone for BufferRange<'_, T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            byte_offset: self.byte_offset,
            byte_size: self.byte_size,
        }
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
        Descriptor::StorageBuffer {
            buffer: self.buffer.as_bytes(),
            offset: self.byte_offset,
            size: self.byte_size,
        }
    }

    pub fn uniform_descriptor(&self) -> Descriptor<'_> {
        Descriptor::UniformBuffer {
            buffer: self.buffer.as_bytes(),
            offset: self.byte_offset,
            size: self.byte_size,
        }
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

/// Describes a color, depth, or stencil attachment.
#[derive(Clone)]
pub struct ColorAttachment<'a> {
    pub image: &'a Image,
    pub clear_value: Option<[f64; 4]> = None,
    /*pub image_view: ImageView,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: [f64; 4],*/
}

impl ColorAttachment<'_> {
    pub(crate) fn get_vk_clear_color_value(&self) -> vk::ClearColorValue {
        if let Some(clear_value) = self.clear_value {
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
                    int32: [
                        clear_value[0] as i32,
                        clear_value[1] as i32,
                        clear_value[2] as i32,
                        clear_value[3] as i32,
                    ],
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

#[derive(Clone)]
pub struct DepthStencilAttachment<'a> {
    pub image: &'a Image,
    pub depth_clear_value: Option<f64> = None,
    pub stencil_clear_value: Option<u32> = None,
    /*pub depth_load_op: vk::AttachmentLoadOp,
    pub depth_store_op: vk::AttachmentStoreOp,
    pub stencil_load_op: vk::AttachmentLoadOp,
    pub stencil_store_op: vk::AttachmentStoreOp,
    pub depth_clear_value: f64,
    pub stencil_clear_value: u32,*/
}

impl DepthStencilAttachment<'_> {
    pub(crate) fn get_vk_clear_depth_stencil_value(&self) -> vk::ClearDepthStencilValue {
        vk::ClearDepthStencilValue {
            depth: self.depth_clear_value.unwrap_or(0.0) as f32,
            stencil: self.stencil_clear_value.unwrap_or(0),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Clone, Debug)]
pub struct VertexBufferDescriptor<'a> {
    pub binding: u32,
    pub buffer_range: BufferRangeUntyped<'a>,
    pub stride: u32,
}

pub trait VertexInput {
    /// Vertex buffer bindings
    fn buffer_layout(&self) -> Cow<'_, [VertexBufferLayoutDescription]>;

    /// Vertex attributes.
    fn attributes(&self) -> Cow<'_, [VertexInputAttributeDescription]>;

    /// Returns an iterator over the vertex buffers referenced in this object.
    fn vertex_buffers(&self) -> impl Iterator<Item = VertexBufferDescriptor<'_>>;
}

#[derive(Copy, Clone, Debug)]
pub struct VertexBufferView<T: Vertex> {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub _phantom: PhantomData<*const T>,
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

/// Describes a shader.
///
/// This type references the SPIR-V code of the shader, as well as the entry point function in the shader
/// and metadata.
#[derive(Debug, Clone, Copy)]
pub struct ShaderEntryPoint<'a> {
    /// Shader stage.
    pub stage: ShaderStage,
    /// SPIR-V code.
    pub code: &'a [u32],
    /// Name of the entry point function in SPIR-V code.
    pub entry_point: &'a str,
    /// Size of the push constants in bytes.
    pub push_constants_size: usize,
    /// Optional path to the source file of the shader.
    ///
    /// Used for diagnostic purposes and as a convenience for hot-reloading shaders.
    pub source_path: Option<&'a str>,
    /// Size of the local workgroup in each dimension, if applicable to the shader type.
    ///
    /// This is valid for compute, task, and mesh shaders.
    pub workgroup_size: [u32; 3],
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
    MeshShading {
        task: Option<ShaderEntryPoint<'a>>,
        mesh: ShaderEntryPoint<'a>,
    },
}

/*
impl<'a> PreRasterizationShaders<'a> {
    /// Creates a new `PreRasterizationShaders` object using mesh shading from the specified source file path.
    ///
    /// The specified source file should contain both task and mesh shaders. The entry point for both shaders is `main`.
    /// Use the `__TASK__` and `__MESH__` macros to distinguish between the two shaders within the source file.
    pub fn mesh_shading_from_source_file(file_path: &'a Path) -> Self {
        let entry_point = "main";
        Self::MeshShading {
            task: Some(ShaderEntryPoint {
                code: ShaderCode::Source(ShaderSource::File(file_path)),
                entry_point,
            }),
            mesh: ShaderEntryPoint {
                code: ShaderCode::Source(ShaderSource::File(file_path)),
                entry_point,
            },
        }
    }

    /// Creates a new `PreRasterizationShaders` object using primitive shading, without tessellation, from the specified source file path.
    pub fn vertex_shader_from_source_file(file_path: &'a Path) -> Self {
        let entry_point = "main";
        Self::PrimitiveShading {
            vertex: ShaderEntryPoint {
                code: ShaderCode::Source(ShaderSource::File(file_path)),
                entry_point,
            },
            tess_control: None,
            tess_evaluation: None,
            geometry: None,
        }
    }
}*/

#[derive(Copy, Clone, Debug)]
pub struct GraphicsPipelineCreateInfo<'a> {
    /// If left empty, use the universal descriptor set layout.
    pub set_layouts: &'a [DescriptorSetLayout],
    // None of the relevant drivers on desktop seem to care about precise push constant ranges,
    // so we just store the total size of push constants.
    // FIXME: this is redundant with the information in ShaderDescriptors
    pub push_constants_size: usize,
    pub vertex_input: VertexInputState<'a>,
    pub pre_rasterization_shaders: PreRasterizationShaders<'a>,
    pub rasterization: RasterizationState,
    pub depth_stencil: Option<DepthStencilState>,
    pub fragment: FragmentState<'a>,
}

#[derive(Copy, Clone, Debug)]
pub struct ComputePipelineCreateInfo<'a> {
    /// If left empty, use the universal descriptor set layout.
    pub set_layouts: &'a [DescriptorSetLayout],
    /// FIXME: this is redundant with the information in `compute_shader`
    pub push_constants_size: usize,
    /// Compute shader.
    pub shader: ShaderEntryPoint<'a>,
}

/// Computes the number of mip levels for a 2D image of the given size.
///
/// # Examples
///
/// ```
/// use gpu::mip_level_count;
/// assert_eq!(mip_level_count(512, 512), 9);
/// assert_eq!(mip_level_count(512, 256), 9);
/// assert_eq!(mip_level_count(511, 256), 8);
/// ```
pub fn mip_level_count(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32
}

pub fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        Format::D16_UNORM_S8_UINT | Format::D24_UNORM_S8_UINT | Format::D32_SFLOAT_S8_UINT
    )
}

pub fn is_depth_only_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        Format::D16_UNORM | Format::X8_D24_UNORM_PACK32 | Format::D32_SFLOAT
    )
}

pub fn is_stencil_only_format(fmt: vk::Format) -> bool {
    matches!(fmt, Format::S8_UINT)
}

pub fn aspects_for_format(fmt: vk::Format) -> vk::ImageAspectFlags {
    if is_depth_only_format(fmt) {
        vk::ImageAspectFlags::DEPTH
    } else if is_stencil_only_format(fmt) {
        vk::ImageAspectFlags::STENCIL
    } else if is_depth_and_stencil_format(fmt) {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum FormatNumericType {
    SInt,
    UInt,
    Float,
}

pub fn format_numeric_type(fmt: vk::Format) -> FormatNumericType {
    match fmt {
        Format::R8_UINT
        | Format::R8G8_UINT
        | Format::R8G8B8_UINT
        | Format::R8G8B8A8_UINT
        | Format::R16_UINT
        | Format::R16G16_UINT
        | Format::R16G16B16_UINT
        | Format::R16G16B16A16_UINT
        | Format::R32_UINT
        | Format::R32G32_UINT
        | Format::R32G32B32_UINT
        | Format::R32G32B32A32_UINT
        | Format::R64_UINT
        | Format::R64G64_UINT
        | Format::R64G64B64_UINT
        | Format::R64G64B64A64_UINT => FormatNumericType::UInt,

        Format::R8_SINT
        | Format::R8G8_SINT
        | Format::R8G8B8_SINT
        | Format::R8G8B8A8_SINT
        | Format::R16_SINT
        | Format::R16G16_SINT
        | Format::R16G16B16_SINT
        | Format::R16G16B16A16_SINT
        | Format::R32_SINT
        | Format::R32G32_SINT
        | Format::R32G32B32_SINT
        | Format::R32G32B32A32_SINT
        | Format::R64_SINT
        | Format::R64G64_SINT
        | Format::R64G64B64_SINT
        | Format::R64G64B64A64_SINT => FormatNumericType::SInt,

        Format::R16_SFLOAT
        | Format::R16G16_SFLOAT
        | Format::R16G16B16_SFLOAT
        | Format::R16G16B16A16_SFLOAT
        | Format::R32_SFLOAT
        | Format::R32G32_SFLOAT
        | Format::R32G32B32_SFLOAT
        | Format::R32G32B32A32_SFLOAT
        | Format::R64_SFLOAT
        | Format::R64G64_SFLOAT
        | Format::R64G64B64_SFLOAT
        | Format::R64G64B64A64_SFLOAT => FormatNumericType::Float,

        // TODO
        _ => FormatNumericType::Float,
    }
}

/// Returns the byte size of one pixel in the specified format.
///
/// # Panics
///
/// Panics if the format is a block-compressed format.
///
pub fn format_pixel_byte_size(fmt: vk::Format) -> u32 {
    match fmt {
        Format::R8_UNORM
        | Format::R8_SNORM
        | Format::R8_USCALED
        | Format::R8_SSCALED
        | Format::R8_UINT
        | Format::R8_SINT
        | Format::R8_SRGB => 1,
        Format::R8G8_UNORM
        | Format::R8G8_SNORM
        | Format::R8G8_USCALED
        | Format::R8G8_SSCALED
        | Format::R8G8_UINT
        | Format::R8G8_SINT
        | Format::R8G8_SRGB => 2,
        Format::R5G6B5_UNORM_PACK16
        | Format::B5G6R5_UNORM_PACK16
        | Format::R5G5B5A1_UNORM_PACK16
        | Format::B5G5R5A1_UNORM_PACK16
        | Format::A1R5G5B5_UNORM_PACK16
        | Format::R16_UNORM
        | Format::R16_SNORM
        | Format::R16_USCALED
        | Format::R16_SSCALED
        | Format::R16_UINT
        | Format::R16_SINT
        | Format::R16_SFLOAT => 2,
        Format::R8G8B8_UNORM
        | Format::R8G8B8_SNORM
        | Format::R8G8B8_USCALED
        | Format::R8G8B8_SSCALED
        | Format::R8G8B8_UINT
        | Format::R8G8B8_SINT
        | Format::R8G8B8_SRGB
        | Format::B8G8R8_UNORM
        | Format::B8G8R8_SNORM
        | Format::B8G8R8_USCALED
        | Format::B8G8R8_SSCALED
        | Format::B8G8R8_UINT
        | Format::B8G8R8_SINT
        | Format::B8G8R8_SRGB => 3,
        Format::R32_UINT | Format::R32_SINT | Format::R32_SFLOAT | Format::D32_SFLOAT | Format::D24_UNORM_S8_UINT => 4,
        Format::R8G8B8A8_UNORM
        | Format::R8G8B8A8_SNORM
        | Format::R8G8B8A8_USCALED
        | Format::R8G8B8A8_SSCALED
        | Format::R8G8B8A8_UINT
        | Format::R8G8B8A8_SINT
        | Format::R8G8B8A8_SRGB
        | Format::B8G8R8A8_UNORM
        | Format::B8G8R8A8_SNORM
        | Format::B8G8R8A8_USCALED
        | Format::B8G8R8A8_SSCALED
        | Format::B8G8R8A8_UINT
        | Format::B8G8R8A8_SINT
        | Format::B8G8R8A8_SRGB
        | Format::A2B10G10R10_UNORM_PACK32
        | Format::A2B10G10R10_UINT_PACK32
        | Format::A2R10G10B10_UNORM_PACK32
        | Format::A2R10G10B10_UINT_PACK32
        | Format::R16G16_UNORM
        | Format::R16G16_SNORM
        | Format::R16G16_USCALED
        | Format::R16G16_SSCALED
        | Format::R16G16_UINT
        | Format::R16G16_SINT
        | Format::R16G16_SFLOAT => 4,
        Format::R32G32_UINT | Format::R32G32_SINT | Format::R32G32_SFLOAT => 8,
        Format::R32G32B32A32_SFLOAT | Format::R32G32B32A32_UINT | Format::R32G32B32A32_SINT => 16,
        _ => panic!("unsupported or block-compressed format"),
    }
}

/*
fn map_buffer_access_to_barrier(state: BufferAccess) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let mut stages = vk::PipelineStageFlags2::empty();
    let mut access = vk::AccessFlags2::empty();
    let shader_stages = vk::PipelineStageFlags2::VERTEX_SHADER
        | vk::PipelineStageFlags2::FRAGMENT_SHADER
        | vk::PipelineStageFlags2::COMPUTE_SHADER;

    if state.contains(BufferAccess::MAP_READ) {
        stages |= vk::PipelineStageFlags2::HOST;
        access |= vk::AccessFlags2::HOST_READ;
    }
    if state.contains(BufferAccess::MAP_WRITE) {
        stages |= vk::PipelineStageFlags2::HOST;
        access |= vk::AccessFlags2::HOST_WRITE;
    }
    if state.contains(BufferAccess::COPY_SRC) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_READ;
    }
    if state.contains(BufferAccess::COPY_DST) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_WRITE;
    }
    if state.contains(BufferAccess::UNIFORM) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::UNIFORM_READ;
    }
    if state.intersects(BufferAccess::STORAGE_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if state.intersects(BufferAccess::STORAGE_READ_WRITE) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
    }
    if state.contains(BufferAccess::INDEX) {
        stages |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access |= vk::AccessFlags2::INDEX_READ;
    }
    if state.contains(BufferAccess::VERTEX) {
        stages |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
    }
    if state.contains(BufferAccess::INDIRECT) {
        stages |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        access |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }

    (stages, access)
}

fn map_image_access_to_barrier(state: ImageAccess) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let mut stages = vk::PipelineStageFlags2::empty();
    let mut access = vk::AccessFlags2::empty();
    let shader_stages = vk::PipelineStageFlags2::VERTEX_SHADER
        | vk::PipelineStageFlags2::FRAGMENT_SHADER
        | vk::PipelineStageFlags2::COMPUTE_SHADER;

    if state.contains(ImageAccess::COPY_SRC) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_READ;
    }
    if state.contains(ImageAccess::COPY_DST) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_WRITE;
    }
    if state.contains(ImageAccess::SAMPLED_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if state.contains(ImageAccess::COLOR_TARGET) {
        stages |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
    }
    if state.intersects(ImageAccess::DEPTH_STENCIL_READ) {
        stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    if state.intersects(ImageAccess::DEPTH_STENCIL_WRITE) {
        stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if state.contains(ImageAccess::IMAGE_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if state.contains(ImageAccess::IMAGE_READ_WRITE) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
    }

    if state == ImageAccess::UNINITIALIZED || state == ImageAccess::PRESENT {
        (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty())
    } else {
        (stages, access)
    }
}

fn map_image_access_to_layout(access: ImageAccess, format: Format) -> vk::ImageLayout {
    let is_color = aspects_for_format(format).contains(vk::ImageAspectFlags::COLOR);
    match access {
        ImageAccess::UNINITIALIZED => vk::ImageLayout::UNDEFINED,
        ImageAccess::COPY_SRC => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ImageAccess::COPY_DST => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ImageAccess::SAMPLED_READ if is_color => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ImageAccess::COLOR_TARGET => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ImageAccess::DEPTH_STENCIL_WRITE => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        _ => {
            if access == ImageAccess::PRESENT {
                vk::ImageLayout::PRESENT_SRC_KHR
            } else if is_color {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            }
        }
    }
}*/

// Implementation detail of the VertexInput macro
#[doc(hidden)]
pub const fn append_attributes<const N: usize>(
    head: &'static [VertexInputAttributeDescription],
    binding: u32,
    base_location: u32,
    tail: &'static [VertexAttributeDescription],
) -> [VertexInputAttributeDescription; N] {
    const NULL_ATTR: VertexInputAttributeDescription = VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: Format::UNDEFINED,
        offset: 0,
    };
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
