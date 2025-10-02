#![feature(default_field_values)]
mod command;
mod device;
mod instance;
pub mod platform;
mod surface;
mod types;
pub mod util;

use std::borrow::Cow;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Bound, RangeBounds};
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::{mem, ptr};

// --- reexports ---

// TODO: make it optional
pub use ash::{self, vk};
pub use gpu_allocator::MemoryLocation;
pub use ordered_float;

pub use command::*;
pub use device::*;
pub use instance::*;
pub use surface::*;
pub use types::*;
// proc-macros
pub use gpu_macros::Vertex;

pub mod prelude {
    pub use crate::util::{CommandStreamExt, DeviceExt};
    pub use crate::{
        vk, Buffer, BufferUsage, ClearColorValue, ColorBlendEquation, ColorTargetState, CommandStream, ComputeEncoder,
        DepthStencilState, Format, FragmentState, GraphicsPipeline, GraphicsPipelineCreateInfo, Image, ImageCreateInfo,
        ImageType, ImageUsage, MemoryLocation, PipelineBindPoint, PipelineLayoutDescriptor, Point2D,
        PreRasterizationShaders, RasterizationState, RcDevice, Rect2D, RenderEncoder, Sampler, SamplerCreateInfo,
        ShaderCode, ShaderEntryPoint, ShaderSource, Size2D, StencilState, Vertex, VertexBufferDescriptor,
        VertexBufferLayoutDescription, VertexInputAttributeDescription, VertexInputState,
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Standard subgroup size.
pub const SUBGROUP_SIZE: u32 = 32;

/// Device address of a GPU buffer.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DeviceAddressUntyped {
    pub address: vk::DeviceAddress,
}

/// Device address of a GPU buffer containing elements of type `T` its associated type.
///
/// The type should be `T: Copy` for a buffer containing a single element of type T,
/// or `[T] where T: Copy` for slices of elements of type T.
#[repr(transparent)]
pub struct DeviceAddress<T: ?Sized + 'static> {
    pub address: vk::DeviceAddress,
    pub _phantom: PhantomData<T>,
}

impl<T: ?Sized + 'static> DeviceAddress<T> {
    /// Null (invalid) device address.
    pub const NULL: Self = DeviceAddress {
        address: 0,
        _phantom: PhantomData,
    };
}

impl<T: 'static> DeviceAddress<[T]> {
    pub fn offset(self, offset: usize) -> Self {
        DeviceAddress {
            address: self.address + (offset * size_of::<T>()) as u64,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized + 'static> Clone for DeviceAddress<T> {
    fn clone(&self) -> Self {
        DeviceAddress {
            address: self.address,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized + 'static> Copy for DeviceAddress<T> {}

/// Bindless handle to an image.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct ImageHandle {
    /// Index of the image in the image descriptor array.
    pub index: u32,
    /// For compatibility with slang.
    _unused: u32,
}

impl ImageHandle {
    pub const INVALID: Self = ImageHandle {
        index: u32::MAX,
        _unused: 0,
    };
}

/// Bindless handle to a 2D texture.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Texture2DHandle(pub ImageHandle);

impl Texture2DHandle {
    pub const INVALID: Self = Texture2DHandle(ImageHandle::INVALID);
}

/// Represents a range of bindless handles to 2D sampled images.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Texture2DHandleRange {
    pub index: u32,
    pub count: u32,
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

#[derive(Debug)]
struct SwapchainImageInner {
    image: Image,
    render_finished: vk::Semaphore,
}

/// Represents a swap chain.
#[derive(Debug)]
pub struct SwapChain {
    pub handle: vk::SwapchainKHR,
    pub surface: vk::SurfaceKHR,
    pub format: vk::SurfaceFormatKHR,
    pub width: u32,
    pub height: u32,
    images: Vec<SwapchainImageInner>,
}

/// Contains information about an image in a swapchain.
#[derive(Debug)]
pub struct SwapchainImage {
    /// Handle of the swapchain that owns this image.
    pub swapchain: vk::SwapchainKHR,
    /// Index of the image in the swap chain.
    pub index: u32,
    pub image: Image,
    /// Used internally by `present` to synchronize rendering to presentation.
    render_finished: vk::Semaphore,
}

/// Graphics pipelines.
///
/// TODO Drop impl
#[derive(Clone)]
pub struct GraphicsPipeline {
    pub(crate) device: RcDevice,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    // Push descriptors require live VkDescriptorSetLayouts (kill me already)
    _descriptor_set_layouts: Vec<DescriptorSetLayout>,
    pub(crate) bindless: bool,
}

impl GraphicsPipeline {
    pub fn set_name(&self, label: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.device.set_object_name(self.pipeline, label);
        }
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

/// Compute pipelines.
///
/// TODO Drop impl
#[derive(Clone)]
pub struct ComputePipeline {
    pub(crate) device: RcDevice,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    _descriptor_set_layouts: Vec<DescriptorSetLayout>,
    pub(crate) bindless: bool,
}

impl ComputePipeline {
    pub fn set_name(&self, label: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.device.set_object_name(self.pipeline, label);
        }
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

/// Samplers
#[derive(Clone, Debug)]
pub struct Sampler {
    // A weak ref is sufficient, the device already owns samplers in its cache
    device: WeakDevice,
    id: SamplerId,
    sampler: vk::Sampler,
}

impl Sampler {
    pub fn set_name(&self, label: &str) {
        unsafe {
            self.device
                .upgrade()
                .expect("the underlying device of this sampler has been destroyed")
                .set_object_name(self.sampler, label);
        }
    }

    pub fn handle(&self) -> vk::Sampler {
        let _device = self
            .device
            .upgrade()
            .expect("the underlying device of this sampler has been destroyed");
        self.sampler
    }

    pub fn descriptor(&self) -> Descriptor<'_> {
        Descriptor::Sampler { sampler: self.clone() }
    }

    pub fn device_handle(&self) -> SamplerHandle {
        SamplerHandle {
            index: self.id.index(),
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

pub trait GpuResource {
    fn set_last_submission_index(&self, submission_index: u64);
}

#[derive(Debug)]
struct BufferInner {
    device: RcDevice,
    id: BufferId,
    memory_location: MemoryLocation,
    last_submission_index: AtomicU64,
    allocation: ResourceAllocation,
    handle: vk::Buffer,
    device_address: vk::DeviceAddress,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: The device resource tracker holds strong references to resources as long as they are in use by the GPU.
        // This prevents `drop` from being called while the resource is still in use, and thus it's safe to delete the
        // resource here.
        unsafe {
            // retire the ID
            self.device.buffer_ids.lock().unwrap().remove(self.id);
            self.device.free_memory(&mut self.allocation);
            self.device.raw.destroy_buffer(self.handle, None);
        }
    }
}

/// A buffer of GPU-visible memory, optionally mapped in host memory, without any associated type.
///
/// TODO: maybe this could be `Buffer<[u8]>` instead of a separate type?
pub struct Buffer<T: ?Sized> {
    inner: Option<Arc<BufferInner>>,
    handle: vk::Buffer,
    size: u64,
    usage: BufferUsage,
    mapped_ptr: Option<NonNull<c_void>>,
    _marker: PhantomData<T>,
}

impl<T: ?Sized> Buffer<T> {
    pub fn set_name(&self, name: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.inner.as_ref().unwrap().device.set_object_name(self.handle, name);
        }
    }

    pub fn device_address(&self) -> DeviceAddress<T> {
        DeviceAddress {
            address: self.inner.as_ref().unwrap().device_address,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn id(&self) -> BufferId {
        self.inner.as_ref().unwrap().id
    }

    /// Returns the size of the buffer in bytes.
    pub fn byte_size(&self) -> u64 {
        self.size
    }

    /// Returns the usage flags of the buffer.
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Returns the buffer's memory location.
    pub fn memory_location(&self) -> MemoryLocation {
        self.inner.as_ref().unwrap().memory_location
    }

    /// Returns the Vulkan buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &RcDevice {
        &self.inner.as_ref().unwrap().device
    }

    /// Returns whether the buffer is host-visible, and mapped in host memory.
    pub fn host_visible(&self) -> bool {
        self.mapped_ptr.is_some()
    }

    /// Returns an untyped reference (`&Buffer<[u8]>`) of the buffer.
    pub fn as_bytes(&self) -> &BufferUntyped {
        // SAFETY: Buffer<T> where T:?Sized has the same layout as BufferUntyped (Buffer<[u8]>), and no stricter
        // alignment constraints for host pointers.
        unsafe { mem::transmute(self) }
    }

    pub fn as_mut_ptr_u8(&self) -> *mut u8 {
        self.mapped_ptr
            .expect("buffer was not mapped in host memory (consider using MemoryLocation::CpuToGpu)")
            .as_ptr() as *mut u8
    }
}

impl<T: Copy> Buffer<T> {
    /// Returns a pointer to the buffer mapped in host memory. Panics if the buffer was not mapped in
    /// host memory.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.as_mut_ptr_u8() as *mut T
    }
}

impl<T: Copy> Buffer<[T]> {
    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        (self.byte_size() / size_of::<T>() as u64) as usize
    }

    fn check_valid_cast<U: Copy>(&self) {
        assert!(
            self.byte_size() % size_of::<U>() as u64 == 0,
            "buffer size is not a multiple of the element size"
        );

        // Check that the host pointer is correctly aligned for T.
        if let Some(ptr) = self.mapped_ptr {
            assert!(
                ptr.addr().get() & (align_of::<U>() - 1) == 0,
                "mapped buffer pointer is not correctly aligned for the element type"
            );
        }
    }

    /// Re-interprets this buffer as a single value of type `T`. Only valid if `self.len() == 1`.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != 1`.
    pub fn as_single(&self) -> &Buffer<T> {
        assert!(self.len() == 1, "buffer does not contain exactly one element");
        // SAFETY: Buffer<[T]> and Buffer<T> have the same layout if the size matches
        unsafe { mem::transmute(self) }
    }

    /// Re-interprets this buffer as a single value of type `T`. Only valid if `self.len() == 1`.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != 1`.
    pub fn single(self) -> Buffer<T> {
        assert!(self.len() == 1, "buffer does not contain exactly one element");
        // SAFETY: Buffer<[T]> and Buffer<T> have the same layout if the size matches
        unsafe { mem::transmute(self) }
    }

    /// Casts the buffer to another element type.
    ///
    /// # Safety
    ///
    /// If the buffer is host mapped, the caller must ensure that casting the buffer to `&[U]` is
    /// valid, that is, either:
    /// * the buffer actually contains elements of type `U`
    /// * OR objects of type `U` are valid for any underlying bit pattern of the correct size
    ///   (see [bytemuck's documentation for AnyBitPattern](https://docs.rs/bytemuck/1.23.2/bytemuck/trait.AnyBitPattern.html) for details)
    ///
    /// # Panics
    ///
    /// Panics if the buffer size is not a multiple of the element size,
    /// or if the host pointer is not correctly aligned for the element type.
    pub unsafe fn as_cast<U: Copy>(&self) -> &Buffer<[U]> {
        self.check_valid_cast::<U>();
        // SAFETY: Buffer<[T]> and Buffer<[U]> have the same layout for all T and U
        mem::transmute(self)
    }

    /// Casts the buffer to another element type.
    ///
    /// # Safety
    ///
    /// If the buffer is host mapped, the caller must ensure that casting the buffer to `&[U]` is
    /// valid, that is, either:
    /// * the buffer actually contains elements of type `U`
    /// * OR objects of type `U` are valid for any underlying bit pattern of the correct size
    ///   (see [bytemuck's documentation for AnyBitPattern](https://docs.rs/bytemuck/1.23.2/bytemuck/trait.AnyBitPattern.html) for details)
    ///
    /// # Panics
    ///
    /// Panics if the buffer size is not a multiple of the element size,
    /// or if the host pointer is not correctly aligned for the element type.
    pub unsafe fn cast<U: Copy>(self) -> Buffer<[U]> {
        self.check_valid_cast::<U>();
        // SAFETY: Buffer<[T]> and Buffer<[U]> have the same layout for all T and U
        mem::transmute(self)
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn as_mut_ptr(&self) -> *mut [T] {
        ptr::slice_from_raw_parts_mut(self.as_mut_ptr_u8() as *mut T, self.len())
    }

    /// If the buffer is mapped in host memory, returns an uninitialized slice of the buffer's elements.
    ///
    /// # Safety
    ///
    /// - All other slices returned by `as_mut_slice` on aliases of this `Buffer` must have been dropped.
    /// - The caller must ensure that nothing else is writing to the buffer while the slice is being accessed.
    ///   i.e. all GPU operations on the buffer have completed.
    ///
    /// FIXME: the first safety condition is hard to track since `Buffer`s have shared ownership.
    ///        Maybe `Buffer`s should have unique ownership instead, i.e. don't make them `Clone`.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut _, self.len()) }
    }

    /// Element range.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> BufferRange<'_, T> {
        let elem_size = size_of::<T>();
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
        assert!(start <= self.size && end <= self.size);

        BufferRange {
            buffer: self,
            byte_offset: start,
            byte_size: end - start,
        }
    }
}

/*
impl Buffer<[u8]> {
    /// Re-interprets this buffer as a single value of type `T`.
    ///
    /// # Safety
    ///
    /// This has the same requirements as `cast<U>()`.
    pub unsafe fn as_cast_from_bytes<T: Copy>(&self) -> &Buffer<T> {
        self.check_valid_cast::<T>();
        // SAFETY: Buffer<[u8]> and Buffer<T> have the same layout if the size matches
        unsafe { mem::transmute(self) }
    }
}*/

impl<T: ?Sized> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            handle: self.handle,
            size: self.size,
            usage: self.usage,
            mapped_ptr: self.mapped_ptr,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> std::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("handle", &self.handle)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl<T: ?Sized> GpuResource for Buffer<T> {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.inner
            .as_ref()
            .unwrap()
            .last_submission_index
            .fetch_max(submission_index, Ordering::Release);
    }
}

impl<T: ?Sized> Drop for Buffer<T> {
    fn drop(&mut self) {
        // If this is the last reference to the buffer, schedule it for deletion once the last
        // submission that used this buffer has completed.
        if let Some(inner) = Arc::into_inner(self.inner.take().unwrap()) {
            let last_submission_index = inner.last_submission_index.load(Ordering::Relaxed);
            inner.device.clone().delete_later(last_submission_index, inner);
        }
    }
}

pub type BufferUntyped = Buffer<[u8]>;

impl<'a, T: Copy> From<&'a Buffer<[T]>> for BufferRange<'a, T> {
    fn from(buffer: &'a Buffer<[T]>) -> Self {
        buffer.slice(..)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Images

#[derive(Debug)]
struct ImageInner {
    device: RcDevice,
    id: ImageId,
    last_submission_index: AtomicU64,
    allocation: ResourceAllocation,
    handle: vk::Image,
    swapchain_image: bool,
    /// Default image view handle.
    default_view: vk::ImageView,
    /// Bindless index (of the default view).
    bindless_handle: ImageViewId,
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        if !self.swapchain_image {
            unsafe {
                //debug!("dropping image {:?} (handle: {:?})", self.id, self.handle);
                self.device.image_ids.lock().unwrap().remove(self.id);
                self.device.image_view_ids.lock().unwrap().remove(self.bindless_handle);
                self.device.free_memory(&mut self.allocation);
                self.device.raw.destroy_image_view(self.default_view, None);
                self.device.raw.destroy_image(self.handle, None);
            }
        }
    }
}

/// Image data stored in CPU-visible memory.
pub struct ImageBuffer {
    /// Host-mapped buffer containing the image data.
    data: BufferUntyped,
    format: Format,
    pitch: u32,
    width: u32,
    height: u32,
    depth: u32,
}

/// Wrapper around a Vulkan image.
#[derive(Clone, Debug)]
pub struct Image {
    inner: Option<Arc<ImageInner>>,
    handle: vk::Image,
    usage: ImageUsage,
    type_: ImageType,
    format: Format,
    size: Size3D,
}

impl Drop for Image {
    fn drop(&mut self) {
        if let Some(inner) = Arc::into_inner(self.inner.take().unwrap()) {
            let last_submission_index = inner.last_submission_index.load(Ordering::Relaxed);
            inner.device.clone().delete_later(last_submission_index, inner);
        }
    }
}

impl GpuResource for Image {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.inner
            .as_ref()
            .unwrap()
            .last_submission_index
            .fetch_max(submission_index, Ordering::Release);
    }
}

impl Image {
    pub fn set_name(&self, label: &str) {
        unsafe {
            self.inner.as_ref().unwrap().device.set_object_name(self.handle, label);
        }
    }

    /// Returns the type (dimensionality) of the image.
    pub fn image_type(&self) -> ImageType {
        self.type_
    }

    /// Returns the format of the image.
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the size in pixels of the image.
    pub fn size(&self) -> Size3D {
        self.size
    }

    /// Returns the width of the image.
    pub fn width(&self) -> u32 {
        self.size.width
    }

    /// Returns the height of the image.
    ///
    /// This is 1 for 1D images.
    pub fn height(&self) -> u32 {
        self.size.height
    }

    /// Returns the depth of the image.
    ///
    /// This is 1 for 1D & 2D images.
    pub fn depth(&self) -> u32 {
        self.size.depth
    }

    /// Returns the usage flags of the image.
    pub fn usage(&self) -> ImageUsage {
        self.usage
    }

    /// Returns the internal tracking ID of the image.
    pub(crate) fn id(&self) -> ImageId {
        self.inner.as_ref().unwrap().id
    }

    /// Returns the image handle.
    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    /// Returns the handle of the default image view.
    pub fn view_handle(&self) -> vk::ImageView {
        self.inner.as_ref().unwrap().default_view
    }

    pub fn device(&self) -> &RcDevice {
        &self.inner.as_ref().unwrap().device
    }

    /// Returns a descriptor for sampling this image in a shader.
    pub fn texture_descriptor(&self, layout: vk::ImageLayout) -> Descriptor<'_> {
        Descriptor::SampledImage { image: self, layout }
    }

    /// Returns a descriptor for accessing this image as a storage image in a shader.
    pub fn storage_image_descriptor(&self, layout: vk::ImageLayout) -> Descriptor<'_> {
        Descriptor::StorageImage { image: self, layout }
    }

    /// Returns the bindless texture handle of this image view.
    pub fn device_image_handle(&self) -> ImageHandle {
        ImageHandle {
            index: self.inner.as_ref().unwrap().bindless_handle.index(),
            _unused: 0,
        }
    }

    /*/// Creates an image view for the base mip level of this image,
    /// suitable for use as a rendering attachment.
    pub fn create_top_level_view(&self) -> ImageView {
        self.create_view(&ImageViewInfo {
            view_type: match self.image_type() {
                ImageType::Image2D => vk::ImageViewType::TYPE_2D,
                _ => panic!("unsupported image type for attachment"),
            },
            format: self.format(),
            subresource_range: ImageSubresourceRange {
                aspect_mask: aspects_for_format(self.format()),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            component_mapping: [
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
            ],
        })
    }*/

    /*/// Creates an `ImageView` object.
    pub(crate) fn create_view(&self, info: &ImageViewInfo) -> ImageView {
        self.inner.as_ref().unwrap().device.create_image_view(self, info)
    }*/
}

/*
#[derive(Debug)]
struct ImageViewInner {
    // Don't hold Arc<ImageInner> here
    //
    // 1. create the image view
    // 2. use it in a submission (#1)
    // 3. use the image in a later submission (#2)
    // 4. drop the image ref -> not added to the deferred deletion list because the image view still holds a reference
    // ImageView now holds the last ref
    // 5. drop the ImageView -> image view added to the deferred deletion list
    // 6. ImageView deleted when #1 finishes, along with the image since it holds the last ref,
    //    but the image might still be in use by #2!
    image: Image,
    id: ImageViewId,
    handle: vk::ImageView,
    last_submission_index: AtomicU64,
}

impl Drop for ImageViewInner {
    fn drop(&mut self) {
        unsafe {
            self.image.device().image_view_ids.lock().unwrap().remove(self.id);
            self.image.device().raw.destroy_image_view(self.handle, None);
        }
    }
}

/// A view over an image subresource or subresource range.
#[derive(Clone, Debug)]
pub struct ImageView {
    inner: Option<Arc<ImageViewInner>>,
    handle: vk::ImageView,
    format: Format,
    size: Size3D,
}

impl Drop for ImageView {
    fn drop(&mut self) {
        if let Some(inner) = Arc::into_inner(self.inner.take().unwrap()) {
            let last_submission_index = inner.last_submission_index.load(Ordering::Relaxed);
            inner.image.device().clone().delete_later(last_submission_index, inner);
        }
    }
}

impl GpuResource for ImageView {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.inner
            .as_ref()
            .unwrap()
            .last_submission_index
            .fetch_max(submission_index, Ordering::Release);
    }
}

impl ImageView {
    /// Returns the format of the image view.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn size(&self) -> Size3D {
        self.size
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn handle(&self) -> vk::ImageView {
        self.handle
    }

    pub fn set_name(&self, label: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.image().device().set_object_name(self.handle, label);
        }
    }

    pub fn image(&self) -> &Image {
        &self.inner.as_ref().unwrap().image
    }

    pub(crate) fn id(&self) -> ImageViewId {
        self.inner.as_ref().unwrap().id
    }

    /*pub fn as_device_texture_2d_handle(&self) -> Texture2DHandle {
        Texture2DHandle(self.device_image_handle())
    }*/
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct DescriptorSetLayout {
    device: RcDevice,
    last_submission_index: Option<Arc<AtomicU64>>,
    pub handle: vk::DescriptorSetLayout,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        if let Some(last_submission_index) = Arc::into_inner(self.last_submission_index.take().unwrap()) {
            let device = self.device.clone();
            let handle = self.handle;
            self.device
                .call_later(last_submission_index.load(Ordering::Relaxed), move || unsafe {
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
    pub mip_level: u32,
    pub origin: vk::Offset3D,
    pub aspect: vk::ImageAspectFlags,
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
    pub clear_value: Option<[f64; 4]>,
    /*pub image_view: ImageView,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: [f64; 4],*/
}

impl ColorAttachment<'_> {
    pub(crate) fn get_vk_clear_color_value(&self) -> vk::ClearColorValue {
        if let Some(clear_value) = self.clear_value {
            match format_numeric_type(self.image.format) {
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
    pub depth_clear_value: Option<f64>,
    pub stencil_clear_value: Option<u32>,
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
