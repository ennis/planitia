use crate::{BufferRange, BufferUsage, Device, Ptr, ResourceAllocation, ResourceId, TrackedResource};
use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use log::{trace, warn};
use std::collections::Bound;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::RangeBounds;
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::{mem, ptr, slice};

impl<T: ?Sized> Drop for Buffer<T> {
    fn drop(&mut self) {
        let mut allocation = mem::take(&mut self.allocation);
        let handle = self.handle;

        Device::global().delete_resource_after_current_submission(self.id, move |device| unsafe {
            trace!("GPU: deleting buffer: {:?}", handle);
            device.raw.destroy_buffer(handle, None);
            device.free_memory(&mut allocation);
        });
    }
}

/// Buffer creation info.
#[derive(Copy, Clone, Debug)]
pub struct BufferCreateInfo<'a> {
    /// Length in number of elements.
    pub len: usize,
    /// Usage flags. Must include all intended uses of the buffer.
    pub usage: BufferUsage = BufferUsage::empty(),
    pub memory_location: MemoryLocation = MemoryLocation::Unknown,
    /// Debug label.
    pub label: &'a str = "",
}

/// A buffer of GPU-visible memory, optionally mapped in host memory, without any associated type.
///
/// `Buffer` objects themselves are Send+Sync, however access to the buffer contents on the host
/// (via raw pointers) is not thread-safe.
/// If the mapped contents are accessed from multiple threads, the caller must
/// ensure proper synchronization.
pub struct Buffer<T: ?Sized> {
    pub(crate) id: ResourceId,
    pub(crate) memory_location: MemoryLocation,
    pub(crate) allocation: ResourceAllocation,
    pub(crate) device_address: vk::DeviceAddress,
    pub(crate) handle: vk::Buffer,
    pub(crate) size: u64,
    pub(crate) usage: BufferUsage,
    pub(crate) mapped_ptr: Option<NonNull<c_void>>,
    _marker: PhantomData<T>,
}

// Safe to send to another thread (buffers do not have shared reference semantics).
unsafe impl<T: ?Sized> Send for Buffer<T> {}
// References can be shared between threads (access to the content of the buffer on the CPU
// is done via unsafe methods, and the called should ensure proper synchronization).
unsafe impl<T: ?Sized> Sync for Buffer<T> {}

impl<T: Copy> Buffer<[T]> {
    /// Creates a new buffer with the specified number of elements.
    pub fn new(create_info: BufferCreateInfo) -> Self {
        let buffer = Device::global().create_buffer(size_of::<T>(), create_info);
        unsafe {
            // SAFETY: the buffer is large enough to hold `len` elements of type `T`
            buffer.cast()
        }
    }

    /// Creates a CpuToGpu buffer and copies data into it.
    pub fn from_slice(data: &[T], label: &str) -> Buffer<[T]> {
        Buffer::from_slice_with_usage(BufferUsage::default(), data, label)
    }

    /// Creates a CpuToGpu buffer and copies data into it.
    pub fn from_slice_with_usage(usage: BufferUsage, data: &[T], label: &str) -> Buffer<[T]> {
        let buffer = Buffer::new(BufferCreateInfo {
            len: data.len(),
            usage,
            label,
            memory_location: MemoryLocation::CpuToGpu,
        });
        unsafe {
            // copy data to mapped buffer
            ptr::copy_nonoverlapping(data.as_ptr(), buffer.as_mut_ptr() as *mut T, data.len());
        }
        buffer
    }
}

// TODO: non-slice buffers are rarely used. Consider removing them, and have `Buffer<T>` be
//       analogous to `Vec<T>`.
impl<T: Copy> Buffer<T> {
    /// Creates a CpuToGpu buffer and copies data into it.
    pub fn from_data(data: &T, label: &str) -> Buffer<T> {
        Buffer::from_data_with_usage(BufferUsage::default(), data, label)
    }

    /// Creates a CpuToGpu buffer and copies data into it.
    pub fn from_data_with_usage(usage: BufferUsage, data: &T, label: &str) -> Buffer<T> {
        let buffer = Buffer::from_slice_with_usage(usage, slice::from_ref(data), label);
        buffer.single()
    }
}

impl<T: ?Sized> Buffer<T> {
    /// Returns the device address of the buffer, for use in shaders.
    pub fn ptr(&self) -> Ptr<T> {
        Ptr {
            raw: self.device_address,
            _phantom: PhantomData,
        }
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
        self.memory_location
    }

    /// Returns the Vulkan buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.handle
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
    /// TODO: raw slice pointers aren't very ergonomic right now. Maybe return `*mut T` instead?
    pub fn as_mut_ptr(&self) -> *mut [T] {
        ptr::slice_from_raw_parts_mut(self.as_mut_ptr_u8() as *mut T, self.len())
    }

    /// If the buffer is mapped in host memory, returns an uninitialized slice of the buffer's elements.
    ///
    /// # Safety
    ///
    /// - ~~All other slices returned by `as_mut_slice` on aliases of this `Buffer` must have been dropped.~~
    /// - The caller must ensure that nothing else is writing to the buffer while the slice is being accessed.
    ///   i.e. all GPU operations on the buffer have completed.
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

/*
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
}*/

// TODO: impl From<IntoIterator> for Buffer

impl<T: ?Sized> std::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("handle", &self.handle)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl<T: ?Sized> TrackedResource for Buffer<T> {
    fn id(&self) -> ResourceId {
        self.id
    }
}

pub type BufferUntyped = Buffer<[u8]>;

impl<'a, T: Copy> From<&'a Buffer<[T]>> for BufferRange<'a, T> {
    fn from(buffer: &'a Buffer<[T]>) -> Self {
        buffer.slice(..)
    }
}

/// Buffer creation.
impl Device {
    /// Creates a new buffer resource.
    pub(crate) fn create_buffer(&self, elem_size: usize, create_info: BufferCreateInfo) -> BufferUntyped {
        //assert!(create_info.len > 0, "buffer size must be greater than zero");

        if create_info.len == 0 || elem_size == 0 {
            warn!(
                "creating a zero-sized buffer: this is not supported by Vulkan and a minimum size of 1 byte will be used"
            );
        }
        let byte_size = elem_size as u64 * create_info.len as u64;

        // We support zero-sized buffers, but Vulkan doesn't, so we enforce a minimum size of 1 byte.
        // The length exposed to the user is still zero.
        const MINIMUM_BUFFER_SIZE: u64 = 1;
        let nonzero_byte_size = byte_size.max(MINIMUM_BUFFER_SIZE);

        // The following flags have no specific logic associated to them in Mesa drivers (at least for desktop GPUs):
        // - VERTEX_BUFFER
        // - INDEX_BUFFER
        // - TRANSFER_SRC
        // - TRANSFER_DST
        // - INDIRECT_BUFFER
        // - SHADER_DEVICE_ADDRESS
        // So we include them by default, that's one less thing for the user to worry about.
        //
        // STORAGE_BUFFER and UNIFORM_BUFFER are only used to impose additional alignment requirements:
        // usually 16 bytes for STORAGE and 64 bytes for UNIFORM.
        // We include STORAGE_BUFFER in the default usage flags, as the impact should be minimal,
        // but we leave out UNIFORM_BUFFER for now (it's not commonly used anymore since we mostly use
        // push constants + BDA with storage buffers).
        let default_flags = vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER;

        unsafe {
            let handle = self
                .raw
                .create_buffer(
                    &vk::BufferCreateInfo {
                        flags: Default::default(),
                        size: nonzero_byte_size,
                        usage: default_flags | create_info.usage.to_vk_buffer_usage_flags(),
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        queue_family_index_count: 0,
                        p_queue_family_indices: ptr::null(),
                        ..Default::default()
                    },
                    None,
                )
                .expect("failed to create buffer");

            let mem_req = self.raw.get_buffer_memory_requirements(handle);
            let allocation = self.allocate_memory_or_panic(&AllocationCreateDesc {
                name: "",
                requirements: mem_req,
                location: create_info.memory_location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
            self.raw
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
                .unwrap();

            let mapped_ptr = allocation.mapped_ptr();
            let allocation = ResourceAllocation::Allocation { allocation };

            let device_address = self.raw.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                buffer: handle,
                ..Default::default()
            });

            if !create_info.label.is_empty() {
                // SAFETY: no concurrent access possible
                self.set_object_name(handle, create_info.label);
            }

            let id = self.allocate_resource_id();
            trace!("GPU: create buffer {:?} {:?} {id:?}", handle, create_info.label);

            BufferUntyped {
                id,
                allocation,
                handle,
                memory_location: create_info.memory_location,
                device_address,
                size: byte_size,
                usage: create_info.usage,
                mapped_ptr,
                _marker: PhantomData,
            }
        }
    }
}
