use crate::{Buffer, BufferUsage, Device, MemoryLocation};
use std::rc::Rc;
use std::{mem, ptr, slice};

// TODO: move this directly into DeviceInner, there's no reason to have it in an extension trait
//       since this is defined in the same crate
pub trait DeviceExt {
    fn upload<T: Copy>(self: &Rc<Self>, usage: BufferUsage, data: &T) -> Buffer<T>;
    fn create_array_buffer<T: Copy>(
        self: &Rc<Self>,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        len: usize,
    ) -> Buffer<[T]>;
    fn upload_slice<T: Copy>(self: &Rc<Self>, usage: BufferUsage, data: &[T]) -> Buffer<[T]>;
}

impl DeviceExt for Device {
    fn upload<T: Copy>(self: &Rc<Self>, usage: BufferUsage, data: &T) -> Buffer<T> {
        let buffer = self.upload_slice(usage, slice::from_ref(data));
        Buffer::new(buffer.untyped)
    }

    fn create_array_buffer<T: Copy>(
        self: &Rc<Self>,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        len: usize,
    ) -> Buffer<[T]> {
        let buffer = self.create_buffer(usage, memory_location, (mem::size_of::<T>() * len) as u64);
        Buffer::new(buffer)
    }

    fn upload_slice<T: Copy>(self: &Rc<Self>, usage: BufferUsage, data: &[T]) -> Buffer<[T]> {
        let buffer = self.create_array_buffer(usage, MemoryLocation::CpuToGpu, data.len());
        unsafe {
            // copy data to mapped buffer
            ptr::copy_nonoverlapping(data.as_ptr(), buffer.as_mut_ptr(), data.len());
        }
        buffer
    }
}
