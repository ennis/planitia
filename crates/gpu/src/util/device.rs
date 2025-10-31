use crate::{Buffer, BufferUsage, Device, MemoryLocation};
use std::{ptr, slice};

impl Device {
    pub fn upload<T: Copy>(&self, usage: BufferUsage, data: &T, label: &str) -> Buffer<T> {
        let buffer = self.upload_slice(usage, slice::from_ref(data), label);
        buffer.single()
    }

    pub fn create_array_buffer<T: Copy>(
        &self,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        len: usize,
        label: &str,
    ) -> Buffer<[T]> {
        let buffer = self.create_buffer(usage, memory_location, (size_of::<T>() * len) as u64, label);
        unsafe {
            // SAFETY: the buffer is large enough to hold `len` elements of type `T`
            buffer.cast()
        }
    }

    pub fn upload_slice<T: Copy>(&self, usage: BufferUsage, data: &[T], label: &str) -> Buffer<[T]> {
        let buffer = self.create_array_buffer(usage, MemoryLocation::CpuToGpu, data.len(), label);
        unsafe {
            // copy data to mapped buffer
            ptr::copy_nonoverlapping(data.as_ptr(), buffer.as_mut_ptr() as *mut T, data.len());
        }
        buffer
    }
}
