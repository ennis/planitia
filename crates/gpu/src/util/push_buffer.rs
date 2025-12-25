use crate::{Buffer, Ptr};

/// GPU-side dynamic array.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushBuffer<T: Copy + 'static> {
    base: Ptr<[T]>,
    offset: u32,
    capacity: u32,
}

impl<T: Copy + 'static> PushBuffer<T> {
    pub fn new(buffer: &Buffer<[T]>) -> Self {
        Self {
            base: buffer.ptr(),
            offset: 0,
            capacity: buffer.len() as u32,
        }
    }
}
