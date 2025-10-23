//! Pipeline files.
//!

// Pipeline files hold a collection of pipelines (which are variants of the same pipeline).
// It contains fixed function state, SPIR-V shader modules, tags for pipeline variants,
// specialization constants, useful reflection data (e.g. push constant sizes),
// and possibly cached pipeline binary blobs.
//
// Information in a pipeline file is sufficient to create a complete pipeline object.
//
// Pipeline files can be directly mapped in memory and read without any copy or parsing step.

use std::alloc::Layout;
use std::marker::PhantomData;
use zerocopy::{Immutable, KnownLayout, TryFromBytes};
use gpu::vk;
use gpu::vk::PolygonMode;

////////////////////////////////////////////////////////////////////////////////////////////////////

fn header_array_layout<Header, Element>(count: usize) -> Layout {
    let (header_array, _) = Layout::new::<Header>().extend(Layout::array::<Element>(count).unwrap()).unwrap();
    header_array.pad_to_align()
}

fn header_array_layout_2<Header1, Header2, Element>(count: usize) -> Layout {
    let (header_array, _) = Layout::new::<Header1>()
        .extend(Layout::new::<Header2>()).unwrap().0
        .extend(Layout::array::<Element>(count).unwrap()).unwrap();
    header_array.pad_to_align()
}

unsafe fn length_prefixed_dst<T: KnownLayout<PointerMetadata=usize> + TryFromBytes + Immutable + ?Sized>(buf: &[u8]) -> Option<&T> {
    let count = u32::read_from_bytes(buf).ok()?;
    Some(T::try_ref_from_prefix_with_elems(buf, count as usize).ok()?.0)
}

unsafe fn try_from_bytes<T>(data: &[u8]) -> Result<(&T, &[u8]), ReadError> {
    // check alignment & size
    if data.as_ptr().addr() & (align_of::<T>() - 1) != 0 {
        return Err(ReadError::InvalidData);
    }
    if data.len() < size_of::<T>() {
        return Err(ReadError::UnexpectedEof);
    }
    let (head, tail) = data.split_at(size_of::<T>());
    // SAFETY: size checked, alignment checked
    Ok((unsafe { &*(head.as_ptr() as *const T) }, tail))
}


unsafe fn try_read<T: Copy>(data: &[u8]) -> Result<(T, &[u8]), ReadError> {
    unsafe {
        try_from_bytes::<T>(data).map(|((v,tail), _)| (*v,tail))
    }
}

///////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////

/// 8-bit tag + 24-bit offset within the data
#[repr(transparent)]
pub struct Offset24<T: ?Sized>(pub u32, PhantomData<*const T>);

impl<T:?Sized> Clone for Offset24<T> {
    fn clone(&self) -> Self {
        Offset24(self.0, PhantomData)
    }
}

impl<T:?Sized> Copy for Offset24<T> {}

pub const MAX_OFFSET: u32 = 0x01000000;

impl<T: ?Sized> Offset24<T> {
    pub fn new(offset: u32) -> Self {
        Self::with_tag(0, offset)
    }

    pub fn with_tag(tag: u8, offset: u32) -> Self {
        assert!(offset < MAX_OFFSET, "offset out of range");
        Offset24((tag as u32) << 24 | offset, PhantomData)
    }

    pub fn tag(&self) -> u8 {
        (self.0 >> 24) as u8
    }

    pub fn offset(&self) -> u32 {
        self.0 & 0x00FFFFFF
    }
}

///////////////////////////////////////////////////////////


#[repr(C)]
// NoPadding
pub struct OffsetSlice<T> {
    ptr: Offset24<T>,
    len: u32,
}


impl<T> Clone for OffsetSlice<T> {
    fn clone(&self) -> Self {
        OffsetSlice {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl<T> Copy for OffsetSlice<T> {}

#[repr(C)]
#[derive(Copy, Clone)]
// NoPadding
struct PipelineHeader {
    /// "PPLL"
    magic: [u8; 4],
    /// 1
    version: u32,
    /// Number of pipelines
    pipeline_count: u32,
}

impl PipelineHeader {
    fn from_bytes(data: &[u8]) -> Result<&Self, ReadError> {
        // SAFETY: NoPadding
        let (header, _rest) = unsafe { try_from_bytes::<Self>(data)? };
        if &header.magic != b"PPLL" {
            return Err(ReadError::InvalidSignature);
        }
        if header.version != 1 {
            return Err(ReadError::InvalidData);
        }
        Ok(header)
    }
}

pub struct Pipeline<'a> {
    header: &'a PipelineHeader,
    /// Array of offsets to each pipeline
    pipeline_entries: &'a [Offset24<PipelineEntry>],
}

impl<'a> Pipeline<'a> {
    pub fn parse(data: &[u8]) {
        // TODO check alignment
    }


}


////////////////////////////////////////////////////////////////////////////////////////////////////

/// Null-terminated UTF-8 string with fixed maximum length.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ZString<const N: usize>([u8; N]);

#[repr(C)]
pub struct VariantKeyData {
    /// Number of tags.
    pub tag_count: u32,
    /// Name of the pipeline.
    pub name: ZString<32>,
    /// Offset to array of tags.
    pub tags: [ZString<32>],
}

impl VariantKeyData {

    pub fn layout(tag_count: usize) -> Layout {
        header_array_layout_2::<u32, ZString<32>, ZString<32>>(tag_count)
    }

    pub fn from_bytes(data: &[u8]) -> Result<&Self, ReadError> {

        // read tag count
        let (tag_count, _rest) = unsafe { try_read::<u32>(data)? };

        // SAFETY: NoPadding
        let (header, _rest) = unsafe { try_from_bytes::<Self>(data)? };
        Ok(header)
    }
}

/// Block that identifies a single pipeline variant within a pipeline file.
pub struct VariantKey<'a> {
    data: &'a VariantKeyData,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PipelineEntry {
    //pub name:
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RasterizationStateBlock {
    pub polygon_mode: PolygonMode,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DepthStencilStateBlock {
    pub format: vk::Format,
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColorBlendEquationBlock {
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
}