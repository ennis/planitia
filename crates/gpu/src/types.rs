// TODO: eventually all vk types should disappear from the public API
use crate::{
    aspects_for_format, is_depth_and_stencil_format, is_depth_only_format, is_stencil_only_format, ShaderEntryPoint,
};
use ash::vk;
use bitflags::bitflags;
use std::borrow::Cow;
use std::path::Path;
use std::slice;

pub type Format = vk::Format;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    /// Front face
    Front = 0,
    /// Back face
    Back = 1,
}

/// Describe a subresource range of an image.
///
/// Same as VkImageSubresourceRange, but implements Eq and PartialEq.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageSubresourceRange {
    pub aspect_mask: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

/// Represents the layout of image data in a buffer.
#[derive(Copy, Clone, Debug)]
pub struct ImageDataLayout {
    /// Offset in bytes from the start of the buffer.
    pub offset: u64,
    /// Size of a row, in texels. Equivalently, the number of texels between each row.
    ///
    /// If `None`, the row length is considered to be tightly packed to the image width.
    pub texel_row_length: Option<u32>,
    /// Height of the image (number of rows).
    pub row_count: Option<u32>,
}

impl ImageDataLayout {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            offset: 0,
            texel_row_length: Some(width),
            row_count: Some(height),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum ImageAspect {
    #[default]
    All = 1,
    Depth = 2,
    Stencil = 4,
}

impl ImageAspect {
    pub fn to_view_aspect_flags(self, format: Format) -> vk::ImageAspectFlags {
        match self {
            ImageAspect::All => aspects_for_format(format),
            ImageAspect::Depth => vk::ImageAspectFlags::DEPTH,
            ImageAspect::Stencil => vk::ImageAspectFlags::STENCIL,
        }
    }

    pub fn to_aspect(self, format: Format) -> vk::ImageAspectFlags {
        if (is_depth_and_stencil_format(format) || is_depth_only_format(format) || is_stencil_only_format(format))
            && self == ImageAspect::All
        {
            panic!("ImageAspect::All is not valid for depth/stencil formats");
        }
        match self {
            ImageAspect::All => vk::ImageAspectFlags::COLOR,
            ImageAspect::Depth => vk::ImageAspectFlags::DEPTH,
            ImageAspect::Stencil => vk::ImageAspectFlags::STENCIL,
        }
    }
}

/*
bitflags! {
    /// Bitmask specifying which aspects of an image are included in a view.
    ///
    /// Use `DEFAULT` to include all relevant aspects for the image format.
    #[derive(Copy,Clone,Debug,PartialEq,Eq,Hash)]
    pub struct ImageAspectMask : u32 {
        // These values must match vk::ImageAspectFlags, except for DEFAULT.
        const COLOR = 1;
        const DEPTH = 2;
        const STENCIL = 4;
        const METADATA = 8;
        const DEFAULT = 0xFFFFFFFF;
    }
}

impl Default for ImageAspectMask {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ImageAspectMask {
    pub fn to_vk_image_aspect_flags(self, format: Format) -> vk::ImageAspectFlags {
        if self == ImageAspectMask::DEFAULT {
            aspects_for_format(format)
        } else {
            vk::ImageAspectFlags::from_raw(self.bits())
        }
    }
}*/

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageSubresourceLayers {
    pub aspect: ImageAspect = ImageAspect::All,
    pub mip_level: u32 = 0,
    pub base_array_layer: u32 = 0,
    pub layer_count: u32 = 1,
}

impl Default for ImageSubresourceLayers {
    fn default() -> Self {
        Self { .. }
    }
}

/// The parameters of an image view.
///
/// Same as VkImageViewCreateInfo, but implements Eq and PartialEq.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageViewInfo {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub subresource_range: ImageSubresourceRange,
    pub component_mapping: [vk::ComponentSwizzle; 4],
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        const UNIFORM_TEXEL_BUFFER = 0b100;
        const STORAGE_TEXEL_BUFFER = 0b1000;
        const UNIFORM = 0b1_0000;
        // Included by default for all buffers; on most GPUs this only adds a 16-byte alignment
        // requirement to the buffer.
        //const STORAGE = 0b10_0000;
    }
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self::empty()
    }
}

impl BufferUsage {
    pub const fn to_vk_buffer_usage_flags(self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::from_raw(self.bits())
    }
}

impl From<BufferUsage> for vk::BufferUsageFlags {
    fn from(usage: BufferUsage) -> Self {
        usage.to_vk_buffer_usage_flags()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PrimitiveTopology {
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
    PatchList,
}

impl PrimitiveTopology {
    pub const fn to_vk_primitive_topology(self) -> vk::PrimitiveTopology {
        match self {
            Self::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            Self::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            Self::LineList => vk::PrimitiveTopology::LINE_LIST,
            Self::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            Self::PointList => vk::PrimitiveTopology::POINT_LIST,
            Self::PatchList => vk::PrimitiveTopology::PATCH_LIST,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PipelineBindPoint {
    Graphics,
    Compute,
}

impl PipelineBindPoint {
    pub const fn to_vk_pipeline_bind_point(self) -> vk::PipelineBindPoint {
        match self {
            Self::Graphics => vk::PipelineBindPoint::GRAPHICS,
            Self::Compute => vk::PipelineBindPoint::COMPUTE,
        }
    }
}

impl From<PipelineBindPoint> for vk::PipelineBindPoint {
    fn from(bind_point: PipelineBindPoint) -> Self {
        bind_point.to_vk_pipeline_bind_point()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ClearColorValue {
    Float([f32; 4]),
    Int([i32; 4]),
    Uint([u32; 4]),
}

impl From<[f32; 4]> for ClearColorValue {
    fn from(v: [f32; 4]) -> Self {
        Self::Float(v)
    }
}

impl From<[i32; 4]> for ClearColorValue {
    fn from(v: [i32; 4]) -> Self {
        Self::Int(v)
    }
}

impl From<[u32; 4]> for ClearColorValue {
    fn from(v: [u32; 4]) -> Self {
        Self::Uint(v)
    }
}

impl From<ClearColorValue> for vk::ClearColorValue {
    fn from(v: ClearColorValue) -> Self {
        match v {
            ClearColorValue::Float(v) => vk::ClearColorValue { float32: v },
            ClearColorValue::Int(v) => vk::ClearColorValue { int32: v },
            ClearColorValue::Uint(v) => vk::ClearColorValue { uint32: v },
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ARGUMENTS

#[derive(Debug, Clone)]
pub struct ArgumentsLayout<'a> {
    pub bindings: Cow<'a, [vk::DescriptorSetLayoutBinding<'static>]>,
}

#[derive(Copy, Clone, Debug)]
pub struct PipelineLayoutDescriptor<'a> {
    pub descriptor_sets: &'a [ArgumentsLayout<'a>],
    // None of the relevant drivers on desktop seem to care about precise push constant ranges,
    // so we just store the total size of push constants.
    pub push_constants_size: usize,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SAMPLERS
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct SamplerCreateInfo {
    pub mag_filter: vk::Filter = vk::Filter::LINEAR,
    pub min_filter: vk::Filter = vk::Filter::LINEAR,
    pub mipmap_mode: vk::SamplerMipmapMode =  vk::SamplerMipmapMode::LINEAR,
    pub address_mode_u: vk::SamplerAddressMode = vk::SamplerAddressMode::CLAMP_TO_EDGE,
    pub address_mode_v: vk::SamplerAddressMode = vk::SamplerAddressMode::CLAMP_TO_EDGE,
    pub address_mode_w: vk::SamplerAddressMode = vk::SamplerAddressMode::CLAMP_TO_EDGE,
    pub mip_lod_bias: f32 = 0.0,
    pub anisotropy_enable: bool = false,
    pub max_anisotropy: f32 = 0.0,
    pub compare_enable: bool = false,
    pub compare_op: vk::CompareOp = vk::CompareOp::ALWAYS,
    pub min_lod: f32 = 0.0,
    pub max_lod: f32 = 0.0,
    pub border_color: vk::BorderColor = vk::BorderColor::INT_OPAQUE_BLACK,
    pub unnormalized_coordinates: bool = false,
}

impl Default for SamplerCreateInfo {
    fn default() -> Self {
        SamplerCreateInfo { .. }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct SamplerCreateInfoHashable {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias_f32_bits: u32,
    pub anisotropy_enable: bool,
    pub max_anisotropy_f32_bits: u32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod_f32_bits: u32,
    pub max_lod_f32_bits: u32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl From<SamplerCreateInfo> for SamplerCreateInfoHashable {
    fn from(info: SamplerCreateInfo) -> Self {
        Self {
            mag_filter: info.mag_filter,
            min_filter: info.min_filter,
            mipmap_mode: info.mipmap_mode,
            address_mode_u: info.address_mode_u,
            address_mode_v: info.address_mode_v,
            address_mode_w: info.address_mode_w,
            mip_lod_bias_f32_bits: info.mip_lod_bias.to_bits(),
            anisotropy_enable: info.anisotropy_enable,
            max_anisotropy_f32_bits: info.max_anisotropy.to_bits(),
            compare_enable: info.compare_enable,
            compare_op: info.compare_op,
            min_lod_f32_bits: info.min_lod.to_bits(),
            max_lod_f32_bits: info.max_lod.to_bits(),
            border_color: info.border_color,
            unnormalized_coordinates: info.unnormalized_coordinates,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point2D {
    pub x: i32,
    pub y: i32,
}

impl Point2D {
    pub const ZERO: Self = Self { x: 0, y: 0 };
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Size2D {
    pub width: u32,
    pub height: u32,
}

impl Size2D {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect2D {
    pub min: Point2D,
    pub max: Point2D,
}

impl Rect2D {
    pub const fn width(&self) -> u32 {
        (self.max.x - self.min.x) as u32
    }
    pub const fn height(&self) -> u32 {
        (self.max.y - self.min.y) as u32
    }

    pub const fn from_origin_size(origin: Point2D, size: Size2D) -> Self {
        Self {
            min: origin,
            max: Point2D {
                x: origin.x + size.width as i32,
                y: origin.y + size.height as i32,
            },
        }
    }

    pub const fn from_xywh(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            min: Point2D { x, y },
            max: Point2D {
                x: x + width as i32,
                y: y + height as i32,
            },
        }
    }
}

/*
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Origin3D {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Origin3D {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };
}*/

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Offset3D {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };
}

impl Into<vk::Offset3D> for Offset3D {
    fn into(self) -> vk::Offset3D {
        vk::Offset3D {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect3D {
    pub min: Offset3D,
    pub max: Offset3D,
}

impl Rect3D {
    pub const fn width(&self) -> u32 {
        (self.max.x - self.min.x) as u32
    }
    pub const fn height(&self) -> u32 {
        (self.max.y - self.min.y) as u32
    }
    pub const fn depth(&self) -> u32 {
        (self.max.z - self.min.z) as u32
    }

    pub const fn size(&self) -> Size3D {
        Size3D {
            width: self.width(),
            height: self.height(),
            depth: self.depth(),
        }
    }

    pub const fn from_origin_size_2d(origin: Point2D, size: Size2D) -> Self {
        Self {
            min: Offset3D {
                x: origin.x,
                y: origin.y,
                z: 0,
            },
            max: Offset3D {
                x: origin.x + size.width as i32,
                y: origin.y + size.height as i32,
                z: 1,
            },
        }
    }

    pub const fn from_size_2d(size: Size2D) -> Self {
        Self::from_origin_size_2d(Point2D::ZERO, size)
    }

    pub const fn from_xywh(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            min: Offset3D { x, y, z: 0 },
            max: Offset3D {
                x: x + width as i32,
                y: y + height as i32,
                z: 1,
            },
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Size3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// BLENDING

#[derive(Copy, Clone, Debug)]
pub struct ColorBlendEquation {
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
}

impl Default for ColorBlendEquation {
    fn default() -> Self {
        Self::REPLACE
    }
}

impl ColorBlendEquation {
    pub const REPLACE: Self = Self {
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
    };

    pub const ALPHA_BLENDING: Self = Self {
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
    };

    // TODO: check if this is correct
    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
    };
}

#[derive(Copy, Clone, Debug)]
pub struct ColorTargetState {
    pub format: vk::Format,
    pub blend_equation: Option<ColorBlendEquation>,
    pub color_write_mask: vk::ColorComponentFlags,
}

impl Default for ColorTargetState {
    fn default() -> Self {
        Self {
            format: vk::Format::UNDEFINED,
            blend_equation: None,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// RASTERIZATION
#[derive(Copy, Clone, Debug)]
pub struct DepthBias {
    pub constant_factor: f32,
    pub clamp: f32,
    pub slope_factor: f32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct RasterizationState {
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_clamp_enable: bool,
    pub conservative_rasterization_mode: vk::ConservativeRasterizationModeEXT,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StencilOpState {
    pub compare: vk::CompareOp,
    pub fail_op: vk::StencilOp,
    pub depth_fail_op: vk::StencilOp,
    pub pass_op: vk::StencilOp,
}

//  Adapted from WGPU
impl StencilOpState {
    pub const IGNORE: Self = StencilOpState {
        compare: vk::CompareOp::ALWAYS,
        fail_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
    };

    /// Returns true if the face state doesn't mutate the target values.
    pub fn is_read_only(&self) -> bool {
        self.pass_op == vk::StencilOp::KEEP
            && self.depth_fail_op == vk::StencilOp::KEEP
            && self.fail_op == vk::StencilOp::KEEP
    }
}

impl StencilOpState {
    pub const fn to_vk_stencil_op_state(&self) -> vk::StencilOpState {
        vk::StencilOpState {
            fail_op: self.fail_op,
            pass_op: self.pass_op,
            depth_fail_op: self.depth_fail_op,
            compare_op: self.compare,
            compare_mask: !0,
            write_mask: !0,
            reference: 0,
        }
    }
}

impl From<StencilOpState> for vk::StencilOpState {
    fn from(state: StencilOpState) -> Self {
        state.to_vk_stencil_op_state()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StencilState {
    pub front: StencilOpState,
    pub back: StencilOpState,
    pub read_mask: u32,
    pub write_mask: u32,
}

//  Adapted from WGPU
impl StencilState {
    /// Returns true if the stencil test is enabled.
    pub fn is_enabled(&self) -> bool {
        (self.front != StencilOpState::IGNORE || self.back != StencilOpState::IGNORE)
            && (self.read_mask != 0 || self.write_mask != 0)
    }
    /// Returns true if the state doesn't mutate the target values.
    pub fn is_read_only(&self, cull_mode: Option<Face>) -> bool {
        // The rules are defined in step 7 of the "Device timeline initialization steps"
        // subsection of the "Render Pipeline Creation" section of WebGPU
        // (link to the section: https://gpuweb.github.io/gpuweb/#render-pipeline-creation)

        if self.write_mask == 0 {
            return true;
        }

        let front_ro = cull_mode == Some(Face::Front) || self.front.is_read_only();
        let back_ro = cull_mode == Some(Face::Back) || self.back.is_read_only();

        front_ro && back_ro
    }
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            front: StencilOpState::IGNORE,
            back: StencilOpState::IGNORE,
            read_mask: 0,
            write_mask: 0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DepthStencilState {
    pub format: vk::Format,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub stencil_state: StencilState,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            format: Format::UNDEFINED,
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::LESS,
            stencil_state: Default::default(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VERTEX STATE

/// Trait implemented by types that represent vertex data in a vertex buffer.
pub unsafe trait Vertex: Copy + 'static {
    const ATTRIBUTES: &'static [VertexInputAttributeDescription];
    const BUFFER_DESC: &'static VertexBufferLayoutDescription;

    fn vertex_input_state() -> VertexInputState<'static> {
        VertexInputState {
            buffers: slice::from_ref(Self::BUFFER_DESC),
            attributes: Self::ATTRIBUTES,
        }
    }
}

/// Trait implemented by types that can serve as indices.
pub unsafe trait VertexIndex: Copy + 'static {
    /// Index type.
    const FORMAT: vk::IndexType;
}

/// Description of a vertex attribute within a vertex layout.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct VertexAttributeDescription {
    pub format: vk::Format,
    pub offset: u32,
}

/// Trait implemented by types that can serve as a vertex attribute.
pub unsafe trait VertexAttribute {
    /// Returns the corresponding data format (the layout of the data in memory).
    const FORMAT: vk::Format;
}

/// Wrapper type for normalized integer attributes.
///
/// Helper for `normalized` in `derive(Vertex)`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct Norm<T>(pub T);

// Vertex attribute types
macro_rules! impl_vertex_attr {
    ($t:ty, $fmt:ident) => {
        unsafe impl VertexAttribute for $t {
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}

// F32
impl_vertex_attr!(f32, R32_SFLOAT);
impl_vertex_attr!([f32; 2], R32G32_SFLOAT);
impl_vertex_attr!([f32; 3], R32G32B32_SFLOAT);
impl_vertex_attr!([f32; 4], R32G32B32A32_SFLOAT);

// U32
impl_vertex_attr!(u32, R32_UINT);
impl_vertex_attr!([u32; 2], R32G32_UINT);
impl_vertex_attr!([u32; 3], R32G32B32_UINT);
impl_vertex_attr!([u32; 4], R32G32B32A32_UINT);

impl_vertex_attr!(i32, R32_SINT);
impl_vertex_attr!([i32; 2], R32G32_SINT);
impl_vertex_attr!([i32; 3], R32G32B32_SINT);
impl_vertex_attr!([i32; 4], R32G32B32A32_SINT);

// U16
impl_vertex_attr!(u16, R16_UINT);
impl_vertex_attr!([u16; 2], R16G16_UINT);
impl_vertex_attr!([u16; 3], R16G16B16_UINT);
impl_vertex_attr!([u16; 4], R16G16B16A16_UINT);

impl_vertex_attr!(i16, R16_SINT);
impl_vertex_attr!([i16; 2], R16G16_SINT);
impl_vertex_attr!([i16; 3], R16G16B16_SINT);
impl_vertex_attr!([i16; 4], R16G16B16A16_SINT);

// UNORM16
impl_vertex_attr!(Norm<u16>, R16_UNORM);
impl_vertex_attr!(Norm<[u16; 2]>, R16G16_UNORM);
impl_vertex_attr!(Norm<[u16; 3]>, R16G16B16_UNORM);
impl_vertex_attr!(Norm<[u16; 4]>, R16G16B16A16_UNORM);

// SNORM16
impl_vertex_attr!(Norm<i16>, R16_SNORM);
impl_vertex_attr!(Norm<[i16; 2]>, R16G16_SNORM);
impl_vertex_attr!(Norm<[i16; 3]>, R16G16B16_SNORM);
impl_vertex_attr!(Norm<[i16; 4]>, R16G16B16A16_SNORM);

// U8
impl_vertex_attr!(u8, R8_UINT);
impl_vertex_attr!([u8; 2], R8G8_UINT);
impl_vertex_attr!([u8; 3], R8G8B8_UINT);
impl_vertex_attr!([u8; 4], R8G8B8A8_UINT);

impl_vertex_attr!(Norm<u8>, R8_UNORM);
impl_vertex_attr!(Norm<[u8; 2]>, R8G8_UNORM);
impl_vertex_attr!(Norm<[u8; 3]>, R8G8B8_UNORM);
impl_vertex_attr!(Norm<[u8; 4]>, R8G8B8A8_UNORM);

impl_vertex_attr!(i8, R8_SINT);
impl_vertex_attr!([i8; 2], R8G8_SINT);
impl_vertex_attr!([i8; 3], R8G8B8_SINT);
impl_vertex_attr!([i8; 4], R8G8B8A8_SINT);

impl_vertex_attr!(math::Vec2, R32G32_SFLOAT);
impl_vertex_attr!(math::Vec3, R32G32B32_SFLOAT);
impl_vertex_attr!(math::Vec4, R32G32B32A32_SFLOAT);

impl_vertex_attr!(math::U16Vec2, R16G16_UINT);
impl_vertex_attr!(math::U16Vec3, R16G16B16_UINT);
impl_vertex_attr!(math::U16Vec4, R16G16B16A16_UINT);

impl_vertex_attr!(Norm<math::U16Vec2>, R16G16_UNORM);
impl_vertex_attr!(Norm<math::U16Vec3>, R16G16B16_UNORM);
impl_vertex_attr!(Norm<math::U16Vec4>, R16G16B16A16_UNORM);

// Index data types --------------------------------------------------------------------------------
macro_rules! impl_index_data {
    ($t:ty, $fmt:ident) => {
        unsafe impl VertexIndex for $t {
            const FORMAT: vk::IndexType = vk::IndexType::$fmt;
        }
    };
}

impl_index_data!(u16, UINT16);
impl_index_data!(u32, UINT32);

#[derive(Copy, Clone, Debug)]
pub struct VertexBufferLayoutDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: vk::VertexInputRate,
}

#[derive(Copy, Clone, Debug)]
pub struct VertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: Format,
    pub offset: u32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexInputState<'a> {
    pub buffers: &'a [VertexBufferLayoutDescription],
    pub attributes: &'a [VertexInputAttributeDescription],
}

pub trait StaticVertexInput {
    /// Vertex buffers
    const BUFFER_LAYOUT: &'static [VertexBufferLayoutDescription];

    /// Vertex attributes.
    const ATTRIBUTES: &'static [VertexInputAttributeDescription];
}

// From WGPU
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FragmentState<'a> {
    pub shader: ShaderEntryPoint<'a>,
    pub multisample: MultisampleState,
    pub color_targets: &'a [ColorTargetState],
    pub blend_constants: [f32; 4],
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SHADERS

#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
    Content(&'a str),
    File(&'a Path),
}

#[derive(Debug, Clone, Copy)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Geometry,
    Compute,
    TessControl,
    TessEvaluation,
    Mesh,
    Task,
}

impl ShaderStage {
    pub fn to_vk_shader_stage(&self) -> vk::ShaderStageFlags {
        match self {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
            ShaderStage::TessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            ShaderStage::TessEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            ShaderStage::Mesh => vk::ShaderStageFlags::MESH_NV,
            ShaderStage::Task => vk::ShaderStageFlags::TASK_NV,
        }
    }
}
