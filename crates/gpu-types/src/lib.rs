#![feature(default_field_values)]
use bitflags::bitflags;
use std::marker::PhantomData;
use std::path::Path;
use std::{fmt, slice};

// Reexports
pub use ash::{self, vk};
use ash::vk::PolygonMode;
pub use vk::Format;

/// 2D point with integer coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point2D {
    pub x: i32,
    pub y: i32,
}

impl Point2D {
    pub const ZERO: Self = Self { x: 0, y: 0 };
}

/// 2D integer size.
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

/// 2D integer rectangle.
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
        Self { min: origin, max: Point2D { x: origin.x + size.width as i32, y: origin.y + size.height as i32 } }
    }

    pub const fn from_xywh(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self { min: Point2D { x, y }, max: Point2D { x: x + width as i32, y: y + height as i32 } }
    }
}

/// 3D integer offset.
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
        vk::Offset3D { x: self.x, y: self.y, z: self.z }
    }
}

/// 3D integer dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Size3D {
    /// Width (extent along the X axis).
    pub width: u32,
    /// Height (extent along the Y axis).
    pub height: u32,
    /// Depth (extent along the Z axis).
    /// When describing a layered 2D image, this represents the number of layers.
    pub depth: u32,
}

impl Size3D {
    /// Creates a new `Size3D` with the specified dimensions.
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self { width, height, depth }
    }
}

impl Into<vk::Extent3D> for Size3D {
    fn into(self) -> vk::Extent3D {
        vk::Extent3D { width: self.width, height: self.height, depth: self.depth }
    }
}

/// 3D integer rectangle.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect3D {
    /// Minimum corner of the rectangle (inclusive).
    pub min: Offset3D,
    /// Maximum corner of the rectangle (exclusive).
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
        Size3D { width: self.width(), height: self.height(), depth: self.depth() }
    }

    pub const fn from_origin_size_2d(origin: Point2D, size: Size2D) -> Self {
        Self {
            min: Offset3D { x: origin.x, y: origin.y, z: 0 },
            max: Offset3D { x: origin.x + size.width as i32, y: origin.y + size.height as i32, z: 1 },
        }
    }

    pub const fn from_size_2d(size: Size2D) -> Self {
        Self::from_origin_size_2d(Point2D::ZERO, size)
    }

    pub const fn from_xywh(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self { min: Offset3D { x, y, z: 0 }, max: Offset3D { x: x + width as i32, y: y + height as i32, z: 1 } }
    }
}

bitflags! {
    /// Describes the intended usages of a buffer.
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

impl BufferUsage {
    pub const fn to_vk_buffer_usage_flags(self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::from_raw(self.bits())
    }
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<BufferUsage> for vk::BufferUsageFlags {
    fn from(usage: BufferUsage) -> Self {
        usage.to_vk_buffer_usage_flags()
    }
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

/// Returns the byte size of one pixel in the specified format.
///
/// # Panics
///
/// Panics if the format is a block-compressed format.
pub const fn format_pixel_byte_size(fmt: vk::Format) -> u32 {
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

/// Specifies the image aspect to consider in image operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum ImageAspect {
    /// All aspects of the image (for a color format, this represents the color aspect,
    /// for a depth/stencil image this represents the depth and/or stencil aspects).
    #[default]
    All = 1,
    /// Operate on the depth aspect of the image (for depth-stencil images).
    Depth = 2,
    /// Operate on the stencil aspect of the image (for depth-stencil images).
    Stencil = 4,
}

impl ImageAspect {
    /// Converts this enum to a `VkImageAspectFlags` value, based on the specified image format.
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

pub fn is_depth_format(fmt: vk::Format) -> bool {
    is_depth_only_format(fmt) || is_depth_and_stencil_format(fmt)
}

pub fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    matches!(fmt, Format::D16_UNORM_S8_UINT | Format::D24_UNORM_S8_UINT | Format::D32_SFLOAT_S8_UINT)
}

pub fn is_depth_only_format(fmt: vk::Format) -> bool {
    matches!(fmt, Format::D16_UNORM | Format::X8_D24_UNORM_PACK32 | Format::D32_SFLOAT)
}

pub fn is_stencil_only_format(fmt: vk::Format) -> bool {
    matches!(fmt, Format::S8_UINT)
}

pub fn aspects_for_format(fmt: Format) -> vk::ImageAspectFlags {
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

/// Dimensionality of an image.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ImageType {
    Image1D,
    Image2D,
    Image3D,
}

impl ImageType {
    pub const fn to_vk_image_type(self) -> vk::ImageType {
        match self {
            Self::Image1D => vk::ImageType::TYPE_1D,
            Self::Image2D => vk::ImageType::TYPE_2D,
            Self::Image3D => vk::ImageType::TYPE_3D,
        }
    }

    pub const fn to_vk_image_view_type(self, layers: u32) -> vk::ImageViewType {
        match self {
            Self::Image1D => {
                if layers > 1 {
                    vk::ImageViewType::TYPE_1D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_1D
                }
            }
            Self::Image2D => {
                if layers > 1 {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                }
            }
            Self::Image3D => vk::ImageViewType::TYPE_3D,
        }
    }
}

impl From<ImageType> for vk::ImageType {
    fn from(ty: ImageType) -> Self {
        ty.to_vk_image_type()
    }
}

bitflags! {
    /// Bitmask describing the intended usage of an image.
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = 0b1;
        const TRANSFER_DST = 0b10;
        const SAMPLED = 0b100;
        const STORAGE = 0b1000;
        const COLOR_ATTACHMENT = 0b1_0000;
        const DEPTH_STENCIL_ATTACHMENT = 0b10_0000;
        const INPUT_ATTACHMENT = 0b1000_0000;
    }
}

impl Default for ImageUsage {
    fn default() -> Self {
        Self::empty()
    }
}

impl ImageUsage {
    pub const fn to_vk_image_usage_flags(self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::from_raw(self.bits())
    }
}

impl From<ImageUsage> for vk::ImageUsageFlags {
    fn from(usage: ImageUsage) -> Self {
        usage.to_vk_image_usage_flags()
    }
}

/// Describes the image subresources to include in an image operation.
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

/// Image view creation parameters.
///
/// Same as VkImageViewCreateInfo, but implements Eq and PartialEq.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageViewInfo {
    pub view_type: vk::ImageViewType,
    pub format: Format,
    pub subresource_range: ImageSubresourceRange,
    pub component_mapping: [vk::ComponentSwizzle; 4],
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
        Self { offset: 0, texel_row_length: Some(width), row_count: Some(height) }
    }
}

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

/// Represents a clear value for use in image clear operations.
#[derive(Clone, Copy, Debug)]
pub enum ClearColorValue {
    /// Floating-point clear value.
    ///
    /// Should be used with floating-point and normalized image formats (UNORM/SNORM).
    Float([f32; 4]),
    /// Integer clear value.
    ///
    /// Should be used with signed integer image formats (SINT).
    Int([i32; 4]),
    /// Unsigned integer clear value.
    ///
    /// Should be used with unsigned integer image formats (UINT).
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

/// Blending parameters for a color attachment.
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

/// Describes a color attachment format and blending parameters.
#[derive(Copy, Clone, Debug)]
pub struct ColorTargetState {
    pub format: Format = vk::Format::UNDEFINED,
    pub blend_equation: Option<ColorBlendEquation> = None,
    pub color_write_mask: vk::ColorComponentFlags = vk::ColorComponentFlags::RGBA,
}

impl Default for ColorTargetState {
    fn default() -> Self {
        Self { .. }
    }
}

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
    pub buffers: &'a [VertexBufferLayoutDescription] = &[],
    pub attributes: &'a [VertexInputAttributeDescription] = &[],
}

pub trait StaticVertexInput {
    /// Vertex buffers
    const BUFFER_LAYOUT: &'static [VertexBufferLayoutDescription];

    /// Vertex attributes.
    const ATTRIBUTES: &'static [VertexInputAttributeDescription];
}

/// Trait implemented by types that represent vertex data in a vertex buffer.
pub unsafe trait Vertex: Copy + 'static {
    const ATTRIBUTES: &'static [VertexInputAttributeDescription];
    const BUFFER_DESC: &'static VertexBufferLayoutDescription;

    fn vertex_input_state() -> VertexInputState<'static> {
        VertexInputState { buffers: slice::from_ref(Self::BUFFER_DESC), attributes: Self::ATTRIBUTES }
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

#[cfg(feature = "color")]
unsafe impl VertexAttribute for color::Srgba8 {
    const FORMAT: Format = Format::R8G8B8A8_UNORM;
}

macro_rules! impl_index_data {
    ($t:ty, $fmt:ident) => {
        unsafe impl VertexIndex for $t {
            const FORMAT: vk::IndexType = vk::IndexType::$fmt;
        }
    };
}

impl_index_data!(u16, UINT16);
impl_index_data!(u32, UINT32);

/// Primitive topology.
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
    /// Converts this enum to a `VkPrimitiveTopology` enum.
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

/// Polygon orientation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    /// Front face of a polygon.
    Front = 0,
    /// Back face of a polygon.
    Back = 1,
}

#[derive(Copy, Clone, Debug)]
pub struct DepthBias {
    pub constant_factor: f32,
    pub clamp: f32,
    pub slope_factor: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct RasterizationState {
    pub polygon_mode: vk::PolygonMode = vk::PolygonMode::FILL,
    pub cull_mode: vk::CullModeFlags = vk::CullModeFlags::NONE,
    pub front_face: vk::FrontFace = vk::FrontFace::CLOCKWISE,
    pub depth_clamp_enable: bool = false,
    pub conservative_rasterization_mode: vk::ConservativeRasterizationModeEXT = vk::ConservativeRasterizationModeEXT::DISABLED,
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
    pub front: StencilOpState = StencilOpState::IGNORE,
    pub back: StencilOpState = StencilOpState::IGNORE,
    pub read_mask: u32 = 0,
    pub write_mask: u32 = 0,
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
        Self { .. }
    }
}

/// Describes how a graphics pipeline reads and modifies a depth-stencil attachment.
#[derive(Copy, Clone, Debug)]
pub struct DepthStencilState {
    pub format: vk::Format = Format::UNDEFINED,
    pub depth_write_enable: bool = false,
    pub depth_compare_op: vk::CompareOp = vk::CompareOp::LESS,
    pub stencil_state: StencilState = StencilState { .. },
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            ..
        }
    }
}

/// Controls multisampling for a graphics pipeline.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MultisampleState {
    /// Number of samples per pixel.
    ///
    /// `1` means that multisampling is disabled.
    pub count: u32 = 1,
    pub mask: u64 = !0,
    pub alpha_to_coverage_enabled: bool = false,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self { .. }
    }
}

/// Controls the fragment shader and color output stages of a graphics pipeline.
#[derive(Copy, Clone, Debug)]
pub struct FragmentState<'a> {
    pub shader: ShaderEntryPoint<'a>,
    pub multisample: MultisampleState = MultisampleState { .. },
    pub color_targets: &'a [ColorTargetState],
    pub blend_constants: [f32; 4] = [0.0, 0.0, 0.0, 0.0],
}

#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
    Content(&'a str),
    File(&'a Path),
}

/// Specifies a pipeline stage.
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

/// Pointers in GPU device address space.
///
/// Like `*mut T`, but in device address space.
/// They can be used in shaders, via VK_KHR_buffer_device_address (required).
#[repr(transparent)]
pub struct Ptr<T: ?Sized + 'static> {
    pub raw: vk::DeviceAddress,
    pub _phantom: PhantomData<T>,
}

impl<T: ?Sized + 'static> fmt::Debug for Ptr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU:{:#016x}", self.raw)
    }
}

impl<T: ?Sized + 'static> Ptr<T> {
    /// Null (invalid) device address.
    pub const NULL: Self = Ptr { raw: 0, _phantom: PhantomData };
}

impl<T: 'static> Ptr<T> {
    pub fn offset(self, offset: usize) -> Self {
        Ptr { raw: self.raw + (offset * size_of::<T>()) as u64, _phantom: PhantomData }
    }
}

impl<T: ?Sized + 'static> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        Ptr { raw: self.raw, _phantom: PhantomData }
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
    pub const INVALID: Self = TextureHandle { index: u32::MAX, _unused: 0 };

    pub const fn new(index: u32) -> Self {
        TextureHandle { index, _unused: 0 }
    }
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
    pub const INVALID: Self = StorageImageHandle { index: u32::MAX, _unused: 0 };

    pub const fn new(index: u32) -> Self {
        StorageImageHandle { index, _unused: 0 }
    }
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
    pub const INVALID: Self = SamplerHandle { index: u32::MAX, _unused: 0 };

    pub const fn new(index: u32) -> Self {
        SamplerHandle { index, _unused: 0 }
    }
}


/// Typedefs for compatibility with slang reflection generated by shadertool.
#[allow(non_camel_case_types)]
pub mod shader_types {
    pub type float = f32;
    pub type vec2f = math::Vec2;
    pub type vec3f = math::Vec3;
    pub type vec4f = math::Vec4;
    pub type int = i32;
    pub type vec2i = math::IVec2;
    pub type vec3i = math::IVec3;
    pub type vec4i = math::IVec4;
    pub type uint = u32;
    pub type vec2u = math::UVec2;
    pub type vec3u = math::UVec3;
    pub type vec4u = math::UVec4;
    pub type uint8_t = u8;
    pub type vec2u8 = math::U8Vec2;
    pub type vec3u8 = math::U8Vec3;
    pub type vec4u8 = math::U8Vec4;
    pub type int8_t = i8;
    pub type vec2i8 = math::I8Vec2;
    pub type vec3i8 = math::I8Vec3;
    pub type vec4i8 = math::I8Vec4;
    pub type uint16_t = u16;
    pub type vec2u16 = math::U16Vec2;
    pub type vec3u16 = math::U16Vec3;
    pub type vec4u16 = math::U16Vec4;
    pub type int16_t = i16;
    pub type vec2i16 = math::I16Vec2;
    pub type vec3i16 = math::I16Vec3;
    pub type vec4i16 = math::I16Vec4;
    pub type mat3x3f = math::Mat3;
    pub type mat4x4f = math::Mat4;
}