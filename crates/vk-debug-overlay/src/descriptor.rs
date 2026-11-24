// Adapted from https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/nil/descriptor.rs

use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use std::ops::RangeInclusive;
use vulkan::*;

pub type NVTextureImageControl = [u32; 8];

fn getbits(word: u32, range: RangeInclusive<u32>) -> u32 {
    let lsb = *range.start();
    let msb = *range.end();
    let mask = (1 << (msb - lsb + 1)) - 1;
    (word >> lsb) & mask
}

//  GPU texture formats
// https://switchbrew.org/wiki/GPU_Texture_Formats#Texture_Formats
// Value 	Format
// 0x01 	R32_G32_B32_A32
// 0x02 	R32_G32_B32
// 0x03 	R16_G16_B16_A16
// 0x04 	R32_G32
// 0x05 	R32_B24G8
// 0x06 	ETC2_RGB
// 0x07 	X8B8G8R8
// 0x08 	A8B8G8R8
// 0x09 	A2B10G10R10
// 0x0a 	ETC2_RGB_PTA
// 0x0b 	ETC2_RGBA
// 0x0c 	R16_G16
// 0x0d 	G8R24
// 0x0e 	G24R8
// 0x0f 	R32
// 0x10 	BC6H_SF16
// 0x11 	BC6H_UF16
// 0x12 	A4B4G4R4
// 0x13 	A5B5G5R1
// 0x14 	A1B5G5R5
// 0x15 	B5G6R5
// 0x16 	B6G5R5
// 0x17 	BC7U
// 0x18 	G8R8
// 0x19 	EAC
// 0x1a 	EACX2
// 0x1b 	R16
// 0x1c 	Y8_VIDEO
// 0x1d 	R8
// 0x1e 	G4R4
// 0x1f 	R1
// 0x20 	E5B9G9R9_SHAREDEXP
// 0x21 	BF10GF11RF11
// 0x22 	G8B8G8R8
// 0x23 	B8G8R8G8
// 0x24 	DXT1 (a.k.a. BC1)
// 0x25 	DXT23 (a.k.a. BC2)
// 0x26 	DXT45 (a.k.a. BC3)
// 0x27 	DXN1
// 0x28 	DXN2
// 0x29 	Z24S8
// 0x2a 	X8Z24
// 0x2b 	S8Z24
// 0x2c 	X4V4Z24__COV4R4V
// 0x2d 	X4V4Z24__COV8R8V
// 0x2e 	V8Z24__COV4R12V
// 0x2f 	ZF32
// 0x30 	ZF32_X24S8
// 0x31 	X8Z24_X20V4S8__COV4R4V
// 0x32 	X8Z24_X20V4S8__COV8R8V
// 0x33 	ZF32_X20V4X8__COV4R4V
// 0x34 	ZF32_X20V4X8__COV8R8V
// 0x35 	ZF32_X20V4S8__COV4R4V
// 0x36 	ZF32_X20V4S8__COV8R8V
// 0x37 	X8Z24_X16V8S8__COV4R12V
// 0x38 	ZF32_X16V8X8__COV4R12V
// 0x39 	ZF32_X16V8S8__COV4R12V
// 0x3a 	Z16
// 0x3b 	V8Z24__COV8R24V
// 0x3c 	X8Z24_X16V8S8__COV8R24V
// 0x3d 	ZF32_X16V8X8__COV8R24V
// 0x3e 	ZF32_X16V8S8__COV8R24V
// 0x40 	ASTC_2D_4X4
// 0x41 	ASTC_2D_5X5
// 0x42 	ASTC_2D_6X6
// 0x44 	ASTC_2D_8X8
// 0x45 	ASTC_2D_10X10
// 0x46 	ASTC_2D_12X12
// 0x50 	ASTC_2D_5X4
// 0x51 	ASTC_2D_6X5
// 0x52 	ASTC_2D_8X6
// 0x53 	ASTC_2D_10X8
// 0x54 	ASTC_2D_12X10
// 0x55 	ASTC_2D_8X5
// 0x56 	ASTC_2D_10X5
// 0x57 	ASTC_2D_10X6
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, FromPrimitive)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum NVTextureFormat {
    R32_G32_B32_A32 = 0x01,
    R32_G32_B32 = 0x02,
    R16_G16_B16_A16 = 0x03,
    R32_G32 = 0x04,
    R32_B24G8 = 0x05,
    ETC2_RGB = 0x06,
    X8B8G8R8 = 0x07,
    A8B8G8R8 = 0x08,
    A2B10G10R10 = 0x09,
    ETC2_RGB_PTA = 0x0a,
    ETC2_RGBA = 0x0b,
    R16_G16 = 0x0c,
    G8R24 = 0x0d,
    G24R8 = 0x0e,
    R32 = 0x0f,
    BC6H_SF16 = 0x10,
    BC6H_UF16 = 0x11,
    A4B4G4R4 = 0x12,
    A5B5G5R1 = 0x13,
    A1B5G5R5 = 0x14,
    B5G6R5 = 0x15,
    B6G5R5 = 0x16,
    BC7U = 0x17,
    G8R8 = 0x18,
    EAC = 0x19,
    EACX2 = 0x1a,
    R16 = 0x1b,
    Y8_VIDEO = 0x1c,
    R8 = 0x1d,
    G4R4 = 0x1e,
    R1 = 0x1f,
    E5B9G9R9_SHAREDEXP = 0x20,
    BF10GF11RF11 = 0x21,
    G8B8G8R8 = 0x22,
    B8G8R8G8 = 0x23,
    DXT1 = 0x24,
    DXT23 = 0x25,
    DXT45 = 0x26,
    DXN1 = 0x27,
    DXN2 = 0x28,
    Z24S8 = 0x29,
    X8Z24 = 0x2a,
    S8Z24 = 0x2b,
    X4V4Z24_COV4R4V = 0x2c,
    X4V4Z24_COV8R8V = 0x2d,
    V8Z24_COV4R12V = 0x2e,
    ZF32 = 0x2f,
    ZF32_X24S8 = 0x30,
    X8Z24_X20V4S8_COV4R4V = 0x31,
    X8Z24_X20V4S8_COV8R8V = 0x32,
    ZF32_X20V4X8_COV4R4V = 0x33,
    ZF32_X20V4X8_COV8R8V = 0x34,
    ZF32_X20V4S8_COV4R4V = 0x35,
    ZF32_X20V4S8_COV8R8V = 0x36,
    X8Z24_X16V8S8_COV4R12V = 0x37,
    ZF32_X16V8X8_COV4R12V = 0x38,
    ZF32_X16V8S8_COV4R12V = 0x39,
    Z16 = 0x3a,
    V8Z24_COV8R24V = 0x3b,
    X8Z24_X16V8S8_COV8R24V = 0x3c,
    ZF32_X16V8X8_COV8R24V = 0x3d,
    ZF32_X16V8S8_COV8R24V = 0x3e,
    ASTC_2D_4X4 = 0x40,
    ASTC_2D_5X5 = 0x41,
    ASTC_2D_6X6 = 0x42,
    ASTC_2D_8X8 = 0x44,
    ASTC_2D_10X10 = 0x45,
    ASTC_2D_12X12 = 0x46,
    ASTC_2D_5X4 = 0x50,
    ASTC_2D_6X5 = 0x51,
    ASTC_2D_8X6 = 0x52,
    ASTC_2D_10X8 = 0x53,
    ASTC_2D_12X10 = 0x54,
    ASTC_2D_8X5 = 0x55,
    ASTC_2D_10X5 = 0x56,
    ASTC_2D_10X6 = 0x57,
    INVALID = 0xFF,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, FromPrimitive)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum NVTextureChannelDataType {
    SNORM = 1,
    UNORM = 2,
    SINT = 3,
    UINT = 4,
    SNORM_FORCE_FP16 = 5,
    UNORM_FORCE_FP16 = 6,
    FLOAT = 7,
    INVALID = 0xFF,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, FromPrimitive)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum NVTextureComponentSwizzle {
    ZERO = 0,
    R = 2,
    G = 3,
    B = 4,
    A = 5,
    ONE_INT = 6,
    ONE_FLOAT = 7,
    INVALID = 0xFF,
}

pub fn nv_texture_format_to_vk_format(
    format: NVTextureFormat,
    channel_data_types: [NVTextureChannelDataType; 4],
) -> Option<VkFormat> {
    match format {
        NVTextureFormat::R32_G32_B32_A32 => match channel_data_types[0] {
            NVTextureChannelDataType::FLOAT => Some(VK_FORMAT_R32G32B32A32_SFLOAT),
            NVTextureChannelDataType::UINT => Some(VK_FORMAT_R32G32B32A32_UINT),
            NVTextureChannelDataType::SINT => Some(VK_FORMAT_R32G32B32A32_SINT),
            NVTextureChannelDataType::SNORM => None,
            NVTextureChannelDataType::UNORM => None,
            _ => None,
        },
        NVTextureFormat::R32_G32_B32 => match channel_data_types[0] {
            NVTextureChannelDataType::FLOAT => Some(VK_FORMAT_R32G32B32_SFLOAT),
            NVTextureChannelDataType::UINT => Some(VK_FORMAT_R32G32B32_UINT),
            NVTextureChannelDataType::SINT => Some(VK_FORMAT_R32G32B32_SINT),
            NVTextureChannelDataType::SNORM => None,
            NVTextureChannelDataType::UNORM => None,
            _ => None,
        },
        NVTextureFormat::R16_G16_B16_A16 => match channel_data_types[0] {
            NVTextureChannelDataType::FLOAT => Some(VK_FORMAT_R16G16B16A16_SFLOAT),
            NVTextureChannelDataType::UINT => Some(VK_FORMAT_R16G16B16A16_UINT),
            NVTextureChannelDataType::SINT => Some(VK_FORMAT_R16G16B16A16_SINT),
            NVTextureChannelDataType::SNORM => Some(VK_FORMAT_R16G16B16A16_SNORM),
            NVTextureChannelDataType::UNORM => Some(VK_FORMAT_R16G16B16A16_UNORM),
            _ => None,
        },
        NVTextureFormat::R32_G32 => match channel_data_types[0] {
            NVTextureChannelDataType::FLOAT => Some(VK_FORMAT_R32G32_SFLOAT),
            NVTextureChannelDataType::UINT => Some(VK_FORMAT_R32G32_UINT),
            NVTextureChannelDataType::SINT => Some(VK_FORMAT_R32G32_SINT),
            NVTextureChannelDataType::SNORM => None,
            NVTextureChannelDataType::UNORM => None,
            _ => None,
        },
        NVTextureFormat::R32_B24G8 => None,
        NVTextureFormat::ETC2_RGB => None, // TODO
        NVTextureFormat::X8B8G8R8 => match channel_data_types[0] {
            NVTextureChannelDataType::UNORM => Some(VK_FORMAT_A8B8G8R8_UNORM_PACK32),
            _ => None,
        },
        NVTextureFormat::A8B8G8R8 => Some(VK_FORMAT_R8G8B8A8_UNORM),
        NVTextureFormat::A2B10G10R10 => Some(VK_FORMAT_A2B10G10R10_UNORM_PACK32),
        NVTextureFormat::ETC2_RGB_PTA => Some(VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK),
        NVTextureFormat::ETC2_RGBA => Some(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK),
        _ => None,
    }
}

pub fn nv_tic_format(tic: &NVTextureImageControl) -> NVTextureFormat {
    NVTextureFormat::from_u8(getbits(tic[0], 0..=6) as u8).unwrap_or(NVTextureFormat::INVALID)
}

pub fn nv_tic_channel_data_types(tic: &NVTextureImageControl) -> [NVTextureChannelDataType; 4] {
    let r = getbits(tic[0], 7..=9) as u8;
    let g = getbits(tic[0], 10..=12) as u8;
    let b = getbits(tic[0], 13..=15) as u8;
    let a = getbits(tic[0], 16..=18) as u8;
    [
        NVTextureChannelDataType::from_u8(r).unwrap_or(NVTextureChannelDataType::INVALID),
        NVTextureChannelDataType::from_u8(g).unwrap_or(NVTextureChannelDataType::INVALID),
        NVTextureChannelDataType::from_u8(b).unwrap_or(NVTextureChannelDataType::INVALID),
        NVTextureChannelDataType::from_u8(a).unwrap_or(NVTextureChannelDataType::INVALID),
    ]
}

pub fn nv_tic_swizzle(tic: &NVTextureImageControl) -> [NVTextureComponentSwizzle; 4] {
    let r = NVTextureComponentSwizzle::from_u8(getbits(tic[0], 19..=21) as u8)
        .unwrap_or(NVTextureComponentSwizzle::INVALID);
    let g = NVTextureComponentSwizzle::from_u8(getbits(tic[0], 22..=24) as u8)
        .unwrap_or(NVTextureComponentSwizzle::INVALID);
    let b = NVTextureComponentSwizzle::from_u8(getbits(tic[0], 25..=27) as u8)
        .unwrap_or(NVTextureComponentSwizzle::INVALID);
    let a = NVTextureComponentSwizzle::from_u8(getbits(tic[0], 28..=30) as u8)
        .unwrap_or(NVTextureComponentSwizzle::INVALID);
    [r, g, b, a]
}

pub fn nv_tic_va(tic: &NVTextureImageControl) -> u64 {
    let lo = (tic[1] & 0xFFFF_FE00) as u64;
    let hi = (tic[2] & 0xFFFF) as u64;
    (hi << 32) | lo
}

pub fn nv_tic_image_size(tic: &NVTextureImageControl) -> [u32; 3] {
    let w = getbits(tic[4], 0..=15);
    let h = getbits(tic[5], 0..=15);
    let d = getbits(tic[5], 16..=29);
    [w + 1, h + 1, d + 1]
}

pub fn nv_tic_min_mip_level(tic: &NVTextureImageControl) -> u32 {
    getbits(tic[7], 0..=3)
}

pub fn nv_tic_max_mip_level(tic: &NVTextureImageControl) -> u32 {
    getbits(tic[7], 4..=7)
}

pub fn format_nv_tic(tic: &NVTextureImageControl) -> String {
    let format = nv_tic_format(tic);
    let channel_types = nv_tic_channel_data_types(tic);
    let swizzle = nv_tic_swizzle(tic);
    let va = nv_tic_va(tic);
    let size = nv_tic_image_size(tic);
    let min_mip = nv_tic_min_mip_level(tic);
    let max_mip = nv_tic_max_mip_level(tic);
    format!(
        "format={format:?}\nchannel_types={channel_types:?}\nswizzle={swizzle:?}\ngpuva=0x{va:016X}\nsize={size:?}\nmin_mip={min_mip}\nmax_mip={max_mip}"
    )
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ImageViewInfo {
    pub format: VkFormat,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl ImageViewInfo {
    pub fn from_descriptor(blob: &[u8; 64]) -> Self {
        let tic = unsafe { &*(blob.as_ptr() as *const NVTextureImageControl) };
        let channel_types = nv_tic_channel_data_types(tic);
        let format = nv_texture_format_to_vk_format(nv_tic_format(tic), channel_types).unwrap_or(VK_FORMAT_UNDEFINED);
        let [width, height, depth] = nv_tic_image_size(tic);
        Self { format, width, height, depth }
    }
}
