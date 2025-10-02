pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

pub fn linear_to_srgb_u8(c: f32) -> u8 {
    (linear_to_srgb(c).clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn srgb_u8_to_linear(c: u8) -> f32 {
    srgb_to_linear(c as f32 / 255.0)
}

/// RGBA color with linear components in the range [0.0, 1.0]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub fn to_srgba32(self) -> Srgba32 {
        Srgba32 {
            r: linear_to_srgb_u8(self.r),
            g: linear_to_srgb_u8(self.g),
            b: linear_to_srgb_u8(self.b),
            a: (self.a.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

/// RGBA color with sRGB components in the range `[0, 255]` packed into 4 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Srgba32 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Srgba32 {
    /// Transparent color (rgb = 0).
    pub const TRANSPARENT: Self = srgba32(0, 0, 0, 0);
    /// Opaque black color.
    pub const BLACK: Self = srgba32(0, 0, 0, 255);
    /// Opaque white color.
    pub const WHITE: Self = srgba32(255, 255, 255, 255);

    pub fn to_rgba(self) -> Rgba {
        Rgba {
            r: srgb_u8_to_linear(self.r),
            g: srgb_u8_to_linear(self.g),
            b: srgb_u8_to_linear(self.b),
            a: self.a as f32 / 255.0,
        }
    }
}

/// Short-hand constructor for `Srgba32`.
pub const fn srgba32(r: u8, g: u8, b: u8, a: u8) -> Srgba32 {
    Srgba32 { r, g, b, a }
}

unsafe impl gpu::VertexAttribute for Srgba32 {
    const FORMAT: gpu::Format = gpu::Format::R8G8B8A8_UNORM;
}