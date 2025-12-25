//! Color types and related functions.


/// Converts a linear sRGB component to non-linear sRGB.
pub fn srgb_linear_to_encoded_f32(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

pub fn srgb_linear_to_encoded(c: f32) -> u8 {
    (srgb_linear_to_encoded_f32(c).clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn srgb_encoded_f32_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn srgb_encoded_to_linear(c: u8) -> f32 {
    srgb_encoded_f32_to_linear(c as f32 / 255.0)
}

/// Linear sRGB color with components in the range [0.0, 1.0]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct LinSrgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl LinSrgba {
    pub fn to_srgba8(self) -> Srgba8 {
        Srgba8 {
            r: srgb_linear_to_encoded(self.r),
            g: srgb_linear_to_encoded(self.g),
            b: srgb_linear_to_encoded(self.b),
            a: (self.a.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

/// Encoded sRGB color with components in the range `[0, 255]` packed into 4 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Srgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Srgba8 {
    /// Transparent color (rgb = 0).
    pub const TRANSPARENT: Self = srgba8(0, 0, 0, 0);
    /// Opaque black color.
    pub const BLACK: Self = srgba8(0, 0, 0, 255);
    /// Opaque white color.
    pub const WHITE: Self = srgba8(255, 255, 255, 255);

    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Srgba8 { r, g, b, a }
    }

    pub fn to_linear(self) -> LinSrgba {
        LinSrgba {
            r: srgb_encoded_to_linear(self.r),
            g: srgb_encoded_to_linear(self.g),
            b: srgb_encoded_to_linear(self.b),
            a: self.a as f32 / 255.0,
        }
    }

    /// Constructs a `Srgba32` color from linear sRGB components.
    pub fn from_linear(r: f32, g: f32, b: f32, a: f32) -> Srgba8 {
        Srgba8 {
            r: srgb_linear_to_encoded(r),
            g: srgb_linear_to_encoded(g),
            b: srgb_linear_to_encoded(b),
            a: (a.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

impl From<[u8;4]> for Srgba8 {
    fn from(arr: [u8; 4]) -> Self {
        Srgba8 {
            r: arr[0],
            g: arr[1],
            b: arr[2],
            a: arr[3],
        }
    }
}

/// Short-hand constructor for `Srgba32`, from encoded sRGB components.
pub const fn srgba8(r: u8, g: u8, b: u8, a: u8) -> Srgba8 {
    Srgba8 { r, g, b, a }
}

#[cfg(feature = "gpu-support")]
unsafe impl gpu::VertexAttribute for Srgba8 {
    const FORMAT: gpu::Format = gpu::Format::R8G8B8A8_UNORM;
}
