//! Color types and conversion utilities.
use math::{Vec4, vec4};

/// Converts one linear-light sRGB channel value to gamma-encoded sRGB.
pub fn srgb_linear_to_encoded_f32(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Converts one linear-light sRGB channel to gamma-encoded sRGB, quantizing the result as a `u8` value in `[0, 255]`.
pub fn srgb_linear_to_encoded(c: f32) -> u8 {
    (srgb_linear_to_encoded_f32(c).clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Converts one gamma-encoded sRGB channel value in `[0, 1]` to linear sRGB.
pub fn srgb_encoded_f32_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Converts a gamma-encoded sRGB `u8` channel value to a linear-light `f32` in `[0.0, 1.0]`.
///
/// Equivalent to `srgb_encoded_f32_to_linear(c as f32 / 255.0)`.
pub fn srgb_encoded_to_linear(c: u8) -> f32 {
    srgb_encoded_f32_to_linear(c as f32 / 255.0)
}

/// A color in linear sRGB space with a straight alpha channel.
///
/// Use [`to_srgba8`](LinSrgba::to_srgba8) to encode as a [`Srgba8`] color for storage or display.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct LinSrgba {
    /// Red channel, linear light intensity.
    pub r: f32,
    /// Green channel, linear light intensity.
    pub g: f32,
    /// Blue channel, linear light intensity.
    pub b: f32,
    /// Alpha (opacity).
    pub a: f32,
}

impl LinSrgba {
    /// Converts this color to a gamma-encoded [`Srgba8`].
    pub fn to_srgba8(self) -> Srgba8 {
        Srgba8 {
            r: srgb_linear_to_encoded(self.r),
            g: srgb_linear_to_encoded(self.g),
            b: srgb_linear_to_encoded(self.b),
            a: (self.a.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }

    /// Returns the RGBA components as a [`Vec4`] in `(x, y, z, w) = (r, g, b, a)` order.
    pub fn to_vec4(self) -> Vec4 {
        vec4(self.r, self.g, self.b, self.a)
    }

    /// Returns whether the color is fully opaque (alpha = 1.0).
    pub fn is_opaque(&self) -> bool {
        self.a >= 1.0
    }
}

/// A color in gamma-encoded sRGB space with a straight alpha channel, stored as four `u8` bytes (8 bits per channel, 32 bits per pixel).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
#[repr(C)]
pub struct Srgba8 {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha (opacity).
    pub a: u8,
}

impl Srgba8 {
    /// Fully transparent black `(0, 0, 0, 0)`.
    pub const TRANSPARENT: Self = srgba8(0, 0, 0, 0);
    /// Fully opaque black `(0, 0, 0, 255)`.
    pub const BLACK: Self = srgba8(0, 0, 0, 255);
    /// Fully opaque white `(255, 255, 255, 255)`.
    pub const WHITE: Self = srgba8(255, 255, 255, 255);

    /// Constructs a [`Srgba8`] color from gamma-encoded 8-bit sRGB + alpha values.
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Srgba8 { r, g, b, a }
    }

    /// Returns whether the color is fully opaque (alpha = 255).
    pub fn is_opaque(&self) -> bool {
        self.a == 255
    }

    /// Decodes this color into linear-light [`LinSrgba`].
    ///
    /// RGB channels are expanded via [`srgb_encoded_to_linear`]; alpha is divided by 255.
    pub fn to_linear(self) -> LinSrgba {
        LinSrgba {
            r: srgb_encoded_to_linear(self.r),
            g: srgb_encoded_to_linear(self.g),
            b: srgb_encoded_to_linear(self.b),
            a: self.a as f32 / 255.0,
        }
    }
    
    /// Decodes this color into linear-light `f32` array.
    pub fn to_linear_array(self) -> [f32; 4] {
        [
            srgb_encoded_to_linear(self.r),
            srgb_encoded_to_linear(self.g),
            srgb_encoded_to_linear(self.b),
            self.a as f32 / 255.0,
        ]
    }
    
    pub fn to_float_array(self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }

    /// Constructs an `Srgba8` from *linear*-light sRGB `f32` components.
    ///
    /// RGB values are encoded via [`srgb_linear_to_encoded`]. Alpha is
    /// clamped to `[0, 1]` and scaled to `[0, 255]`.
    pub fn from_linear(r: f32, g: f32, b: f32, a: f32) -> Srgba8 {
        Srgba8 {
            r: srgb_linear_to_encoded(r),
            g: srgb_linear_to_encoded(g),
            b: srgb_linear_to_encoded(b),
            a: (a.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }

    /// Constructs an `Srgba8` from HSL + alpha values.
    ///
    /// TODO review doc
    ///
    /// Uses the standard HSL → encoded-sRGB conversion (no linear-light intermediate step, as HSL
    /// is defined in terms of encoded sRGB channel values).
    ///
    /// # Parameters
    /// - `h`: hue in degrees, `[0.0, 360.0)` values outside this range are wrapped.
    /// - `s`: saturation, `[0.0, 1.0]` 0 gives a grey, 1 gives a fully saturated hue.
    /// - `l`: lightness, `[0.0, 1.0]` 0 is black, 0.5 is the pure hue, 1 is white.
    /// - `a`: alpha (opacity), `[0.0, 1.0]`.
    pub fn from_hsla(h: f32, s: f32, l: f32, a: f32) -> Self {
        let h = h / 360.0;

        let (r, g, b) = if s == 0.0 {
            (l, l, l)
        } else {
            let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
            let p = 2.0 * l - q;
            (hue_to_channel(p, q, h + 1.0 / 3.0), hue_to_channel(p, q, h), hue_to_channel(p, q, h - 1.0 / 3.0))
        };

        Srgba8 {
            r: (r.clamp(0.0, 1.0) * 255.0).round() as u8,
            g: (g.clamp(0.0, 1.0) * 255.0).round() as u8,
            b: (b.clamp(0.0, 1.0) * 255.0).round() as u8,
            a: (a.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

/// Maps an HSL hue fraction `t` to a single channel value using the piecewise HSL interpolation
/// formula. `p` and `q` are the lower and upper luminance bounds derived from lightness and
/// saturation. `t` is automatically wrapped into `[0, 1]`.
fn hue_to_channel(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

impl From<[u8; 4]> for Srgba8 {
    /// Constructs an `Srgba8` from a raw `[r, g, b, a]` byte array.
    fn from(arr: [u8; 4]) -> Self {
        Srgba8 { r: arr[0], g: arr[1], b: arr[2], a: arr[3] }
    }
}

/// Constructs a [`Srgba8`] color from gamma-encoded 8-bit sRGB + alpha values.
///
/// Shorthand for `Srgba8::new(r, g, b, a)`.
pub const fn srgba8(r: u8, g: u8, b: u8, a: u8) -> Srgba8 {
    Srgba8 { r, g, b, a }
}
