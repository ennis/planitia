use color::Srgba8;
use math::Vec4;
use std::ops::Range;

/// Gradient color stop.
#[derive(Debug, Clone, Copy)]
pub struct ColorStop {
    /// Normalized position of the color stop on the gradient, in the range `[0, 1]`.
    pub position: f32,
    /// Color.
    pub color: Srgba8,
}

/// Describes how a gradient extends beyond its start and end points.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum GradientExtendMode {
    /// The gradient is clamped to the start and end colors beyond the start and end points.
    Clamp = 0,
    /// The gradient is repeated in a normal fashion (cycled) beyond the start and end points.
    Repeat = 1,
    /// The gradient is repeated in a mirrored fashion beyond the start and end points.
    Mirror = 2,
}

/// Segment of the integral of a gradient ramp.
///
/// Directly sampling the color ramp at a pixel may cause aliasing if the gradient changes rapidly.
/// Instead, we get the gradient color at a pixel by integrating the gradient ramp over the
/// pixel span.
/// This integral can be evaluated numerically, but for gradient ramps, which are piecewise linear,
/// the integral is a piecewise quadratic function which can be evaluated easily.
/// This struct represents a segment of this piecewise quadratic function between `t_min` and `t_max`.
///
/// The function represented by this segment is a vector-valued polynomial
/// `f(t) = a * t^2 + b * t + c` for `t` in `[t_min, t_max]`,
/// where `a`, `b` and `c` are derived from the color stops.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct GradientIntegralSegment {
    pub(crate) a: Vec4,
    pub(crate) b: Vec4,
    pub(crate) c: Vec4,
    pub(crate) t_min: f32,
    pub(crate) t_max: f32,
    pub(crate) _padding: [f32; 2],
}

pub(crate) struct GradientRampData {
    pub(crate) segments: Range<usize>,
    pub(crate) integral: Vec4,
    pub(crate) opaque: bool,
}

pub(crate) fn compute_gradient_integral(
    color_stops: &[ColorStop],
    segments: &mut Vec<GradientIntegralSegment>,
) -> GradientRampData {
    let seg_start = segments.len();

    assert!(!color_stops.is_empty(), "Gradient must have at least one color stop");

    let n = color_stops.len();
    let opaque = color_stops.iter().all(|stop| stop.color.is_opaque());

    if n == 1 {
        let color = color_stops[0].color.to_linear().to_vec4();
        segments.push(GradientIntegralSegment {
            a: Vec4::ZERO,
            b: color,
            c: Vec4::ZERO,
            t_min: 0.0,
            t_max: 1.0,
            _padding: [0.0; 2],
        });
        return GradientRampData { segments: seg_start..(seg_start + 1), integral: color, opaque };
    }

    let mut acc = Vec4::splat(0.0);
    for i in 0..(n - 1) {
        let first = i;
        let next = (i + 1) % n;

        let t_l = color_stops[first].position;
        let t_r = color_stops[next].position;
        if (t_l - t_r).abs() < 1e-6 {
            continue;
        }

        let c_l = color_stops[first].color.to_linear().to_vec4();
        let c_r = color_stops[next].color.to_linear().to_vec4();

        let dt = t_r - t_l;
        let a = 0.5 * (c_r - c_l) / dt;
        let b = (c_l * t_r - c_r * t_l) / dt;
        let c = acc - t_l * (a * t_l + b);

        acc = t_r * (a * t_r + b) + c;
        segments.push(GradientIntegralSegment { a, b, c, t_min: t_l, t_max: t_r, _padding: [0.0; 2] });
    }

    /*for (i,s) in segments.iter().enumerate() {
        let a = s.a.x;
        let b = s.b.x;
        let c = s.c.x;
        let t_min = s.t_min;
        let t_max = s.t_max;
        let integral_min = a * t_min * t_min + b * t_min + c;
        let integral_max = a * t_max * t_max + b * t_max + c;
        eprintln!("[{i}] {a} * t² + {b} * t + {c} for t in [{t_min}, {t_max}] => integral in [{integral_min}, {integral_max}]");
    }*/

    GradientRampData { segments: seg_start..segments.len(), integral: acc, opaque }
}
