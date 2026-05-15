use crate::{Vec2, Vec3};
use std::iter;
macro_rules! impl_bezier {
    ($(#[$attr:meta])* $name:ident, $vec:ty) => {
        $(#[$attr])*
        #[derive(Copy, Clone, Debug)]
        pub struct $name {
            pub p0: $vec,
            pub p1: $vec,
            pub p2: $vec,
            pub p3: $vec,
        }

        impl $name {
            /// Splits this curve into two Bézier segments at the specified parameter `t` (0 <= t <= 1).
            pub fn subdivide(&self, t: f32) -> (Self, Self) {
                let q0 = self.p0.lerp(self.p1, t);
                let q1 = self.p1.lerp(self.p2, t);
                let q2 = self.p2.lerp(self.p3, t);
                let r0 = q0.lerp(q1, t);
                let r1 = q1.lerp(q2, t);
                let p = r0.lerp(r1, t);

                (
                    Self {
                        p0: self.p0,
                        p3: p,
                        p1: q0,
                        p2: r0,
                    },
                    Self {
                        p0: p,
                        p3: self.p3,
                        p1: r1,
                        p2: q2,
                    },
                )
            }

            /// Whether this curve is flat enough to be approximated by a line segment.
            pub fn is_flat(&self, tolerance: f32) -> bool {
                let p0 = self.p0;
                let p1 = self.p1;
                let p2 = self.p2;
                let p3 = self.p3;
                let t = tolerance * tolerance;
                (0.5 * (p0 + p2) - p1).length_squared() <= t && (0.5 * (p1 + p3) - p2).length_squared() <= t
            }

            fn flatten_inner(&self, points: &mut Vec<$vec>, tolerance: f32) {
                //points.push(self.start);
                if self.is_flat(tolerance) {
                    points.push(self.p3);
                } else {
                    let (a, b) = self.subdivide(0.5);
                    a.flatten_inner(points, tolerance);
                    b.flatten_inner(points, tolerance);
                }
            }

            /// Flattens this curve segment to a polyline.
            ///
            /// # Arguments
            ///
            /// * `points` - A vector to which the resulting points will be appended. The first point of the curve (p0) will be added only if the vector is empty, otherwise only the subsequent points will be added.
            /// * `tolerance` - The maximum allowed deviation from the curve to the polyline. Smaller values result in more points and a closer approximation.
            pub fn flatten(&self, points: &mut Vec<$vec>, tolerance: f32) {
                if points.is_empty() {
                    points.push(self.p0);
                }
                self.flatten_inner(points, tolerance);
            }
        }
    }
}

impl_bezier!(
     /// 2D cubic Bézier curve segment.
     CubicBezier2,
     Vec2
);

impl_bezier!(
     /// 3D cubic Bézier curve segment.
     CubicBezier3,
     Vec3
);

/*
/// Checks if a cubic Bézier lies within a distance from the origin.
///
/// This only checks the control points, not the endpoints, which are assumed to lie within the
/// tolerance already.
///
/// Port of https://github.com/fonttools/fonttools/blob/3b9a73ff8379ab49d3ce35aaaaf04b3a7d9d1655/Lib/fontTools/cu2qu/cu2qu.py#L281
fn cubic_farthest_fit_inside(cubic: CubicBezier2, tolerance: f32) -> bool {
    if cubic.p2.length() <= tolerance && cubic.p1.length() <= tolerance {
        return true;
    }

    let p0 = cubic.p0;
    let p1 = cubic.p1;
    let p2 = cubic.p2;
    let p3 = cubic.p3;

    let mid = (p0 + 3.0 * (p1 + p2) + p3) * 0.125;
    if mid.length() > tolerance {
        return false;
    }
    let (left, right) = cubic.subdivide(0.5);
    cubic_farthest_fit_inside(left, tolerance) && cubic_farthest_fit_inside(right, tolerance)
}

/// Approximates a cubic Bézier segment with a *single* quadratic Bézier segment.
///
/// Returns the control point of the quadratic Bézier.
fn cubic_to_quadratic_single(cubic: CubicBezier2) -> Vec2 {
    let p0 = cubic.p0;
    let p1 = cubic.p1;
    let p2 = cubic.p2;
    let p3 = cubic.p3;
    (3.0 * (p1 + p2) - p0 - p3) / 4.0
}

fn approx_quad_control(c: &CubicBezier2, t: f32) -> Vec2 {
    let p1 = c.p0 + (c.p1 - c.p0) * 1.5;
    let p2 = c.p3 + (c.p2 - c.p3) * 1.5;
    p1.lerp(p2, t)
}

// adapted from kurbo
fn try_approx_cubic_to_quadratic_control(cubic: CubicBezier2, accuracy: f32) -> Option<Vec2> {
    if let Some(q1) = calc_intersect(cubic.p0, cubic.p1, cubic.p3, cubic.p2)
        .or_else(|| (cubic.p1 == cubic.p2 && (cubic.p0 == cubic.p1 || cubic.p3 == cubic.p2)).then_some(cubic.p1))
    {
        let c1 = cubic.p0.lerp(q1, 2.0 / 3.0);
        let c2 = cubic.p3.lerp(q1, 2.0 / 3.0);
        if !cubic_farthest_fit_inside(
            CubicBezier2 {
                p0: Vec2::ZERO,
                p1: c1,
                p2: c2,
                p3: Vec2::ZERO,
            },
            accuracy,
        ) {
            return None;
        }
        Some(q1)
    } else {
        None
    }
}

/// Calculates the intersection point of two lines (a,b) and (c,d).
/// Adapted from https://github.com/googlefonts/cu2qu/blob/4cbc9b6bd48acad95003995428132f859aeba4df/Lib/cu2qu/cu2qu.py#L154
fn calc_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> Option<Vec2> {
    let ab = b - a;
    let cd = d - c;
    let p = Vec2::new(-ab.y, ab.x);
    let denom = cd.dot(p);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (c - a).dot(p) / denom;
    Some(c + cd * t)
}

fn cubic_coeffs(cubic: CubicBezier2) -> (Vec2, Vec2, Vec2, Vec2) {
    let p0 = cubic.p0;
    let p1 = cubic.p1;
    let p2 = cubic.p2;
    let p3 = cubic.p3;

    let a = -p0 + 3.0 * (p1 - p2) + p3;
    let b = 3.0 * (p0 - 2.0 * p1 + p2);
    let c = 3.0 * (p1 - p0);
    let d = p0;

    (a, b, c, d)
}

fn cubic_from_coeffs(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> CubicBezier2 {
    let p0 = d;
    let p1 = d + c / 3.0;
    let p2 = p1 + (c + b) / 3.0;
    let p3 = (d + c + b + a) / 3.0;

    CubicBezier2 { p0, p1, p2, p3 }
}

// split a cubic into n segments
fn cubic_subdiv_n(cubic: CubicBezier2, n: usize) -> impl Iterator<Item = CubicBezier2> {
    let mut i = 0;
    let (a, b, c, d) = cubic_coeffs(cubic);
    let dt = 1.0f32 / n as f32;
    let delta_2 = dt * dt; // 1 / n^2
    let delta_3 = dt * delta_2; // 1 / n^3

    iter::from_fn(move || {
        if i >= n {
            return None;
        }
        let t1 = i as f32 * dt; // i / n
        let t1_2 = t1 * t1; // (i / n)^2
        let a1 = a * delta_3;
        let b1 = (3.0 * a * t1 + b) * delta_2;
        let c1 = (3.0 * a * t1_2 + 2.0 * b * t1 + c) * dt;
        let d1 = a * t1 * t1_2 + b * t1_2 + c * t1 + d;
        let result = cubic_from_coeffs(a1, b1, c1, d1);
        i += 1;
        Some(result)
    })
}

// adapted from kurbo
// https://docs.rs/kurbo/latest/src/kurbo/cubicbez.rs.html
fn approx_cubic_to_quadratic_spline_n(c: CubicBezier2, n: usize, accuracy: f32, out: &mut Vec<Vec2>) -> bool {
    if n == 1 {
        if let Some(q1) = try_approx_cubic_to_quadratic_control(c, accuracy) {
            out.push(q1);
            out.push(c.p3);
            true
        } else {
            false
        }
    } else {
        let mut cubics = cubic_subdiv_n(c, n);
        let mut next_cubic = cubics.next().unwrap();
        let mut next_q1 = approx_quad_control(&next_cubic, 0.0);
        let mut q2 = c.p0;
        let mut d1 = Vec2::ZERO;

        let save = out.len();
        out.push(c.p0);
        out.push(next_q1);

        for i in 1..=n {
            let current_cubic = next_cubic;
            let q0 = q2;
            let q1 = next_q1;
            q2 = if i < n {
                next_cubic = cubics.next().unwrap();
                next_q1 = approx_quad_control(&next_cubic, i as f32 / (n - 1) as f32);

                out.push(next_q1);
                q1.midpoint(next_q1)
            } else {
                current_cubic.p3
            };
            let d0 = d1;
            d1 = q2 - current_cubic.p3;

            if d1.length() > accuracy
                || !cubic_farthest_fit_inside(
                    CubicBezier2 {
                        p0: d0,
                        p1: q0.lerp(q1, 2.0 / 3.0) - current_cubic.p1,
                        p2: q2.lerp(q1, 2.0 / 3.0) - current_cubic.p2,
                        p3: d1,
                    },
                    accuracy,
                )
            {
                out.truncate(save);
                return false;
            }
        }
        out.push(c.p3);
        true
    }
}*/

/*
/// Approximates a 2D cubic Bézier segment with a sequence of quadratic Bézier curves, and appends the control points to `out`.
pub fn cubic_to_quadratic_spline(segment: CubicBezier2, out: &mut Vec<Vec2>) -> bool {

    let mut split_order = 0;
    let save = out.len();

    const MAX_SPLINE_SPLIT : usize = 10;
    while split_order <= MAX_SPLINE_SPLIT {
        split_order += 1;
        out.truncate(save);

        match curve.approx_spline_n(split_order, accuracy) {
            Some(spline) => result.push(spline),
            None => break,
        }

        if result.len() == curves.len() {
            return Some(result);
        }
    }
    None

}
*/