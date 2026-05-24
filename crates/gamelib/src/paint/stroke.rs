//! Stroke expansion

use crate::paint::flatten::{approx_equal, flatten_path};
use crate::paint::{PathBuilder, PathSlice};
use math::{Mat3, Vec2, vec2};

/// Stroke options.
#[derive(Clone, Copy, Debug)]
pub struct StrokeOptions {
    /// Stroke width in pixels.
    pub width: f32,
    pub miter_limit: f32 = 4.0,
}

pub(crate) fn expand_stroke(
    path: PathSlice,
    transform: &Mat3,
    stroke_options: &StrokeOptions,
    output: &mut PathBuilder,
) {
    // flatten path to polyline
    let mut points = Vec::new();
    let mut contours = Vec::new();
    flatten_path(path, &Mat3::IDENTITY, 0.5, &mut points, &mut contours);

    // expand each contour
    for contour in contours {
        expand_stroke_contour(&points[contour], stroke_options, output);
    }
}

fn miter_vector(n0: Vec2, n1: Vec2, half_width: f32, miter_limit: f32) -> Vec2 {
    let sum = n0 + n1;
    let len = sum.length();
    if len < 1e-6 {
        return n0 * half_width;
    }
    let miter_dir = sum / len;
    let cos_half = miter_dir.dot(n0).max(0.05);
    let miter_len = (half_width / cos_half).min(miter_limit * half_width);
    miter_dir * miter_len
}

fn expand_stroke_contour(pts: &[Vec2], stroke_options: &StrokeOptions, output: &mut PathBuilder) {
    let n = pts.len();
    if n < 2 {
        return;
    }

    let is_closed = approx_equal(pts[0], pts[n - 1]);

    let normals = (0..n - 1)
        .map(|i| {
            let a = pts[i];
            let b = pts[i + 1];
            let dir = (b - a).normalize_or_zero();
            vec2(-dir.y, dir.x)
        })
        .collect::<Vec<_>>();

    if !is_closed {
        let offsets = (0..n)
            .map(|i| {
                if i == 0 {
                    normals[0]
                } else if i == n - 1 {
                    normals[normals.len() - 1]
                } else {
                    let n0 = normals[i - 1];
                    let n1 = normals[i];
                    miter_vector(n0, n1, stroke_options.width * 0.5, stroke_options.miter_limit)
                }
            })
            .collect::<Vec<_>>();

        output.move_to(pts[0] + offsets[0]);
        for i in 0..n {
            output.line_to(pts[i] + offsets[i]);
        }
        // TODO end cap
        for i in (0..n).rev() {
            output.line_to(pts[i] - offsets[i]);
        }
        // TODO start cap
        output.close();
    } else {
        let offsets = (0..n - 1)
            .map(|i| {
                let n0 = normals[if n == 0 { n - 2 } else { i }];
                let n1 = normals[i + 1];
                miter_vector(n0, n1, stroke_options.width * 0.5, stroke_options.miter_limit)
            })
            .collect::<Vec<_>>();
        // outer loop
        output.move_to(pts[0] + offsets[0]);
        for i in 0..(n - 1) {
            output.line_to(pts[i] + offsets[i]);
        }
        // inner loop
        output.move_to(pts[0] - offsets[0]);
        for i in (0..(n - 1)).rev() {
            output.line_to(pts[i] - offsets[i]);
        }
    }
}
