//! Flatten paths into a list of line segments.

use crate::paint::{PathSegment, PathSlice};
use math::geom::CubicBezier2;
use math::{Mat3, Vec2};
use std::ops::Range;

/// Returns whether two points are approximately equal, within a small epsilon.
fn approx_equal(a: Vec2, b: Vec2) -> bool {
    const EPSILON: f32 = 0.03125; // 1/32th of a pixel
    a.distance_squared(b) < EPSILON * EPSILON
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Contour {
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) path: u32,
}

/// Flattens a path into a list of line segments.
///
/// Returns a tuple `(point_count, contour_count)` representing the number of added points and contours.
pub(super) fn flatten_path(
    path: PathSlice,
    transform: &Mat3,
    accuracy: f32,
    path_index: u32,
    points: &mut Vec<Vec2>,
    contours: &mut Vec<Contour>,
) -> (u32, u32) {
    let start_points_len = points.len();
    let start_contours_len = contours.len();

    let mut cur_contour = None;
    let mut pos = Vec2::ZERO;
    for segment in path.iter() {
        match segment {
            PathSegment::MoveTo(to) => {
                // finish current contour and begin a new one
                if let Some(start) = cur_contour {
                    contours.push(Contour {
                        start: start as u32,
                        end: points.len() as u32,
                        path: path_index,
                    });
                }

                cur_contour = Some(points.len());
                let tto = transform.transform_point2(to);
                points.push(tto);
                pos = tto;
            }
            PathSegment::LineTo(to) => {
                let tto = transform.transform_point2(to);
                points.push(tto);
                pos = tto;
            }
            PathSegment::QuadTo { ctrl, to } => {
                // transform to cubic and flatten that
                // TODO: flatten quads directly
                let tctrl = transform.transform_point2(ctrl);
                let tto = transform.transform_point2(to);
                flatten_cubic(&quadratic_to_cubic(pos, tctrl, tto), accuracy, points);
                pos = tto;
            }
            PathSegment::CubicTo { ctrl1, ctrl2, to } => {
                let tctrl1 = transform.transform_point2(ctrl1);
                let tctrl2 = transform.transform_point2(ctrl2);
                let tto = transform.transform_point2(to);
                flatten_cubic(&CubicBezier2 { p0: pos, p1: tctrl1, p2: tctrl2, p3: tto }, accuracy, points);
                pos = tto;
            }
            PathSegment::Close => {
                let start = cur_contour.take().expect("malformed path: close outside of active contour");
                if !approx_equal(pos, points[start]) {
                    points.push(points[start]);
                }
                contours.push(Contour {
                    start: start as u32,
                    end: points.len() as u32,
                    path: path_index,
                });
            }
        }
    }
    if let Some(start) = cur_contour {
        contours.push(Contour {
            start: start as u32,
            end: points.len() as u32,
            path: path_index,
        });
    }

    let point_count = points.len() - start_points_len;
    let contour_count = contours.len() - start_contours_len;
    (point_count as u32, contour_count as u32)
}

fn quadratic_to_cubic(from: Vec2, ctrl: Vec2, to: Vec2) -> CubicBezier2 {
    CubicBezier2 { p0: Vec2::ZERO, p1: ctrl * (2.0 / 3.0), p2: to * (2.0 / 3.0) + ctrl * (1.0 / 3.0), p3: to }
}

fn flatten_cubic(cubic: &CubicBezier2, accuracy: f32, points: &mut Vec<Vec2>) {
    if cubic.is_flat(accuracy) {
        points.push(cubic.p3);
    } else {
        let (a, b) = cubic.subdivide(0.5);
        flatten_cubic(&a, accuracy, points);
        flatten_cubic(&b, accuracy, points);
    }
}
