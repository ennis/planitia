//! 2D vector path representation and construction.
//!
//! A [`Path`] (or its borrowed counterpart [`PathSlice`]) describes a sequence of contours made up
//! of straight lines and Bézier curves. Paths are built incrementally with [`PathBuilder`].

use math::{Mat3, Vec2};
use std::f32::consts::FRAC_2_PI;

/// A single segment of a path, with its points already extracted.
///
/// In general, avoid storing path data as `PathSegment` arrays, instead prefer the more compact [`Path`] representation which
/// packs points and segment types into separate arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathSegment {
    /// Move the current point without drawing anything.
    MoveTo(Vec2),
    /// Draw a straight line from the current point to the given endpoint.
    LineTo(Vec2),
    /// Draw a quadratic Bézier curve through `ctrl` to `to`.
    QuadTo { ctrl: Vec2, to: Vec2 },
    /// Draw a cubic Bézier curve with two control points.
    CubicTo { ctrl1: Vec2, ctrl2: Vec2, to: Vec2 },
    /// Close the current contour by connecting back to its starting point.
    Close,
}

/// Defines the type of path segment in a [`Path`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum PathVerb {
    /// Start a new contour at a point.
    MoveTo,
    /// Straight line to a point.
    LineTo,
    /// Quadratic Bézier curve (1 control point + endpoint).
    QuadTo,
    /// Cubic Bézier curve (2 control points + endpoint).
    CubicTo,
    /// Close the current contour.
    Close,
}

/// Owned, immutable 2D path.
///
/// Paths are encoded as an array of verbs with associated points.
/// A path may contain any number of disconnected "contours", each starting with a `MoveTo` verb.
/// The starting `MoveTo` sets the initial point of the contour, and is followed by any number
/// of `LineTo`, `QuadTo`, and `CubicTo` verbs that append curve segments.
/// `Close` verbs make the contour closed by connecting the last point back to the starting `MoveTo` of the contour.
///
/// Can be borrowed as a [`PathSlice`] via the [`AsPathSlice`] trait.
/// To iterate over each verb, see [`PathSlice::iter`].
///
/// # Encoding
///
/// Each verb consumes a certain number of points from the `points` array, in order:
///
/// - `MoveTo` and `LineTo`: 1
/// - `QuadTo`: 2 (control, endpoint)
/// - `CubicTo`: 3 (control1, control2, endpoint)
/// - `Close`: 0
#[derive(Clone, Debug)]
pub struct Path {
    /// List of verbs defining the type of each segment.
    pub verbs: Box<[PathVerb]>,
    /// Flat array of control/endpoint positions for all verbs, in order.
    pub points: Box<[Vec2]>,
}

impl Path {
    /// Applies `transform` to every point in the path in-place.
    pub fn transform_in_place(&mut self, transform: Mat3) {
        for p in self.points.iter_mut() {
            *p = transform.transform_point2(*p);
        }
    }
}

/// Borrowed view of a [`Path`] (or [`PathBuilder`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PathSlice<'a> {
    /// List of verbs defining the type of each segment.
    pub verbs: &'a [PathVerb],
    /// Flat array of control/endpoint positions for all verbs, in order.
    pub points: &'a [Vec2],
}

impl<'a> From<&'a PathBuilder> for PathSlice<'a> {
    fn from(builder: &'a PathBuilder) -> Self {
        PathSlice { verbs: builder.verbs.as_slice(), points: builder.points.as_slice() }
    }
}

impl<'a> From<&'a Path> for PathSlice<'a> {
    fn from(path: &'a Path) -> Self {
        PathSlice { verbs: &path.verbs, points: &path.points }
    }
}

impl<'a> PathSlice<'a> {
    /// Iterates over all verbs of the path as decoded [`PathSegment`] values.
    pub fn iter(&self) -> impl Iterator<Item = PathSegment> + '_ {
        let mut point_index = 0;
        self.verbs.iter().map(move |verb| match verb {
            PathVerb::MoveTo => {
                let to = self.points[point_index];
                point_index += 1;
                PathSegment::MoveTo(to)
            }
            PathVerb::LineTo => {
                let to = self.points[point_index];
                point_index += 1;
                PathSegment::LineTo(to)
            }
            PathVerb::QuadTo => {
                let ctrl = self.points[point_index];
                let to = self.points[point_index + 1];
                point_index += 2;
                PathSegment::QuadTo { ctrl, to }
            }
            PathVerb::CubicTo => {
                let ctrl1 = self.points[point_index];
                let ctrl2 = self.points[point_index + 1];
                let to = self.points[point_index + 2];
                point_index += 3;
                PathSegment::CubicTo { ctrl1, ctrl2, to }
            }
            PathVerb::Close => PathSegment::Close,
        })
    }
}

/// Incremental builder for [`Path`] values.
///
/// Call [`finish`](PathBuilder::finish) to convert the
/// builder into an immutable [`Path`], or use [`AsPathSlice`] to borrow the in-progress data
/// without consuming the builder.
///
/// # Example
/// ```ignore
/// let mut b = PathBuilder::new();
/// b.move_to(vec2(0.0, 0.0))
///  .line_to(vec2(100.0, 0.0))
///  .line_to(vec2(100.0, 100.0))
///  .close();
/// let path = b.finish();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PathBuilder {
    verbs: Vec<PathVerb>,
    points: Vec<Vec2>,
}

/// Approximates a circular arc (centred at the origin, radius 1) as a sequence of cubic Bézier
/// curves and appends the control/endpoint triples to `out`.
///
/// Returns the number of cubic segments emitted.
///
/// The caller is responsible for emitting the initial `MoveTo` and adding the corresponding
/// [`PathVerb::CubicTo`] entries.
fn arc_to_beziers(angle_start: f32, angle_extent: f32, out: &mut Vec<Vec2>) -> u32 {
    let segment_count = (angle_extent.abs() * FRAC_2_PI).ceil() as usize;
    let segment_angle = angle_extent / (segment_count as f32);
    let k = 4.0 / 3.0 * f32::tan(0.5 * segment_angle);
    out.reserve(segment_count * 3);
    for i in 0..segment_count {
        let theta1 = angle_start + i as f32 * segment_angle;
        let theta2 = theta1 + segment_angle;
        let cos1 = theta1.cos();
        let sin1 = theta1.sin();
        let cos2 = theta2.cos();
        let sin2 = theta2.sin();
        out.push(Vec2::new(cos1 - k * sin1, sin1 + k * cos1));
        out.push(Vec2::new(cos2 + k * sin2, sin2 - k * cos2));
        out.push(Vec2::new(cos2, sin2));
    }
    segment_count as u32
}

impl PathBuilder {
    /// Creates a new, empty path builder.
    pub fn new() -> Self {
        Self { verbs: Vec::new(), points: Vec::new() }
    }

    /// Consumes the builder and returns the finished [`Path`].
    pub fn finish(self) -> Path {
        Path { verbs: self.verbs.into_boxed_slice(), points: self.points.into_boxed_slice() }
    }

    /// Begins a new sub-path at `to`, without drawing anything.
    ///
    /// If the builder is not empty this starts a new, disconnected contour.
    pub fn move_to(&mut self, to: Vec2) -> &mut Self {
        self.verbs.push(PathVerb::MoveTo);
        self.points.push(to);
        self
    }

    /// Appends a straight line from the current point to `to`.
    pub fn line_to(&mut self, to: Vec2) -> &mut Self {
        self.verbs.push(PathVerb::LineTo);
        self.points.push(to);
        self
    }

    /// Appends a quadratic Bézier curve.
    pub fn quad_to(&mut self, control: Vec2, endpoint: Vec2) -> &mut Self {
        self.verbs.push(PathVerb::QuadTo);
        self.points.push(control);
        self.points.push(endpoint);
        self
    }

    /// Appends a cubic Bézier curve.
    pub fn cubic_to(&mut self, ctrl1: Vec2, ctrl2: Vec2, endpoint: Vec2) -> &mut Self {
        self.verbs.push(PathVerb::CubicTo);
        self.points.push(ctrl1);
        self.points.push(ctrl2);
        self.points.push(endpoint);
        self
    }

    /// Appends an elliptical arc segment
    pub fn arc_to_endpoint(&mut self, to: Vec2, radii: Vec2, phi: f32, large_arc: bool, sweep: bool) -> &mut Self {
        let from = self.points.last().copied().unwrap_or(Vec2::ZERO);

        // convert to center parameterization
        // https://www.w3.org/TR/SVG11/implnote.html#ArcConversionEndpointToCenter
        let cosphi = phi.cos();
        let sinphi = phi.sin();
        let x1p = (from.x - to.x) / 2.0 * cosphi + (from.y - to.y) / 2.0 * sinphi;
        let y1p = -(from.x - to.x) / 2.0 * sinphi + (from.y - to.y) / 2.0 * cosphi;
        let rx = radii.x.abs();
        let ry = radii.y.abs();
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let x1p2 = x1p * x1p;
        let y1p2 = y1p * y1p;

        let rr = f32::sqrt((rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2) / (rx2 * y1p2 + ry2 * x1p2))
            * if large_arc == sweep { -1.0 } else { 1.0 };
        let cxp = rr * rx * y1p / ry;
        let cyp = rr * -ry * x1p / rx;
        let cx = cxp * cosphi - cyp * sinphi + (from.x + to.x) / 2.0;
        let cy = cxp * sinphi + cyp * cosphi + (from.y + to.y) / 2.0;

        let u = Vec2::new((x1p - cxp) / rx, (y1p - cyp) / ry);
        let v = Vec2::new((-x1p - cxp) / rx, (-y1p - cyp) / ry);

        let unorm = u.length();
        let vnorm = v.length();
        let un = u / unorm;
        let vn = v / vnorm;
        let mut theta = f32::acos(u.x / unorm).copysign(u.y);
        let mut extent = f32::acos(un.dot(vn)).copysign(u.x * v.y - u.y * v.x);

        if !sweep && extent > 0.0 {
            extent -= 2.0 * std::f32::consts::PI;
        } else if sweep && extent < 0.0 {
            extent += 2.0 * std::f32::consts::PI;
        }
        theta = theta.rem_euclid(std::f32::consts::TAU);
        extent = extent.rem_euclid(std::f32::consts::TAU);

        let transform = |p: Vec2| {
            let xp = p.x * rx * phi.cos() - p.y * ry * phi.sin() + cx;
            let yp = p.x * rx * phi.sin() + p.y * ry * phi.cos() + cy;
            Vec2::new(xp, yp)
        };

        let ptcount = self.points.len();
        let bezier_count = arc_to_beziers(theta, extent, &mut self.points);
        for i in ptcount..self.points.len() {
            self.points[i] = transform(self.points[i]);
        }
        for _ in 0..bezier_count {
            self.verbs.push(PathVerb::CubicTo);
        }
        self
    }

    /// Closes the current contour by connecting the last point back to the starting `MoveTo` point.
    pub fn close(&mut self) {
        self.verbs.push(PathVerb::Close);
    }

    /// Applies `transform` to every point in the builder in-place.
    pub fn transform_in_place(&mut self, transform: Mat3) {
        for p in self.points.iter_mut() {
            *p = transform.transform_point2(*p);
        }
    }

    /// Clears all verbs and points, resetting the builder to an empty state.
    pub fn clear(&mut self) {
        self.verbs.clear();
        self.points.clear();
    }
}

/// Implemented by types that can be borrowed as a [`PathSlice`].
///
/// Both [`Path`] and [`PathBuilder`] implement this.
pub trait AsPathSlice {
    fn as_path_slice(&self) -> PathSlice<'_>;
}

impl AsPathSlice for Path {
    fn as_path_slice(&self) -> PathSlice<'_> {
        PathSlice::from(self)
    }
}

impl AsPathSlice for PathBuilder {
    fn as_path_slice(&self) -> PathSlice<'_> {
        PathSlice::from(self)
    }
}
