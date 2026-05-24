//! 2D vector path representation and construction.
//!
//! A [`Path`] (or its borrowed counterpart [`PathSlice`]) describes a sequence of contours made up
//! of straight lines and Bézier curves. Paths are built incrementally with [`PathBuilder`].

use math::{vec2, Mat3, Rect, Vec2};
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
#[derive(Debug, Clone, PartialEq, Default)]
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
    // TODO review

    // Split the arc into segments of at most 90° so the cubic approximation stays accurate.
    let segment_count = (angle_extent.abs() * FRAC_2_PI).ceil() as usize;
    if segment_count == 0 {
        return 0;
    }
    let segment_angle = angle_extent / segment_count as f32;

    // Magic constant for circular arc approximation with cubics
    let k = 4.0 / 3.0 * (segment_angle * 0.25).tan();

    out.reserve(segment_count * 3);
    for i in 0..segment_count {
        let theta1 = angle_start + i as f32 * segment_angle;
        let theta2 = theta1 + segment_angle;
        let (sin1, cos1) = theta1.sin_cos();
        let (sin2, cos2) = theta2.sin_cos();
        out.push(Vec2::new(cos1 - k * sin1, sin1 + k * cos1)); // ctrl1
        out.push(Vec2::new(cos2 + k * sin2, sin2 - k * cos2)); // ctrl2
        out.push(Vec2::new(cos2,             sin2));           // endpoint
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

    /// Appends an elliptical arc from the current point to `to`.
    ///
    /// Follows the SVG endpoint-to-center conversion:
    /// <https://www.w3.org/TR/SVG11/implnote.html#ArcConversionEndpointToCenter>
    ///
    /// # Arguments
    /// - `to`: arc endpoint.
    /// - `radii`: x and y radii of the ellipse. Negative values are treated as their absolute value.
    /// - `phi`: rotation angle of the ellipse x-axis relative to the screen x-axis, in radians.
    /// - `large_arc`: chooses the larger of the two possible arcs when `true`.
    /// - `sweep`: draws the arc in the positive-angle (clockwise in screen space) direction when `true`.
    ///
    /// If the current point equals `to`, or either radius is zero, the arc degenerates to a
    /// straight line and a `LineTo` is emitted instead.
    pub fn arc_to_endpoint(&mut self, to: Vec2, radii: Vec2, phi: f32, large_arc: bool, sweep: bool) -> &mut Self {
        // TODO review
        let from = self.points.last().copied().unwrap_or(Vec2::ZERO);

        // F.6.6 Correction of out-of-range radii
        // "If rx = 0 or ry = 0, then treat this as a straight line from (x1, y1) to (x2, y2)"
        if from == to || radii.x == 0.0 || radii.y == 0.0 {
            return self.line_to(to);
        }

        // F.6.5 Conversion from endpoint to center parameterization
        // https://www.w3.org/TR/SVG11/implnote.html#ArcConversionEndpointToCenter

        let (sin_phi, cos_phi) = phi.sin_cos();

        // Correction of out-of-range radii
        // (F.6.6.1)
        let rx = radii.x.abs();
        let ry = radii.y.abs();

        // Step 1: Compute (x1′, y1′)
        // (F.6.5.1)
        let mid = (from - to) * 0.5;
        let x1p =  cos_phi * mid.x + sin_phi * mid.y;
        let y1p = -sin_phi * mid.x + cos_phi * mid.y;

        // F.6.6 Correction of out-of-range radii.
        // Step 3: Ensure radii are large enough
        let x1p2 = x1p * x1p;
        let y1p2 = y1p * y1p;
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let lambda = x1p2 / rx2 + y1p2 / ry2; // (F.6.6.2)
        let (rx, ry, rx2, ry2) = if lambda > 1.0 {
            // (F.6.6.3)
            let s = lambda.sqrt();
            let rx = s * rx;
            let ry = s * ry;
            (rx, ry, rx * rx, ry * ry)
        } else {
            (rx, ry, rx2, ry2)
        };

        // Step 2: Compute (cx′, cy′)
        // (F.6.5.2)
        let sq = {
            let num = rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2;
            let den = rx2 * y1p2 + ry2 * x1p2;
            (num / den).max(0.0).sqrt()
        };
        let sign = if large_arc == sweep { -1.0_f32 } else { 1.0_f32 };
        let cxp =  sign * sq * rx * y1p / ry;
        let cyp = -sign * sq * ry * x1p / rx;

        // Step 3: Compute (cx, cy) from (cx′, cy′)
        // (F.6.5.3)
        let center = Vec2::new(
            cos_phi * cxp - sin_phi * cyp + (from.x + to.x) * 0.5,
            sin_phi * cxp + cos_phi * cyp + (from.y + to.y) * 0.5,
        );

        // Step 4: Compute θ1 and Δθ

        let ux = (x1p - cxp) / rx;
        let uy = (y1p - cyp) / ry;
        let vx = (-x1p - cxp) / rx;
        let vy = (-y1p - cyp) / ry;

        // (F.6.5.5) (adapted)
        let theta = uy.atan2(ux);

        // (F.6.5.6)
        let u_len = (ux * ux + uy * uy).sqrt();
        let cos_d = ((ux * vx + uy * vy) / u_len).clamp(-1.0, 1.0);
        let mut extent = cos_d.acos().copysign(ux * vy - uy * vx);

        // Force the sign to match the requested sweep direction.
        if !sweep && extent > 0.0 {
            extent -= std::f32::consts::TAU;
        } else if sweep && extent < 0.0 {
            extent += std::f32::consts::TAU;
        }

        // convert the arc to cubic Bézier segments

        // arc_to_beziers produces points on a unit circle; transform them into the
        // target ellipse by scaling by (rx, ry) and rotating by phi
        let transform = |p: Vec2| Vec2::new(
            cos_phi * p.x * rx - sin_phi * p.y * ry + center.x,
            sin_phi * p.x * rx + cos_phi * p.y * ry + center.y,
        );

        let pt_start = self.points.len();
        let bezier_count = arc_to_beziers(theta, extent, &mut self.points);
        for i in pt_start..self.points.len() {
            self.points[i] = transform(self.points[i]);
        }
        for _ in 0..bezier_count {
            self.verbs.push(PathVerb::CubicTo);
        }
        self
    }

    /// Appends a closed circle contour centred at `center` with the given `radius`.
    pub fn circle(&mut self, center: Vec2, radius: f32) -> &mut Self {
        let right = center + Vec2::new(radius, 0.0);
        let left  = center - Vec2::new(radius, 0.0);
        let r = Vec2::splat(radius);
        self.move_to(right);
        self.arc_to_endpoint(left,  r, 0.0, false, true);
        self.arc_to_endpoint(right, r, 0.0, false, true);
        self.close();
        self
    }

    /// Appends a closed ellipse contour centred at `center` with semi-axes `radii` and
    /// x-axis rotation `phi` (in radians).
    pub fn ellipse(&mut self, center: Vec2, radii: Vec2, phi: f32) -> &mut Self {
        let (sin_phi, cos_phi) = phi.sin_cos();
        let start = center + Vec2::new(radii.x * cos_phi, radii.x * sin_phi);
        let end   = center - Vec2::new(radii.x * cos_phi, radii.x * sin_phi);
        self.move_to(start);
        self.arc_to_endpoint(end,   radii, phi, false, true);
        self.arc_to_endpoint(start, radii, phi, false, true);
        self.close();
        self
    }

    /// Appends a rectangle with top-left corner `origin` and size `size`.
    pub fn rect(&mut self, rect: &Rect) -> &mut Self {
        self.move_to(rect.min);
        self.line_to(vec2(rect.max.x, rect.min.y));
        self.line_to(rect.max);
        self.line_to(vec2(rect.min.x, rect.max.y));
        self.close();
        self
    }

    /// Appends a rectangle with top-left corner `origin` and size `size`.
    pub fn rect_origin_size(&mut self, origin: Vec2, size: Vec2) -> &mut Self {
        self.rect(&Rect { min: origin, max: origin + size })
    }

    /// Appends a rounded rectangle with top-left corner `origin`, size `size`, and corner radii `radii`.
    pub fn rrect(&mut self, rect: &Rect, radii: Vec2) -> &mut Self {
        let r = radii.min(rect.size() * 0.5);
        self.move_to(rect.min + vec2(r.x, 0.0));
        self.line_to(vec2(rect.max.x - r.x, rect.min.y));
        self.arc_to_endpoint(vec2(rect.max.x, rect.min.y + r.y), r, 0.0, false, true);
        self.line_to(vec2(rect.max.x, rect.max.y - r.y));
        self.arc_to_endpoint(vec2(rect.max.x - r.x, rect.max.y), r, 0.0, false, true);
        self.line_to(vec2(rect.min.x + r.x, rect.max.y));
        self.arc_to_endpoint(vec2(rect.min.x, rect.max.y - r.y), r, 0.0, false, true);
        self.line_to(vec2(rect.min.x, rect.min.y + r.y));
        self.arc_to_endpoint(rect.min + vec2(r.x, 0.0), r, 0.0, false, true);
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
