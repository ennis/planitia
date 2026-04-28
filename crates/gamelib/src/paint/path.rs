use std::f32::consts::FRAC_2_PI;
use math::Vec2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathVerb {
    MoveTo,
    LineTo,
    QuadTo,
    CubicTo,
    Close,
}

#[derive(Clone, Debug)]
pub struct Path {
    verbs: Box<[PathVerb]>,
    points: Box<[Vec2]>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PathSlice<'a> {
    pub verbs: &'a [PathVerb],
    pub points: &'a [Vec2],
}

impl<'a> From<&'a PathBuilder> for PathSlice<'a> {
    fn from(builder: &'a PathBuilder) -> Self {
        PathSlice {
            verbs: builder.verbs.as_slice(),
            points: builder.points.as_slice(),
        }
    }
}

impl<'a> From<&'a Path> for PathSlice<'a> {
    fn from(path: &'a Path) -> Self {
        PathSlice {
            verbs: &path.verbs,
            points: &path.points,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathBuilder {
    verbs: Vec<PathVerb>,
    points: Vec<Vec2>,
}

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
    pub fn new() -> Self {
        Self {
            verbs: Vec::new(),
            points: Vec::new(),
        }
    }

    pub fn finish(self) -> Path {
        Path {
            verbs: self.verbs.into_boxed_slice(),
            points: self.points.into_boxed_slice(),
        }
    }

    pub fn move_to(&mut self, to: Vec2) -> &mut Self {
        self.verbs.push(PathVerb::MoveTo);
        self.points.push(to);
        self
    }

    pub fn line_to(&mut self, to: Vec2) -> &mut Self {
        self.verbs.push(PathVerb::LineTo);
        self.points.push(to);
        self
    }

    pub fn quad_to(&mut self, ctrl: Vec2, to: Vec2) -> &mut Self {
        self.verbs.push(PathVerb::QuadTo);
        self.points.push(ctrl);
        self.points.push(to);
        self
    }

    pub fn cubic_to(&mut self, ctrl1: Vec2, ctrl2: Vec2, to: Vec2) -> &mut Self {
        self.verbs.push(PathVerb::CubicTo);
        self.points.push(ctrl1);
        self.points.push(ctrl2);
        self.points.push(to);
        self
    }

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

        let rr = f32::sqrt((rx2*ry2-rx2*y1p2-ry2*x1p2) / (rx2*y1p2+ry2*x1p2)) * if large_arc == sweep { -1.0 } else { 1.0 };
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
        theta  = theta.rem_euclid(std::f32::consts::TAU);
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
        for i in 0..bezier_count {
            self.verbs.push(PathVerb::CubicTo);
        }
        self
    }

    /*pub fn arc(&mut self, center: Vec2, radii: Vec2, phi: f32, from: f32, to: f32) -> &mut Self {
        // TODO
        self
    }*/

    pub fn close(&mut self) {
        self.verbs.push(PathVerb::Close);
    }
}
