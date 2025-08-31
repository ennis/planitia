use crate::Vec3;

/// A cubic Bézier curve segment defined by four control points.
#[derive(Copy, Clone, Debug)]
pub struct CubicBezierSegment {
    pub p0: Vec3,
    pub p1: Vec3,
    pub p2: Vec3,
    pub p3: Vec3,
}


impl CubicBezierSegment {
    fn subdivide(&self, t: f32) -> (Self, Self) {
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

    fn is_flat(&self, tolerance: f32) -> bool {
        let p0 = self.p0;
        let p1 = self.p1;
        let p2 = self.p2;
        let p3 = self.p3;
        let t = tolerance * tolerance;
        (0.5 * (p0 + p2) - p1).length_squared() <= t && (0.5 * (p1 + p3) - p2).length_squared() <= t
    }

    fn flatten_inner(&self, points: &mut Vec<Vec3>, tolerance: f32) {
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
    pub fn flatten(&self, points: &mut Vec<Vec3>, tolerance: f32) {
        if points.is_empty() {
            points.push(self.p0);
        }
        self.flatten_inner(points, tolerance);
    }
}