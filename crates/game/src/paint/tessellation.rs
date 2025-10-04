use crate::paint::shape::RectShape;
use crate::paint::{FeatherVertex, Srgba32};
use math::{Vec2, vec2, U16Vec2};
use std::f32::consts::PI;

#[derive(Clone, Copy, Debug)]
struct PN {
    p: Vec2,
    /// Not necessarily normalized
    n: Vec2,
    c: Srgba32,
}

impl PN {
    const fn new(p: Vec2, n: Vec2, c: Srgba32) -> Self {
        Self { p, n, c }
    }
}

#[derive(Default)]
struct GeomSink {
    vertices: Vec<FeatherVertex>,
    indices: Vec<u32>,
}

impl GeomSink {
    fn reserve(&mut self, vertex_count: usize, index_count: usize) {
        self.vertices.reserve(vertex_count);
        self.indices.reserve(index_count);
    }
}

fn stroke_path(path: &[PN], closed: bool, width: f32, feather: f32, out: &mut GeomSink) {
    out.reserve(
        path.len() * 2,
        if closed { path.len() * 6 } else { (path.len() - 1) * 6 },
    );

    for (i, v) in path.iter().enumerate() {
        out.vertices
            .push(FeatherVertex::new(v.p - v.n * (width + feather), -1.0, v.c));
        out.vertices
            .push(FeatherVertex::new(v.p + v.n * (width + feather), 1.0, v.c));
        if i > 0 {
            let base = (i * 2) as u32;
            out.indices.extend([base - 2, base - 1, base, base, base - 1, base + 1]);
        }
    }

    if closed {
        let base = (path.len() * 2) as u32;
        out.indices.extend([base - 2, base - 1, 0, 0, base - 1, 1]);
    }
}

fn tess_feathered_polygon(polygon: &[PN], feather_in: f32, feather_out: f32, out: &mut GeomSink) {
    assert!(feather_in >= 0.0);
    assert!(feather_out >= 0.0);

    if feather_in > 0.0 || feather_out > 0.0 {
        out.reserve(
            polygon.len() * 2,                           // main poly + feather line
            (polygon.len() - 2) * 3 + polygon.len() * 6, // main tris + feather strip
        );
    } else {
        out.reserve(polygon.len(), (polygon.len() - 2) * 3);
    }

    // tess main polygon
    let base = out.vertices.len() as u32;
    out.vertices.extend(
        polygon
            .iter()
            .map(|v| FeatherVertex::new(v.p - feather_in * v.n, 0.0, v.c)),
    );
    for i in 1..(polygon.len() - 1) {
        let i = i as u32;
        out.indices.extend([base, base + i, base + i + 1]);
    }

    // feather strip
    if feather_in > 0.0 || feather_out > 0.0 {
        let base_feather = out.vertices.len() as u32;
        for v in polygon.iter() {
            out.vertices.push(FeatherVertex::new(v.p + feather_out * v.n, 1.0, v.c));
        }
        for i in 0..polygon.len() {
            let i = i as u32;
            let j = (i + 1) % (polygon.len() as u32);
            out.indices.extend([base + i, base + j, base_feather + i]);
            out.indices.extend([base_feather + i, base + j, base_feather + j]);
        }
    }
}

fn flatten_rrect(shape: RectShape, out: &mut Vec<PN>) {
    let rect = shape.rect;
    let r = shape.radius;
    let corner_centers = [
        Vec2::new(rect.min.x + r, rect.min.y + r),
        Vec2::new(rect.max.x - r, rect.min.y + r),
        Vec2::new(rect.max.x - r, rect.max.y - r),
        Vec2::new(rect.min.x + r, rect.max.y - r),
    ];

    if r > 0.0 {
        // Approximate each corner with a quarter circle
        let tol = 2.0f32;
        let nseg = ((PI * r) / tol).ceil() as usize;
        let mut angle = PI;
        for (icorner, center) in corner_centers.iter().enumerate() {
            for i in 0..=nseg {
                let dir = Vec2::new(angle.cos(), -angle.sin());
                let p = center + dir * r;
                out.push(PN::new(p, dir, shape.colors[icorner]));
                angle = PI * (1.0 - 0.5 * (icorner as f32 + i as f32 / nseg as f32));
            }
        }
    } else {
        // No radius, just push the corners
        out.push(PN::new(vec2(rect.min.x, rect.min.y), vec2(-1.0, -1.0), shape.colors[0]));
        out.push(PN::new(vec2(rect.max.x, rect.min.y), vec2(1.0, -1.0), shape.colors[1]));
        out.push(PN::new(vec2(rect.max.x, rect.max.y), vec2(1.0, 1.0), shape.colors[2]));
        out.push(PN::new(vec2(rect.min.x, rect.max.y), vec2(-1.0, 1.0), shape.colors[3]));
    }
}

#[derive(Default)]
pub struct Tessellator {
    geometry: GeomSink,
}

impl Tessellator {
    pub fn new() -> Self {
        Self {
            geometry: GeomSink::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.geometry.vertices.is_empty()
    }

    pub fn fill_rrect(&mut self, rect: RectShape) {
        let mut p = Vec::new();
        flatten_rrect(rect, &mut p);

        if rect.feather > 0.0 {
            tess_feathered_polygon(&p, 0.0, rect.feather, &mut self.geometry);
        } else {
            // 1px wide AA feather
            tess_feathered_polygon(&p, 0.5, 0.5, &mut self.geometry);
        }
    }

    pub fn stroke_rrect(&mut self, rect: RectShape, width: f32) {
        let mut p = Vec::new();
        flatten_rrect(rect, &mut p);

        if rect.feather > 0.0 {
            stroke_path(&p, true, width, rect.feather, &mut self.geometry);
        } else {
            // 1px wide AA feather
            stroke_path(&p, true, width, 0.5, &mut self.geometry);
        }
    }

    pub fn quad(&mut self, p0: Vec2, p1: Vec2, uv0: U16Vec2, uv1: U16Vec2, color: Srgba32) {
        let base = self.geometry.vertices.len() as u32;
        self.geometry.vertices.extend(
        [
            FeatherVertex {
                p: Vec2::new(p0.x, p0.y),
                uv: U16Vec2::new(uv0.x, uv0.y),
                color,
                feather: 0.0,
            },
            FeatherVertex {
                p: Vec2::new(p1.x, p0.y),
                uv: U16Vec2::new(uv1.x, uv0.y),
                color,
                feather: 0.0,
            },
            FeatherVertex {
                p: Vec2::new(p1.x, p1.y),
                uv: U16Vec2::new(uv1.x, uv1.y),
                color,
                feather: 0.0,
            },
            FeatherVertex {
                p: Vec2::new(p0.x, p1.y),
                uv: U16Vec2::new(uv0.x, uv1.y),
                color,
                feather: 0.0,
            },
        ]);
        self.geometry.indices.extend([base+0, base+1, base+2, base+0, base+2, base+3]);
    }

    pub fn finish_and_reset(&mut self) -> Mesh {

        let vertices = self.geometry.vertices.clone();
        let indices = self.geometry.indices.clone();
        self.geometry.vertices.clear();
        self.geometry.indices.clear();

        Mesh {
            vertices: vertices,
            indices: indices,
        }
    }
}

#[derive(Default, Debug)]
pub struct Mesh {
    pub vertices: Vec<FeatherVertex>,
    pub indices: Vec<u32>,
}
