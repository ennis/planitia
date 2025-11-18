use image::ImageBuffer;
use math::{Vec2, vec2, vec3};
use priority_queue::PriorityQueue;
use spade::handles::{FaceHandle, InnerTag};
use spade::{DelaunayTriangulation, HasPosition, HierarchyHintGenerator, Point2, Triangulation};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::BufWriter;
use color_print::cprintln;

struct PointWithHeight {
    position: Point2<f32>,
    height: f32,
}

impl PointWithHeight {
    fn new(x: f32, y: f32, height: f32) -> Self {
        Self {
            position: Point2::new(x, y),
            height,
        }
    }
}

impl HasPosition for PointWithHeight {
    type Scalar = f32;

    fn position(&self) -> Point2<f32> {
        self.position
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OrdF32(pub f32);

impl Eq for OrdF32 {}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[derive(Default)]
struct FaceData {
    heap_index: usize,
}

// cached point errors
struct Candidate {
    face: usize,
    x: u32,
    y: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.face == other.face
    }
}

impl Eq for Candidate {}

impl Hash for Candidate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.face.hash(state);
    }
}

type DT = DelaunayTriangulation<PointWithHeight, (), (), FaceData, HierarchyHintGenerator<f32>>;
type CandidateHeap = PriorityQueue<Candidate, OrdF32>;
type HeightmapImage = ImageBuffer<image::Luma<f32>, Vec<f32>>;
type InnerFaceHandle<'a> = FaceHandle<'a, InnerTag, PointWithHeight, (), (), FaceData>;

fn point_inside_triangle(a: Vec2, b: Vec2, c: Vec2, p: Vec2) -> bool {
    let ab = b - a;
    let bc = c - b;
    let ca = a - c;
    let ap = p - a;
    let bp = p - b;
    let cp = p - c;
    let abxap = ab.x * ap.y - ab.y * ap.x; // cross product AB x AP
    let bcxbp = bc.x * bp.y - bc.y * bp.x; // cross product BC x BP
    let caxcp = ca.x * cp.y - ca.y * cp.x;
    (abxap < 0.0) == (caxcp < 0.0) && (bcxbp < 0.0) == (caxcp < 0.0)
}

fn scan_triangle(face: InnerFaceHandle, heightmap: &HeightmapImage, error_threshold: f32, heap: &mut CandidateHeap) {
    let [a, b, c] = face.vertices();
    let plane = {
        let p0 = a.position();
        let p1 = b.position();
        let p2 = c.position();
        let v0 = vec3(p1.x - p0.x, b.data().height - a.data().height, p1.y - p0.y);
        let v1 = vec3(p2.x - p0.x, c.data().height - a.data().height, p2.y - p0.y);
        v0.cross(v1)
    };

    let pa = a.position();
    let pb = b.position();
    let pc = c.position();

    // triangle bbox
    let min_x = pa.x.min(pb.x.min(pc.x)).floor() as u32;
    let max_x = pa.x.max(pb.x.max(pc.x)).ceil() as u32;
    let min_y = pa.y.min(pb.y.min(pc.y)).floor() as u32;
    let max_y = pa.y.max(pb.y.max(pc.y)).ceil() as u32;

    let mut max_error = 0.0;
    let mut max_error_point = None;

    // scan over bbox
    for y in min_y..max_y {
        for x in min_x..max_x {
            let p = vec2(x as f32 + 0.5, y as f32 + 0.5);
            if point_inside_triangle(vec2(pa.x, pa.y), vec2(pb.x, pb.y), vec2(pc.x, pc.y), p) {
                // compute height at p using plane equation
                let dx = p.x - pa.x;
                let dz = p.y - pa.y;
                let height = (-plane.x * dx - plane.z * dz) / plane.y + a.data().height;

                // do something with the height at (x,y)
                let h = heightmap.get_pixel(x, y)[0];
                let error = f32::abs(height - h);

                if error > max_error {
                    max_error = error;
                    max_error_point = Some((x, y, h));
                }
            }
        }
    }

    if let Some((x, y, h)) = max_error_point {
        if max_error < error_threshold {
            return;
        }
        heap.push(
            Candidate {
                face: face.index(),
                x,
                y,
            },
            OrdF32(max_error),
        );
    }
}

fn insert(
    dt: &mut DT,
    candidate: &Candidate,
    heightmap: &HeightmapImage,
    error_threshold: f32,
    heap: &mut CandidateHeap,
) {
    let x = candidate.x;
    let y = candidate.y;
    let h = heightmap.get_pixel(x, y)[0];

    let v = dt
        .insert(PointWithHeight::new(x as f32 + 0.5, y as f32 + 0.5, h))
        .unwrap();

    // scan affected (newly created) faces
    for out_edge in dt.vertex(v).out_edges() {
        let Some(f) = out_edge.face().as_inner() else { continue };
        scan_triangle(f, heightmap, error_threshold, heap);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TriangulationOptions {
    pub error_threshold: Option<f32>,
    pub triangle_count_target: Option<usize>,
}

/// Generates a triangle mesh from a heightmap image using an adaptive triangulation approach.
///
/// See Garland, Michael, and Paul S. Heckbert. "Fast polygonal approximation of terrains and height fields.", 1995.
///
/// The progress callback is called with the current number of triangles in the triangulation.
pub(super) fn tessellate_heightmap(
    heightmap: &HeightmapImage,
    options: &TriangulationOptions,
    progress_callback: &mut dyn FnMut(u32, f32),
) {
    let mut dt = DT::new();
    let height = heightmap.height();
    let width = heightmap.width();

    // intialize with the four corner points
    {
        let sh = |x, y| -> f32 { heightmap.get_pixel(x, y)[0] };
        dt.insert(PointWithHeight::new(0.5, 0.5, sh(0, 0))).unwrap();
        dt.insert(PointWithHeight::new(width as f32 - 0.5, 0.5, sh(width - 1, 0)))
            .unwrap();
        dt.insert(PointWithHeight::new(0.5, height as f32 - 0.5, sh(0, height - 1)))
            .unwrap();
        dt.insert(PointWithHeight::new(
            width as f32 - 0.5,
            height as f32 - 0.5,
            sh(width - 1, height - 1),
        ))
        .unwrap();
    }

    let error_threshold = match (options.triangle_count_target, options.error_threshold) {
        (_, Some(threshold)) => threshold,
        (Some(_), None) => 0.0,
        (None, None) => 0.01, // default error threshold
    };

    // initial scan of all faces
    let mut candidate_heap = CandidateHeap::new();
    for face in dt.inner_faces() {
        scan_triangle(face, heightmap, error_threshold, &mut candidate_heap);
    }

    // report progess every 100 triangles
    let mut last_reported_triangle_count = 0;
    let mut last_max_error = 0.0;
    loop {
        // get the point with the maximum error
        let Some((candidate, OrdF32(max_error))) = candidate_heap.pop() else {
            break;
        };
        last_max_error = max_error;

        // break if error is small enough...
        if max_error < error_threshold {
            break;
        }

        insert(&mut dt, &candidate, heightmap, error_threshold, &mut candidate_heap);

        if last_reported_triangle_count == 100 {
            progress_callback(dt.num_inner_faces() as u32, max_error);
            last_reported_triangle_count = 0;
        } else {
            last_reported_triangle_count += 1;
        }

        if let Some(triangle_count_target) = options.triangle_count_target {
            if dt.num_inner_faces() >= triangle_count_target {
                break;
            }
        }

        //eprintln!("max error: {max_error}, remaining candidates: {}", candidate_heap.len());
    }

    progress_callback(dt.num_inner_faces() as u32, last_max_error);

    // dump the generated mesh to an OBJ file for inspection
    {
        cprintln!("Dumping heightmap...");

        use std::fs::File;
        use std::io::Write;

        let mut obj_file = File::create("../../terrain_mesh.obj").unwrap();
        let mut buf_writer = BufWriter::new(&mut obj_file);

        let mut vertex_map = HashMap::<usize, usize>::new();

        for (i, v) in dt.vertices().enumerate() {
            vertex_map.insert(v.index(), i);
            let h = v.data().height;
            writeln!(buf_writer, "v {} {} {}", v.position().x, h * 100.0, v.position().y).unwrap();
        }

        for f in dt.inner_faces() {
            let [a, b, c] = f.vertices();
            let ia = vertex_map[&a.index()];
            let ib = vertex_map[&b.index()];
            let ic = vertex_map[&c.index()];
            writeln!(buf_writer, "f {} {} {}", ib + 1, ia + 1, ic + 1).unwrap(); // OBJ is 1-indexed
        }
    }
}
