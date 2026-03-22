use math::Vec3;
use std::collections::HashMap;
use gpu::Ptr;

#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct WingedEdge {
    pub src_vtx: u32,
    pub dst_vtx: u32,
    /// Right (CW) face
    pub r_face: u32,
    /// Left (CCW) face
    pub l_face: u32,
    /// CW R edge
    pub r_cw: u32,
    /// CCW R edge
    pub r_ccw: u32,
    /// CCW L edge
    pub l_ccw: u32,
    /// CW L edge
    pub l_cw: u32,
}

#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct WEFace {
    pub edges: [u32; 3],
}

#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct WEPoint<Data> {
    pub data: Data,
}

#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct WEFaceVertex<Data> {
    pub data: Data,
    //normal: Vec3,
    //vertex: u32,
    pub incident_edge: u32,
}

pub struct WingedEdgeMesh<PointData, FaceVertexData> {
    points: Vec<WEPoint<PointData>>,
    face_vertices: Vec<WEFaceVertex<FaceVertexData>>,
    edges: Vec<WingedEdge>,
    faces: Vec<WEFace>,
    points_gpu: gpu::Buffer<WEPoint<PointData>>,
    face_vertices_gpu: gpu::Buffer<WEFaceVertex<FaceVertexData>>,
    edges_gpu: gpu::Buffer<WingedEdge>,
    faces_gpu: gpu::Buffer<WEFace>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct WEMeshDataGPU<PointData: Copy+'static, FaceVertexData: Copy+'static> {
    points: Ptr<WEPoint<PointData>>,
    face_vertices: Ptr<WEFaceVertex<FaceVertexData>>,
    edges: Ptr<WingedEdge>,
    faces: Ptr<WEFace>,
    point_count: u32,
    face_vertex_count: u32,
    edge_count: u32,
    face_count: u32,
}

impl<PointData: Copy, FaceVertexData: Copy> WingedEdgeMesh<PointData, FaceVertexData> {
    pub fn new(
        points: &[PointData],
        face_vertices: &[FaceVertexData],
        get_point: impl Fn(&FaceVertexData) -> u32,
    ) -> WingedEdgeMesh<PointData, FaceVertexData> {
        assert!(
            face_vertices.len() % 3 == 0,
            "invalid number of indices: must be a multiple of 3"
        );

        let mut faces: Vec<WEFace> = Vec::with_capacity(face_vertices.len() / 3);
        let mut edge_map: HashMap<(u32, u32), u32> = HashMap::new();
        let mut vertex_to_edge: HashMap<u32, u32> = HashMap::new();
        let mut edges: Vec<WingedEdge> = Vec::new();

        let vertices: Vec<_> = points.iter().map(|p| WEPoint { data: *p }).collect();

        let face_vertices: Vec<_> = face_vertices
            .iter()
            .enumerate()
            .map(|(i, fv)| {
                WEFaceVertex {
                    data: *fv,
                    incident_edge: u32::MAX,
                }
            })
            .collect();

        for fvs in face_vertices.chunks(3) {
            let i_face = faces.len() as u32;

            let mut face_edges = [0u32; 3];

            for i in 0..3 {
                let src_fv = fvs[i];
                let dst_fv = fvs[(i + 1) % 3];
                let src_vtx = get_point(&src_fv.data);
                let dst_vtx = get_point(&dst_fv.data);

                // find if the edge already exists
                let mut edge_index = edge_map.get(&(src_vtx.min(dst_vtx), src_vtx.max(dst_vtx))).cloned();

                if let Some(ei) = edge_index {
                    edges[ei as usize].l_face = i_face;
                    face_edges[i] = ei;
                } else {
                    // create new edge
                    let new_edge = WingedEdge {
                        src_vtx,
                        dst_vtx,
                        r_face: faces.len() as u32,
                        l_face: u32::MAX,
                        r_cw: u32::MAX,
                        r_ccw: u32::MAX,
                        l_ccw: u32::MAX,
                        l_cw: u32::MAX,
                    };
                    let new_ei = edges.len() as u32;
                    edges.push(new_edge);
                    face_edges[i] = new_ei;
                    vertex_to_edge.insert(src_vtx, new_ei);
                }
            }

            for i in 0..3 {
                let curr_ei = face_edges[i];
                let next_ei = face_edges[(i + 1) % 3];
                let prev_ei = face_edges[(i + 2) % 3];
                let src_vtx = get_point(&fvs[i].data);

                if src_vtx == edges[curr_ei as usize].src_vtx {
                    edges[curr_ei as usize].r_cw = next_ei;
                    edges[curr_ei as usize].r_ccw = prev_ei;
                } else {
                    edges[curr_ei as usize].l_ccw = prev_ei;
                    edges[curr_ei as usize].l_cw = next_ei;
                }
            }

            faces.push(WEFace { edges: face_edges });
        }

        // Upload to GPU
        let face_vertices_gpu = gpu::Buffer::from_slice(&face_vertices);
        let edges_gpu = gpu::Buffer::from_slice(&edges);
        let faces_gpu = gpu::Buffer::from_slice(&faces);
        let vertices_gpu = gpu::Buffer::from_slice(&vertices);

        WingedEdgeMesh {
            face_vertices,
            edges,
            faces,
            points: vertices,
            points_gpu: vertices_gpu,
            face_vertices_gpu,
            edges_gpu,
            faces_gpu,
        }
    }
}
