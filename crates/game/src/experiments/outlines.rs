use crate::experiments::lines::draw_lines;
use crate::{SceneInfo, SceneInfoUniforms};
use bytesize::ByteSize;
use color::{Srgba8, srgba8};
use gamelib::asset::{AssetLoadError, AssetReadGuard, Handle, VfsPath, VfsPathBuf};
use gamelib::input::InputEvent;
use gamelib::pipeline_cache::{get_compute_pipeline, get_graphics_pipeline};
use gamelib::{static_assets, tweak};
use gpu::PrimitiveTopology::TriangleList;
use gpu::{
    BarrierFlags, Buffer, BufferCreateInfo, DrawIndirectCommand, Image, MemoryLocation, Ptr, RootParams, Size3D,
};
use hgeo::util::polygons_to_triangle_mesh;
use log::{info, warn};
use math::{Mat4, Vec3};
use smallvec::{SmallVec, smallvec};
use std::alloc::Layout;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::Path;
use std::{fmt, ptr};
use math::geom::Camera;

/// Vertices emitted by the outline extraction compute shader.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ContourVertex {
    pub(crate) clip_pos: math::Vec4,
    pub(crate) angle: f32,
    pub(crate) flags: u16,
    pub(crate) group_id: u16,
    pub(crate) pointy: f32,
    pub(crate) reserved_0: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ContourEdge {
    position_0: Vec3,
    point_0: u32,
    position_1: Vec3,
    point_1: u32,
    group_id: u16,
}

/// Loads geometry from the specified file path.
pub struct OutlineExperiment {
    // --- Preprocessed mesh data ---
    mesh: Mesh,
    processed_mesh: Option<ProcessedMesh>,

    // --- GPU pipeline resources ---

    // contour extraction
    expanded_contour_vertices: Buffer<ContourVertex>,
    expanded_contour_indices: Buffer<u32>,
    contour_edges: Buffer<ContourEdgeBuffer>,

    // edge linking
    contour_point_list: Buffer<ContourPoint>,
    contour_point_list_subdiv: Buffer<ContourPoint>,
    global_to_contour_index_map: Buffer<u32>,
    contour_rank_successors: Buffer<u64>, // low 32 bits: rank, high 32 bits: successor index, root index after list ranking pass
    contour_rank_successors_1: Buffer<u64>, // low 32 bits: rank, high 32 bits: successor index, root index after list ranking pass

    // contour rendering
    expanded_contours_draw_command: Buffer<DrawIndirectCommand>,
    angle_texture: Image,
    normal_texture: Image,
    jfa_0: Image,
    jfa_1: Image,

    lock_view: bool,
    locked_eye: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct NormalCone {
    axis: Vec3,
    angle: f32,
    center: Vec3,
    radius: f32,
}

struct EdgeCluster {
    edges: Vec<u32>,
    faces: Vec<u32>,
    cone: NormalCone,
}

const MAX_EDGES_PER_CLUSTER: usize = 128;
const JFA_TILE_SIZE: u32 = 16;

////////////////////////////////////////////////////////////////////////////////////////////////////

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct MeshEdge {
    normal_cw: u16,
    normal_ccw: u16,
    vertex_0: u16,
    vertex_1: u16,
}

const CLUSTER_EDGE_FLAG_BORDER: u16 = 1 << 0;
const CLUSTER_EDGE_FLAG_CREASE: u16 = 1 << 1;

impl fmt::Debug for MeshEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(na={},nb={},v0={},v1={})",
            self.normal_cw, self.normal_ccw, self.vertex_0, self.vertex_1
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
struct MeshVertex {
    position: Vec3,
    point_index: u32,
    normal: Vec3,
    pointy: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct MeshFace {
    vertices: [u16; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct MeshData {
    vertices: Ptr<MeshVertex>,
    edges: Ptr<MeshEdge>,
    faces: Ptr<MeshFace>,
    face_normals: Ptr<Vec3>,
    meshlets: Ptr<Meshlet>,
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            vertices: Ptr::NULL,
            edges: Ptr::NULL,
            faces: Ptr::NULL,
            face_normals: Ptr::NULL,
            meshlets: Ptr::NULL,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Meshlet {
    normal_cone: NormalCone,
    base_vertex: u32,
    base_edge: u32,
    base_face: u32,
    base_face_normal: u32,
    vertex_count: u16,
    edge_count: u16,
    face_count: u16,
    face_normal_count: u16,
    group_id: u16,
}

struct ProcessedMesh {
    vertices: gpu::Buffer<MeshVertex>,
    edges: gpu::Buffer<MeshEdge>,
    faces: gpu::Buffer<MeshFace>,
    face_normals: gpu::Buffer<Vec3>,
    meshlets: gpu::Buffer<Meshlet>,
    draw_commands: gpu::Buffer<DrawIndirectCommand>,
    gpu: MeshData,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Copy, Clone)]
struct Vertex {
    point: u32,
    normal: Vec3,
}

#[derive(Copy, Clone)]
struct Point {
    position: Vec3,
    pointy: f32,
    id: i32,
}

#[derive(Copy, Clone)]
struct Face {
    normal: Vec3,
    vertices: [u32; 3],
    edges: [u32; 3],
}

#[derive(Copy, Clone)]
struct Edge {
    /// 1st point index
    p0: u32,
    /// 2nd point index
    p1: u32,
    /// 1st adjacent face index
    f_cw: u32,
    /// 2nd adjacent face index
    f_ccw: u32,
    /// Current cluster
    cluster: usize,
}

const NO_FACE: u32 = u32::MAX;

#[derive(Default)]
struct Mesh {
    /// Triplets of vertex indices + face normal.
    faces: Vec<Face>,
    /// Pairs of vertex indices + edge type.
    edges: Vec<Edge>,
    vertices: Vec<Vertex>,
    points: Vec<Point>,
    incident_edges: Vec<SmallVec<u32, 6>>, // per-point incident edge indices
}

impl Mesh {
    fn new(points: Vec<Point>, vertices: Vec<Vertex>, indices: &[u32]) -> Mesh {
        assert!(
            indices.len() % 3 == 0,
            "invalid number of indices: must be a multiple of 3"
        );

        let mut faces: Vec<Face> = Vec::with_capacity(indices.len() / 3);
        // (min(P0,P1), max(P0,P1)) -> Edge index
        let mut edge_map: HashMap<(u32, u32), u32> = HashMap::new();
        let mut edges: Vec<Edge> = Vec::new();

        for index in indices.chunks(3) {
            let v0 = index[0];
            let v1 = index[1];
            let v2 = index[2];

            let normal = {
                let p0 = points[vertices[v0 as usize].point as usize];
                let p1 = points[vertices[v1 as usize].point as usize];
                let p2 = points[vertices[v2 as usize].point as usize];
                let edge1 = p1.position - p0.position;
                let edge2 = p2.position - p0.position;
                Vec3::cross(edge1, edge2).normalize()
            };

            let mut insert_edge = |va: u32, vb: u32, face| -> u32 {
                let pa = vertices[va as usize].point;
                let pb = vertices[vb as usize].point;

                *edge_map
                    .entry((pa.min(pb), pa.max(pb)))
                    .and_modify(|edge_index: &mut u32| {
                        edges[*edge_index as usize].f_ccw = face;
                    })
                    .or_insert_with(|| {
                        let edge_index = edges.len() as u32;
                        edges.push(Edge {
                            p0: pa,
                            p1: pb,
                            f_cw: face,
                            f_ccw: NO_FACE,
                            cluster: 0,
                        });
                        edge_index
                    })
            };

            let face_index = faces.len() as u32;
            let e0 = insert_edge(v0, v1, face_index);
            let e1 = insert_edge(v1, v2, face_index);
            let e2 = insert_edge(v2, v0, face_index);

            faces.push(Face {
                normal,
                vertices: [v0, v1, v2],
                edges: [e0, e1, e2],
            });
        }

        let mut incident_edges: Vec<SmallVec<_, 6>> = vec![smallvec![]; points.len()];
        for (i_edge, edge) in edges.iter().enumerate() {
            incident_edges[edge.p0 as usize].push(i_edge as u32);
            incident_edges[edge.p1 as usize].push(i_edge as u32);
        }

        Mesh {
            faces,
            edges,
            vertices,
            points,
            incident_edges,
        }
    }
}

// Compute the normal cone of a cluster
fn compute_normal_cone(mesh: &Mesh, cluster_edges: &[u32]) -> NormalCone {
    // average the face normals of all edges in the cluster
    let mut n = Vec3::ZERO;
    let mut faces = HashSet::new();
    let mut count = 0;

    let mut accumulate = |face_index: u32| -> usize {
        if face_index == NO_FACE {
            return 0;
        }
        if faces.insert(face_index) {
            let face = &mesh.faces[face_index as usize];
            n += face.normal;
            1
        } else {
            0
        }
    };

    for edge_index in cluster_edges {
        let edge = &mesh.edges[*edge_index as usize];
        count += accumulate(edge.f_cw);
        count += accumulate(edge.f_ccw);
    }

    n /= count as f32;
    n = n.normalize();

    // determine the normal that deviates the most from the average
    let mut min_dot = 0.0f32;
    let mut out_normal = Vec3::ZERO;

    let mut update_min = |face_index: u32| {
        if face_index == NO_FACE {
            return;
        }
        let face = &mesh.faces[face_index as usize];
        let angle = Vec3::dot(face.normal, n);
        if angle < min_dot {
            min_dot = angle;
            out_normal = face.normal;
        }
    };

    let mut center = Vec3::ZERO;
    for edge_index in cluster_edges {
        let edge = &mesh.edges[*edge_index as usize];
        update_min(edge.f_cw);
        update_min(edge.f_ccw);
        let midpoint = (mesh.points[edge.p0 as usize].position + mesh.points[edge.p1 as usize].position) / 2.0;
        center += midpoint;
    }
    center /= cluster_edges.len() as f32;

    let mut radius = 0.0f32;
    for edge_index in cluster_edges {
        let edge = &mesh.edges[*edge_index as usize];
        let midpoint = (mesh.points[edge.p0 as usize].position + mesh.points[edge.p1 as usize].position) / 2.0;
        let dist = (midpoint - center).length();
        radius = f32::max(radius, dist);
    }

    NormalCone {
        axis: n,
        angle: min_dot.acos(),
        radius,
        center,
    }
}

fn acceptance_heuristic(cone: &NormalCone, cam_dist: f32) -> f32 {
    let beta_s = (cone.radius / cam_dist).asin();
    f32::sin(cone.angle + beta_s)
}

fn cluster_edges(mesh: &mut Mesh, cam_dist: f32) -> Vec<EdgeCluster> {
    // start with one cluster per edge
    let mut clusters: Vec<EdgeCluster> = Vec::new();

    for i in 0..mesh.edges.len() {
        let cone = compute_normal_cone(mesh, &[i as u32]);
        let cluster = EdgeCluster {
            edges: vec![i as u32],
            faces: vec![],
            cone,
        };
        mesh.edges[i].cluster = i;
        clusters.push(cluster);
    }

    //
    let nclusters = clusters.len();
    loop {
        let mut progress = false;
        for i_cluster in 0..nclusters {
            // find adjacent edge belonging to another cluster
            let nedges = clusters[i_cluster].edges.len();
            'try_merge_with_incident: for i_cluster_edge in 0..nedges {
                let i_edge = clusters[i_cluster].edges[i_cluster_edge] as usize;
                for point in [mesh.edges[i_edge].p0, mesh.edges[i_edge].p1] {
                    for &e2 in &mesh.incident_edges[point as usize] {
                        let e2 = &mesh.edges[e2 as usize];
                        if e2.cluster == i_cluster {
                            continue;
                        }

                        let i_cluster_1 = e2.cluster;
                        let cluster_1 = &clusters[i_cluster_1];

                        // don't merge if it would exceed the maximum number of edges per cluster
                        if nedges + cluster_1.edges.len() > MAX_EDGES_PER_CLUSTER {
                            continue;
                        }

                        let merged_edges: Vec<u32> = clusters[i_cluster]
                            .edges
                            .iter()
                            .cloned()
                            .chain(cluster_1.edges.iter().cloned())
                            .collect();
                        let merged_cone = compute_normal_cone(mesh, &merged_edges);

                        let accept_0 = acceptance_heuristic(&clusters[i_cluster].cone, cam_dist);
                        let accept_1 = acceptance_heuristic(&cluster_1.cone, cam_dist);
                        let accept_merged = acceptance_heuristic(&merged_cone, cam_dist);

                        let gain = accept_0 + accept_1 - accept_merged;
                        if gain > 0.0 {
                            for edge in &cluster_1.edges {
                                mesh.edges[*edge as usize].cluster = i_cluster;
                            }
                            clusters[i_cluster_1].edges.clear();
                            clusters[i_cluster].edges = merged_edges;
                            clusters[i_cluster].cone = merged_cone;
                            progress = true;
                            break 'try_merge_with_incident;
                        } else {
                            continue;
                        }
                    }
                }
            }
        }
        if !progress {
            break;
        }
    }

    let num_clusters = clusters.iter().filter(|c| !c.edges.is_empty()).count();

    // Now cluster faces according to the most common cluster among their incident edges.
    for (fi, face) in &mut mesh.faces.iter_mut().enumerate() {
        let mut votes = [(0, 0); 3]; // (cluster index, vote count)

        let mut vote = |cluster_index| {
            if let Some(p) = votes.iter().position(|(ci, _)| *ci == cluster_index) {
                votes[p].1 += 1;
            } else {
                *votes.iter_mut().find(|(_, count)| *count == 0).unwrap() = (cluster_index, 1);
            }
        };

        for &ei in &face.edges {
            //eprintln!("face {} edge {} cluster {}", fi, ei, mesh.edges[ei as usize].cluster);
            vote(mesh.edges[ei as usize].cluster as u32);
        }

        let cluster = votes.iter().max_by_key(|&(_, count)| count).unwrap().0;
        //eprintln!("face {} assigned to cluster {}", fi, cluster);
        clusters[cluster as usize].faces.push(fi as u32);
    }

    info!("merged {} edges into {} clusters", mesh.edges.len(), num_clusters);
    clusters
}

fn convert_edge_clusters_to_meshlets(mesh: &Mesh, clusters: &[EdgeCluster]) -> ProcessedMesh {
    let num_clusters = clusters.iter().filter(|c| !c.edges.is_empty()).count();

    // Generate GPU vertices.
    // We generate one GPU vertex per mesh vertex, unless the mesh vertex shares its attributes
    // (position and normal) with another vertex.
    #[derive(Copy, Clone, PartialOrd, Ord, Eq, PartialEq, Hash)]
    struct VertexKey {
        key: [u32; 4],
    }
    impl VertexKey {
        fn new(point_index: u32, normal: Vec3) -> Self {
            let nx = normal.x.to_bits(); // sue me
            let ny = normal.y.to_bits();
            let nz = normal.z.to_bits();
            Self {
                key: [point_index, nx, ny, nz],
            }
        }
    }

    // same as gpu_vertices, but partitioned by meshlet
    let mut partitioned_vertices: Vec<MeshVertex> = Vec::new();
    let mut partitioned_normals: Vec<Vec3> = Vec::new();
    let mut partitioned_faces: Vec<MeshFace> = Vec::with_capacity(mesh.faces.len());
    let mut partitioned_edges: Vec<MeshEdge> = Vec::with_capacity(mesh.edges.len());
    let mut meshlets: Vec<Meshlet> = Vec::with_capacity(num_clusters);
    let mut draw_commands: Vec<DrawIndirectCommand> = Vec::with_capacity(num_clusters);

    // Generate meshlets that contain all the faces and vertices incident to
    // the edges in a clusters.

    // nomenclature:
    //      VI  = vertex index (mesh.vertices[VI])
    //      FI  = face index (mesh.faces[FI])
    //      LVI = local vertex index (partitioned_vertices[base_vertex + LVI])
    //      LNI = local normal index (partitioned_normals[base_face_normal + LNI])
    //
    // local_vertex_map     maps VI  -> LVI relative to current base_vertex
    // local_face_normals   maps FI  -> LNI relative to current base_face_normal
    // local_vertex_dedup:  maps (point index, normal) -> LVI relative to current base_vertex
    for (_i_cluster, cluster) in clusters.iter().filter(|c| !c.edges.is_empty()).enumerate() {
        let mut local_vertex_map: HashMap<u32, u16> = HashMap::new();
        let mut local_vertex_dedup: HashMap<VertexKey, u16> = HashMap::new();
        let mut local_face_normals: HashMap<u32, u16> = HashMap::new();

        let base_vertex = partitioned_vertices.len() as u32;
        let base_edge = partitioned_edges.len() as u32;
        let base_face = partitioned_faces.len() as u32;
        let base_face_normal = partitioned_normals.len() as u32;

        let mut emit_local_vertex = |vi: u32| -> u16 {
            *local_vertex_map.entry(vi).or_insert_with(|| {
                // insert or fuse with equivalent vertex
                let vertex = &mesh.vertices[vi as usize];
                let key = VertexKey::new(vertex.point, vertex.normal);
                let lvi = *local_vertex_dedup.entry(key).or_insert_with(|| {
                    partitioned_vertices.push(MeshVertex {
                        position: mesh.points[vertex.point as usize].position,
                        normal: vertex.normal,
                        point_index: vertex.point,
                        pointy: mesh.points[vertex.point as usize].pointy,
                    });
                    (partitioned_vertices.len() as u32 - 1 - base_vertex) as u16
                });
                lvi
            })
        };

        // --- emit meshlet vertices ---

        for (i_edge, &edge_index) in cluster.edges.iter().enumerate() {
            let edge = &mesh.edges[edge_index as usize];

            let mut incident_face_normals = [0; 2];

            for (i, &fi) in [edge.f_cw, edge.f_ccw].iter().enumerate() {
                if fi == NO_FACE {
                    incident_face_normals[i] = 0xFFFF;
                } else {
                    let lni = local_face_normals.entry(fi).or_insert_with(|| {
                        let face = &mesh.faces[fi as usize];
                        partitioned_normals.push(face.normal);
                        (partitioned_normals.len() as u32 - 1 - base_face_normal) as u16
                    });
                    incident_face_normals[i] = *lni;
                }
            }

            let fi = match (edge.f_cw, edge.f_ccw) {
                (NO_FACE, NO_FACE) => panic!("edge has no incident faces"),
                (NO_FACE, _) => edge.f_ccw,
                (_, NO_FACE) => edge.f_cw,
                _ => edge.f_cw,
            } as usize;

            // find vertices that corresponds to the edge endpoints
            // for crease edges, this could be the vertices coming from either incident face
            // in any case, for edges, only the position is valid
            let edge_v0 = *mesh.faces[fi]
                .vertices
                .iter()
                .find(|&vi| mesh.vertices[*vi as usize].point == edge.p0)
                .unwrap();
            let edge_v1 = *mesh.faces[fi]
                .vertices
                .iter()
                .find(|&vi| mesh.vertices[*vi as usize].point == edge.p1)
                .unwrap();

            // map vertices into meshlet data
            let vertex_0 = emit_local_vertex(edge_v0);
            let vertex_1 = emit_local_vertex(edge_v1);

            partitioned_edges.push(MeshEdge {
                normal_cw: incident_face_normals[0],
                normal_ccw: incident_face_normals[1],
                vertex_0,
                vertex_1,
            });
        }

        // --- emit meshlet faces ---
        let mut group_id = None;
        for &fi in cluster.faces.iter() {
            let face = &mesh.faces[fi as usize];
            let lv0 = emit_local_vertex(face.vertices[0]);
            let lv1 = emit_local_vertex(face.vertices[1]);
            let lv2 = emit_local_vertex(face.vertices[2]);

            partitioned_faces.push(MeshFace {
                vertices: [lv0, lv1, lv2],
            });

            // check group ID consistency
            for &v in &face.vertices {
                let point = mesh.vertices[v as usize].point;
                let id = mesh.points[point as usize].id;
                if group_id.is_none() {
                    group_id = Some(id as u16);
                } else {
                    if group_id.unwrap() != id as u16 {
                        warn!(
                            "meshlet contains faces from multiple groups: {} and {}",
                            group_id.unwrap(),
                            id
                        );
                    }
                }
            }
        }

        let face_count = cluster.faces.len() as u16;

        meshlets.push(Meshlet {
            normal_cone: cluster.cone,
            base_vertex,
            base_edge,
            base_face,
            base_face_normal,
            vertex_count: (partitioned_vertices.len() as u32 - base_vertex) as u16,
            edge_count: cluster.edges.len() as u16,
            face_count: cluster.faces.len() as u16,
            face_normal_count: local_face_normals.len() as u16,
            group_id: group_id.unwrap_or(0),
        });

        draw_commands.push(DrawIndirectCommand {
            vertex_count: (face_count * 3) as u32,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        /*eprintln!(
            "cluster: {} edges, {} vertices, {} faces",
            cluster_gpu.edge_count, cluster_gpu.vertex_count, cluster_gpu.face_count
        );
        eprintln!("- edges: {:?}", cluster_gpu.edges);
        eprintln!("- vertices: {:?}", &cluster_gpu.vertices);*/
    }

    let vertices = gpu::Buffer::from_slice(&partitioned_vertices);
    let edges = gpu::Buffer::from_slice(&partitioned_edges);
    let faces = gpu::Buffer::from_slice(&partitioned_faces);
    let face_normals = gpu::Buffer::from_slice(&partitioned_normals);
    let meshlets = gpu::Buffer::from_slice(&meshlets);
    let draw_commands = gpu::Buffer::from_slice(&draw_commands);

    let gpu = MeshData {
        vertices: vertices.ptr(),
        edges: edges.ptr(),
        faces: faces.ptr(),
        face_normals: face_normals.ptr(),
        meshlets: meshlets.ptr(),
    };
    ProcessedMesh {
        vertices,
        edges,
        faces,
        face_normals,
        meshlets,
        draw_commands,
        gpu,
    }
}

#[repr(C)]
struct ContourEdgeBuffer {
    count: u32,
    edges: [ContourEdge], // unsized
}

impl ContourEdgeBuffer {
    fn layout(count: usize) -> Layout {
        let (layout, _array_offset) = Layout::new::<u32>()
            .extend(Layout::array::<ContourEdge>(count).unwrap())
            .unwrap();
        layout.pad_to_align()
    }
}

/*
struct ContourRootParams {
    // input mesh
    SceneInfo* scene_info;
    MeshData mesh;
    uint cluster_count;
    uint vertex_count;

    // contour extraction output
    ContourEdgeBuffer* contour_edges;
    uint contour_point_count;
    uint* contour_point_list;     // successor list of contour points: global point index -> successor point index

    // contour ranking data
    uint64_t* contours_rank_successors_0;
    uint64_t* contours_rank_successors_1;

    // expanded contour data
    OutlineVertex* expanded_contour_vertices;
    uint* expanded_contour_indices;
    DrawIndirectCommand* expanded_contours_draw_command;

    // shading
    float3 main_light_direction;    // in world space
}*/

#[repr(C)]
#[derive(Copy, Clone)]
struct ContourPoint {
    position: Vec3,
    next: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ContoursRootParams {
    scene_info: Ptr<SceneInfoUniforms>,
    mesh_data: MeshData,
    meshlet_count: u32,
    eye: Vec3,

    contour_edges: Ptr<ContourEdgeBuffer>,
    // Number of contour points in the list (atomic counter)
    contour_point_count: u32,
    // Contour point linked list
    contour_point_list: Ptr<ContourPoint>,
    contour_point_list_subdiv: Ptr<ContourPoint>,
    // Size of the global -> contour remapping table
    global_point_count: u32,
    global_to_contour_map: Ptr<u32>,

    contour_rank_successors_0: Ptr<u64>,
    contour_rank_successors_1: Ptr<u64>,

    expanded_contour_vertex_count: u32,
    expanded_contour_vertices: Ptr<ContourVertex>,
    expanded_contour_indices: Ptr<u32>,
    expanded_contours_draw_command: Ptr<DrawIndirectCommand>,

    main_light_direction: Vec3,

    depth_texture: gpu::TextureHandle,
    angle_texture: gpu::TextureHandle,
    normal_texture: gpu::TextureHandle,

    jfa_result: gpu::StorageImageHandle,
    jfa_in_texture: gpu::StorageImageHandle,
    jfa_out_texture: gpu::StorageImageHandle,
    jfa_step_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct RankContoursRootParams {
    common: Ptr<ContoursRootParams>,
    len: u32,
    contour_rank_successors_0: Ptr<u64>,
    contour_rank_successors_1: Ptr<u64>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct SubdivideContoursRootParams {
    common: Ptr<ContoursRootParams>,
    point_count: u32,
}

/*
#[repr(C)]
#[derive(Copy, Clone)]
struct DepthPassRootParams {
    scene_info: Ptr<SceneInfoUniforms>,
    mesh_data: MeshData,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BaseRenderRootParams {
    scene_info: Ptr<SceneInfoUniforms>,
    mesh_data: MeshData,
    main_light_direction: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RenderOutlinesRootParams {
    pub(crate) scene_info: gpu::Ptr<SceneInfoUniforms>,
    pub(crate) vertices: gpu::Ptr<ContourVertex>,
    pub(crate) indices: gpu::Ptr<u32>,
}*/

/*
#[repr(C)]
#[derive(Clone, Copy)]
struct CornerDetectionRootParams {
    pub(crate) scene_info: gpu::Ptr<SceneInfoUniforms>,
    pub(crate) angle_tex: gpu::TextureHandle,
    pub(crate) normal_tex: gpu::TextureHandle,
    pub(crate) jfa_result: gpu::StorageImageHandle,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct JfaInitRootParams {
    pub(crate) scene_info: gpu::Ptr<SceneInfoUniforms>,
    pub(crate) angle_tex: gpu::StorageImageHandle,
    pub(crate) jfa_init_tex: gpu::StorageImageHandle,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct JfaStepRootParams {
    pub(crate) scene_info: gpu::Ptr<SceneInfoUniforms>,
    pub(crate) jfa_in_tex: gpu::StorageImageHandle,
    pub(crate) jfa_out_tex: gpu::StorageImageHandle,
    pub(crate) step_size: u32,
}
*/

static_assets! {
    static EXTRACT_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#extract_contours";
    static EXPAND_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#expand_contours";
    static RENDER_OUTLINES: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#render_outlines";
    static DEPTH_PASS: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#depth_pass";
    static CORNER_DETECTION: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#corner_detection";
    static BASE_RENDER: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#base_render";
    static JFA_INIT: gpu::ComputePipeline = "/shaders/game_shaders.sharc#jfa_init";
    static JFA_STEP: gpu::ComputePipeline = "/shaders/game_shaders.sharc#jfa_step";

    static BREAK_CONTOURS_INIT: gpu::ComputePipeline = "/shaders/game_shaders.sharc#break_contours_init";
    static BREAK_CONTOURS_STEP: gpu::ComputePipeline = "/shaders/game_shaders.sharc#break_contours_step";
    static RANK_CONTOURS_INIT: gpu::ComputePipeline = "/shaders/game_shaders.sharc#rank_contours_init";
    static RANK_CONTOURS_STEP: gpu::ComputePipeline = "/shaders/game_shaders.sharc#rank_contours_step";

    static SETUP_SUBDIVIDE_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#setup_subdivide_contours";
    static FINISH_SUBDIVIDE_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#finish_subdivide_contours";
    static SUBDIVIDE_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#subdivide_contours";
}

const RANK_CONTOURS_GROUP_SIZE: u32 = 256;
const EXPAND_CONTOURS_GROUP_SIZE: u32 = 256;
const SUBDIVIDE_CONTOURS_GROUP_SIZE: u32 = 256;

impl OutlineExperiment {
    pub fn new() -> Self {
        Self {
            mesh: Mesh::default(),
            processed_mesh: None,
            expanded_contour_vertices: gpu::Buffer::from_slice(&[]),
            expanded_contour_indices: gpu::Buffer::from_slice(&[]),
            contour_edges: unsafe { gpu::Buffer::from_layout(ContourEdgeBuffer::layout(1)) },
            contour_point_list: gpu::Buffer::from_slice(&[]),
            contour_point_list_subdiv: gpu::Buffer::from_slice(&[]),
            global_to_contour_index_map: gpu::Buffer::from_slice(&[]),
            contour_rank_successors: gpu::Buffer::from_slice(&[]),
            contour_rank_successors_1: gpu::Buffer::from_slice(&[]),
            expanded_contours_draw_command: gpu::Buffer::from_slice(&[DrawIndirectCommand {
                vertex_count: 0,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            }]),
            angle_texture: Image::new(gpu::ImageCreateInfo {
                width: 1,
                height: 1,
                format: gpu::Format::R16G16B16A16_UINT,
                usage: gpu::ImageUsage::SAMPLED | gpu::ImageUsage::STORAGE | gpu::ImageUsage::COLOR_ATTACHMENT,
                ..
            }),
            normal_texture: Image::new(gpu::ImageCreateInfo {
                width: 1,
                height: 1,
                format: gpu::Format::A2B10G10R10_UNORM_PACK32,
                usage: gpu::ImageUsage::SAMPLED | gpu::ImageUsage::STORAGE | gpu::ImageUsage::COLOR_ATTACHMENT,
                ..
            }),
            jfa_0: Image::new(gpu::ImageCreateInfo {
                width: 1,
                height: 1,
                format: gpu::Format::R16G16_SINT,
                usage: gpu::ImageUsage::STORAGE,
                ..
            }),
            jfa_1: Image::new(gpu::ImageCreateInfo {
                width: 1,
                height: 1,
                format: gpu::Format::R16G16_SINT,
                usage: gpu::ImageUsage::STORAGE,
                ..
            }),
            lock_view: false,
            locked_eye: Vec3::ZERO,
        }
    }

    /// Loads geometry from a houdini geometry file.
    fn load_geometry(&mut self, path: &Path) {
        let geo = hgeo::Geo::load(path).unwrap();

        let points = (0..geo.point_count)
            .map(|ptnum| {
                let position = geo.point(ptnum as u32, "P");
                let pointy = geo.point(ptnum as u32, "pointy");
                let id = geo.point(ptnum as u32, "id");
                Point { position, pointy, id }
            })
            .collect::<Vec<_>>();

        geo.point_attribute_typed::<Vec3>("P").unwrap().as_slice().to_vec();

        let (vertices, indices) = {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            hgeo::util::polygons_to_triangle_mesh(
                &geo,
                |g, vtxnum| {
                    let normal: Vec3 = g.vertex(vtxnum, "N");
                    let point = g.vertexpoint(vtxnum);
                    //let point = g.vertexpoint(vtxnum);
                    //eprintln!("vertex {vtxnum}: pt = {point}");
                    vertices.push(Vertex {
                        point: g.vertexpoint(vtxnum),
                        normal,
                    });
                    vertices.len() as u32 - 1
                },
                |i0, i1, i2| {
                    indices.push(i0);
                    indices.push(i1);
                    indices.push(i2);
                },
            );
            (vertices, indices)
        };

        let mut mesh = Mesh::new(points, vertices, &indices);

        let edge_clusters = cluster_edges(&mut mesh, 10.0);
        let processed_mesh = convert_edge_clusters_to_meshlets(&mesh, &edge_clusters);

        self.contour_edges = unsafe { gpu::Buffer::from_layout(ContourEdgeBuffer::layout(mesh.edges.len())) };
        self.global_to_contour_index_map.discard_resize(mesh.points.len());
        self.contour_rank_successors.discard_resize(mesh.points.len());
        self.contour_rank_successors_1.discard_resize(mesh.points.len());

        // num points + some for subdivision
        let subdiv_points = mesh.points.len() * 8;
        self.contour_point_list.discard_resize(subdiv_points);
        self.contour_point_list_subdiv.discard_resize(subdiv_points);
        self.expanded_contour_vertices.discard_resize(subdiv_points * 2);
        self.expanded_contour_indices.discard_resize(subdiv_points * 6);

        let total_gpu_size = processed_mesh.vertices.byte_size()
            + processed_mesh.edges.byte_size()
            + processed_mesh.faces.byte_size()
            + processed_mesh.face_normals.byte_size()
            + processed_mesh.meshlets.byte_size()
            + processed_mesh.draw_commands.byte_size()
            + self.contour_point_list.byte_size()
            + self.contour_rank_successors.byte_size()
            + self.contour_rank_successors_1.byte_size()
            + self.contour_edges.byte_size();

        info!(
            "loaded mesh from {}:
                   Count        Byte size
    vertices       {:<8}     {:<8} ({} B per elem)
    faces          {:<8}     {:<8} ({} B per elem)
    face normals   {:<8}     {:<8} ({} B per elem)
    edges          {:<8}     {:<8} ({} B per elem)
    meshlets       {:<8}     {:<8} ({} B per elem)
    draw cmds      {:<8}     {:<8} ({} B per elem)
    contour list   {:<8}     {:<8} ({} B per elem)
    contour ranks  {:<8}     {:<8} ({} B per elem)
    contour edges               {:<8}
    Total:                      {}",
            path.display(),
            processed_mesh.vertices.len(),
            ByteSize::b(processed_mesh.vertices.byte_size()).display().si(),
            processed_mesh.vertices.byte_size() / processed_mesh.vertices.len() as u64,
            processed_mesh.faces.len(),
            ByteSize::b(processed_mesh.faces.byte_size()).display().si(),
            processed_mesh.faces.byte_size() / processed_mesh.faces.len() as u64,
            processed_mesh.face_normals.len(),
            ByteSize::b(processed_mesh.face_normals.byte_size()).display().si(),
            processed_mesh.face_normals.byte_size() / processed_mesh.face_normals.len() as u64,
            processed_mesh.edges.len(),
            ByteSize::b(processed_mesh.edges.byte_size()).display().si(),
            processed_mesh.edges.byte_size() / processed_mesh.edges.len() as u64,
            processed_mesh.meshlets.len(),
            ByteSize::b(processed_mesh.meshlets.byte_size()).display().si(),
            processed_mesh.meshlets.byte_size() / processed_mesh.meshlets.len() as u64,
            processed_mesh.draw_commands.len(),
            ByteSize::b(processed_mesh.draw_commands.byte_size()).display().si(),
            processed_mesh.draw_commands.byte_size() / processed_mesh.draw_commands.len() as u64,
            self.contour_point_list.len(),
            ByteSize::b(self.contour_point_list.byte_size()).display().si(),
            self.contour_point_list.byte_size() / self.contour_point_list.len() as u64,
            self.contour_rank_successors.len(),
            ByteSize::b(self.contour_rank_successors.byte_size()).display().si(),
            self.contour_rank_successors.byte_size() / self.contour_rank_successors.len() as u64,
            ByteSize::b(self.contour_edges.byte_size()).display().si(),
            ByteSize::b(total_gpu_size).display().si(),
        );

        self.processed_mesh = Some(processed_mesh);
        self.mesh = mesh;
    }

    pub(crate) fn input(&mut self, input_event: &InputEvent) {
        if input_event.is_shortcut("Ctrl+O") {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Houdini Geometry", &["geo", "bgeo"])
                .pick_file()
            {
                self.load_geometry(&path);
            }
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.angle_texture.discard_resize(Size3D::new(width, height, 1));
        self.normal_texture.discard_resize(Size3D::new(width, height, 1));
        self.jfa_0.discard_resize(Size3D::new(width, height, 1));
        self.jfa_1.discard_resize(Size3D::new(width, height, 1));
    }

    pub(crate) fn render(
        &mut self,
        cmd: &mut gpu::CommandBuffer,
        color_target: &gpu::Image,
        depth_target: &gpu::Image,
        scene_info: &SceneInfo,
    ) -> Result<(), AssetLoadError> {
        let Some(ref mesh) = self.processed_mesh else {
            return Ok(());
        };

        self.lock_view = tweak!(lock_view = false);
        if !self.lock_view {
            self.locked_eye = scene_info.eye;
        }

        let root_params = cmd.upload(&ContoursRootParams {
            scene_info: scene_info.gpu,
            mesh_data: mesh.gpu,
            meshlet_count: mesh.meshlets.len() as u32,

            eye: self.locked_eye,
            contour_edges: self.contour_edges.ptr(),
            contour_point_count: 0,
            contour_point_list: self.contour_point_list.ptr(),
            contour_point_list_subdiv: self.contour_point_list_subdiv.ptr(),
            global_point_count: self.mesh.points.len() as u32,
            global_to_contour_map: self.global_to_contour_index_map.ptr(),

            contour_rank_successors_0: self.contour_rank_successors.ptr(),
            contour_rank_successors_1: self.contour_rank_successors_1.ptr(),

            expanded_contour_vertex_count: 0,
            expanded_contour_vertices: self.expanded_contour_vertices.ptr(),
            expanded_contour_indices: self.expanded_contour_indices.ptr(),
            expanded_contours_draw_command: self.expanded_contours_draw_command.ptr(),

            main_light_direction: tweak!(main_light_direction = Vec3::new(0.5, -1.0, 0.5).normalize()),

            depth_texture: depth_target.texture_handle(),
            angle_texture: self.angle_texture.texture_handle(),
            normal_texture: self.normal_texture.texture_handle(),

            jfa_result: self.jfa_0.storage_handle(),
            jfa_in_texture: self.jfa_0.storage_handle(),
            jfa_out_texture: self.jfa_1.storage_handle(),
            jfa_step_size: 1,
        });

        /////////////////////////////////////////////////////////
        // base render & depth pass
        {
            cmd.barrier_dst(BarrierFlags::DEPTH_STENCIL);
            let mut encoder = cmd.begin_rendering(
                &[
                    gpu::ColorAttachment {
                        image: color_target,
                        clear_value: None,
                    },
                    gpu::ColorAttachment {
                        image: &self.normal_texture,
                        clear_value: Some([0.0, 0.0, 0.0, 0.0]),
                    },
                ],
                Some(gpu::DepthStencilAttachment {
                    image: &depth_target,
                    depth_clear_value: None,
                    stencil_clear_value: None,
                }),
            );
            encoder.bind_graphics_pipeline(&*BASE_RENDER.read()?);
            encoder.draw_indirect(
                TriangleList,
                None,
                &mesh.draw_commands,
                0..mesh.draw_commands.len() as u32,
                root_params,
            );
            encoder.finish();
        }

        unsafe {
            // clear DrawIndirectCommand::vertex
            cmd.update_buffer(&self.expanded_contours_draw_command.as_bytes(), 0, &[0; 4]);
        }

        /////////////////////////////////////////////////////////
        // contour extraction
        cmd.bind_compute_pipeline(&*EXTRACT_CONTOURS.read()?);

        unsafe {
            // clear ContourEdgeBuffer::count
            // not very pretty
            cmd.update_buffer(&self.contour_edges.as_bytes(), 0, &[0; 4]);
            cmd.fill_buffer(&self.global_to_contour_index_map.as_bytes().slice(..), 0xFFFF_FFFF);
            cmd.fill_buffer(&self.contour_point_list.as_bytes().slice(..), 0xFFFF_FFFF);
            cmd.fill_buffer(&self.contour_point_list_subdiv.as_bytes().slice(..), 0xFFFF_FFFF);
            cmd.barrier(BarrierFlags::TRANSFER_WRITE, BarrierFlags::COMPUTE_SHADER | BarrierFlags::STORAGE);
        }

        cmd.dispatch(mesh.meshlets.len() as u32, 1, 1, root_params);

        cmd.barrier(
            BarrierFlags::COMPUTE_SHADER | BarrierFlags::STORAGE | BarrierFlags::DEPTH_STENCIL,
            BarrierFlags::COMPUTE_SHADER |
                BarrierFlags::FRAGMENT_SHADER
                | BarrierFlags::SAMPLED_READ
                | BarrierFlags::STORAGE
                | BarrierFlags::INDIRECT_READ,
        );

        /////////////////////////////////////////////////////////
        // contour loop breaking
        // find contour loops and determine a "break" point for each loop that defines the starting
        // point for ranking the list
        {
            // init

            let n = self.mesh.points.len() as u32;
            let groups_count = n.div_ceil(RANK_CONTOURS_GROUP_SIZE) as u32;

            cmd.bind_compute_pipeline(&*BREAK_CONTOURS_INIT.read()?);
            let mut params = RankContoursRootParams {
                common: root_params,
                len: n,
                contour_rank_successors_0: self.contour_rank_successors.ptr(),
                contour_rank_successors_1: self.contour_rank_successors_1.ptr(),
            };
            cmd.dispatch(groups_count, 1, 1, &params);

            let cs_barrier = BarrierFlags::COMPUTE_SHADER | BarrierFlags::STORAGE;

            cmd.barrier(cs_barrier, cs_barrier);

            let mut rank_steps = (n as f32).log2().ceil() as u32 + tweak!(extra_ranking_steps = 0u32);
            // round up to an even number so that we end up writing to the correct buffer
            if rank_steps % 2 != 0 {
                rank_steps += 1;
            }

            cmd.bind_compute_pipeline(&*BREAK_CONTOURS_STEP.read()?);
            for _ in 0..rank_steps {
                cmd.dispatch(groups_count, 1, 1, &params);
                std::mem::swap(
                    &mut params.contour_rank_successors_0,
                    &mut params.contour_rank_successors_1,
                );
                cmd.barrier(cs_barrier, cs_barrier);
            }

            // contour ranking
            cmd.bind_compute_pipeline(&*RANK_CONTOURS_INIT.read()?);
            cmd.dispatch(groups_count, 1, 1, &params);

            cmd.barrier(cs_barrier, cs_barrier);

            cmd.bind_compute_pipeline(&*RANK_CONTOURS_STEP.read()?);
            for _ in 0..rank_steps {
                cmd.dispatch(groups_count, 1, 1, &params);
                std::mem::swap(
                    &mut params.contour_rank_successors_0,
                    &mut params.contour_rank_successors_1,
                );
                cmd.barrier(cs_barrier, cs_barrier);
            }
        }

        /////////////////////////////////////////////////////////
        // contour subdivision
        {
            let point_count = self.contour_point_list.len() as u32;
            let groups_count = point_count.div_ceil(SUBDIVIDE_CONTOURS_GROUP_SIZE);


            let subdivision_levels = tweak!(contour_subdivision_levels = 0u32);
            for _ in 0..subdivision_levels {
                let params = cmd.upload(&SubdivideContoursRootParams {
                    common: root_params,
                    point_count,
                });

                let cs_barrier = BarrierFlags::COMPUTE_SHADER | BarrierFlags::STORAGE;
                cmd.bind_compute_pipeline(&*SETUP_SUBDIVIDE_CONTOURS.read()?);
                cmd.dispatch(1, 1, 1, params);
                cmd.barrier(cs_barrier, cs_barrier);
                cmd.bind_compute_pipeline(&*SUBDIVIDE_CONTOURS.read()?);
                cmd.dispatch(groups_count, 1, 1, params);
                cmd.barrier(cs_barrier, cs_barrier);
                cmd.bind_compute_pipeline(&*FINISH_SUBDIVIDE_CONTOURS.read()?);
                cmd.dispatch(groups_count, 1, 1, params);
                cmd.barrier(cs_barrier, cs_barrier);
            }
        }



        /////////////////////////////////////////////////////////
        // expand contours to quad geometry
        {
            let n = self.mesh.edges.len() as u32;
            let groups_count = n.div_ceil(EXPAND_CONTOURS_GROUP_SIZE);
            cmd.bind_compute_pipeline(&*EXPAND_CONTOURS.read()?);
            cmd.dispatch(groups_count, 1, 1, root_params);
            cmd.barrier_source(
                BarrierFlags::COMPUTE_SHADER | BarrierFlags::STORAGE,
            );
        }

        /////////////////////////////////////////////////////////
        // render contours
        {
            cmd.barrier_dst(BarrierFlags::VERTEX_SHADER | BarrierFlags::STORAGE | BarrierFlags::INDIRECT_READ);

            let mut encoder = cmd.begin_rendering(
                &[gpu::ColorAttachment {
                    image: &self.angle_texture,
                    clear_value: Some([0.0, 0.0, 0.0, 0.0]),
                    ..
                }],
                None,
            );

            encoder.bind_graphics_pipeline(&*RENDER_OUTLINES.read()?);
            encoder.draw_indirect(TriangleList, None, &self.expanded_contours_draw_command, 0..1, root_params);
            encoder.finish();

            cmd.barrier_source(BarrierFlags::FRAGMENT_SHADER | BarrierFlags::COLOR_ATTACHMENT);
        }

        /*/////////////////////////////////////////////////////////
        // Jump flooding
        {
            let w = self.angle_texture.width();
            let h = self.angle_texture.height();
            let ntiles_x = w.div_ceil(JFA_TILE_SIZE);
            let ntiles_y = h.div_ceil(JFA_TILE_SIZE);

            let wp = w.next_power_of_two();
            let hp = h.next_power_of_two();
            let mut step_size = wp.max(hp) / 2;

            // init
            cmd.bind_compute_pipeline(&*JFA_INIT.read()?);
            cmd.dispatch(
                ntiles_x,
                ntiles_y,
                1,
                &JfaInitRootParams {
                    scene_info: scene_info.gpu,
                    angle_tex: self.angle_texture.storage_handle(),
                    jfa_init_tex: self.jfa_0.storage_handle(),
                },
            );
            let bf = BarrierFlags::COMPUTE_SHADER | BarrierFlags::STORAGE;
            cmd.barrier_source(bf);

            cmd.bind_compute_pipeline(&*JFA_STEP.read()?);
            let mut swapped = false;
            let mut extra_passes = 0;
            loop {
                cmd.barrier_dst(bf);
                cmd.dispatch(
                    ntiles_x,
                    ntiles_y,
                    1,
                    &JfaStepRootParams {
                        scene_info: scene_info.gpu,
                        jfa_in_tex: self.jfa_0.storage_handle(),
                        jfa_out_tex: self.jfa_1.storage_handle(),
                        step_size,
                    },
                );
                if step_size == 1 {
                    extra_passes += 1;
                    if extra_passes > 2 {
                        break;
                    }
                } else {
                    step_size /= 2;
                }
                std::mem::swap(&mut self.jfa_0, &mut self.jfa_1);
                swapped = !swapped;
                cmd.barrier_source(bf);
            }
            // make sure the final result is in jfa_1
            if swapped {
                std::mem::swap(&mut self.jfa_0, &mut self.jfa_1);
            }
        }*/

        /////////////////////////////////////////////////////////
        // corner detection
        if tweak!(show_corner_detection = true) {
            cmd.barrier_dst(BarrierFlags::FRAGMENT_SHADER | BarrierFlags::SAMPLED_READ);
            let mut encoder = cmd.begin_rendering(
                &[gpu::ColorAttachment {
                    image: color_target,
                    clear_value: None,
                }],
                Some(gpu::DepthStencilAttachment {
                    image: &depth_target,
                    depth_clear_value: None,
                    stencil_clear_value: None,
                }),
            );
            encoder.bind_graphics_pipeline(&*CORNER_DETECTION.read()?);
            encoder.draw(
                TriangleList,
                None,
                0..6,
                0..1,
                root_params,
            );
            encoder.finish();
        }

        if tweak!(show_edge_clusters = false) {
            let mut encoder = cmd.begin_rendering(
                &[gpu::ColorAttachment {
                    image: color_target,
                    clear_value: None,
                }],
                Some(gpu::DepthStencilAttachment {
                    image: &depth_target,
                    depth_clear_value: None,
                    stencil_clear_value: None,
                }),
            );

            fn random_color(index: usize) -> Srgba8 {
                let random_colors = &[
                    srgba8(255, 0, 0, 255),
                    srgba8(0, 255, 0, 255),
                    srgba8(0, 0, 255, 255),
                    srgba8(255, 255, 0, 255),
                    srgba8(0, 255, 255, 255),
                    srgba8(255, 0, 255, 255),
                    srgba8(192, 192, 192, 255),
                    srgba8(128, 0, 0, 255),
                    srgba8(128, 128, 0, 255),
                    srgba8(0, 128, 0, 255),
                    srgba8(128, 0, 128, 255),
                    srgba8(0, 128, 128, 255),
                    srgba8(0, 0, 128, 255),
                ];
                random_colors[index % random_colors.len()]
            }

            let mut vertices = Vec::with_capacity(self.mesh.edges.len() * 2);
            let mut lines = Vec::with_capacity(self.mesh.edges.len());
            for edge in self.mesh.edges.iter() {
                let p0 = &self.mesh.points[edge.p0 as usize];
                let p1 = &self.mesh.points[edge.p1 as usize];
                let color = random_color(edge.cluster);
                vertices.push(crate::experiments::lines::LineVertex {
                    position: p0.position,
                    color,
                });
                vertices.push(crate::experiments::lines::LineVertex {
                    position: p1.position,
                    color,
                });
                lines.push(crate::experiments::lines::Line {
                    start_vertex: (vertices.len() - 2) as u32,
                    vertex_count: 2,
                    ..
                });
            }

            draw_lines(&mut encoder, &vertices, &lines, scene_info);
            encoder.finish();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::SliceRandom;
    use rand::rng;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::IntoParallelIterator;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    #[test]
    fn test_edge_linking() {
        let mut edges = {
            let mut rng = rng();
            let mut edges = Vec::new();
            let n = 20;
            let nloops = 10;
            let mut points = (0..200000).collect::<Vec<_>>();
            points.shuffle(&mut rng);

            let mut i = 0;

            for _ in 0..nloops {
                let loop_base = i;
                while (i - loop_base) < n {
                    let p0 = points[i];
                    let p1 = points[loop_base + (i - loop_base + 1) % n];
                    edges.push((p0, p1));
                    i += 1;
                }
            }

            edges.shuffle(&mut rng);
            edges
        };

        fn print(edges: &[(usize, usize)]) {
            for (i, (p0, p1)) in edges.iter().enumerate() {
                eprintln!("{:>5} -> {:<5}", p0, p1);
            }
        }

        // print(&edges);

        //-------------------------------------------
        // 1st pass: renumber edges to have indices 0..N

        const NULL: usize = usize::MAX;

        let max_point = edges.iter().map(|&(p0, p1)| p0.max(p1)).max().unwrap();
        let mut point_map: Vec<usize> = vec![NULL; max_point + 1];

        // compact point indices
        for (i, &(p0, _)) in edges.iter().enumerate() {
            point_map[p0] = i;
        }
        for (i, edge) in edges.iter_mut().enumerate() {
            edge.0 = point_map[edge.0];
            edge.1 = point_map[edge.1];
        }

        eprintln!("Compacted edges");
        //print(&edges);

        // pointer jumping
        {
            let mut s = vec![NULL; edges.len() * 2];
            //let mut ss = vec![NULL; edges.len() * 2];
            let mut x = vec![0; edges.len() * 2];

            for &(p0, p1) in &edges {
                s[p0] = p1;
                x[p0] = 1;
                x[p1] = 1;
            }
            //
            //eprintln!("Successor map");
            //for (i, s) in s.iter().enumerate() {
            //    eprintln!("{:>5} -> {}", i, s);
            //}

            let start = std::time::Instant::now();
            for i in 0..s.len() {
                loop {
                    if s[i] == NULL || s[s[i]] == NULL || s[i] == s[s[i]] {
                        break;
                    }
                    // atomic
                    //dbg!((i, s[i], s[s[i]], x[i], x[s[i]]));
                    x[i] += x[s[i]];
                    s[i] = s[s[i]];
                }
            }
            let duration = start.elapsed();
            eprintln!("Pointer jumping: {}us", duration.as_micros());

            //
            eprintln!("Ranks:");
            for (i, rank) in x.iter().enumerate() {
                eprintln!("{:>5}: {} root={}", i, rank, s[i]);
            }

            edges.sort_by(|&(p0a, _), &(p0b, _)| (s[p0b], x[p0b]).cmp(&(s[p0a], x[p0a])));

            eprintln!("sorted edges:");
            print(&edges);
        }

        // parallel ver.
        {
            let mut sx: Vec<AtomicU64> = vec![0xFFFF_FFFF_0000_0000; edges.len() * 2]
                .into_iter()
                .map(AtomicU64::new)
                .collect();

            const NULL32: usize = 0xFFFF_FFFF;
            for &(p0, p1) in &edges {
                sx[p0].store(1u64 | (p1 as u64) << 32, Relaxed);
                //sx[p1].store(1u64 | (NULL32 as u64) << 32, Relaxed);
            }

            let start = std::time::Instant::now();
            (0..sx.len()).into_par_iter().for_each(|i| {
                loop {
                    let sxi = sx[i].load(Relaxed);
                    let si = (sxi >> 32) as usize;
                    let xi = (sxi & 0xFFFF_FFFF) as usize;

                    if si == NULL32 {
                        break;
                    }

                    let ssxi = sx[si].load(Relaxed);
                    let xsi = (ssxi & 0xFFFF_FFFF) as usize;
                    let ssi = (ssxi >> 32) as usize;

                    if ssi == NULL32 || si == ssi {
                        break;
                    }

                    // atomic
                    //dbg!((i, s[i], s[s[i]], x[i], x[s[i]]));

                    //x[i] += x[s[i]];
                    //s[i] = s[s[i]];

                    let new_sxi = (xi + xsi) as u64 | (ssi as u64) << 32;
                    sx[i].store(new_sxi, Relaxed);
                }
            });
            let duration = start.elapsed();
            eprintln!("Parallel pointer jumping: {}us", duration.as_micros());

            eprintln!("Ranks (parallel):");
            for (i, rank) in sx.iter().enumerate() {
                let sx = rank.load(Relaxed);
                let xi = (sx & 0xFFFF_FFFF) as usize;
                let si = (sx >> 32) as usize;
                eprintln!("{:>5}: rank={}, s={}", i, xi, si);
            }

            edges.sort_by(|&(p0a, _), &(p0b, _)| {
                let sxa = sx[p0a].load(Relaxed);
                let sxb = sx[p0b].load(Relaxed);
                sxb.cmp(&sxa)
            });

            eprintln!("sorted edges (parallel):");
            print(&edges);
        }
    }
}
