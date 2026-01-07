use crate::experiments::coat::ExpandedVertex;
use crate::experiments::lines::draw_lines;
use crate::{SceneInfo, SceneInfoUniforms};
use color::{Srgba8, srgba8};
use gamelib::asset::Handle;
use gamelib::input::InputEvent;
use gamelib::pipeline_cache::get_graphics_pipeline;
use gpu::PrimitiveTopology::TriangleList;
use gpu::{Barrier, BarrierFlags, Buffer, BufferCreateInfo, DrawIndirectCommand, MemoryLocation, RenderPassInfo, RootParams};
use hgeo::util::polygons_to_triangle_mesh;
use log::info;
use math::Vec3;
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Range;
use std::path::Path;

/// Loads geometry from the specified file path.
pub struct OutlineExperiment {
    pipeline: Handle<gpu::ComputePipeline>,
    debug_strokes: Handle<gpu::GraphicsPipeline>,
    depth_pass: Handle<gpu::GraphicsPipeline>,
    mesh: Mesh,
    clusters: Buffer<[Cluster]>,
    cluster_draw_commands: Buffer<[DrawIndirectCommand]>,
    outline_vertices: Buffer<[ExpandedVertex]>,
    outline_indices: Buffer<[u32]>,
}

struct Vertex {
    position: Vec3,
    normal: Vec3,
}

struct Face {
    normal: Vec3,
    vertices: [u32; 3],
    edges: [u32; 3],
}

struct Edge {
    /// 1st vertex index
    v0: u32,
    /// 2nd vertex index
    v1: u32,
    /// 1st adjacent face index
    f0: u32,
    /// 2nd adjacent face index
    f1: u32,
    /// Current cluster
    cluster: usize,
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
    cone: NormalCone,
}

const MAX_EDGES_PER_CLUSTER: usize = 128;

////////////////////////////////////////////////////////////////////////////////////////////////////

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct ClusterEdge {
    face_0: u16,
    face_1: u16,
    vertex_0: u16,
    vertex_1: u16,
}

impl fmt::Debug for ClusterEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(f0={},f1={},v0={},v1={})",
            self.face_0, self.face_1, self.vertex_0, self.vertex_1
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
struct ClusterVertex {
    position: Vec3,
    normal: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct ClusterFace {
    vertices: [u16; 3],
    normal: Vec3,
}

/// Edge "meshlet" for GPU consumption
#[repr(C)]
#[derive(Copy, Clone)]
struct Cluster {
    /// Vertices in this cluster.
    vertices: [ClusterVertex; MAX_EDGES_PER_CLUSTER],
    /// All edges (represented by the two faces they connect) in this cluster
    edges: [ClusterEdge; MAX_EDGES_PER_CLUSTER],
    /// All faces in this cluster
    faces: [ClusterFace; MAX_EDGES_PER_CLUSTER * 2],
    normal_cone: NormalCone,
    vertex_count: u16,
    edge_count: u16,
    face_count: u16,
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            vertices: [ClusterVertex::default(); MAX_EDGES_PER_CLUSTER],
            edges: [ClusterEdge::default(); MAX_EDGES_PER_CLUSTER],
            faces: [ClusterFace::default(); MAX_EDGES_PER_CLUSTER * 2],
            vertex_count: 0,
            edge_count: 0,
            face_count: 0,
            normal_cone: NormalCone::default(),
        }
    }
}

struct Mesh {
    face: Vec<Face>,
    edges: Vec<Edge>, // pairs of vertex indices
    vertices: Vec<Vertex>,
    incident_edges: Vec<SmallVec<u32, 6>>, // per-vertex incident edge indices
}

impl Mesh {
    fn from_triangle_mesh(vertices: Vec<Vertex>, indices: &[u32]) -> Mesh {
        assert!(
            indices.len() % 3 == 0,
            "invalid number of indices: must be a multiple of 3"
        );

        let mut faces = Vec::with_capacity(indices.len() / 3);
        let mut edges = HashMap::new();

        for index in indices.chunks(3) {
            let v0 = index[0];
            let v1 = index[1];
            let v2 = index[2];

            let normal = {
                let p0 = vertices[v0 as usize].position;
                let p1 = vertices[v1 as usize].position;
                let p2 = vertices[v2 as usize].position;
                let edge1 = p1 - p0;
                let edge2 = p2 - p0;
                Vec3::cross(edge1, edge2).normalize()
            };

            faces.push(Face {
                normal,
                vertices: [v0, v1, v2],
                edges: [0; 3], // to be filled later
            });

            let mut insert_edge = |va: u32, vb: u32, face| {
                edges
                    .entry((va.min(vb), va.max(vb)))
                    .and_modify(|edge: &mut Edge| {
                        edge.f1 = face;
                    })
                    .or_insert_with(|| Edge {
                        v0: va.min(vb),
                        v1: va.max(vb),
                        f0: face,
                        f1: u32::MAX,
                        cluster: 0,
                    });
            };

            let face_index = (faces.len() - 1) as u32;
            insert_edge(v0, v1, face_index);
            insert_edge(v1, v2, face_index);
            insert_edge(v2, v0, face_index);
        }

        let edges: Vec<Edge> = edges.into_values().collect();

        let mut incident_edges: Vec<SmallVec<_, 6>> = vec![smallvec![]; vertices.len()];
        for (i_edge, edge) in edges.iter().enumerate() {
            incident_edges[edge.v0 as usize].push(i_edge as u32);
            incident_edges[edge.v1 as usize].push(i_edge as u32);
        }

        Mesh {
            face: faces,
            edges,
            vertices,
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
        if face_index == u32::MAX {
            return 0;
        }
        if faces.insert(face_index) {
            let face = &mesh.face[face_index as usize];
            n += face.normal;
            1
        } else {
            0
        }
    };

    for edge_index in cluster_edges {
        let edge = &mesh.edges[*edge_index as usize];
        count += accumulate(edge.f0);
        count += accumulate(edge.f1);
    }

    n /= count as f32;
    n = n.normalize();

    // determine the normal that deviates the most from the average
    let mut min_dot = 0.0f32;
    let mut out_normal = Vec3::ZERO;

    let mut update_min = |face_index: u32| {
        if face_index == u32::MAX {
            return;
        }
        let face = &mesh.face[face_index as usize];
        let angle = Vec3::dot(face.normal, n);
        if angle < min_dot {
            min_dot = angle;
            out_normal = face.normal;
        }
    };

    let mut center = Vec3::ZERO;
    for edge_index in cluster_edges {
        let edge = &mesh.edges[*edge_index as usize];
        update_min(edge.f0);
        update_min(edge.f1);
        let midpoint = (mesh.vertices[edge.v0 as usize].position + mesh.vertices[edge.v1 as usize].position) / 2.0;
        center += midpoint;
    }
    center /= cluster_edges.len() as f32;

    let mut radius = 0.0f32;
    for edge_index in cluster_edges {
        let edge = &mesh.edges[*edge_index as usize];
        let midpoint = (mesh.vertices[edge.v0 as usize].position + mesh.vertices[edge.v1 as usize].position) / 2.0;
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

struct ClusterEdgesParams {
    cam_dist: f32,
}

fn cluster_edges(
    mesh: &mut Mesh,
    params: &ClusterEdgesParams,
    draw_commands: &mut Vec<DrawIndirectCommand>,
) -> Buffer<[Cluster]> {
    // start with one cluster per edge
    let mut clusters: Vec<EdgeCluster> = Vec::new();

    for i in 0..mesh.edges.len() {
        let cone = compute_normal_cone(mesh, &[i as u32]);
        let cluster = EdgeCluster {
            edges: vec![i as u32],
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
                for vertex in [mesh.edges[i_edge].v0, mesh.edges[i_edge].v1] {
                    for &e2 in &mesh.incident_edges[vertex as usize] {
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

                        let accept_0 = acceptance_heuristic(&clusters[i_cluster].cone, params.cam_dist);
                        let accept_1 = acceptance_heuristic(&cluster_1.cone, params.cam_dist);
                        let accept_merged = acceptance_heuristic(&merged_cone, params.cam_dist);

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

    info!("merged {} edges into {} clusters", mesh.edges.len(), num_clusters);

    let mut clusters_gpu = gpu::Buffer::new(BufferCreateInfo {
        len: num_clusters,
        usage: Default::default(),
        memory_location: MemoryLocation::CpuToGpu,
        label: "",
    });

    // Convert to GPU data
    for (i_cluster, cluster) in clusters.iter().filter(|c| !c.edges.is_empty()).enumerate() {
        let mut cluster_gpu = Cluster::default();

        let mut vertex_map: HashMap<u32, u16> = HashMap::new();
        let mut face_map: HashMap<u32, u16> = HashMap::new();

        {
            let mut next_vertex_index = 0;
            for (i_edge, &edge_index) in cluster.edges.iter().enumerate() {
                let edge = &mesh.edges[edge_index as usize];

                let quad_vertices;
                let vertices = if edge.f0 == u32::MAX {
                    &mesh.face[edge.f1 as usize].vertices[..]
                } else if edge.f1 == u32::MAX {
                    &mesh.face[edge.f0 as usize].vertices[..]
                } else {
                    let v0 = edge.v0;
                    let v1 = edge.v1;
                    quad_vertices = [
                        edge.v0,
                        edge.v1,
                        mesh.face[edge.f0 as usize]
                            .vertices
                            .iter()
                            .find(|&&v| v != v0 && v != v1)
                            .unwrap()
                            .clone(),
                        mesh.face[edge.f1 as usize]
                            .vertices
                            .iter()
                            .find(|&&v| v != v0 && v != v1)
                            .unwrap()
                            .clone(),
                    ];
                    &quad_vertices
                };

                for &v_index in vertices {
                    if !vertex_map.contains_key(&v_index) {
                        let vertex = &mesh.vertices[v_index as usize];
                        cluster_gpu.vertices[next_vertex_index] = ClusterVertex {
                            position: vertex.position,
                            normal: vertex.normal,
                        };
                        vertex_map.insert(v_index, next_vertex_index as u16);
                        next_vertex_index += 1;
                    }
                }
            }
        }

        for &edge_index in cluster.edges.iter() {
            let edge = &mesh.edges[edge_index as usize];
            //eprintln!("edge count : {}", cluster.edges.len());
            for &f_index in &[edge.f0, edge.f1] {
                if f_index == u32::MAX {
                    continue;
                }
                if !face_map.contains_key(&f_index) {
                    let face = &mesh.face[f_index as usize];
                    //eprintln!("face vertices: {:?} ", face.vertices);

                    let v0 = vertex_map[&face.vertices[0]];
                    let v1 = vertex_map[&face.vertices[1]];
                    let v2 = vertex_map[&face.vertices[2]];
                    cluster_gpu.faces[cluster_gpu.face_count as usize] = ClusterFace {
                        vertices: [v0, v1, v2],
                        normal: face.normal,
                    };
                    face_map.insert(f_index, cluster_gpu.face_count);
                    cluster_gpu.face_count += 1;
                }
            }
        }

        for (i_edge, &edge_index) in cluster.edges.iter().enumerate() {
            let edge = &mesh.edges[edge_index as usize];
            let f0 = if edge.f0 == u32::MAX {
                u16::MAX
            } else {
                face_map[&edge.f0]
            };
            let f1 = if edge.f1 == u32::MAX {
                u16::MAX
            } else {
                face_map[&edge.f1]
            };
            let v0 = vertex_map[&edge.v0];
            let v1 = vertex_map[&edge.v1];

            cluster_gpu.edges[i_edge] = ClusterEdge {
                face_0: f0,
                face_1: f1,
                vertex_0: v0,
                vertex_1: v1,
            };
        }

        cluster_gpu.edge_count = cluster.edges.len() as u16;
        cluster_gpu.normal_cone = cluster.cone;

        //eprintln!(
        //    "cluster: {} edges, {} vertices, {} faces",
        //    cluster_gpu.edge_count, cluster_gpu.vertex_count, cluster_gpu.face_count
        //);
        //eprintln!("- edges: {:?}", cluster_gpu.edges);
        //eprintln!("- vertices: {:?}", &cluster_gpu.vertices);

        draw_commands.push(DrawIndirectCommand {
            vertex_count: (cluster_gpu.face_count as u32) * 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        unsafe {
            clusters_gpu.as_mut_slice()[i_cluster].write(cluster_gpu);
        }
    }
    clusters_gpu
}

/*
struct OutlineRootParams {
    SceneInfo* scene_info;
    Cluster* clusters;
    uint32_t cluster_count;
    PushBuffer<OutlineVertex> out_outline_vertices;
}
*/

#[repr(C)]
#[derive(Copy, Clone)]
struct OutlineExperimentRootParams {
    scene_info: gpu::Ptr<SceneInfoUniforms>,
    clusters: gpu::Ptr<[Cluster]>,
    cluster_count: u32,
    vertex_count: u32,
    out_vertices: gpu::Ptr<[ExpandedVertex]>,
    out_indices: gpu::Ptr<[u32]>,
    out_draw_command: gpu::Ptr<[gpu::DrawIndirectCommand]>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct DepthPassRootParams {
    scene_info: gpu::Ptr<SceneInfoUniforms>,
    clusters: gpu::Ptr<[Cluster]>,
    cluster_count: u32,
}

impl OutlineExperiment {
    pub fn new() -> Self {
        Self {
            pipeline: gamelib::pipeline_cache::get_compute_pipeline("/shaders/game_shaders.sharc#outline"),
            debug_strokes: get_graphics_pipeline("/shaders/game_shaders.sharc#debug_strokes"),
            depth_pass: get_graphics_pipeline("/shaders/game_shaders.sharc#depth_pass"),
            mesh: Mesh {
                face: vec![],
                edges: vec![],
                vertices: vec![],
                incident_edges: vec![],
            },
            clusters: gpu::Buffer::from_slice(&[], "outline_clusters"),
            cluster_draw_commands: gpu::Buffer::from_slice(&[], "outline_cluster_draw_commands"),
            outline_vertices: gpu::Buffer::from_slice(&[], "outline_vertices"),
            outline_indices: gpu::Buffer::from_slice(&[], "outline_indices"),
        }
    }

    /// Loads geometry from a houdini geometry file.
    fn load_geometry(&mut self, path: &Path) {
        let geo = hgeo::Geo::load(path).unwrap();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // load all polygon meshes
        polygons_to_triangle_mesh(
            &geo,
            |g, ptnum| {
                let position: Vec3 = g.point(ptnum, "P");
                let normal: Vec3 = g.point(ptnum, "N");
                vertices.push(Vertex { position, normal });
                vertices.len() as u32 - 1
            },
            |i0, i1, i2| {
                indices.push(i0);
                indices.push(i1);
                indices.push(i2);
            },
        );

        let mut mesh = Mesh::from_triangle_mesh(vertices, &indices);
        let cluster_params = ClusterEdgesParams { cam_dist: 10.0 };
        let mut draw_commands = Vec::new();
        self.clusters = cluster_edges(&mut mesh, &cluster_params, &mut draw_commands);
        self.cluster_draw_commands = gpu::Buffer::from_slice(&draw_commands, "outline_cluster_draw_commands");
        self.mesh = mesh;
        self.outline_vertices = gpu::Buffer::<[ExpandedVertex]>::new(gpu::BufferCreateInfo {
            len: self.mesh.vertices.len() * 20, // not great
            label: "outline_vertices",
            ..
        });
        self.outline_indices = gpu::Buffer::<[u32]>::new(gpu::BufferCreateInfo {
            len: self.mesh.vertices.len() * 40, // no idea
            label: "outline_indices",
            ..
        });
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

    pub(crate) fn render(
        &mut self,
        cmd: &mut gpu::CommandStream,
        color_target: &gpu::Image,
        depth_target: &gpu::Image,
        scene_info: &SceneInfo,
    ) {
        let outline_pipeline = match self.pipeline.read() {
            Ok(p) => p,
            Err(_) => return,
        };

        let Ok(debug_stroke_pipeline) = self.debug_strokes.read() else {
            return;
        };
        let Ok(depth_pass_pipeline) = self.depth_pass.read() else {
            return;
        };

        /////////////////////////////////////////////////////////
        // depth pass
        cmd.barrier(BarrierFlags::DEPTH_STENCIL);
        let mut encoder = cmd.begin_rendering(RenderPassInfo {
            color_attachments: &[],
            depth_stencil_attachment: Some(gpu::DepthStencilAttachment {
                image: &depth_target,
                ..
            }),
        });
        encoder.bind_graphics_pipeline(&*depth_pass_pipeline);
        encoder.draw_indirect(
            TriangleList,
            None,
            &self.cluster_draw_commands,
            0..self.cluster_draw_commands.len() as u32,
            RootParams::Immediate(&DepthPassRootParams {
                scene_info: scene_info.gpu,
                clusters: self.clusters.ptr(),
                cluster_count: self.clusters.len() as u32,
            }),
        );
        drop(encoder);

        /////////////////////////////////////////////////////////
        // contour extraction
        cmd.bind_compute_pipeline(&*outline_pipeline);

        let draw_indirect_command_buffer = gpu::Buffer::from_slice(
            &[gpu::DrawIndirectCommand {
                vertex_count: 0,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            }],
            "draw_indirect",
        );

        cmd.dispatch(
            self.clusters.len() as u32,
            1,
            1,
            RootParams::Immediate(&OutlineExperimentRootParams {
                scene_info: scene_info.gpu,
                clusters: self.clusters.ptr(),
                cluster_count: self.clusters.len() as u32,
                vertex_count: 0,
                out_vertices: self.outline_vertices.ptr(),
                out_indices: self.outline_indices.ptr(),
                out_draw_command: draw_indirect_command_buffer.ptr(),
            }),
        );


        cmd.barrier_source(BarrierFlags::COMPUTE_SHADER | BarrierFlags::STORAGE | BarrierFlags::DEPTH_STENCIL);
        cmd.barrier(BarrierFlags::FRAGMENT_SHADER | BarrierFlags::SAMPLED_READ | BarrierFlags::STORAGE | BarrierFlags::INDIRECT_READ);

        /////////////////////////////////////////////////////////
        // render contours
        let mut encoder = cmd.begin_rendering(RenderPassInfo {
            color_attachments: &[gpu::ColorAttachment {
                image: color_target,
                ..
            }],
            depth_stencil_attachment: None,
        });

        encoder.bind_graphics_pipeline(&*debug_stroke_pipeline);
        encoder.draw_indirect(
            TriangleList,
            None,
            &draw_indirect_command_buffer,
            0..1,
            RootParams::Immediate(&crate::experiments::coat::DebugStrokesRootParams {
                scene_info: scene_info.gpu,
                vertices: self.outline_vertices.ptr(),
                indices: self.outline_indices.ptr(),
                depth_texture: depth_target.texture_descriptor_index(),
            }),
        );

        //draw_lines(&mut encoder, &line_vertices, &lines, scene_info);

        /*fn random_color(index: usize) -> Srgba8 {
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
            let v0 = &self.mesh.vertices[edge.v0 as usize];
            let v1 = &self.mesh.vertices[edge.v1 as usize];
            let color = random_color(edge.cluster);
            vertices.push(crate::experiments::lines::LineVertex {
                position: v0.position,
                color,
            });
            vertices.push(crate::experiments::lines::LineVertex {
                position: v1.position,
                color,
            });
            lines.push(crate::experiments::lines::Line {
                start_vertex: (vertices.len() - 2) as u32,
                vertex_count: 2,
                ..
            });
        }

        draw_lines(&mut encoder, &vertices, &lines, scene_info);*/
    }
}
