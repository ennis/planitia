use crate::experiments::lines::draw_lines;
use crate::experiments::winged_edge_mesh::{WEMeshDataGPU, WingedEdgeMesh};
use crate::{SceneInfo, SceneInfoUniforms};
use bytesize::ByteSize;
use color::{srgba8, Srgba8};
use gamelib::asset::{AssetLoadError, AssetReadGuard, Handle, VfsPath, VfsPathBuf};
use gamelib::input::InputEvent;
use gamelib::render::pipeline_cache::{get_compute_pipeline, get_graphics_pipeline};
use gamelib::render::RenderTarget;
use gamelib::{static_assets, tweak};
use gpu::PrimitiveTopology::TriangleList;
use gpu::{Buffer, BufferCreateInfo, DrawIndirectCommand, Image, InvalidateFlags, MemoryLocation, Ptr, RootParams, Size3D};
use hgeo::util::polygons_to_triangle_mesh;
use log::{info, warn};
use math::geom::Camera;
use math::{IVec2, Mat4, Vec3};
use smallvec::{smallvec, SmallVec};
use std::alloc::Layout;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::Path;
use std::{fmt, ptr};

#[repr(C)]
#[derive(Copy, Clone)]
struct PointData {
    position: Vec3,
    id: u32
}

#[repr(C)]
#[derive(Copy, Clone)]
struct FaceVertexData {
    point: u32,
    normal: Vec3,
}

pub struct OutlineExperiment {
    // --- Preprocessed mesh data ---
    mesh: WingedEdgeMesh<PointData, FaceVertexData>,

    // --- GPU pipeline resources ---

    // contour extraction
    expanded_contour_vertices: Buffer<ContourVertex>,
    expanded_contour_indices: Buffer<u32>,
    segments: Buffer<ContourSegmentBuffer>,

    // edge linking
    contour_point_list: Buffer<ContourPoint>,
    contour_point_list_subdiv: Buffer<ContourPoint>,
    global_to_contour_index_map: Buffer<u32>,
    contour_rank_successors: Buffer<u64>, // low 32 bits: rank, high 32 bits: successor index, root index after list ranking pass
    contour_rank_successors_1: Buffer<u64>, // low 32 bits: rank, high 32 bits: successor index, root index after list ranking pass

    // contour rendering
    expanded_contours_draw_command: Buffer<DrawIndirectCommand>,
    angle_texture: RenderTarget,
    normal_texture: RenderTarget,
    shading_texture: RenderTarget,

    lock_view: bool,
    locked_eye: Vec3,
}

/// GPU root parameters for contour extraction, ranking, and expansion passes.
#[repr(C)]
#[derive(Copy, Clone)]
struct ContoursRootParams {
    scene_info: Ptr<SceneInfoUniforms>,
    mesh: WEMeshDataGPU<PointData, FaceVertexData>,
    eye: Vec3,

    contour_edges: Ptr<ContourSegmentBuffer>,

    contour_rank_successors_0: Ptr<u64>,
    contour_rank_successors_1: Ptr<u64>,

    expanded_contour_vertex_count: u32,
    expanded_contour_vertices: Ptr<ContourVertex>,
    expanded_contour_indices: Ptr<u32>,
    expanded_contours_draw_command: Ptr<DrawIndirectCommand>,

    main_light_direction: Vec3,
    silhouette_color: Srgba8,
    line_width: f32,
    filter_width: f32,
    isophote_offset: f32,

    depth_texture: gpu::TextureHandle,
    angle_texture: gpu::TextureHandle,
    normal_texture: gpu::TextureHandle,
    shading_texture: gpu::TextureHandle,
}


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

/// Segment emitted by contour extraction.
#[repr(C)]
#[derive(Clone, Copy)]
struct ContourSegment {
    position_0: Vec3,
    point_0: u32,
    position_1: Vec3,
    point_1: u32,
    group_id: u16,
}

#[repr(C)]
struct ContourSegmentBuffer {
    count: u32,
    edges: [ContourSegment], // unsized
}

impl ContourSegmentBuffer {
    fn layout(count: usize) -> Layout {
        let (layout, _array_offset) = Layout::new::<u32>()
            .extend(Layout::array::<ContourSegment>(count).unwrap())
            .unwrap();
        layout.pad_to_align()
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ContourPoint {
    position: Vec3,
    group_id: u32,
    next: u32,
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

#[repr(C)]
#[derive(Copy, Clone)]
struct SubdivideContoursRootParams2 {
    common: Ptr<ContoursRootParams>,
    point_count: u32,
}

static_assets! {
    static BASE_RENDER: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#base_render";
    static DEPTH_PASS: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#depth_pass";
    static EXTRACT_INTERPOLATED_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#extract_interpolated_contours";
    static EXPAND_INTERPOLATED_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#expand_interpolated_contours";

    static BREAK_CONTOURS_INIT: gpu::ComputePipeline = "/shaders/game_shaders.sharc#break_contours_init";
    static BREAK_CONTOURS_STEP: gpu::ComputePipeline = "/shaders/game_shaders.sharc#break_contours_step";
    static RANK_CONTOURS_INIT: gpu::ComputePipeline = "/shaders/game_shaders.sharc#rank_contours_init";
    static RANK_CONTOURS_STEP: gpu::ComputePipeline = "/shaders/game_shaders.sharc#rank_contours_step";

    static SETUP_SUBDIVIDE_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#setup_subdivide_contours";
    static FINISH_SUBDIVIDE_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#finish_subdivide_contours";
    static SUBDIVIDE_CONTOURS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#subdivide_contours";

    static RENDER_OUTLINES: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#render_outlines";
}

const RANK_CONTOURS_GROUP_SIZE: u32 = 256;
const EXPAND_CONTOURS_GROUP_SIZE: u32 = 256;
const SUBDIVIDE_CONTOURS_GROUP_SIZE: u32 = 256;
const EXTRACT_CONTOURS_THREAD_GROUP_SIZE: u32 = 128;

impl OutlineExperiment {
    pub fn new() -> Self {
        Self {
            mesh: WingedEdgeMesh::default(),
            expanded_contour_vertices: gpu::Buffer::from_slice(&[]),
            expanded_contour_indices: gpu::Buffer::from_slice(&[]),
            segments: unsafe { gpu::Buffer::from_layout(ContourSegmentBuffer::layout(1)) },
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
            angle_texture: RenderTarget::new(gpu::Format::R16G16B16A16_UINT, gpu::ImageUsage::SAMPLED | gpu::ImageUsage::STORAGE | gpu::ImageUsage::COLOR_ATTACHMENT),
            normal_texture: RenderTarget::new(gpu::Format::R16G16B16A16_SFLOAT, gpu::ImageUsage::SAMPLED | gpu::ImageUsage::STORAGE | gpu::ImageUsage::COLOR_ATTACHMENT),
            shading_texture: RenderTarget::new(gpu::Format::R8G8B8A8_UNORM, gpu::ImageUsage::SAMPLED | gpu::ImageUsage::STORAGE | gpu::ImageUsage::COLOR_ATTACHMENT),
            lock_view: false,
            locked_eye: Vec3::ZERO,
        }
    }

    /// Loads geometry from a houdini geometry file.
    fn load_geometry(&mut self, path: &Path) {
        let geo = hgeo::Geo::load(path).unwrap();

        // convert points
        let points = (0..geo.point_count)
            .map(|ptnum| {
                let position = geo.point(ptnum as u32, "P");
                let id = geo.point(ptnum as u32, "id");
                PointData { position, id }
            })
            .collect::<Vec<_>>();

        // convert polygons to triangle mesh and collect vertices
        let vertices = {
            let mut vertices = Vec::new();
            for prim in geo.polygons() {
                if !prim.closed {
                    continue;
                }
                let first_vertex = vertices.len();
                for (i, vi) in prim.vertices().enumerate() {
                    vertices.push(FaceVertexData {
                        point: geo.vertexpoint(vi),
                        normal: geo.vertex(vi, "N"),
                    });
                    if i > 2 {
                        let v0 = vertices[first_vertex];
                        let v1 = vertices[first_vertex + i - 1];
                        //let v2 = vertices[first_vertex + i];
                        vertices.push(v0);
                        vertices.push(v1);
                    }
                }
            }
            vertices
        };

        let point_count = points.len();
        let vertex_count = vertices.len();
        self.mesh = WingedEdgeMesh::new(&points, &vertices, |fv: &FaceVertexData| fv.point);



        self.segments = unsafe { Buffer::from_layout(ContourSegmentBuffer::layout(self.mesh.edges.len())) };
        self.global_to_contour_index_map.resize_no_copy(point_count);
        self.contour_rank_successors.resize_no_copy(point_count);
        self.contour_rank_successors_1.resize_no_copy(point_count);


        // num points + some for subdivision
        let subdiv_points = point_count * 8;
        self.contour_point_list.resize_no_copy(subdiv_points);
        self.contour_point_list_subdiv.resize_no_copy(subdiv_points);
        self.expanded_contour_vertices.resize_no_copy(subdiv_points * 2);
        self.expanded_contour_indices.resize_no_copy(subdiv_points * 6);

        unsafe {
            gpu::set_debug_name(&self.segments, "segments");
            gpu::set_debug_name(&self.mesh.edges_gpu, "edges");
            gpu::set_debug_name(&self.mesh.points_gpu, "points");
            gpu::set_debug_name(&self.mesh.faces_gpu, "faces");
            gpu::set_debug_name(&self.expanded_contours_draw_command, "expanded_contours_draw_command");
            gpu::set_debug_name(&self.expanded_contour_indices, "expanded_contour_indices");
            gpu::set_debug_name(&self.expanded_contour_vertices, "expanded_contour_vertices");
        }

        let total_gpu_size = self.mesh.points_gpu.byte_size()
            + self.mesh.edges_gpu.byte_size()
            + self.mesh.faces_gpu.byte_size()
            + self.mesh.face_vertices_gpu.byte_size()
            + self.contour_point_list.byte_size()
            + self.contour_rank_successors.byte_size()
            + self.contour_rank_successors_1.byte_size()
            + self.segments.byte_size();

        info!(
            "loaded mesh from {}:
                   Count        Byte size
    points         {:<8}     {:<8} ({} B per elem)
    faces          {:<8}     {:<8} ({} B per elem)
    face vertices  {:<8}     {:<8} ({} B per elem)
    edges          {:<8}     {:<8} ({} B per elem)
    Total:                      {}
",
            path.display(),

            self.mesh.points.len(),
            ByteSize::b(self.mesh.points_gpu.byte_size()).display().si(),
            self.mesh.points_gpu.byte_size() / self.mesh.points_gpu.len() as u64,

            self.mesh.faces.len(),
            ByteSize::b(self.mesh.faces_gpu.byte_size()).display().si(),
            self.mesh.faces_gpu.byte_size() / self.mesh.faces_gpu.len() as u64,

            self.mesh.face_vertices.len(),
            ByteSize::b(self.mesh.face_vertices_gpu.byte_size()).display().si(),
            self.mesh.face_vertices_gpu.byte_size() / self.mesh.face_vertices_gpu.len() as u64,

            self.mesh.edges.len(),
            ByteSize::b(self.mesh.edges_gpu.byte_size()).display().si(),
            self.mesh.edges_gpu.byte_size() / self.mesh.edges_gpu.len() as u64,

            total_gpu_size
        );
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

    pub(crate) fn resize(&mut self, _width: u32, _height: u32) {
        // nothing
    }

    pub(crate) fn render(
        &mut self,
        cmd: &mut gpu::CommandBuffer,
        color_target: &gpu::Image,
        depth_target: &gpu::Image,
        scene_info: &SceneInfo,
    ) -> Result<(), AssetLoadError> {

        if self.mesh.points.is_empty() {
            // nothing to render
            return Ok(());
        }

        let width = color_target.width();
        let height = color_target.height();

        self.angle_texture.setup(width, height);
        self.normal_texture.setup(width, height);
        self.shading_texture.setup(width, height);

        use InvalidateFlags as IF;

        // handle view lock
        self.lock_view = tweak!(lock_view = false);
        if !self.lock_view {
            self.locked_eye = scene_info.eye;
        }

        let root_params = cmd.upload(&ContoursRootParams {
            mesh: self.mesh.gpu_data(),
            scene_info: scene_info.gpu,

            eye: self.locked_eye,
            contour_edges: self.segments.ptr(),

            contour_rank_successors_0: self.contour_rank_successors.ptr(),
            contour_rank_successors_1: self.contour_rank_successors_1.ptr(),

            expanded_contour_vertex_count: 0,
            expanded_contour_vertices: self.expanded_contour_vertices.ptr(),
            expanded_contour_indices: self.expanded_contour_indices.ptr(),
            expanded_contours_draw_command: self.expanded_contours_draw_command.ptr(),

            main_light_direction: tweak!(main_light_direction = Vec3::new(0.5, -1.0, 0.5).normalize()),
            silhouette_color: tweak!(silhouette_color = Srgba8::new(255, 255, 255, 255)),
            line_width: tweak!(line_width = 1.0),
            filter_width: tweak!(filter_width = 0.7),
            isophote_offset: tweak!(isophote_offset = 0.0),

            depth_texture: depth_target.texture_handle(),
            angle_texture: self.angle_texture.texture_handle(),
            normal_texture: self.normal_texture.texture_handle(),
            shading_texture: self.shading_texture.texture_handle(),
        });

        /////////////////////////////////////////////////////////
        // base render & depth pass
        {
            let mut pass = cmd.begin_rendering(
                &[
                    self.shading_texture.as_color_attachment([0.0, 0.0, 0.0, 0.0]),
                    self.normal_texture.as_color_attachment([0.0, 0.0, 0.0, 0.0]),
                    self.angle_texture.as_color_attachment([0.0, 0.0, 0.0, 0.0]),
                ],
                Some(depth_target.as_depth_stencil_attachment(None, None)),
            );
            pass.bind_graphics_pipeline(&*BASE_RENDER.read()?);
            pass.draw(TriangleList, None, 0..self.mesh.face_vertices.len() as u32, 0..1, root_params);
            pass.finish();
        }

        unsafe {
            // clear DrawIndirectCommand::vertex
            cmd.update_buffer(&self.expanded_contours_draw_command.as_bytes(), 0, &[0; 4]);
        }

        /////////////////////////////////////////////////////////
        // contour extraction

        unsafe {
            // clear ContourEdgeBuffer::count
            // not very pretty
            cmd.update_buffer(&self.segments.as_bytes(), 0, &[0; 4]);
            cmd.fill_buffer(&self.global_to_contour_index_map.as_bytes().slice(..), 0xFFFF_FFFF);
            cmd.fill_buffer(&self.contour_point_list.as_bytes().slice(..), 0xFFFF_FFFF);
            cmd.fill_buffer(&self.contour_point_list_subdiv.as_bytes().slice(..), 0xFFFF_FFFF);
        }

        cmd.barrier(IF::STORAGE);

        cmd.bind_compute_pipeline(&*EXTRACT_INTERPOLATED_CONTOURS.read()?);
        cmd.dispatch(self.mesh.faces.len().div_ceil(EXTRACT_CONTOURS_THREAD_GROUP_SIZE as usize) as u32, 1, 1, root_params);

        cmd.barrier(IF::STORAGE | IF::INDIRECT | IF::TEXTURE);

        /////////////////////////////////////////////////////////
        // contour loop breaking
        // find contour loops and determine a "break" point for each loop that defines the starting
        // point for ranking the list
        /*{
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

            cmd.barrier(IF::STORAGE);

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
                cmd.barrier(IF::STORAGE);
            }

            // contour ranking
            cmd.bind_compute_pipeline(&*RANK_CONTOURS_INIT.read()?);
            cmd.dispatch(groups_count, 1, 1, &params);

            cmd.barrier(IF::STORAGE);

            cmd.bind_compute_pipeline(&*RANK_CONTOURS_STEP.read()?);
            for _ in 0..rank_steps {
                cmd.dispatch(groups_count, 1, 1, &params);
                std::mem::swap(
                    &mut params.contour_rank_successors_0,
                    &mut params.contour_rank_successors_1,
                );
                cmd.barrier(IF::STORAGE);
            }
        }*/

        /////////////////////////////////////////////////////////
        // contour subdivision
        /*if !use_interpolated_contours {
            let point_count = self.contour_point_list.len() as u32;
            let groups_count = point_count.div_ceil(SUBDIVIDE_CONTOURS_GROUP_SIZE);

            let subdivision_levels = tweak!(contour_subdivision_levels = 0u32);
            for _ in 0..subdivision_levels {
                let params = cmd.upload(&SubdivideContoursRootParams {
                    common: root_params,
                    point_count,
                });

                cmd.bind_compute_pipeline(&*SETUP_SUBDIVIDE_CONTOURS.read()?);
                cmd.dispatch(1, 1, 1, params);
                cmd.barrier(IF::STORAGE);
                cmd.bind_compute_pipeline(&*SUBDIVIDE_CONTOURS.read()?);
                cmd.dispatch(groups_count, 1, 1, params);
                cmd.barrier(IF::STORAGE);
                cmd.bind_compute_pipeline(&*FINISH_SUBDIVIDE_CONTOURS.read()?);
                cmd.dispatch(groups_count, 1, 1, params);
                cmd.barrier(IF::STORAGE);
            }
        }*/

        /////////////////////////////////////////////////////////
        // expand contours to quad geometry
        {
            let n = self.mesh.edges.len() as u32;
            let groups_count = n.div_ceil(EXPAND_CONTOURS_GROUP_SIZE);
            cmd.bind_compute_pipeline(&*EXPAND_INTERPOLATED_CONTOURS.read()?);
            cmd.dispatch(groups_count, 1, 1, root_params);
        }

        cmd.barrier(IF::STORAGE | IF::INDIRECT);

        /////////////////////////////////////////////////////////
        // render contours
        {
            let mut encoder = cmd.begin_rendering(
                &[color_target.as_color_attachment(None)],
                None,
            );
            encoder.bind_graphics_pipeline(&*RENDER_OUTLINES.read()?);
            encoder.draw_indirect(
                TriangleList,
                None,
                &self.expanded_contours_draw_command,
                0..1,
                root_params,
            );
            encoder.finish();
        }

        cmd.barrier(IF::STORAGE | IF::TEXTURE);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::SliceRandom;
    use rand::rng;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::IntoParallelIterator;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::atomic::AtomicU64;

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
