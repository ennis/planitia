use crate::{SceneInfo, SceneInfoUniforms};
use gamelib::{egui, static_assets, tweak};
use gpu::{Buffer, BufferCreateInfo, Image, ImageCreateInfo, InvalidateFlags, PrimitiveTopology, Size3D};
use math::{IVec2, Vec3};
use std::path::Path;
use gamelib::asset::AssetLoadError;
use gamelib::egui::{DragValue, Slider};
use gamelib::input::InputEvent;

static_assets! {
    static BASE_RENDER: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#automaton_base_render";
    static CONTOUR_DETECTION: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#contour_detection";
    static SIM_INIT_EMITTERS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#sim_init_emitters";
    static SIM_INIT_TRAILS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#sim_init_trails";
    static SIM_STEP_EMITTERS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#sim_step_emitters";
    static SIM_STEP_TRAILS: gpu::ComputePipeline = "/shaders/game_shaders.sharc#sim_step_trails";
    static SIM_STEP_NEXT: gpu::ComputePipeline = "/shaders/game_shaders.sharc#sim_step_next";
    static SIM_RENDER: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#sim_render";
    static AUTOMATON_DEBUG: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#automaton_debug";
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Point {
    position: Vec3,
    id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    point: u32,
    normal: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MeshData {
    points: gpu::Ptr<Point>,
    vertices: gpu::Ptr<Vertex>,
    point_count: u32,
    vertex_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Emitter {
    position: IVec2,
}

#[repr(u32)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum DebugMode {
    None = 0,
    Shading = 1,
    Normals = 2,
    Depth = 3,
    Aux = 4,
    ContoursMaxCurv = 5,
    ContoursAngle = 6,
    SimTrails = 7,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RootParams {
    scene_info: gpu::Ptr<SceneInfoUniforms>,
    mesh: MeshData,
    shading_texture: gpu::TextureHandle,
    normal_texture: gpu::TextureHandle,
    depth_texture: gpu::TextureHandle,
    aux_texture: gpu::TextureHandle,
    contour_target: gpu::TextureHandle,
    trails_0: gpu::StorageImageHandle,
    trails_1: gpu::StorageImageHandle,
    emitters: gpu::Ptr<Emitter>,
    emitter_count: u32,
    light_direction: Vec3,
    sim_step: u32,
    debug_mode: DebugMode,
}

//#[repr(C)]
//#[derive(Clone, Copy)]
//struct SimParams {
//    root_params: gpu::Ptr<RootParams>,
//    step: u32,
//    input: gpu::TextureHandle,
//    output: gpu::TextureHandle,
//}

/// Loads geometry from the specified file path.
pub struct AutomatonExperiment {
    // --- Preprocessed mesh data ---
    points: gpu::Buffer<Point>,
    vertices: gpu::Buffer<Vertex>,
    point_count: u32,
    vertex_count: u32,

    shading_texture: Image,
    normal_texture: Image,
    depth_texture: Image,
    aux_texture: Image,

    contour_target: Image,
    emitters: Buffer<Emitter>,
    trails_0: Image,
    trails_1: Image,
    sim_step: u32,
    max_sim_steps: u32,
    debug_mode: DebugMode
}

const EMITTERS_COUNT: usize = 1000;

impl AutomatonExperiment {
    pub fn new() -> Self {
        Self {
            points: Buffer::from_slice(&[]),
            vertices: Buffer::from_slice(&[]),
            point_count: 0,
            vertex_count: 0,
            shading_texture: Image::new_color_attachment(1, 1, gpu::Format::R8G8B8A8_UNORM),
            normal_texture: Image::new_color_attachment(1, 1, gpu::Format::A2B10G10R10_UNORM_PACK32),
            depth_texture: Image::new(ImageCreateInfo {
                width: 1,
                height: 1,
                format: gpu::Format::D32_SFLOAT_S8_UINT,
                usage: gpu::ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..
            }),
            aux_texture: Image::new_color_attachment(1, 1, gpu::Format::R16G16B16A16_UINT),
            contour_target: Image::new_color_attachment(1, 1, gpu::Format::R32G32B32A32_SFLOAT),
            trails_0: Image::new_texture(1, 1, gpu::Format::R32G32B32A32_SFLOAT),
            trails_1: Image::new_texture(1, 1, gpu::Format::R32G32B32A32_SFLOAT),
            emitters: Buffer::new(BufferCreateInfo {len: EMITTERS_COUNT, .. }),
            sim_step: 0,
            max_sim_steps: 30,
            debug_mode: DebugMode::Shading,
        }
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

    pub(crate) fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new("Automaton Experiment").show(ctx, |ui| {
            ui.label("Debug mode:");
            ui.vertical(|ui| {
                ui.selectable_value(&mut self.debug_mode, DebugMode::None, "None");
                ui.selectable_value(&mut self.debug_mode, DebugMode::Shading, "Shading");
                ui.selectable_value(&mut self.debug_mode, DebugMode::Normals, "Normals");
                ui.selectable_value(&mut self.debug_mode, DebugMode::Depth, "Depth");
                ui.selectable_value(&mut self.debug_mode, DebugMode::Aux, "Aux");
                ui.selectable_value(&mut self.debug_mode, DebugMode::ContoursMaxCurv, "Contours (max curvature)");
                ui.selectable_value(&mut self.debug_mode, DebugMode::ContoursAngle, "Contours (angle)");
                ui.selectable_value(&mut self.debug_mode, DebugMode::SimTrails, "Simulation trails");
            });
            if ui.button("Reset simulation").clicked() {
                self.sim_step = 0;
            }
            //ui.add(Slider::new(&mut self.sim_step, 0..=self.max_sim_steps).text("Sim step").step_by(1.0));
            ui.add(DragValue::new(&mut self.max_sim_steps).range(0..=1000).prefix("Max sim steps: ").speed(1.0));
        });
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.shading_texture.discard_resize(Size3D::new(width, height, 1));
        self.normal_texture.discard_resize(Size3D::new(width, height, 1));
        self.depth_texture.discard_resize(Size3D::new(width, height, 1));
        self.contour_target.discard_resize(Size3D::new(width, height, 1));
        self.aux_texture.discard_resize(Size3D::new(width, height, 1));
        self.trails_0.discard_resize(Size3D::new(width, height, 1));
        self.trails_1.discard_resize(Size3D::new(width, height, 1));
    }

    pub(crate) fn load_geometry(&mut self, path: &Path) {
        let geo = hgeo::Geo::load(path).unwrap();

        let points = (0..geo.point_count)
            .map(|ptnum| {
                let position = geo.point(ptnum as u32, "P");
                let id = geo.point(ptnum as u32, "id");
                Point { position, id }
            })
            .collect::<Vec<_>>();

        //geo.point_attribute_typed::<Vec3>("P").unwrap().as_slice().to_vec();

        let vertices = {
            let mut vertices = Vec::new();
            //let mut indices = Vec::new();
            hgeo::util::polygons_to_triangle_mesh_2(
                &geo,
                |g, vtxnum, primnum| {
                    let normal: Vec3 = g.vertex(vtxnum, "N");
                    let point = g.vertexpoint(vtxnum);
                    Vertex { point, normal }
                },
                |v0, v1, v2| {
                    vertices.push(*v0);
                    vertices.push(*v1);
                    vertices.push(*v2);
                },
            );
            vertices
        };

        self.points = Buffer::from_slice(&points);
        self.vertices = Buffer::from_slice(&vertices);
        self.point_count = points.len() as u32;
        self.vertex_count = vertices.len() as u32;
    }


    pub(crate) fn render(&mut self, cmd: &mut gpu::CommandBuffer, color_target: &Image, depth_target: &Image, scene_info: &SceneInfo) -> Result<(), AssetLoadError>  {

        let params = cmd.upload(&RootParams {
            scene_info: scene_info.gpu,
            mesh: MeshData {
                points: self.points.ptr(),
                vertices: self.vertices.ptr(),
                point_count: self.point_count,
                vertex_count: self.vertex_count,
            },
            shading_texture: self.shading_texture.texture_handle(),
            normal_texture: self.normal_texture.texture_handle(),
            depth_texture: self.depth_texture.texture_handle(),
            aux_texture: self.aux_texture.texture_handle(),
            contour_target: self.contour_target.texture_handle(),
            trails_0: self.trails_0.storage_handle(),
            trails_1: self.trails_1.storage_handle(),
            emitters: self.emitters.ptr(),
            emitter_count: self.emitters.len() as u32,
            light_direction: tweak!(light_direction: Vec3 = Vec3::new(0.0, 0.0, -1.0)),
            sim_step: self.sim_step,
            debug_mode: self.debug_mode,
        });

        // base render
        {
            let mut encoder = cmd.begin_rendering(
                &[
                    gpu::ColorAttachment {
                        image: &self.shading_texture,
                        clear: Some([0.0, 0.0, 0.0, 0.0]),
                    },
                    gpu::ColorAttachment {
                        image: &self.normal_texture,
                        clear: Some([0.0, 0.0, 0.0, 0.0]),
                    },
                    gpu::ColorAttachment {
                        image: &self.aux_texture,
                        clear: Some([0.0, 0.0, 0.0, 0.0]),
                    }
                ],
                Some(gpu::DepthStencilAttachment {
                    image: &self.depth_texture,
                    depth_clear: Some(1.0),
                    stencil_clear: None,
                }),
            );


            encoder.bind_graphics_pipeline(&*BASE_RENDER.read()?);
            encoder.draw(PrimitiveTopology::TriangleList, None, 0..self.vertex_count, 0..1, params);
            encoder.finish();
        }

        // contour detection
        {
            let mut encoder = cmd.begin_rendering(
                &[gpu::ColorAttachment {
                    image: &self.contour_target,
                    clear: Some([0.0, 0.0, 0.0, 0.0]),
                }],
                None,
            );

            encoder.bind_graphics_pipeline(&*CONTOUR_DETECTION.read()?);
            encoder.draw_screen_quad(params);
            encoder.finish();
        }

        // sim
        {
            let emitter_workgroup_size = 64;
            let trails_workgroup_tile_size = 16;
            let emitter_workgroup_count = self.emitters.len().div_ceil(emitter_workgroup_size as usize) as u32;
            let trails_workgroup_count_x = self.trails_0.width().div_ceil(trails_workgroup_tile_size);
            let trails_workgroup_count_y = self.trails_0.height().div_ceil(trails_workgroup_tile_size);

            cmd.bind_compute_pipeline(&*SIM_INIT_EMITTERS.read()?);
            cmd.dispatch(emitter_workgroup_count, 1, 1, params);
            cmd.bind_compute_pipeline(&*SIM_INIT_TRAILS.read()?);
            cmd.dispatch(trails_workgroup_count_x, trails_workgroup_count_y, 1, params);


            for _ in 0..self.max_sim_steps {

                cmd.barrier(InvalidateFlags::STORAGE);

                cmd.bind_compute_pipeline(&*SIM_STEP_EMITTERS.read()?);
                cmd.dispatch(emitter_workgroup_count, 1, 1, params);

                cmd.barrier(InvalidateFlags::STORAGE);

                cmd.bind_compute_pipeline(&*SIM_STEP_TRAILS.read()?);
                cmd.dispatch(trails_workgroup_count_x, trails_workgroup_count_y, 1, params);

                cmd.barrier(InvalidateFlags::STORAGE);

                cmd.bind_compute_pipeline(&*SIM_STEP_NEXT.read()?);
                cmd.dispatch(1, 1, 1, params);
            }

            cmd.barrier(InvalidateFlags::STORAGE);

            let mut encoder = cmd.begin_rendering(
                &[gpu::ColorAttachment {
                    image: &color_target,
                    clear: None,
                }],
                Some(gpu::DepthStencilAttachment {
                    image: &depth_target,
                    depth_clear: None,
                    stencil_clear: None,
                }),
            );
            encoder.bind_graphics_pipeline(&*SIM_RENDER.read()?);
            encoder.draw_screen_quad(params);
            encoder.finish();
        }

        // debug
        {
            let mut encoder = cmd.begin_rendering(
                &[gpu::ColorAttachment {
                    image: &color_target,
                    clear: None,
                }],
                Some(gpu::DepthStencilAttachment {
                    image: &depth_target,
                    depth_clear: None,
                    stencil_clear: None,
                }),
            );
            encoder.bind_graphics_pipeline(&*AUTOMATON_DEBUG.read()?);
            encoder.draw_screen_quad(params);
            encoder.finish();
        }

        Ok(())
    }
}
