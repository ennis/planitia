#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]

use gamelib::asset::{AssetCache, Handle};
use gamelib::camera_control::{CameraControl, CameraControlInput};
use gamelib::context::{App, AppHandler, LoopHandler};
use gamelib::egui;
use gamelib::egui::{Color32, Scene};
use gamelib::input::{InputEvent, PointerButton};
use gamelib::paint::{DrawGlyphRunOptions, PaintRenderParams, PaintScene, Painter, TextFormat, TextLayout};
use gamelib::pipeline_cache::get_graphics_pipeline;
use gamelib::platform::{EventToken, InitOptions, RenderTargetImage, UserEvent};
use std::ops::Deref;

use color::{Srgba8, srgba8};
use egui_demo_lib::{View, WidgetGallery};
use gpu::PrimitiveTopology::TriangleList;
use gpu::{Image, Ptr, RenderPassInfo, RootParams, root_params};
use log::debug;
use math::geom::Camera;
use math::{Mat4, Vec2, Vec3};

mod experiments;

/// Global application singleton.
static APP: App<Game> = App::new();

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SceneInfoUniforms {
    view_matrix: Mat4,
    proj_matrix: Mat4,
    view_proj_matrix: Mat4,
    screen_size: Vec2,
    time: f32,
    frame: u32,
    eye: Vec3,
}

/// Scene info and GPU buffer containing it.
pub struct SceneInfo {
    info: SceneInfoUniforms,
    pub gpu: Ptr<SceneInfoUniforms>,
}

impl Deref for SceneInfo {
    type Target = SceneInfoUniforms;

    fn deref(&self) -> &Self::Target {
        &self.info
    }
}

struct Game {
    width: u32,
    height: u32,
    demo: WidgetGallery,
    color: Color32,
    painter: Painter,
    camera_control: CameraControl,
    depth_stencil_buffer: gpu::Image,
    grid_shader: Handle<gpu::GraphicsPipeline>,
    background_shader: Handle<gpu::GraphicsPipeline>,
    frame_count: u32,
    start_time: std::time::Instant,
    coat_experiment: experiments::coat::CoatExperiment,
    outline_experiment: experiments::outlines::OutlineExperiment,
}

fn create_depth_buffer(width: u32, height: u32) -> Image {
    Image::new(gpu::ImageCreateInfo {
        width,
        height,
        format: gpu::Format::D32_SFLOAT_S8_UINT,
        usage: gpu::ImageUsage::DEPTH_STENCIL_ATTACHMENT | gpu::ImageUsage::TRANSFER_DST,
        ..
    })
}

impl Default for Game {
    fn default() -> Self {
        Self {
            demo: WidgetGallery::default(),
            color: Default::default(),
            painter: Painter::new(gpu::Format::R8G8B8A8_UNORM, None),
            depth_stencil_buffer: create_depth_buffer(WIDTH, HEIGHT),
            camera_control: CameraControl::default(),
            grid_shader: get_graphics_pipeline("/shaders/pipelines.parc#grid"),
            background_shader: get_graphics_pipeline("/shaders/pipelines.parc#background"),
            frame_count: 0,
            width: WIDTH,
            height: HEIGHT,
            start_time: std::time::Instant::now(),
            coat_experiment: experiments::coat::CoatExperiment::new(),
            outline_experiment: experiments::outlines::OutlineExperiment::new(),
        }
    }
}

impl Game {
    fn render_scene(&mut self, encoder: &mut gpu::RenderEncoder, _camera: &Camera, scene_info: &SceneInfo) {
        //----------------------------------
        // Draw background
        {
            if let Ok(background_shader) = self.background_shader.read() {
                encoder.bind_graphics_pipeline(&*background_shader);
                encoder.draw(
                    TriangleList,
                    None,
                    0..6,
                    0..1,
                    root_params! {
                        scene_uniforms: Ptr<SceneInfoUniforms> = scene_info.gpu,
                        bottom_color: Srgba8 = srgba8(20, 20, 40, 255),
                        top_color: Srgba8 = srgba8(100, 150, 255, 255)
                    },
                );
            }
        }

        //----------------------------------
        // Draw grid
        {
            if let Ok(grid_shader) = self.grid_shader.read() {
                encoder.bind_graphics_pipeline(&*grid_shader);
                encoder.draw(
                    TriangleList,
                    None,
                    0..6,
                    0..1,
                    root_params! {
                        scene_uniforms: Ptr<SceneInfoUniforms> = scene_info.gpu,
                        grid_scale: f32 = 100.0
                    },
                );
            }
        }

        //----------------------------------
        // Draw editor
    }

    fn render_overlay(&mut self, cmd: &mut gpu::CommandStream, target: &gpu::Image) {
        experiments::painting_test(
            &mut self.painter,
            cmd,
            target,
            Srgba8::from(self.color.to_srgba_unmultiplied()),
        );
    }
}

impl AppHandler for Game {
    fn input(&mut self, input_event: InputEvent) {
        // --- SHORTCUTS ---

        // App exit
        if input_event.is_shortcut("Ctrl+Q") {
            debug!("Quit requested via Ctrl+Q");
            APP.quit();
        }

        // Home camera
        if input_event.is_shortcut("Home") {
            debug!("Home camera");
            //self.camera_control.home();
        }

        // --- CAMERA ---
        if self.camera_control.handle_input(&input_event) {
            return;
        }

        // --- experiments ---
        //self.coat_experiment.input(&input_event);
        self.outline_experiment.input(&input_event);
    }

    fn event(&mut self, event: UserEvent) {}

    fn resized(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.camera_control.resize(width, height);
        self.depth_stencil_buffer = create_depth_buffer(width, height);
    }

    fn vsync(&mut self) {}

    fn render(&mut self, target: RenderTargetImage) {
        // TODO: that's not the right time to reload assets.
        //       Ideally this should be done asynchronously, on another thread, so as not
        //       to block the GUI and rendering.
        AssetCache::instance().do_reload();

        let mut cmd = gpu::CommandStream::new();
        let time = self.start_time.elapsed().as_secs_f32();
        let frame = self.frame_count;
        self.frame_count += 1;

        //-------------------------------
        // Render 3D scene
        {
            let camera = self.camera_control.camera();
            let mut scene_info = SceneInfo {
                info: SceneInfoUniforms {
                    view_matrix: camera.view,
                    proj_matrix: camera.projection,
                    view_proj_matrix: camera.view_projection(),
                    screen_size: camera.screen_size.as_vec2(),
                    time,
                    frame,
                    eye: camera.eye().as_vec3(),
                },
                gpu: Ptr::NULL,
            };
            scene_info.gpu = cmd.upload_temporary(&scene_info.info);

            let mut encoder = cmd.begin_rendering(RenderPassInfo {
                color_attachments: &[gpu::ColorAttachment {
                    image: target.image,
                    clear_value: Some([0.0, 0.0, 0.0, 1.0]),
                }],
                depth_stencil_attachment: Some(gpu::DepthStencilAttachment {
                    image: &self.depth_stencil_buffer,
                    depth_clear_value: Some(1.0),
                    stencil_clear_value: Some(0),
                }),
            });

            self.render_scene(&mut encoder, &camera, &scene_info);
            encoder.finish();

            //self.coat_experiment
            //    .render(&mut cmd, &target.image, &self.depth_stencil_buffer, &scene_info);
            self.outline_experiment
                .render(&mut cmd, &target.image, &self.depth_stencil_buffer, &scene_info);
        }

        //-------------------------------
        // Render 2D overlays
        self.render_overlay(&mut cmd, &target.image);

        //-------------------------------
        // Render GUI
        APP.render_imgui(&mut cmd, &target.image);

        cmd.flush(&[target.ready], &[target.rendering_finished], None).unwrap();
    }

    fn close_requested(&mut self) {
        APP.quit();
    }

    fn imgui(&mut self, ctx: &egui::Context) {
        egui::Window::new("imgui").show(ctx, |ui| {
            ui.color_edit_button_srgba(&mut self.color);
            self.demo.ui(ui);
        });
    }
}

fn main() {
    // register asset directories
    AssetCache::register_directory(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
    gamelib::register_asset_directory();

    APP.run(&InitOptions {
        width: WIDTH,
        height: HEIGHT,
        window_title: "Planitia",
        ..Default::default()
    });
}
