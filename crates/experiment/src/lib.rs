#![feature(default_field_values)]

mod terrain;

use std::path::PathBuf;
use crate::background::SceneInfo;
use chrono::{DateTime, Local};
use gamelib::camera_control::CameraControl;
use gamelib::color::{Srgba8, srgba8};
use gamelib::math::vec2;
use gamelib::paint::{ColorStop, GradientExtendMode, LinearGradientFill, PaintScene};
use gamelib::platform::RenderTargetImage;
use gamelib::{format_message, gpu_span, register_plugin, AppHandler, InputEvent, WindowHandle};
use gpu::{self, Ptr, root_params};
use std::time::Instant;
use gpu::PrimitiveTopology::TriangleList;

#[gpu::shader_module("shaders/background.slang#1511")]
mod background {}

#[gpu::shader_module("shaders/grid.slang#487")]
mod grid {}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExperimentApp {
    #[serde(skip)]
    last_time: Option<Instant>,
    load_time: DateTime<Local>,
    frames_rendered: usize,
    camera_control: CameraControl,
    geometry_file: PathBuf,
    #[serde(skip)]
    geometry: hgeo::Geo,
}

impl Default for ExperimentApp {
    fn default() -> Self {
        Self::new()
    }
}

impl ExperimentApp {
    fn new() -> ExperimentApp {
        let start = Instant::now();
        ExperimentApp {
            last_time: Some(start),
            load_time: Local::now(),
            camera_control: Default::default(),
            frames_rendered: 0,
            geometry: Default::default(),
            geometry_file: Default::default(),
        }
    }

    fn update(&mut self) {}
}

impl AppHandler for ExperimentApp {
    fn input(&mut self, window: WindowHandle, input_event: &InputEvent) {
        self.camera_control.handle_input(input_event);
        if input_event.is_shortcut("Ctrl+O") {
            if let Some(geometry_file) = gamelib::pick_file("Houdini Geometry File", &["bgeo", "geo"]) {
                eprintln!("Picked file: {}", geometry_file.display());
                self.geometry_file = geometry_file;
            }
        }
    }

    fn started(&mut self) {}

    fn loaded(&mut self) {
        self.load_time = Local::now();
        if !self.geometry_file.as_os_str().is_empty() {
            match hgeo::Geo::load(&self.geometry_file) {
                Ok(geometry) => {
                    self.geometry = geometry;
                    eprintln!("Loaded geometry file: {}", self.geometry_file.display());
                }
                Err(err) => {
                    eprintln!("Failed to load geometry file: {}: {}", self.geometry_file.display(), err);
                }
            }
        }
    }

    fn render(&mut self, window: WindowHandle, image: RenderTargetImage<'_>) {

        let _span = gamelib::span!("plugin render");

        let image = image.image;

        let gpu_device_name = gpu::get_physical_device_name();
        //eprintln!("GPU device name: {}", gpu_device_name);

        let camera = self.camera_control.camera();
        let scene_info = SceneInfo {
            view_matrix: camera.view,
            proj_matrix: camera.projection,
            view_proj_matrix: camera.projection * camera.view,
            screen_size: vec2(image.width() as f32, image.height() as f32),
            time: self.frames_rendered as f32,
            frame: self.frames_rendered as u32,
            eye: camera.eye().as_vec3(),
        };
        let scene_info_gpu = gpu::upload(&scene_info);

        self.frames_rendered += 1;

        let time = Instant::now();
        let delta = time.duration_since(self.last_time.unwrap_or(Instant::now()));
        self.last_time = Some(time);

        let fps = if delta.as_secs_f32() > 0.0 { 1.0 / delta.as_secs_f32() } else { 0.0 };

        format_message!("GPU   : {}\n", gpu_device_name);
        format_message!("DT    : {delta:?}\nFPS   : {fps:.1}\n");
        format_message!("FRAME : {}\n", self.frames_rendered);
        let time_since_reload = Local::now().signed_duration_since(self.load_time);
        format_message!(
            "Last reload  : {} ({}m {}s ago)",
            self.load_time.format("%Y-%m-%d %H:%M:%S"),
            time_since_reload.num_minutes(),
            time_since_reload.num_seconds() % 60
        );

        let mut scene = PaintScene::new(Srgba8::TRANSPARENT);
        scene.fill_circle(vec2(100.5, 100.5), 60.0, srgba8(255, 255, 255, 255));
        let gradient = scene.create_gradient_ramp(&[
            ColorStop { position: 0.0, color: srgba8(255, 0, 0, 255) },
            ColorStop { position: 0.5, color: srgba8(0, 255, 0, 255) },
            ColorStop { position: 1.0, color: srgba8(0, 0, 255, 255) },
        ]);
        scene.fill_circle(
            vec2(300.5, 100.5),
            60.0,
            LinearGradientFill {
                start: vec2(240.5, 100.5),
                end: vec2(360.5, 100.5),
                ramp: gradient,
                extend_mode: GradientExtendMode::Repeat,
            },
        );
        scene.render(image);

        {
            let _span = gpu_span!("render background");
            gpu::render(&[gpu::ColorAttachment { image, clear: None }], None, |encoder| {
                encoder.bind_graphics_pipeline(&background::background);
                encoder.draw_screen_quad(root_params! {
                    scene: Ptr<SceneInfo> = scene_info_gpu,
                    bottom_color: u32 = 0xFF000000,
                    top_color: u32 = 0xFF0000FF
                });

                encoder.bind_graphics_pipeline(&grid::grid);
                encoder.draw(
                    TriangleList,
                    None,
                    0..6,
                    0..1,
                    root_params! {
                        scene_uniforms: Ptr<SceneInfo> = scene_info_gpu,
                        grid_scale: f32 = 100.0
                    },
                );
            });
        }
    }

    fn resized(&mut self, window: WindowHandle, width: u32, height: u32) {
        // nothing
    }

    fn vsync(&mut self) {
        // nothing
    }
}

register_plugin!(ExperimentApp::new);
