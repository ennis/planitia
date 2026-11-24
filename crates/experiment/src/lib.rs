#![feature(default_field_values)]

mod terrain;
mod terrain2;

use crate::terrain::{Terrain, load_terrain_from_heightmap};
use chrono::{DateTime, Local};
use gamelib::camera_control::CameraControl;
use gamelib::color::{Srgba8, srgba8};
use gamelib::math::vec2;
use gamelib::paint::{ColorStop, GradientExtendMode, LinearGradientFill, PaintScene};
use gamelib::platform::RenderTargetImage;
use gamelib::{AppHandler, InputEvent, WindowHandle, format_message, gpu_span, register_plugin};
use gpu::PrimitiveTopology::TriangleList;
use gpu::{self, Ptr, root_params};
use std::path::PathBuf;
use std::time::Instant;

#[derive(serde::Serialize, serde::Deserialize)]
struct ExperimentApp {
    #[serde(skip)]
    last_time: Option<Instant>,
    load_time: DateTime<Local>,
    frames_rendered: usize,
    camera_control: CameraControl,
    #[serde(skip)]
    terrain: Terrain,
    #[serde(skip)]
    terrain2: Option<terrain2::Terrain>,
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
            terrain: Default::default(),
            terrain2: Default::default(),
        }
    }

    fn update(&mut self) {}
}

impl AppHandler for ExperimentApp {
    fn input(&mut self, _window: WindowHandle, input_event: &InputEvent) {
        self.camera_control.handle_input(input_event);
        if input_event.is_shortcut("Home") {
            let size = self.camera_control.screen_size;
            eprintln!("Reset camera");
            self.camera_control = CameraControl::new(size.x as u32, size.y as u32);
        }
        if input_event.is_shortcut(")") {
            self.camera_control.fov_y_radians *= 0.9;
            self.camera_control.last_cam.set(None);
        }
        if input_event.is_shortcut("=") {
            self.camera_control.fov_y_radians *= 1.1;
            self.camera_control.last_cam.set(None);
        }
        if input_event.is_shortcut("-") {
            self.camera_control.z_far *= 0.9;
            self.camera_control.last_cam.set(None);
        }
        if input_event.is_shortcut("+") {
            self.camera_control.z_far *= 1.1;
            self.camera_control.last_cam.set(None);
        }
        if input_event.is_shortcut("Shift++") {
            self.camera_control.z_near *= 1.1;
            self.camera_control.last_cam.set(None);
        }
        if input_event.is_shortcut("Shift+-") {
            self.camera_control.z_near *= 0.9;
            self.camera_control.last_cam.set(None);
        }
    }

    fn started(&mut self) {}

    fn loaded(&mut self) {
        self.load_time = Local::now();
        load_terrain_from_heightmap("data/heightmap2.png", 2.5, vec2(0.5, 0.5))
            .map(|terrain| {
                self.terrain = terrain;
                eprintln!("Loaded terrain from heightmap");
            })
            .unwrap_or_else(|err| {
                eprintln!("Failed to load terrain from heightmap: {}", err);
            });

        self.terrain2 = terrain2::Terrain::new("data/heightmap2.png").ok();
    }


    fn render(&mut self, window: WindowHandle, image: RenderTargetImage<'_>) {
        let _span = gamelib::span!("plugin render");
        let gpu_device_name = gpu::get_physical_device_name();
        //eprintln!("GPU device name: {}", gpu_device_name);
        let camera = self.camera_control.camera();
        self.frames_rendered += 1;
        let time = Instant::now();
        let delta = time.duration_since(self.last_time.unwrap_or(Instant::now()));
        self.last_time = Some(time);
        let fps = if delta.as_secs_f32() > 0.0 { 1.0 / delta.as_secs_f32() } else { 0.0 };
        format_message!("GPU   : {}\n", gpu_device_name);
        format_message!("DT    : {delta:?}\nFPS   : {fps:.1}\n");
        format_message!("FRAME : {}\n", self.frames_rendered);
        format_message!("NEAR  : {}    FAR : {}\n", camera.frustum.near_plane, camera.frustum.far_plane);
        let time_since_reload = Local::now().signed_duration_since(self.load_time);
        format_message!(
            "Last reload  : {} ({}m {}s ago)\n",
            self.load_time.format("%Y-%m-%d %H:%M:%S"),
            time_since_reload.num_minutes(),
            time_since_reload.num_seconds() % 60
        );
        format_message!("EYE : {:.2},{:.2},{:.2}", camera.eye().x, camera.eye().y, camera.eye().z);
        self.terrain.render(&camera, image.image);
        //if let Some(terrain2) = self.terrain2.as_mut() {
        //    terrain2.render(&camera, image.image);
        //}
    }

    fn resized(&mut self, window: WindowHandle, width: u32, height: u32) {
        self.camera_control.resize(width, height);
        // nothing
    }

    fn vsync(&mut self) {
        // nothing
    }
}

register_plugin!(ExperimentApp::new);
