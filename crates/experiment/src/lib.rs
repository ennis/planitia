#![feature(default_field_values)]

mod terrain;

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
    geometry_file: PathBuf,
    #[serde(skip)]
    geometry: hgeo::Geo,
    #[serde(skip)]
    terrain: Terrain,
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
            terrain: Default::default(),
        }
    }

    fn update(&mut self) {}
}

impl AppHandler for ExperimentApp {
    fn input(&mut self, _window: WindowHandle, input_event: &InputEvent) {
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
        load_terrain_from_heightmap("data/heightmap2.png", 1.0, vec2(0.5, 0.5))
            .map(|terrain| {
                self.terrain = terrain;
                eprintln!("Loaded terrain from heightmap");
            })
            .unwrap_or_else(|err| {
                eprintln!("Failed to load terrain from heightmap: {}", err);
            });
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
        let time_since_reload = Local::now().signed_duration_since(self.load_time);
        format_message!(
            "Last reload  : {} ({}m {}s ago)",
            self.load_time.format("%Y-%m-%d %H:%M:%S"),
            time_since_reload.num_minutes(),
            time_since_reload.num_seconds() % 60
        );
        self.terrain.render(&camera, image.image);
    }

    fn resized(&mut self, window: WindowHandle, width: u32, height: u32) {
        // nothing
    }

    fn vsync(&mut self) {
        // nothing
    }
}

register_plugin!(ExperimentApp::new);
