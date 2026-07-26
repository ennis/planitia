#![feature(default_field_values)]

use crate::background::SceneInfo;
use chrono::{DateTime, Local};
use gamelib::camera_control::CameraControl;
use gamelib::color::{Srgba8, srgba8};
use gamelib::math::vec2;
use gamelib::paint::{ColorStop, GradientExtendMode, LinearGradientFill, PaintScene};
use gamelib::platform::RenderTargetImage;
use gamelib::{format_message, register_plugin, AppHandler, InputEvent, WindowHandle};
use gpu::{self, Ptr, root_params};
use std::time::{Duration, Instant};

#[gpu::shader_module("shaders/background.slang#447")]
mod background {}

#[derive(serde::Serialize, serde::Deserialize)]
struct HotReloadTest {
    #[serde(skip)]
    last_time: Option<Instant>,
    load_time: DateTime<Local>,
    frames_rendered: usize,
    camera_control: CameraControl,
}

impl Default for HotReloadTest {
    fn default() -> Self {
        Self::new()
    }
}

impl HotReloadTest {
    fn new() -> HotReloadTest {
        let start = Instant::now();
        HotReloadTest {
            last_time: Some(start),
            load_time: Local::now(),
            camera_control: Default::default(),
            frames_rendered: 0,
        }
    }

    fn update(&mut self) {}
}

impl AppHandler for HotReloadTest {
    fn input(&mut self, window: WindowHandle, input_event: &InputEvent) {
        //
    }

    fn started(&mut self) {}

    fn loaded(&mut self) {
        self.load_time = Local::now();
    }

    fn render(&mut self, window: WindowHandle, image: RenderTargetImage<'_>) {
        let image = image.image;

        let gpu_device_name = gpu::get_physical_device_name();
        //eprintln!("GPU device name: {}", gpu_device_name);

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

        gpu::render(&[gpu::ColorAttachment { image, clear: None }], None, |encoder| {
            encoder.bind_graphics_pipeline(&background::background);
            let scene_info = encoder.upload(&SceneInfo {
                view_matrix: Default::default(),
                proj_matrix: Default::default(),
                view_proj_matrix: Default::default(),
                screen_size: Default::default(),
                time: self.frames_rendered as f32,
                frame: self.frames_rendered as u32,
                eye: Default::default(),
            });

            encoder.draw_screen_quad(root_params! {
                scene: Ptr<SceneInfo> = scene_info,
                bottom_color: u32 = 0xFF000000,
                top_color: u32 = 0xFF0000FF
            });
        })
    }

    fn resized(&mut self, window: WindowHandle, width: u32, height: u32) {
        // nothing
    }

    fn vsync(&mut self) {
        // nothing
    }
}

register_plugin!(HotReloadTest::new);
