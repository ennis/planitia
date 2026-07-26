#![feature(default_field_values)]

use crate::background::SceneInfo;
use chrono::{DateTime, Local};
use gamelib::camera_control::CameraControl;
use gamelib::color::{Srgba8, srgba8};
use gamelib::math::vec2;
use gamelib::paint::{ColorStop, GradientExtendMode, LinearGradientFill, PaintScene};
use gamelib::{PluginEvent, PluginInterface, PluginResult, register_plugin};
use gpu::{self, Ptr, root_params};
use std::time::{Duration, Instant};

#[gpu::shader_module("shaders/background.slang#447")]
mod background {}

#[derive(serde::Serialize, serde::Deserialize)]
struct HotReloadTest {
    #[serde(skip)]
    last_time: Option<Instant>,
    load_time: DateTime<Local>,
    reload_count: usize,
    camera_control: CameraControl,
}

impl HotReloadTest {
    fn new() -> HotReloadTest {
        HotReloadTest { last_time: None, load_time: Local::now(), reload_count: 0, camera_control: Default::default() }
    }

    fn update(&mut self) {}
}

impl PluginInterface for HotReloadTest {
    fn event(&mut self, event: &PluginEvent) -> PluginResult {
        let gpu_device_name = gpu::get_physical_device_name();
        //eprintln!("GPU device name: {}", gpu_device_name);

        let time = Instant::now();
        let delta = if let Some(last_time) = self.last_time { time.duration_since(last_time) } else { Duration::ZERO };
        self.last_time = Some(time);

        let fps = if delta.as_secs_f32() > 0.0 { 1.0 / delta.as_secs_f32() } else { 0.0 };

        gamelib::print_message(format!("GPU device name: {}\n", gpu_device_name));
        gamelib::print_message(format!("DT  : {delta:?}\nFPS : {fps:.1}\n"));
        gamelib::print_message(format!("Last reload: {} (reloaded {} times)\n", self.load_time, self.reload_count));

        match event {
            PluginEvent::Init => {
                eprintln!("Plugin init");
                self.load_time = Local::now();
                self.reload_count += 1;
            }
            PluginEvent::Deinit => {
                eprintln!("Plugin deinit");
            }
            PluginEvent::VSync(image) => {
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
                        time: time.elapsed().as_secs_f32(),
                        frame: 0,
                        eye: Default::default(),
                    });

                    encoder.draw_screen_quad(root_params! {
                        scene: Ptr<SceneInfo> = scene_info,
                        bottom_color: u32 = 0xFF000000,
                        top_color: u32 = 0xFF0000FF
                    });
                })
            }
            _ => {}
        }

        PluginResult::WaitVSync
    }
}

register_plugin!(HotReloadTest::new);
