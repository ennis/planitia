#![expect(unused, reason = "noisy")]

use crate::context::{AppHandler, LoopHandler, get_gpu_device, quit, render_imgui, run};
use crate::input::InputEvent;
use crate::paint::{PaintRenderParams, PaintScene, Painter, Srgba32};
use crate::platform::{EventToken, InitOptions, LoopEvent, RenderTargetImage};
use egui::Color32;
use egui_demo_lib::{Demo, DemoWindows, View, WidgetGallery};
use futures::{FutureExt, StreamExt};
use math::geom::rect_xywh;
use math::{Rect, vec2};

mod context;
mod event;
mod executor;
mod imgui;
mod input;
mod notifier;
mod paint;
mod platform;
mod script;
mod shaders;
mod task;
mod timer;
mod util;
mod world;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

struct Handler {
    physics_timer: EventToken,
    render_target_time: EventToken,
    demo: WidgetGallery,
    color: Color32,
    painter: Painter,
}

impl Default for Handler {
    fn default() -> Self {
        let painter = Painter::new(get_gpu_device(), gpu::Format::R8G8B8A8_UNORM, None);

        Self {
            physics_timer: EventToken(1),
            render_target_time: EventToken(2),
            demo: WidgetGallery::default(),
            color: Default::default(),
            painter,
        }
    }
}

impl Handler {
    fn paint_test_scene(&mut self, cmd: &mut gpu::CommandStream, target: &gpu::Image) {


        let mut scene = self.painter.build_scene();
        let [r, g, b, a] = self.color.to_srgba_unmultiplied();
        let color = Srgba32 { r, g, b, a };
        scene.fill_rrect(rect_xywh(100.0, 100.0, 200.0, 200.0), 20.0, color);

       /* scene.draw_glyph_run(
            vec2(400.0, 200.0),
            "Hello, world! こんにちは",
            48.0,
        );*/

        scene.finish(
            cmd,
            &PaintRenderParams {
                camera: Default::default(),
                color_target: target.create_top_level_view(),
                depth_target: None,
            },
        );
    }
}

impl AppHandler for Handler {
    fn input(&mut self, input_event: InputEvent) {}

    fn event(&mut self, token: EventToken) {}

    fn resized(&mut self, width: u32, height: u32) {
        //todo!()
    }

    fn vsync(&mut self) {}

    fn render(&mut self, target: RenderTargetImage) {
        let device = get_gpu_device();
        let mut cmd = device.create_command_stream();
        cmd.clear_image(&target.image, gpu::ClearColorValue::Float([0.5, 0.5, 0.5, 1.0]));

        //cmd.clear_image(&target.image, gpu::ClearColorValue::Float([1.0, 1.0, 1.0, 1.0]));
        self.paint_test_scene(&mut cmd, &target.image);
        render_imgui(&mut cmd, &target.image);

        cmd.flush(&[target.ready], &[target.rendering_finished]).unwrap();
    }

    fn close_requested(&mut self) {
        quit();
    }

    fn imgui(&mut self, ctx: &egui::Context) {
        egui::Window::new("imgui").show(ctx, |ui| {
            ui.color_edit_button_srgba(&mut self.color);
            dbg!(self.color);
            self.demo.ui(ui);
        });
    }
}

fn main() {
    run::<Handler>(&InitOptions {
        width: WIDTH,
        height: HEIGHT,
        ..Default::default()
    });
}
