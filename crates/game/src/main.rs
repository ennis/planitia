#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]

use crate::context::{AppHandler, LoopHandler, get_gpu_device, quit, render_imgui, run};
use crate::input::{InputEvent, PointerButton};
use crate::paint::{DrawGlyphRunOptions, PaintRenderParams, PaintScene, Painter, Srgba32, TextFormat, TextLayout};
use crate::platform::{EventToken, InitOptions, LoopEvent, RenderTargetImage};
use egui::Color32;
use egui_demo_lib::{Demo, DemoWindows, View, WidgetGallery};
use futures::{FutureExt, StreamExt};
use log::debug;
use serde_json::json;
use math::geom::rect_xywh;
use math::{Rect, vec2};
use crate::camera_control::{CameraControl, CameraControlInput};
use crate::pipeline_editor::PipelineEditor;

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
mod camera_control;
mod pipeline_editor;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;


struct Handler {
    physics_timer: EventToken,
    render_target_time: EventToken,
    demo: WidgetGallery,
    color: Color32,
    painter: Painter,
    camera_control: CameraControl,
    pipeline_editor: PipelineEditor,
}

impl Default for Handler {
    fn default() -> Self {
        let painter = Painter::new(get_gpu_device(), gpu::Format::R8G8B8A8_UNORM, None);
        let device = get_gpu_device();
        /*let grid_pipeline = device.create_graphics_pipeline(&gpu::GraphicsPipelineCreateInfo {
            vertex: gpu::ShaderModule::from_spirv_bytes(device, include_bytes!("../shaders/grid.vert.spv")).unwrap(),
            fragment: Some(gpu::ShaderModule::from_spirv_bytes(device, include_bytes!("../shaders/grid.frag.spv")).unwrap()),
            ..Default::default()
        }).unwrap();*/

        Self {
            physics_timer: EventToken(1),
            render_target_time: EventToken(2),
            demo: WidgetGallery::default(),
            color: Default::default(),
            painter,
            camera_control: CameraControl::default(),
            pipeline_editor: Default::default(),
            //grid_pipeline: ,
        }
    }
}

impl Handler {
    fn paint_test_scene(&mut self, cmd: &mut gpu::CommandStream, target: &gpu::Image) {
        let mut scene = self.painter.build_scene();
        let [r, g, b, a] = self.color.to_srgba_unmultiplied();
        let color = Srgba32 { r, g, b, a };
        scene.fill_rrect(rect_xywh(100.0, 100.0, 200.0, 200.0), 20.0, color);

        let mut text = TextLayout::new(
            &TextFormat {
                size: 48.0,
                ..Default::default()
            },
            r"Innumerable force of Spirits armed,
That durst dislike his reign, and, me preferring,
His utmost power with adverse power opposed
In dubious battle on the plains of Heaven
And shook his throne. What though the field be lost?
All is not lost—the unconquerable will,
And study of revenge, immortal hate,
And courage never to submit or yield:
And what is else not to be overcome?",
        );
        text.layout(1000.0);

        for glyph_run in text.glyph_runs() {
            scene.draw_glyph_run(vec2(0.0, 0.0), &glyph_run, &DrawGlyphRunOptions::default());
        }

        /*scene.draw_glyph_run(
            vec2(400.0, 200.0),
            "Hello, world! こんにちは",
            48.0,
        );*/

        scene.finish(
            cmd,
            &PaintRenderParams {
                camera: Default::default(),
                color_target: target,
                depth_target: None,
            },
        );

    }

    fn draw_grid(&mut self, cmd: &mut gpu::CommandStream, target: &gpu::Image) {

        //
        /*let pipeline = self.pipeline_db.query("grid", [
            ColorFormat(),
            DepthFormat(),

            ]);*/

        // self-contained pipeline files:
        // - custom tags (for vertex signature, etc.)
        // - SPIR-V shader modules
        // - fixed-function state (rasterization, depth-stencil, blending)
        // - specialization constants (formats, etc.)
        // - useful reflection data (push constant sizes)
    }
}

impl AppHandler for Handler {
    fn input(&mut self, input_event: InputEvent) {

        // --- SHORTCUTS ---

        // App exit
        if input_event.is_shortcut("Ctrl+Q") {
            debug!("Quit requested via Ctrl+Q");
            quit();
        }

        // Home camera
        if input_event.is_shortcut("Home") {
            debug!("Home camera");
            //self.camera_control.home();
        }

        // --- CAMERA ---
        self.camera_control.handle_input(&input_event);
    }

    fn event(&mut self, token: EventToken) {}

    fn resized(&mut self, width: u32, height: u32) {
        self.camera_control.resize(width, height);
    }

    fn vsync(&mut self) {}

    fn render(&mut self, target: RenderTargetImage) {
        let device = get_gpu_device();
        let mut cmd = device.create_command_stream();
        cmd.clear_image(&target.image, gpu::ClearColorValue::Float([0.0, 0.0, 0.0, 1.0]));

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
            //dbg!(self.color);
            self.demo.ui(ui);
        });
        self.pipeline_editor.show_gui(ctx);
    }
}

fn main() {
    run::<Handler>(&InitOptions {
        width: WIDTH,
        height: HEIGHT,
        window_title: "Planitia",
        ..Default::default()
    });
}
