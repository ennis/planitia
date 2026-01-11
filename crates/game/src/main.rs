#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]

use gamelib::asset::{AssetCache, FileSystemEvent, Handle};
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
use gpu::{Image, Ptr, RootParams, root_params};
use log::debug;
use math::geom::{Camera, rect_xywh};
use math::{Mat4, Vec2, Vec3, vec2};

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
    bg_top_color: Color32,
    bg_bottom_color: Color32,
    painter: Painter,
    camera_control: CameraControl,
    depth_stencil_buffer: gpu::Image,
    grid_shader: Handle<gpu::GraphicsPipeline>,
    background_shader: Handle<gpu::GraphicsPipeline>,
    frame_count: u32,
    start_time: std::time::Instant,
    coat_experiment: experiments::coat::CoatExperiment,
    outline_experiment: experiments::outlines::OutlineExperiment,
    show_grid: bool,
    show_painting_demo: bool,
    show_background: bool,
    show_imgui: bool,
}

fn create_depth_buffer(width: u32, height: u32) -> Image {
    Image::new(gpu::ImageCreateInfo {
        width,
        height,
        format: gpu::Format::D32_SFLOAT_S8_UINT,
        usage: gpu::ImageUsage::DEPTH_STENCIL_ATTACHMENT | gpu::ImageUsage::TRANSFER_DST | gpu::ImageUsage::SAMPLED,
        ..
    })
}

impl Default for Game {
    fn default() -> Self {
        Self {
            demo: WidgetGallery::default(),
            color: Default::default(),
            bg_top_color: Color32::from_rgb(100, 149, 237),
            bg_bottom_color: Color32::from_rgb(25, 25, 112),
            painter: Painter::new(gpu::Format::R8G8B8A8_UNORM, None),
            depth_stencil_buffer: create_depth_buffer(WIDTH, HEIGHT),
            camera_control: CameraControl::default(),
            grid_shader: get_graphics_pipeline("/shaders/game_shaders.sharc#grid"),
            background_shader: get_graphics_pipeline("/shaders/game_shaders.sharc#background"),
            frame_count: 0,
            width: WIDTH,
            height: HEIGHT,
            start_time: std::time::Instant::now(),
            coat_experiment: experiments::coat::CoatExperiment::new(),
            outline_experiment: experiments::outlines::OutlineExperiment::new(),
            show_grid: true,
            show_painting_demo: false,
            show_background: true,
            show_imgui: false,
        }
    }
}

impl Game {
    fn render_scene(&mut self, encoder: &mut gpu::RenderEncoder, _camera: &Camera, scene_info: &SceneInfo) {
        //----------------------------------
        // Draw background
        let bottom_color = Srgba8::from(self.bg_bottom_color.to_srgba_unmultiplied());
        let top_color = Srgba8::from(self.bg_top_color.to_srgba_unmultiplied());
        if self.show_background {
            if let Ok(background_shader) = self.background_shader.read() {
                encoder.bind_graphics_pipeline(&*background_shader);
                encoder.draw(
                    TriangleList,
                    None,
                    0..6,
                    0..1,
                    root_params! {
                        scene_uniforms: Ptr<SceneInfoUniforms> = scene_info.gpu,
                        bottom_color: Srgba8 = bottom_color,
                        top_color: Srgba8 = top_color
                    },
                );
            }
        }

        //----------------------------------
        // Draw grid
        if self.show_grid {
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
    }

    fn render_overlay(&mut self, cmd: &mut gpu::CommandBuffer, target: &gpu::Image) {
        let mut scene = self.painter.build_scene();

        let mut text = TextLayout::new(
            &TextFormat {
                size: 20.0,
                ..Default::default()
            },
            concat!(
                "[Home] Home camera\n",
                "[G] Toggle grid\n",
                "[H] Toggle background\n",
                "[P] Toggle painting demo\n",
                "[I] Toggle imgui\n"
            ),
        );
        text.layout(1000.0);

        for glyph_run in text.glyph_runs() {
            scene.draw_glyph_run(vec2(0.0, 0.0), &glyph_run, &DrawGlyphRunOptions::default());
        }

        scene.finish(
            cmd,
            &PaintRenderParams {
                camera: Default::default(),
                color_target: target,
                depth_target: None,
            },
        );

        if self.show_painting_demo {
            experiments::painting_test(
                &mut self.painter,
                cmd,
                target,
                Srgba8::from(self.color.to_srgba_unmultiplied()),
            );
        }
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

        if input_event.is_shortcut("G") {
            self.show_grid = !self.show_grid;
        }
        if input_event.is_shortcut("H") {
            self.show_background = !self.show_background;
        }
        if input_event.is_shortcut("P") {
            self.show_painting_demo = !self.show_painting_demo;
        }
        if input_event.is_shortcut("I") {
            self.show_imgui = !self.show_imgui;
        }

        // --- CAMERA ---
        if self.camera_control.handle_input(&input_event) {
            return;
        }

        // --- experiments ---
        //self.coat_experiment.input(&input_event);
        self.outline_experiment.input(&input_event);
    }

    fn event(&mut self, event: UserEvent) {
        /*if let Ok(FileSystemEvent { .. }) = event.downcast::<FileSystemEvent>() {
            self.
        }*/
    }

    fn resized(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.camera_control.resize(width, height);
        self.depth_stencil_buffer = create_depth_buffer(width, height);
        self.outline_experiment.resize(width, height);
    }

    fn vsync(&mut self) {}

    fn render(&mut self, target: RenderTargetImage) {
        // TODO: that's not the right time to reload assets.
        //       Ideally this should be done asynchronously, on another thread, so as not
        //       to block the GUI and rendering.
        AssetCache::instance().do_reload();

        let mut cmd = gpu::CommandBuffer::new();
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
            scene_info.gpu = cmd.upload(&scene_info.info);

            let mut encoder = cmd.begin_rendering(&[gpu::ColorAttachment {
                    image: target.image,
                    clear_value: Some([0.0, 0.0, 0.0, 1.0]),
                }],
                Some(gpu::DepthStencilAttachment {
                    image: &self.depth_stencil_buffer,
                    depth_clear_value: Some(1.0),
                    stencil_clear_value: Some(0),
                }));

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
        if self.show_imgui {
            APP.render_imgui(&mut cmd, &target.image);
        }

        gpu::submit(cmd).unwrap();
    }

    fn close_requested(&mut self) {
        APP.quit();
    }

    fn imgui(&mut self, ctx: &egui::Context) {
        egui::Window::new("imgui").show(ctx, |ui| {
            egui::Grid::new("params")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Show grid");
                    ui.checkbox(&mut self.show_grid, "");
                    ui.end_row();

                    ui.label("Show background");
                    ui.checkbox(&mut self.show_background, "");
                    ui.end_row();

                    ui.label("Show painting demo");
                    ui.checkbox(&mut self.show_painting_demo, "");
                    ui.end_row();

                    ui.label("Show imgui");
                    ui.checkbox(&mut self.show_imgui, "");
                    ui.end_row();

                    ui.label("Painting demo color");
                    ui.color_edit_button_srgba(&mut self.color);
                    ui.end_row();

                    ui.label("BG top color");
                    ui.color_edit_button_srgba(&mut self.bg_top_color);
                    ui.end_row();

                    ui.label("BG bottom color");
                    ui.color_edit_button_srgba(&mut self.bg_bottom_color);
                    ui.end_row();
                });
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
