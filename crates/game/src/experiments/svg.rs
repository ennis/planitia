use color::Srgba8;
use gamelib::input::{InputEvent, MouseScrollDelta, PointerButton};
use gamelib::paint::{
    ColorStop, GradientExtendMode, LinearGradientFill, PaintRenderParams, PaintScene, Painter, PathBuilder, TextFormat,
    TextLayout,
};
use gamelib::tracy_client;
use math::{Mat3, Vec2, rect_xywh, vec2};
use usvg::Node;

pub(crate) struct SvgExperiment {
    document: usvg::Tree,
    global_offset: Vec2,
    global_scale: Vec2,
    /// Current cursor position in window pixels.
    cursor_pos: Vec2,
    /// Whether the left mouse button is held for panning.
    is_panning: bool,
}

fn render_node(scene: &mut PaintScene, node: &Node) {
    let mut path_builder = PathBuilder::new();
    match node {
        Node::Image(image) => {
            // TODO render image
        }
        Node::Path(path) => {
            // TODO render path
            for segment in path.data().segments() {
                match segment {
                    usvg::tiny_skia_path::PathSegment::MoveTo(to) => {
                        path_builder.move_to(vec2(to.x, to.y));
                    }
                    usvg::tiny_skia_path::PathSegment::LineTo(to) => {
                        path_builder.line_to(vec2(to.x, to.y));
                    }
                    usvg::tiny_skia_path::PathSegment::QuadTo(ctrl, to) => {
                        path_builder.quad_to(vec2(ctrl.x, ctrl.y), vec2(to.x, to.y));
                    }
                    usvg::tiny_skia_path::PathSegment::CubicTo(ctrl1, ctrl2, to) => {
                        path_builder.cubic_to(vec2(ctrl1.x, ctrl1.y), vec2(ctrl2.x, ctrl2.y), vec2(to.x, to.y));
                    }
                    usvg::tiny_skia_path::PathSegment::Close => {
                        path_builder.close();
                    }
                }
            }
            path_builder.close();
            if let Some(fill) = path.fill() {
                match fill.paint() {
                    usvg::Paint::Color(c) => {
                        let conv_color = Srgba8::new(c.red, c.green, c.blue, fill.opacity().to_u8());
                        scene.fill_path(&path_builder, conv_color);
                    }
                    usvg::Paint::LinearGradient(gradient) => {
                        let stops = gradient
                            .stops()
                            .iter()
                            .map(|stop| {
                                let color = stop.color();
                                let conv_color =
                                    Srgba8::new(color.red, color.green, color.blue, fill.opacity().to_u8());
                                let position = stop.offset().get_finite().get();
                                ColorStop { position, color: conv_color }
                            })
                            .collect::<Vec<_>>();
                        let ramp = scene.create_gradient_ramp(&stops);

                        scene.fill_path(
                            &path_builder,
                            LinearGradientFill {
                                start: vec2(gradient.x1(), gradient.y1()),
                                end: vec2(gradient.x2(), gradient.y2()),
                                ramp,
                                extend_mode: match gradient.spread_method() {
                                    usvg::SpreadMethod::Pad => GradientExtendMode::Clamp,
                                    usvg::SpreadMethod::Repeat => GradientExtendMode::Repeat,
                                    usvg::SpreadMethod::Reflect => GradientExtendMode::Mirror,
                                },
                            },
                        );
                    }
                    _ => {
                        // TODO other paints
                    }
                }
            }
            path_builder.clear();
        }
        Node::Group(group) => {
            render_group(scene, group);
        }
        Node::Text(text) => {
            // TODO render text
        }
    }
}

fn render_group(scene: &mut PaintScene, group: &usvg::Group) {
    let tr = group.transform();
    let node_transform = Mat3::from_cols_array(&[tr.sx, tr.kx, 0.0, tr.ky, tr.sy, 0.0, tr.tx, tr.ty, 1.0]);
    scene.save();
    scene.transform(node_transform);
    for child in group.children() {
        render_node(scene, child);
    }
    scene.restore();
}

impl SvgExperiment {
    pub fn new() -> Self {
        let svg_data = include_str!("grid.svg");
        let document = usvg::Tree::from_str(svg_data, &usvg::Options::default()).unwrap();
        Self {
            document,
            global_offset: Default::default(),
            global_scale: vec2(1.0, 1.0),
            cursor_pos: Default::default(),
            is_panning: false,
        }
    }

    pub fn input(&mut self, input: &InputEvent) {
        match *input {
            InputEvent::CursorMoved { x, y } => {
                let new_pos = vec2(x as f32, y as f32);
                if self.is_panning {
                    self.global_offset += new_pos - self.cursor_pos;
                }
                self.cursor_pos = new_pos;
            }
            InputEvent::PointerDown { button: PointerButton::LEFT, x, y } => {
                self.cursor_pos = vec2(x as f32, y as f32);
                self.is_panning = true;
            }
            InputEvent::PointerUp { button: PointerButton::LEFT, .. } => {
                self.is_panning = false;
            }
            InputEvent::MouseWheel(delta) => {
                // Zoom towards the current cursor position.
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta { y, .. } => y,
                    MouseScrollDelta::PixelDelta { y, .. } => y / 40.0,
                };
                let zoom_factor = (1.1_f32).powf(scroll_y);
                // Adjust offset so the point under the cursor stays fixed:
                //   new_offset = cursor - zoom_factor * (cursor - old_offset)
                self.global_offset = self.cursor_pos + zoom_factor * (self.global_offset - self.cursor_pos);
                self.global_scale *= zoom_factor;
            }
            _ => {}
        }
    }

    pub fn render(&mut self, painter: &mut Painter, cmd: &mut gpu::CommandBuffer, target: &gpu::Image) {
        let mut scene = painter.build_scene();

        {
            let _span = tracy_client::span!("svg_build_scene");
            // Apply global pan and zoom.
            let sx = self.global_scale.x;
            let sy = self.global_scale.y;
            let tx = self.global_offset.x;
            let ty = self.global_offset.y;
            let global_transform = Mat3::from_cols_array(&[sx, 0.0, 0.0, 0.0, sy, 0.0, tx, ty, 1.0]);
            scene.save();
            scene.transform(global_transform);
            render_group(&mut scene, self.document.root());
            scene.restore();
        }

        {
            let _span = tracy_client::span!("svg_render_scene");
            scene.finish(
                cmd,
                &PaintRenderParams { camera: Default::default(), color_target: target, depth_target: None },
            );
        }
    }
}
