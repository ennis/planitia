use color::Srgba8;
use gamelib::paint::{PaintRenderParams, PaintScene, Painter, PathBuilder, TextFormat, TextLayout};
use gamelib::tracy_client;
use math::geom::rect_xywh;
use math::{Mat3, vec2};
use usvg::Node;

pub(crate) struct SvgExperiment {
    document: usvg::Tree,
}

fn render_node(scene: &mut PaintScene, node: &Node) {
    match node {
        Node::Image(image) => {
            // TODO render image
        }
        Node::Path(path) => {
            // TODO render path
            let mut path_builder = PathBuilder::new();
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
            let p = path_builder.finish();
            if let Some(fill) = path.fill() {
                match fill.paint() {
                    usvg::Paint::Color(c) => {
                        let conv_color = Srgba8::new(c.red, c.green, c.blue, fill.opacity().to_u8());
                        scene.fill_path(&p, conv_color);
                    }
                    _ => {
                        // TODO other paints
                    }
                }
            }
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
        let svg_data = include_str!("QF.svg");
        let document = usvg::Tree::from_str(svg_data, &usvg::Options::default()).unwrap();
        Self { document }
    }

    pub fn render(&mut self, painter: &mut Painter, cmd: &mut gpu::CommandBuffer, target: &gpu::Image) {
        let mut scene = painter.build_scene();

        {
            let _span = tracy_client::span!("svg_build_scene");
            render_group(&mut scene, self.document.root());
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
