pub mod coat;
mod dcel;
pub mod lines;
pub mod outlines;
mod sweep;

use color::Srgba8;
use gamelib::paint::{DrawGlyphRunOptions, PaintRenderParams, Painter, TextFormat, TextLayout};
use math::geom::rect_xywh;
use math::vec2;

pub(super) fn painting_test(painter: &mut Painter, cmd: &mut gpu::CommandStream, target: &gpu::Image, color: Srgba8) {
    let mut scene = painter.build_scene();
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

    scene.finish(
        cmd,
        &PaintRenderParams {
            camera: Default::default(),
            color_target: target,
            depth_target: None,
        },
    );
}
