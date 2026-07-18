pub mod automaton;
pub mod coat;
mod dcel;
pub mod lines;
pub mod outlines;
mod ss_contours;
pub mod svg;
mod sweep;
mod winged_edge_mesh;


use color::Srgba8;
use gamelib::paint::{render_scene, DrawGlyphRunOptions, PaintScene, TextFormat, TextLayout};
use math::{rect_xywh, vec2};

pub(super) fn painting_test(target: &gpu::Image, color: Srgba8)
{
    let mut scene = PaintScene::new(Srgba8::TRANSPARENT);

    scene.fill_rrect(rect_xywh(100.0, 100.0, 200.0, 200.0), 20.0, color);
    scene.fill_circle(vec2(400.0, 400.0), 100.0, color);

    let mut text = TextLayout::new(
        &TextFormat { size: 48.0, ..Default::default() },
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

    render_scene(target, scene);
}
