mod curve;
pub(crate) mod egui_backend;
mod icon_button;
pub(crate) mod input_state;
mod popup_button;

pub use popup_button::*;

use crate::imgui;
use crate::imgui::input_state::EguiInputState;
use crate::input::InputEvent;
use egui::{Align, Color32, FontDefinitions, RichText, Style, TextFormat, TextStyle};
use gpu::CommandStream;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;

// see https://github.com/rerun-io/rerun/blob/main/crates/re_ui/src/lib.rs#L599
fn generic_list_header<R>(ui: &mut egui::Ui, label: &str, right_buttons: impl Fn(&mut egui::Ui) -> R) {
    ui.allocate_ui_with_layout(
        egui::Vec2::new(ui.available_width(), 20.0),
        egui::Layout::left_to_right(Align::Center),
        |ui| {
            let mut rect = ui.available_rect_before_wrap();
            let hline_stroke = ui.style().visuals.widgets.noninteractive.bg_stroke;
            rect.extend_with_x(ui.clip_rect().right());
            rect.extend_with_x(ui.clip_rect().left());
            ui.painter().hline(rect.x_range(), rect.top(), hline_stroke);
            ui.painter().hline(rect.x_range(), rect.bottom(), hline_stroke);

            ui.strong(label);
            ui.allocate_ui_with_layout(
                ui.available_size(),
                egui::Layout::right_to_left(egui::Align::Center),
                right_buttons,
            )
            .inner
        },
    );
}

pub(crate) fn style_to_text_format(style: &egui::Style) -> TextFormat {
    let mut text_format = TextFormat::default();
    text_format.color = style.visuals.text_color();
    text_format.font_id = style
        .override_text_style
        .clone()
        .unwrap_or(TextStyle::Body)
        .resolve(style);
    text_format
}

fn color_icon(icon: &str, color: Color32) -> RichText {
    RichText::new(icon).size(20.0).color(color)
}

pub(crate) struct ImguiContext {
    renderer: imgui::egui_backend::Renderer,
    ctx: egui::Context,
    input: EguiInputState,
    output: egui::FullOutput,
}

impl ImguiContext {
    pub(crate) fn new(gpu: &gpu::RcDevice) -> Self {
        let ctx = egui::Context::default();
        ctx.set_fonts(FontDefinitions::default());
        ctx.set_style(Style::default());
        let renderer = egui_backend::Renderer::new(gpu);
        Self {
            renderer,
            ctx,
            input: EguiInputState::default(),
            output: egui::FullOutput::default(),
        }
    }

    pub(crate) fn handle_input(&mut self, event: &InputEvent) -> bool {
        self.input.update(&self.ctx, event)
    }

    pub(crate) fn run(&mut self, f: impl FnMut(&egui::Context)) {
        let raw_input = self.input.raw.take();
        let output = self.ctx.run(raw_input, f);
        self.input.handle_platform_output(&output.platform_output);
        self.output = output;
    }

    pub(crate) fn render(&mut self, cmd: &mut CommandStream, image: &gpu::Image) {
        self.renderer.render(
            cmd,
            &image.create_top_level_view(),
            &self.ctx,
            mem::take(&mut self.output.textures_delta),
            mem::take(&mut self.output.shapes),
            mem::take(&mut self.output.pixels_per_point),
        );
    }
}
