use egui::{Align2, Color32, FontId, Response, Ui};

pub fn icon_button(ui: &mut Ui, icon: &str, color: Color32) -> Response {
    let (rect, response) = ui.allocate_exact_size(egui::Vec2::new(20.0, 20.0), egui::Sense::click());
    if response.hovered() {
        let color = ui.style().visuals.selection.bg_fill;
        ui.painter().rect_filled(rect, 0.0, color);
    }
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        icon,
        FontId::proportional(16.0),
        color,
    );
    response
}
