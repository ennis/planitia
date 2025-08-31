use egui::{
    Align, Align2, Area, Color32, Direction, FontId, Frame, InnerResponse, Key, Layout, Order, Pos2, Rect, Response,
    RichText, Sense, Stroke, TextEdit, TextFormat, TextStyle, Ui, Vec2, WidgetText,
};

pub fn popup_button(ui: &mut Ui, text: impl Into<WidgetText>, contents: impl FnOnce(&mut Ui) -> Response) -> Response {
    let popup_id = ui.auto_id_with("popup");
    //let open = ui.memory(|mem| mem.is_popup_open(popup_id));
    let mut button_response = ui.button(text);

    if button_response.clicked() {
        ui.memory_mut(|mem| mem.toggle_popup(popup_id));
    }

    if ui.memory(|mem| mem.is_popup_open(popup_id)) {
        let area_response = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(button_response.rect.max)
            .show(ui.ctx(), |ui| {
                Frame::popup(ui.style()).show(ui, |ui| {
                    let resp = contents(ui);
                    if resp.changed() {
                        button_response.mark_changed();
                    }
                });
            })
            .response;

        if !button_response.clicked() && (ui.input(|i| i.key_pressed(Key::Escape)) || area_response.clicked_elsewhere())
        {
            ui.memory_mut(|mem| mem.close_popup(popup_id));
        }
    }

    button_response
}
