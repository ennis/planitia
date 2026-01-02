use egui::emath::{GuiRounding, RectTransform};
use egui::{Color32, Pos2, Rect, Response, Sense, Stroke, Ui, Vec2, pos2};
use uniform_cubic_splines::basis::CatmullRom;
use uniform_cubic_splines::{spline, spline_inverse};

pub fn curve_editor(ui: &mut Ui, abscissae: &mut [f64], values: &mut [f64]) -> Response {
    assert_eq!(abscissae.len(), values.len());
    let size = egui::vec2(400., 400.);
    let handle_radius = 5.;
    //
    let (mut resp, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    let rect = Rect::from_min_size(resp.rect.min, size);
    let to_area = RectTransform::from_to(
        Rect::from_min_max(Pos2::new(0.0, 1.0), Pos2::new(1.0, 0.0)),
        Rect::from_min_size(resp.rect.min, size),
    );
    let to_normalized = to_area.inverse();

    // bg color
    painter.rect_filled(resp.rect, 0., Color32::from_gray(32));

    // grid
    let grid_divisions = 10;
    let grid_stroke = Stroke::new(1., Color32::from_gray(64));
    for x in 1..grid_divisions {
        for y in 1..grid_divisions {
            let mut pos = to_area.transform_pos(pos2(x as f32, y as f32) / grid_divisions as f32);
            pos = pos.round_to_pixels(painter.pixels_per_point());
            painter.line_segment(
                [Pos2::new(pos.x + 0.5, rect.min.y), Pos2::new(pos.x + 0.5, rect.max.y)],
                grid_stroke,
            );
            painter.line_segment(
                [Pos2::new(rect.min.x, pos.y + 0.5), Pos2::new(rect.max.x, pos.y + 0.5)],
                grid_stroke,
            );
        }
    }

    // draw handles in the space
    for i in 1..abscissae.len() - 1 {
        let mut center = to_area.transform_pos(pos2(abscissae[i] as f32, values[i] as f32));
        let handle_size = Vec2::splat(2. * handle_radius);
        let handle_rect = Rect::from_center_size(center, handle_size);
        let handle_resp = ui.interact(handle_rect, resp.id.with(i), Sense::drag());

        if handle_resp.dragged() {
            resp.mark_changed();
            center += handle_resp.drag_delta();
            center = center.clamp(rect.min, rect.max);
            let pos = to_normalized.transform_pos(center);
            abscissae[i] = pos.x as f64;
            values[i] = pos.y as f64;

            abscissae[0] = abscissae[1];
            values[0] = values[1];
            abscissae[abscissae.len() - 1] = abscissae[abscissae.len() - 2];
            values[values.len() - 1] = values[values.len() - 2];
        }

        painter.circle(
            center,
            handle_radius,
            Color32::TRANSPARENT,
            Stroke::new(1., Color32::WHITE),
        );
    }

    // draw curve
    let mut prev = None;

    for i in 0..=size.x as u32 {
        let t = i as f64 / size.x as f64;

        let v = spline_inverse::<CatmullRom, _>(t, abscissae).unwrap_or_default();
        let v = spline::<CatmullRom, _, _>(v, values).unwrap();

        let pos = to_area.transform_pos(pos2(t as f32, v as f32));
        if let Some(prev) = prev {
            painter.line_segment([prev, pos], Stroke::new(1., Color32::GRAY));
        }
        prev = Some(pos);
    }

    resp
}

//pub fn curve_editor_button(ui: &mut Ui, abscissae: &mut [f64], values: &mut [f64]) {
//    curve_editor(ui, abscissae, values);
//}
