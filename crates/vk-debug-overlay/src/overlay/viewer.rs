//! Interactive image viewer widget with pan & zoom controls.
//!
//! [`image_viewer`] displays a Vulkan texture (already bound to an [`imgui::TextureId`] by the
//! overlay renderer, see `TEXID_FONT` / `TEXID_SWAPCHAIN` in `overlay::renderer`) inside a
//! scrollable canvas, with toolbar controls to reset/fit the view and to zoom numerically, plus
//! mouse-wheel zoom (centered on the cursor) and click-drag panning.

use imgui::{ButtonFlags, MouseButton, SliderFlags, TextureId, Ui};

/// Persistent pan/zoom state for a single [`image_viewer`] instance.
///
/// Keep one of these around per viewer (e.g. in your `GuiState`) across frames so that the view
/// isn't reset every time the window is redrawn.
#[derive(Copy, Clone, Debug)]
pub struct ImageViewerState {
    /// Zoom factor; `1.0` means one texel maps to one screen pixel.
    pub zoom: f32,
    /// Screen-space offset (in pixels, relative to the canvas top-left) of the image origin.
    pub pan: [f32; 2],
    /// When set, the view is reset to fit the canvas on the next call to [`image_viewer`].
    pub fit_requested: bool,
}

impl Default for ImageViewerState {
    fn default() -> Self {
        ImageViewerState { zoom: 1.0, pan: [0.0, 0.0], fit_requested: true }
    }
}

impl ImageViewerState {
    /// Requests that the view be reset to fit the whole image in the canvas on the next frame.
    pub fn fit(&mut self) {
        self.fit_requested = true;
    }

    /// Resets to 1:1 pixel scale, centering the origin on the canvas top-left.
    pub fn reset_to_100_percent(&mut self) {
        self.zoom = 1.0;
        self.pan = [0.0, 0.0];
        self.fit_requested = false;
    }
}

const MIN_ZOOM: f32 = 0.02;
const MAX_ZOOM: f32 = 64.0;

/// Displays an interactive image viewer for a Vulkan texture.
///
/// * `label` uniquely identifies the widget (used to build child-window/control ids); it is not
///   displayed.
/// * `texture_id` is the imgui texture id that the overlay renderer has bound to the underlying
///   `VkImageView` of the texture to display.
/// * `image_size` is the size of the texture, in texels, used to compute UVs and the initial fit.
/// * `state` holds the pan/zoom state across frames; create one with `ImageViewerState::default()`
///   and keep it alive for the lifetime of the viewer.
///
/// Controls: "Fit" resets the view to fit the canvas, "100%" resets to a 1:1 pixel scale, and the
/// "Zoom" slider sets the zoom factor directly. The mouse wheel zooms in/out centered on the
/// cursor, and dragging with the left or middle mouse button pans the view.
pub fn image_viewer(ui: &Ui, label: &str, texture_id: TextureId, image_size: [f32; 2], state: &mut ImageViewerState) {
    let id = ui.push_id(label);

    // --- Toolbar -------------------------------------------------------------------------------
    ui.text(format!("{:.0} x {:.0}", image_size[0], image_size[1]));
    ui.same_line();
    if ui.button("Fit") {
        state.fit();
    }
    ui.same_line();
    if ui.button("100%") {
        state.reset_to_100_percent();
    }
    ui.same_line();
    ui.set_next_item_width(140.0);
    ui.slider_config("Zoom", MIN_ZOOM, MAX_ZOOM).flags(SliderFlags::LOGARITHMIC).build(&mut state.zoom);
    ui.same_line();
    ui.text(format!("{:.0}%", state.zoom * 100.0));

    // --- Canvas ----------------------------------------------------------------------------------
    // Reserve all remaining space in the host window/child for the image canvas.
    let avail = ui.content_region_avail();
    let canvas_size = [avail[0].max(1.0), avail[1].max(1.0)];

    ui.child_window("##canvas")
        .size(canvas_size)
        .border(true)
        .scroll_bar(false)
        .scrollable(false)
        .build(|| {
            let canvas_pos = ui.cursor_screen_pos();
            let canvas_size = ui.content_region_avail();

            if state.fit_requested && canvas_size[0] > 0.0 && image_size[0] > 0.0 && image_size[1] > 0.0 {
                let zoom_x = canvas_size[0] / image_size[0];
                let zoom_y = canvas_size[1] / image_size[1];
                state.zoom = zoom_x.min(zoom_y).clamp(MIN_ZOOM, MAX_ZOOM);
                state.pan = [
                    (canvas_size[0] - image_size[0] * state.zoom) * 0.5,
                    (canvas_size[1] - image_size[1] * state.zoom) * 0.5,
                ];
                state.fit_requested = false;
            }

            // Background so that transparent images are still visible as a rectangle.
            ui.get_window_draw_list()
                .add_rect(canvas_pos, [canvas_pos[0] + canvas_size[0], canvas_pos[1] + canvas_size[1]], 0x40404040)
                .filled(true)
                .build();

            // Invisible button covering the whole canvas: it both captures mouse input for
            // dragging and reports hover for wheel-zoom, without drawing anything itself.
            ui.invisible_button_flags(
                "##canvas_input",
                canvas_size,
                ButtonFlags::MOUSE_BUTTON_LEFT | ButtonFlags::MOUSE_BUTTON_MIDDLE,
            );
            let hovered = ui.is_item_hovered();
            let active = ui.is_item_active();

            // Mouse-wheel zoom, keeping the texel under the cursor stationary on screen.
            let wheel = ui.io().mouse_wheel;
            if hovered && wheel != 0.0 {
                let mouse_pos = ui.io().mouse_pos;
                let local = [mouse_pos[0] - canvas_pos[0], mouse_pos[1] - canvas_pos[1]];
                let image_pt =
                    [(local[0] - state.pan[0]) / state.zoom, (local[1] - state.pan[1]) / state.zoom];
                let new_zoom = (state.zoom * (1.0 + wheel * 0.1)).clamp(MIN_ZOOM, MAX_ZOOM);
                state.pan[0] -= image_pt[0] * (new_zoom - state.zoom);
                state.pan[1] -= image_pt[1] * (new_zoom - state.zoom);
                state.zoom = new_zoom;
            }

            // Click-drag panning with either the left or middle mouse button.
            if active && (ui.is_mouse_dragging(MouseButton::Left) || ui.is_mouse_dragging(MouseButton::Middle)) {
                let delta = if ui.is_mouse_down(MouseButton::Middle) {
                    ui.mouse_drag_delta_with_button(MouseButton::Middle)
                } else {
                    ui.mouse_drag_delta_with_button(MouseButton::Left)
                };
                state.pan[0] += delta[0];
                state.pan[1] += delta[1];
                ui.reset_mouse_drag_delta(MouseButton::Left);
                ui.reset_mouse_drag_delta(MouseButton::Middle);
            }

            // Draw the image at the current pan/zoom.
            let p_min = [canvas_pos[0] + state.pan[0], canvas_pos[1] + state.pan[1]];
            let p_max = [p_min[0] + image_size[0] * state.zoom, p_min[1] + image_size[1] * state.zoom];
            ui.get_window_draw_list().add_image(texture_id, p_min, p_max).build();

            // Status line: texel coordinates under the cursor, if hovered.
            if hovered {
                let mouse_pos = ui.io().mouse_pos;
                let tx = (mouse_pos[0] - p_min[0]) / state.zoom;
                let ty = (mouse_pos[1] - p_min[1]) / state.zoom;
                if tx >= 0.0 && ty >= 0.0 && tx < image_size[0] && ty < image_size[1] {
                    ui.get_window_draw_list().add_text(
                        [canvas_pos[0] + 4.0, canvas_pos[1] + 4.0],
                        0xFFFFFFFF,
                        format!("({:.0}, {:.0})", tx, ty),
                    );
                }
            }
        });

    id.pop();
}