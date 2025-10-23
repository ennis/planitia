use crate::util::make_unique_path;
use log::{error, info};
use serde_json::json;
use std::path::Path;
use gpu::vk;

pub struct PipelineEditor {
    file_name: String,
    vertex_entry_point: String,
    fragment_entry_point: String,
    depth_test_enable: bool,
    depth_write_enable: bool,
    depth_compare_op: vk::CompareOp,
    color_formats: Vec<gpu::Format>,
}

impl Default for PipelineEditor {
    fn default() -> Self {
        Self {
            file_name: "new_pipeline".to_string(),
            vertex_entry_point: "vertex_main".to_string(),
            fragment_entry_point: "fragment_main".to_string(),
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare_op: Default::default(),
            color_formats: vec![gpu::Format::R8G8B8A8_UNORM],
        }
    }
}

impl PipelineEditor {
    fn save(&self) {
        let path = Path::new("crates/game/shaders")
            .join(&self.file_name)
            .with_extension("json");
        let path = make_unique_path(&path);

        // serialize to json
        let json = json!({
            "vertex_entry_point": self.vertex_entry_point,
            "fragment_entry_point": self.fragment_entry_point,
            "depth_test_enable": self.depth_test_enable,
            "depth_write_enable": self.depth_write_enable,
            "color_formats": self.color_formats.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>(),
        });

        let json_str = serde_json::to_string_pretty(&json).unwrap();
        match std::fs::write(&path, json_str) {
            Ok(_) => {
                info!("Pipeline configuration saved to {}", path.display());
            }
            Err(e) => {
                error!("Failed to save pipeline configuration to {}: {}", path.display(), e);
            }
        }
    }

    pub fn show_gui(&mut self, ctx: &egui::Context) {

        fn prop_grid(ui: &mut egui::Ui, heading: &str, contents: impl FnOnce(&mut egui::Ui)) {
            //ui.collapsing(heading, |ui| {
                egui::Grid::new(heading)
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, contents);
            //});
        }

        egui::Window::new("Pipeline Editor").show(ctx, |ui| {
            prop_grid(ui, "", |ui| {
                self.edit_vertex_properties(ui);
                self.edit_fragment_properties(ui);
                self.edit_output_properties(ui);
            });

            // Load/Save/Clone
            ui.separator();
            if ui.button("Save").clicked() {
                self.save();
            }
        });
    }

    fn edit_vertex_properties(&mut self, ui: &mut egui::Ui) {
        // Vertex Stage
        ui.label("Vertex Entry Point");
        ui.text_edit_singleline(&mut self.vertex_entry_point);
        ui.end_row();
    }

    fn edit_fragment_properties(&mut self, ui: &mut egui::Ui) {

        ui.label("Fragment Entry Point");
        ui.text_edit_singleline(&mut self.fragment_entry_point);
        ui.end_row();

        // Depth Stencil State
        ui.label("Depth Test Enable");
        ui.checkbox(&mut self.depth_test_enable, "");
        ui.end_row();

        ui.label("Depth Write Enable");
        ui.checkbox(&mut self.depth_write_enable, "");
        ui.end_row();

        // Depth comparison
        ui.add_enabled_ui(self.depth_test_enable, |ui| {
            ui.label("Depth Compare Op");

        });
        ui.add_enabled_ui(self.depth_test_enable, |ui| {
            egui::ComboBox::new("depth_compare_op", "")
                .selected_text(format!("{:?}", self.depth_compare_op))
                .show_ui(ui, |ui| {
                    for &op in &[
                        vk::CompareOp::NEVER,
                        vk::CompareOp::LESS,
                        vk::CompareOp::EQUAL,
                        vk::CompareOp::LESS_OR_EQUAL,
                        vk::CompareOp::GREATER,
                        vk::CompareOp::NOT_EQUAL,
                        vk::CompareOp::GREATER_OR_EQUAL,
                        vk::CompareOp::ALWAYS,
                    ] {
                        ui.selectable_value(&mut self.depth_compare_op, op, format!("{:?}", op));
                    }
                });
        });
        ui.end_row();
    }

    fn edit_output_properties(&mut self, ui: &mut egui::Ui) {
        // Output Stage

        ui.label("Output Color Format");
        egui::ComboBox::from_label("Color Format").selected_text(format!("{:?}", self.color_formats[0])).show_ui(ui, |ui| {
            for &format in &[
                gpu::Format::R8G8B8A8_UNORM,
                gpu::Format::R16G16B16A16_SFLOAT,
                gpu::Format::R32G32B32A32_SFLOAT,
            ] {
                ui.selectable_value(&mut self.color_formats[0], format, format!("{:?}", format));
            }
        });

        ui.end_row();
    }
}
