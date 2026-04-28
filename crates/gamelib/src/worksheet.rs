use color::Srgba8;
use egui::emath::GuiRounding;
use math::{IVec2, Vec2};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::path::Path;

/// Base grid cell size in physical pixels.
const CSZ: f32 = 22.0;

// The workbench acts as a GUI designer in which you can add and organize widgets that control shader parameters.
// The layout is stored in a separate JSON file next to the shader archive.
//

/// Grid rectangle
fn g_rect(x: i32, y: i32, w: i32, h: i32) -> egui::Rect {
    egui::Rect::from_min_size(
        egui::pos2(x as f32 * CSZ, y as f32 * CSZ),
        egui::vec2(w as f32 * CSZ, h as f32 * CSZ),
    )
}

fn edit_cell(ui: &mut egui::Ui) {
    ui.add(egui::Slider::new(&mut 0.5, 0.0..=3.0).text("right"));
    ui.add(egui::Slider::new(&mut 0.5, 0.0..=3.0).text("left"));
    ui.add(egui::Slider::new(&mut 0.5, 0.0..=3.0).text("top"));
    ui.add(egui::Slider::new(&mut 0.5, 0.0..=3.0).text("bottom"));
    if ui.button("OK").clicked() {
        // dismiss
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct CellStyle {
    background: Srgba8,
}

#[derive(Clone, Serialize, Deserialize)]
enum CellData {
    Empty,
    Text(String),
    Number(f64),
}

#[derive(Clone, Serialize, Deserialize)]
enum CellFormat {
    Text,
    Number {
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
    },
}

/*
enum CellInteraction {
    None,
    TextField,
    Slider,
}*/

#[derive(Clone, Serialize, Deserialize)]
struct Cell {
    /// Index into the styles array.
    style: usize,
    /// Data.
    data: CellData,
    /// Format of the cell data, used for editing.
    format: CellFormat,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct BorderStyle {
    width: i32,
    color: Srgba8,
}

#[derive(Clone, Serialize, Deserialize)]
struct Track {
    /// Size of the track (column width or row height) in physical pixels.
    size: u32,
    /// Whether the track is currently collapsed.
    collapsed: bool,
}

#[derive(Clone, Serialize, Deserialize)]
struct Grid {
    pos: IVec2,
    /// Column widths.
    columns: Vec<Track>,
    /// Row heights.
    rows: Vec<Track>,
    cells: Vec<Cell>,

    /// Indexed by row, then column. `h_borders[0]` is top edge of first row.
    h_borders: Vec<Vec<BorderStyle>>,
    /// Indexed by column, then row. `v_borders[0]` is left edge of first column.
    v_borders: Vec<Vec<BorderStyle>>,

    styles: Vec<CellStyle>,
    subgrids: Vec<Subgrid>,

    #[serde(skip)]
    selection: Selection,
}

#[derive(Clone, Default)]
struct Selection {
    col_range: Range<usize>,
    row_range: Range<usize>,
}

fn color_to_egui(c: Srgba8) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(c.r, c.g, c.b, c.a)
}

fn cell_gui(ui: &mut egui::Ui, pos: Vec2, size: Vec2, cell: &mut Cell, styles: &mut [CellStyle]) {
    //let w = self.columns[addr.x as usize].size as f32;
    //let h = self.rows[addr.y as usize].size as f32;

    let rect = egui::Rect::from_min_size(egui::pos2(pos.x, pos.y), egui::vec2(size.x, size.y));
    let resp = ui.allocate_rect(rect, egui::Sense::click());
    let sty = &styles[cell.style];
    let painter = ui.painter();

    painter.rect_filled(resp.rect, 0.0, color_to_egui(sty.background));

    // interaction
    if resp.clicked() {
        edit_cell(ui);
    }
}

impl Grid {
    fn gui(&mut self, ui: &mut egui::Ui) {

        self.paint_borders(ui.painter());

        let mut x_pos = 0;
        let mut y_pos = 0;
        for i_row in 0..self.rows.len() {
            let row = &self.rows[i_row];
            if row.collapsed {
                continue;
            }
            for i_col in 0..self.columns.len() {
                let col = &self.columns[i_col];
                if col.collapsed {
                    continue;
                }
                let addr = IVec2::new(i_col as i32, i_row as i32);
                let i_cell = (addr.y as usize) * self.columns.len() + (addr.x as usize);

                cell_gui(
                    ui,
                    Vec2::new(x_pos as f32, y_pos as f32),
                    Vec2::new(col.size as f32, row.size as f32),
                    &mut self.cells[i_cell],
                    &mut self.styles,
                );
                x_pos += col.size;
            }
            y_pos += row.size;
        }
    }

    fn paint_borders(&self, painter: &egui::Painter) {
        // horizontal borders
        let mut x_pos = 0;
        let mut y_pos = 0;
        let ppp = painter.ctx().pixels_per_point();

        for i_row_line in 0..=self.rows.len() {
            let borders = &self.h_borders[i_row_line];
            for i_col in 0..self.columns.len() {
                let col = &self.columns[i_col];
                if col.collapsed {
                    continue;
                }
                painter.line_segment(
                    [
                        egui::pos2(x_pos as f32, y_pos as f32).round_to_pixel_center(ppp),
                        egui::pos2((x_pos + col.size) as f32, y_pos as f32).round_to_pixel_center(ppp),
                    ],
                    egui::Stroke::new(borders[i_col].width as f32, color_to_egui(borders[i_col].color)),
                );
                x_pos += col.size;
            }
            x_pos = 0;
            if i_row_line < self.rows.len() {
                y_pos += self.rows[i_row_line].size;
            }
        }

        x_pos = 0;
        y_pos = 0;

        // vertical borders
        for i_col_line in 0..=self.columns.len() {
            let borders = &self.v_borders[i_col_line];
            for i_row in 0..self.rows.len() {
                let row = &self.rows[i_row];
                if row.collapsed {
                    continue;
                }
                painter.line_segment(
                    [
                        egui::pos2(x_pos as f32, y_pos as f32).round_to_pixel_center(ppp),
                        egui::pos2(x_pos as f32, (y_pos + row.size) as f32).round_to_pixel_center(ppp),
                    ],
                    egui::Stroke::new(borders[i_row].width as f32, color_to_egui(borders[i_row].color)),
                );
                y_pos += row.size;
            }
            y_pos = 0;
            if i_col_line < self.columns.len() {
                x_pos += self.columns[i_col_line].size;
            }
        }
    }

    pub fn new(column_count: usize, row_count: usize) -> Self {
        let columns = vec![
            Track {
                size: 100,
                collapsed: false
            };
            column_count
        ];
        let rows = vec![
            Track {
                size: 22,
                collapsed: false
            };
            row_count
        ];

        let cells = vec![
            Cell {
                style: 0,
                data: CellData::Text("Hello".to_string()),
                format: CellFormat::Text,
            };
            column_count * row_count
        ];
        let styles = vec![CellStyle {
            background: Srgba8::new(0, 0, 0, 255),
        }];

        let h_borders = vec![
            vec![
                BorderStyle {
                    width: 1,
                    color: Srgba8::new(255, 255, 255, 255),
                };
                column_count
            ];
            row_count + 1
        ];
        let v_borders = vec![
            vec![
                BorderStyle {
                    width: 1,
                    color: Srgba8::new(255, 255, 255, 255),
                };
                row_count
            ];
            column_count + 1
        ];

        Grid {
            pos: IVec2::ZERO,
            columns,
            rows,
            cells,
            h_borders,
            v_borders,
            styles,
            subgrids: vec![],
            selection: Selection::default(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Subgrid {
    columns: Range<usize>,
    rows: Range<usize>,
    grid: Grid,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Worksheet {
    root_grid: Grid,
}

impl Default for Worksheet {
    fn default() -> Self {
        Self::new(100, 100)
    }
}

impl Worksheet {
    pub fn new(column_count: usize, row_count: usize) -> Self {
        Self {
            root_grid: Grid::new(column_count, row_count),
        }
    }

    pub fn gui(&mut self, ui: &mut egui::Ui) {
        self.root_grid.gui(ui);
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let ron_str = ron::to_string(&self).expect("failed to serialize worksheet");
        std::fs::write(path, ron_str)
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let ron_str = std::fs::read_to_string(path)?;
        let worksheet = ron::from_str(&ron_str).expect("failed to deserialize worksheet");
        Ok(worksheet)
    }
}

