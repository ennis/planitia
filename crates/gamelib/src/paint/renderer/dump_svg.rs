use crate::paint::renderer::{GpuSceneData, TILE_SIZE};
use color::Srgba8;
use math::{Rect, vec2};

impl GpuSceneData {
    /// Writes the prepared scene to a SVG file for debugging purposes.
    pub(super) fn write_svg<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        let bbox = self.covers.iter().fold(Rect::ZERO, |bbox, b| {
            let tile_coords = b.tile_coords();
            let tile_rect = Rect::from_origin_size(
                vec2(tile_coords.x as f32 * TILE_SIZE as f32, tile_coords.y as f32 * TILE_SIZE as f32),
                vec2(TILE_SIZE as f32, TILE_SIZE as f32),
            );
            bbox.union(&tile_rect)
        });
        writeln!(file, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">"#, bbox.max.x, bbox.max.y)?;

        // write one group per tile
        // the group transform is the tile's coordinates multiplied by the tile size
        // paint a red rect for tiles with non-zero winding, and a black line for each segment in the tile
        for b in self.covers.iter() {
            let tile_coords = b.tile_coords();
            writeln!(
                file,
                r#"<g transform="translate({} {})">"#,
                tile_coords.x * TILE_SIZE as i32,
                tile_coords.y * TILE_SIZE as i32
            )?;
            if b.winding > 0 {
                writeln!(
                    file,
                    r#"<rect x="0" y="0" width="{}" height="{}" fill="red" fill-opacity="0.5"/>"#,
                    TILE_SIZE, TILE_SIZE
                )?;
            } else if b.winding < 0 {
                writeln!(
                    file,
                    r#"<rect x="0" y="0" width="{}" height="{}" fill="blue" fill-opacity="0.5"/>"#,
                    TILE_SIZE, TILE_SIZE
                )?;
            }
            // show tile bounds in light gray
            writeln!(
                file,
                r#"<rect x="0" y="0" width="{}" height="{}" fill="none" stroke="lightgray"/>"#,
                TILE_SIZE, TILE_SIZE
            )?;
            for seg in &self.clip_segs[b.seg_offset as usize..(b.seg_offset + b.seg_count) as usize] {
                // colorize segments by path ID
                let path_id = b.path();
                let color = Srgba8::from_hsla(path_id as f32, 0.5, 0.5, 1.0);

                let x1 = (seg.x & 0x7F) as f32 / 127.0 * TILE_SIZE as f32;
                let y1 = (seg.y & 0x7F) as f32 / 127.0 * TILE_SIZE as f32;
                let x2 = (seg.z & 0x7F) as f32 / 127.0 * TILE_SIZE as f32;
                let y2 = (seg.w & 0x7F) as f32 / 127.0 * TILE_SIZE as f32;

                writeln!(
                    file,
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="rgb({} {} {})"/>"#,
                    x1, y1, x2, y2, color.r, color.g, color.b
                )?;
            }
            writeln!(file, "</g>")?;
        }

        writeln!(file, "</svg>")?;
        Ok(())
    }
}
