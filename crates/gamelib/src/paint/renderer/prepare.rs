//! Preparation of Scenes for rendering by the GPU.

use crate::paint::PathSlice;
use crate::paint::fill::Fill;
use crate::paint::flatten::flatten_path;
use crate::paint::renderer::{
    DrawCommand, FillData, PreparedSceneData, RASTER_TILE_SIZE, Scene, TileCover, TileCoverList, pack_tile_cover_id,
};
use color::Srgba8;
use math::{Mat3, Rect, U8Vec2, U8Vec4, UVec2, Vec2, ivec2, u8vec2, u8vec4, vec2};
use std::ops::Range;

/// Prepares the given scene for rendering.
///
/// This does multiple things:
/// - flattens the paths in the scene to line segments
/// - builds a list of tiles for each path, and for each tile, builds a list of clipped line segments
///   that intersect the tile
/// - computes tile-level winding numbers
/// - builds `TileCoverLists` for each tile which can be interpreted by the
///   rasterization shader
pub(super) fn prepare_scene(scene: &Scene, viewport: UVec2) -> PreparedSceneData {
    let _span = tracy_client::span!("prepare_scene");

    const TILE: i32 = RASTER_TILE_SIZE as i32;

    let viewport_tile_size = ivec2(viewport.x.div_ceil(TILE as u32) as i32, viewport.y.div_ceil(TILE as u32) as i32);

    // convert fills to GPU format
    let fills: Vec<FillData> = scene
        .fills
        .iter()
        .map(|f| match f {
            Fill::Solid(color) => FillData { color: *color },
            Fill::Texture { .. } => FillData { color: Srgba8::WHITE }, // TODO handle textured fills
        })
        .collect();

    // stats
    let mut culled_viewport_tiles = 0;
    let mut culled_fully_covered_tiles = 0;

    let mut covers: Vec<TileCover> = vec![];
    let mut clip_segs: Vec<U8Vec4> = vec![];

    let mut cur_transform = Mat3::IDENTITY;
    // temp buffers for holding flattening results for each path
    let mut flattened_vertices: Vec<Vec2> = vec![];
    let mut contours: Vec<Range<usize>> = vec![];
    let mut path_index = 0;

    #[derive(Copy, Clone, Debug)]
    struct Tile {
        x: i32,
        y: i32,
        a: U8Vec2,
        b: U8Vec2,
        winding: i32,
        path: u32,
    }
    let mut tiles: Vec<Tile> = vec![];

    for cmd in scene.draw_commands.iter() {
        let (verb_range, base_vertex, fill) = match cmd {
            DrawCommand::FillPath { verb_range, base_vertex, fill } => (verb_range.clone(), *base_vertex, *fill as u32),
            DrawCommand::SetTransform(transform) => {
                cur_transform = *transform;
                continue;
            }
        };

        // flatten path to segments
        {
            let _span = tracy_client::span!("flatten_path");
            flatten_path(
                PathSlice { verbs: &scene.path_verbs[verb_range], points: &scene.path_points[base_vertex..] },
                &cur_transform,
                1.0,
                &mut flattened_vertices,
                &mut contours,
            );
        }

        for c in contours.iter() {
            let range = c.start as usize..c.end as usize;

            {
                let _span = tracy_client::span!("contour");
                for [p, q] in flattened_vertices[range].array_windows() {
                    // quantize coordinates to 1/128th of a tile
                    //let quant = TILE as f32 / 128.0;
                    //let quantize_in = |v: f32| (v / quant as f32).round() * quant as f32;

                    //let p = vec2(quantize_in(p_orig.x), quantize_in(p_orig.y));
                    //let q = vec2(quantize_in(q_orig.x), quantize_in(q_orig.y));

                    // coarse conservative rasterization of line segment pq
                    let d = q - p;
                    let dxi = d.x.signum() as i32;
                    let dyi = d.y.signum() as i32;

                    let p_t_x = (p.x / TILE as f32).floor() as i32;
                    //let q_t_x = q.x as i32 / TILE;
                    let p_t_y = (p.y / TILE as f32).floor() as i32;
                    let q_t_y = (q.y / TILE as f32).floor() as i32;

                    let quantize = |v: f32| (v / TILE as f32 * 127.0f32).round() as u8;

                    let mut entry_t_x = p_t_x; // current row entry tile x coord
                    let mut entry_loc_x = p.x - (entry_t_x * TILE) as f32; // current row entry x coord in tile coord space
                    //let mut row_entry_x_quant = quantize(row_entry_x);
                    //let mut row_entry_y = p.y - (p_t_y * TILE) as f32; // current row entry y coord in tile coord space

                    let mut t_y = p_t_y;
                    while t_y != q_t_y + dyi {
                        // tile-local Y coordinate of exit of segment
                        //let row_exit_y = ((dyi.max(0) * TILE) as f32).min(q.y);

                        let y_1 = ((t_y + dyi.max(0)) * TILE) as f32;
                        // solve intersection of the segment with the bottom or top edge of the tile row
                        let t = (y_1 - p.y) / d.y;

                        let isect = if t_y != q_t_y { p.x + t * d.x } else { q.x };
                        // x coord of tile that contains the intersection, aka the "exit tile" for the current row
                        let exit_t_x = (isect / TILE as f32).floor() as i32;

                        debug_assert!((exit_t_x >= entry_t_x && d.x >= 0.0) || (exit_t_x <= entry_t_x && d.x <= 0.0));

                        let exit_loc_x = isect - (exit_t_x * TILE) as f32;
                        let mut t_x = entry_t_x;
                        while t_x != exit_t_x + dxi {
                            // compute winding number delta given segment entry and exits from the top edge
                            // If the segment enters from (0,0) we don't count that as an entry or exit
                            // and this won't contribute to the winding number. However, the vertical winding number
                            // calculation inside the shader will pick it up and add the contribution properly.

                            // entry from the top edge if
                            // - going down (dyi > 0)
                            // - and we're at the entry tile of the row (t_x == row_t_x)
                            // - and not entering from (0,0)
                            let top_in = if t_x == entry_t_x && t_y != p_t_y && dyi > 0 { -1 } else { 0 };
                            // exit through the top edge:
                            // - going up (dyi < 0)
                            // - and we're at the intersection tile (t_x == isc_t_x)
                            // - and not exiting through (0,0)
                            let top_out = if t_x == exit_t_x && t_y != q_t_y && dyi < 0 { 1 } else { 0 };
                            let delta_cov = top_in + top_out /* + right_in + left_out*/;

                            let left_in = t_x != entry_t_x && dxi > 0;
                            let left_out = t_x != exit_t_x && dxi < 0;

                            //eprintln!("segment {} (path {}) intersects tile ({}, {}) isc_t_x={}, row_t_x={}, t_y={}, p_t_y={}, q_t_y={}, d=({:.1}, {:.1}), dxi={}/dyi={}",
                            //          i_segment, c.path, t_x, t_y, isc_t_x, row_t_x, t_y, p_t_y, q_t_y, d.x, d.y, dxi, dyi
                            //);

                            // clip pq to tile
                            let p_clip_x = if t_x == entry_t_x { entry_loc_x } else { ((-dxi).max(0) * TILE) as f32 };
                            let q_clip_x = if t_x == exit_t_x { exit_loc_x } else { (dxi.max(0) * TILE) as f32 };
                            // Local Y of the segment's entry point into this tile row:
                            // - first row: the start point p itself
                            // - going down: segment entered from the top edge (y=0)
                            // - going up:   segment entered from the bottom edge (y=TILE)
                            let entry_y_local = if t_y == p_t_y {
                                p.y - (t_y * TILE) as f32
                            } else if dyi > 0 {
                                0.0
                            } else {
                                TILE as f32
                            };
                            let p_clip_y;
                            let q_clip_y;
                            if exit_t_x == entry_t_x {
                                p_clip_y = (p.y - (t_y * TILE) as f32).clamp(0.0, TILE as f32);
                                q_clip_y = (q.y - (t_y * TILE) as f32).clamp(0.0, TILE as f32);
                            } else {
                                p_clip_y = (entry_y_local
                                    + d.y / d.x * (p_clip_x + ((t_x - entry_t_x) * TILE) as f32 - entry_loc_x))
                                    .clamp(0.0, TILE as f32);
                                q_clip_y = (entry_y_local
                                    + d.y / d.x * (q_clip_x + ((t_x - entry_t_x) * TILE) as f32 - entry_loc_x))
                                    .clamp(0.0, TILE as f32);
                            }
                            //eprintln!("  tile ({}, {}), delta_cov={}, p_clip=({:.1}, {:.1}), q_clip=({:.1}, {:.1}) row_loc_x={row_loc_x}", t_x, t_y, delta_cov, p_clip_x, p_clip_y, q_clip_x, q_clip_y);

                            //debug_assert!(p_clip_x >= 0.0 && p_clip_x <= TILE as f32, "p_clip_x={}", p_clip_x);
                            //debug_assert!(q_clip_x >= 0.0 && q_clip_x <= TILE as f32, "q_clip_x={}", q_clip_x);
                            //debug_assert!(p_clip_y >= 0.0 && p_clip_y <= TILE as f32, "p_clip_y={}", p_clip_y);
                            //debug_assert!(q_clip_y >= 0.0 && q_clip_y <= TILE as f32, "q_clip_y={}", q_clip_y);

                            let mut p_clip_x_q = quantize(p_clip_x);
                            let mut p_clip_y_q = quantize(p_clip_y);
                            p_clip_x_q |= if left_in { 0x80 } else { 0 };
                            p_clip_y_q |= if left_out { 0x80 } else { 0 };
                            let q_clip_x_q = quantize(q_clip_x);
                            let q_clip_y_q = quantize(q_clip_y);

                            tiles.push(Tile {
                                x: t_x,
                                y: t_y,
                                a: u8vec2(p_clip_x_q, p_clip_y_q),
                                b: u8vec2(q_clip_x_q, q_clip_y_q),
                                winding: delta_cov,
                                path: path_index,
                            });

                            t_x += dxi;
                        }

                        entry_t_x = exit_t_x;
                        entry_loc_x = exit_loc_x;
                        t_y += dyi;
                    }
                }
            }
        }

        if tiles.is_empty() {
            continue;
        }

        {
            let _span = tracy_client::span!("scanline_sort");
            // sort newly created tiles of the current contour in scanline order
            tiles.sort_by(|a, b| a.y.cmp(&b.y).then(a.x.cmp(&b.x)));
        }

        {
            let _span = tracy_client::span!("merge_into_covers");
            let mut x = tiles[0].x;
            let mut y = tiles[0].y;
            let mut i = 0;
            let mut winding = 0; // winding for the tile to the right
            let mut cover = TileCover {
                id: pack_tile_cover_id(x, y, tiles[0].path),
                winding: 0,
                fill,
                seg_offset: clip_segs.len() as u32,
                seg_count: 0,
            };

            let mut cull_and_push_cover = |cover: TileCover| {
                let tile_coords = cover.tile_coords();
                if tile_coords.x < 0
                    || tile_coords.x >= viewport_tile_size.x
                    || tile_coords.y < 0
                    || tile_coords.y >= viewport_tile_size.y
                {
                    // skip tiles outside the viewport
                    culled_viewport_tiles += 1;
                    return;
                }
                covers.push(cover);
            };

            while i < tiles.len() {
                let tile = tiles[i];

                if tile.y == y && tile.x != x {
                    // end current bin
                    cull_and_push_cover(cover);
                    cover.winding = winding;
                    cover.seg_count = 0;
                    if winding != 0 {
                        // emit empty bins for skipped tiles if we are inside the shape
                        for xi in x + 1..tile.x {
                            cover.id = pack_tile_cover_id(xi, tile.y, tile.path);
                            cull_and_push_cover(cover);
                        }
                    }
                    cover.id = pack_tile_cover_id(tile.x, tile.y, tile.path);
                    cover.seg_offset = clip_segs.len() as u32;
                } else if tile.y != y {
                    // new row
                    // end current bin
                    cull_and_push_cover(cover);
                    cover.id = pack_tile_cover_id(tile.x, tile.y, tile.path);
                    cover.winding = 0;
                    cover.seg_offset = clip_segs.len() as u32;
                    cover.seg_count = 0;
                }

                clip_segs.push(u8vec4(tile.a.x, tile.a.y, tile.b.x, tile.b.y));
                x = tile.x;
                y = tile.y;
                i += 1;
                cover.seg_count += 1;
                winding += tile.winding;
            }

            if cover.seg_count > 0 {
                cull_and_push_cover(cover);
            }
        }

        flattened_vertices.clear();
        contours.clear();
        tiles.clear();
        path_index += 1;
    }

    if covers.is_empty() {
        // possibly all cover tiles have been culled
        return PreparedSceneData { clip_segs, tile_covers: covers, tile_cover_lists: vec![], fills };
    }

    // sort covers by row, column, path
    {
        let _span = tracy_client::span!("cover_sort");
        covers.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    }

    let mut cover_lists: Vec<TileCoverList> = vec![];

    {
        let _span = tracy_client::span!("cover_commands");
        let mut cur_coords = covers[0].tile_coords();
        let mut offset = 0;
        for (i, b) in covers.iter().enumerate() {
            let coords = b.tile_coords();
            if cur_coords != coords {
                cover_lists.push(TileCoverList { offset, count: i as u32 - offset });
                offset = i as u32;
            } else {
                // if the cover is fully opaque and has a non-zero winding, and doesn't have any segment,
                // then it fully covers the previous cover tiles, and we can trim the previous covers
                if b.winding != 0 && b.seg_count == 0 {
                    if fills[b.fill as usize].color.a == 255 {
                        culled_fully_covered_tiles += i as u32 - offset;
                        offset = i as u32;
                    }
                }
            }
            cur_coords = coords;
        }
        if offset < covers.len() as u32 {
            cover_lists.push(TileCoverList { offset, count: covers.len() as u32 - offset });
        }
    }

    //let total_tiles_to_render = covers.len() as u32 - culled_fully_covered_tiles;

    //info!("{} cover tiles to render", total_tiles_to_render);
    //info!("culled {} fully covered tiles", culled_fully_covered_tiles);
    //info!("culled {} tiles outside viewport", culled_viewport_tiles);

    PreparedSceneData { clip_segs, tile_covers: covers, tile_cover_lists: cover_lists, fills }
}
