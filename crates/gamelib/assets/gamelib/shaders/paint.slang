// Manifest: shaders.toml
import path;
import common;

// NOTE: keep in sync with `paint::renderer::TILE_SIZE`
static const uint TILE_SIZE = 16;

struct GradientIntegralSegment {
    float4 a;
    float4 b;
    float4 c;
    float t_min;
    float t_max;
    uint2 padding;
};

enum GradientExtendMode : uint8_t {
    Clamp = 0,
    Repeat = 1,
    Mirror = 2,
};

struct SolidColorFill {
    uint tag;
    Srgba8 color;
};

struct TextureFill {
    uint tag;
    Texture2D<float4>.Handle texture;
    SamplerState.Handle sampler;
    float3x3 deviceToTexCoord;
    Srgba8 color;
};

struct LinearGradientFill {
    uint tag;
    GradientExtendMode extend_mode;
    float2 start;
    float2 end;
    uint2 seg_range;     // Range into `gradient_segments`.
    float4 integral;    // integral over the whole gradient ramp (t in [0,1])
};

enum FillType : uint {
    SolidColor = 0,
    Texture = 1,
    LinearGradient = 2,
};

struct FillData {
    uint tag;
    uint payload[15];
};

struct TileCover {
    uint64_t id;
    int winding;
    uint fill;
    // TODO: we could store the segments next to the each cover instead of this indirection
    uint seg_offset;
    uint seg_count;

    int2 tile_coord() {
        return int2(int(id >> 32 & 0xFFFF), int(id >> 48 & 0xFFFF));
    }

    uint2 path_id() {
        return uint2(id & 0xFFFFFFFF);
    }
};

struct TileDrawData {
    uint cover_offset;
    uint cover_count;
};

struct RasterizeTilesParams {
    uint8_t4* segments;
    TileCover* covers;
    uint cover_count;
    TileDrawData* tiles;
    FillData* fills;
    GradientIntegralSegment* gradient_segments;
    RWTexture2D<float4>.Handle output;
    uint start_cover_list;
};


float extend_gradient_coord(float t, GradientExtendMode mode) {
    if (mode == GradientExtendMode::Clamp) {
        return saturate(t);
    } else if (mode == GradientExtendMode::Repeat) {
        return t - floor(t);
    } else if (mode == GradientExtendMode::Mirror) {
        return abs(t - 2.0 * round(0.5 * t));
    }
    return t; // should never happen
}

GradientIntegralSegment find_gradient_segment(GradientIntegralSegment* segments, uint2 range, float t) {
    for (uint i_seg = range.x; i_seg < range.y; i_seg++) {
        if (t >= segments[i_seg].t_min && t <= segments[i_seg].t_max) {
            return segments[i_seg];
        }
    }
    return segments[range.x]; // should never happen
}

// Evaluates a linear gradient.
float4 eval_linear_gradient(GradientIntegralSegment* segments, LinearGradientFill fill, float2 pixel_center) {
    let d        = fill.end - fill.start;
    let inv_len2 = 1.0 / dot(d, d);

    // Gradient coordinate at the pixel center.
    let t_center = dot(pixel_center - fill.start, d) * inv_len2;
    let half_extent = 0.5 * (abs(d.x) + abs(d.y)) * inv_len2;

    int period = int(floor(t_center + half_extent)) - int(floor(t_center - half_extent));
    float t_min = extend_gradient_coord(t_center - half_extent, fill.extend_mode);
    float t_max = extend_gradient_coord(t_center + half_extent, fill.extend_mode);

    //t_max = max(t_max, t_min + 1e-3); // avoid numerical issues when t_min == t_max
    //t_min = min(t_min, t_max - 1e-3);

    let left = find_gradient_segment(segments, fill.seg_range, t_min);
    let right = find_gradient_segment(segments, fill.seg_range, t_max);

    let color = (period * fill.integral + t_max * (right.a * t_max + right.b) + right.c - (t_min * (left.a * t_min + left.b) + left.c)) / (float(period) + t_max - t_min);
    return color; //float4(t_center, t_center, t_center, 1.0);
}

void unpack_segment(uint8_t4 segment, out float4 decoded, out uint left_cross) {
    decoded = float4(segment & uint8_t4(0x7F)) / 127.0 * TILE_SIZE;
    left_cross = (segment.x & 0x80) >> 7 | (segment.y & 0x80) >> 6;
}

// Winding numbers along the left edge of the tile.
// First filled with winding number delta, the absolute values are calculated with a prefix sum in a second step.
groupshared int s_winding_v[TILE_SIZE];

// Fractional winding numbers on every pixel of the left edge of the tile, quantized to [0,255].
groupshared int s_winding_v_area[TILE_SIZE];

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void rasterize_tiles(uint3 workgroup_id: SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uniform RasterizeTilesParams* u)
{
    // One workgroup is dispatched per `TileDrawData` in u.tiles.
    // There is one `TileDrawData` per spatial tile to be drawn (there are no overlapping `TileDrawData`).
    // Each `TileDrawData` points to a list of `TileCover`s in `u.covers`.
    // Each `TileCover` represents a path that covers the tile, and points to a list of segments in `u.segments` that intersect the tile
    // and contribute to the coverage calculation.

    var tile = u.tiles[workgroup_id.x + u.start_cover_list];
    let tile_coord = u.covers[tile.cover_offset].tile_coord();    // all covers for the tile should have the same tile coords

    let tsize = int2(TILE_SIZE, TILE_SIZE);
    let local     = int2(group_thread_id.xy);               // coordinates within the tile
    let tile_top_left = tile_coord * tsize;                      // tile top-left corner position in screen coordinates
    let pixel_center  = float2(local) + 0.5;  // pixel center in tile-local coordinates
    let pixel_center_screen = tile_top_left + pixel_center; // pixel center in screen coordinates

    var color = float4(0.0);

    // for each tile cover
    for (uint i_cover = tile.cover_offset; i_cover < tile.cover_offset + tile.cover_count; i_cover++) {

        var tile_cover = u.covers[i_cover];

        //----------------------------------------------------------------------------
        // 1. compute winding numbers on the left edge of the tile

        // 1.1 clear shared memory
        //
        // parallel over all entries in the array
        //
        let local_index = group_thread_id.y * TILE_SIZE + group_thread_id.x;
        if (local_index < TILE_SIZE) {
            s_winding_v_area[local_index] = 0;
            s_winding_v[local_index] = 0;
        }

        GroupMemoryBarrierWithGroupSync();

        // 1.2 each thread processes one segment and writes the winding delta if the segment
        //     crosses the left edge of the tile
        //
        // parallel over all segments of the cover
        for (uint i_seg = 0; i_seg < tile_cover.seg_count; i_seg += TILE_SIZE*TILE_SIZE) {
            if (i_seg + local_index < tile_cover.seg_count) {
                float4 seg;
                uint left_cross;
                unpack_segment(u.segments[tile_cover.seg_offset + i_seg + local_index], seg, left_cross);

                if (left_cross != 0) {
                    // crossing left to right: -1, crossing right to left: +1
                    // ... or maybe the opposite, idk
                    let sign = left_cross == 2 ? -1 : 1;
                    let y = sign < 0 ? seg.w : seg.y;       // sign < 0 ? b.y : a.y
                    let yi = int(floor(y)); // pixel row index of intersection
                    let yf = 1 - (y - yi);        // 1 - fractional part
                    let yf_quant = int(yf * 255.0); // quantized to 0..255
                    InterlockedAdd(s_winding_v[yi], sign);
                    InterlockedAdd(s_winding_v_area[yi], sign * yf_quant);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // 1.3. compute absolute winding numbers with prefix sum
        // this can be done with subgroup ops as long as TILE_SIZE <= wave size
        //
        // parallel over all entries in the array
        //
        if (local_index < TILE_SIZE) {
             int winding = s_winding_v[local_index];
             winding = WavePrefixSum(winding);

             s_winding_v[local_index] = winding;
        }

        GroupMemoryBarrierWithGroupSync();

        //----------------------------------------------------------------------------
        // 2. compute coverage for every pixel in the tile
        //
        // parallel over all pixels in the tile
        //

        // base winding number at tile-local coordinate (0, y)
        float winding = (float) tile_cover.winding + s_winding_v[local.y] + float(s_winding_v_area[local.y]) / 255.0;

        for (uint i = 0; i < tile_cover.seg_count; i++) {
            uint8_t4 rawseg = u.segments[tile_cover.seg_offset + i];
            float4 seg;
            uint left_cross;
            unpack_segment(rawseg, seg, left_cross);
            float2 a = seg.xy - float2(local);
            float2 b = seg.zw - float2(local);
            // The line below produces very weird behavior when the line is horizontal (d.x == 0),
            // as if the both the d.x == 0 and d.x != 0 branches are both taken.
            // For some reason this behavior disappears if we access b.x in the d.x != 0 branch.
            // This seems like a driver issue.
            //float2 d = seg.zw - seg.xy;
            float2 d = float2(int2(rawseg.zw & uint8_t2(0x7F)) - int2(rawseg.xy & uint8_t2(0x7F))) / 127.0 * TILE_SIZE;

            if (d.y == 0.0) {
                // horizontal segments don't contribute anything to the winding
                continue;
            }

            if (d.x == 0.0) {
                // special-case for vertical segments
                float y_min = saturate(min(a.y, b.y));
                float y_max = saturate(max(a.y, b.y));
                float x = a.x;
                winding += sign(-d.y) * abs(y_max - y_min) * saturate(1.0 - x);
                continue;
            }

            // General case: compute pixel coverage.
            // The derivation is a bit long to write down here without pictures.
            // Very roughly, this is the pixel area between the min/max y covered by a segment,
            // adjusted by adding or subtracting the areas of triangles formed by the intersection of the segment
            // and the pixel bounds.

            // FIXME: for near-vertical segments this might be unstable
            float y_0 = a.y - a.x * d.y / d.x;
            float y_1 = a.y + (1.0 - a.x) * d.y / d.x;
            float y_min = saturate(min(a.y, b.y));
            float y_max = saturate(max(a.y, b.y));
            float x_0 = a.x + (y_min - a.y) / d.y * d.x;
            float x_1 = a.x + (y_max - a.y) / d.y * d.x;
            float a_1 = abs(y_min - y_0) * max(0.0, x_0);
            float a_2 = abs(y_max - y_0) * max(0.0, x_1);
            float a_3 = abs(y_min - y_1) * max(0.0, x_0 - 1.0);
            float a_4 = abs(y_max - y_1) * max(0.0, x_1 - 1.0);
            float area = (y_max - y_min) + 0.5 * (abs(a_3 - a_4) - abs(a_1 - a_2));
            winding += area * sign(-d.y);
        }

        //----------------------------------------------------------------------------
        // 3. evaluate fill

        float4 fill_color;
        uint fill_index = tile_cover.fill;
        switch (u.fills[fill_index].tag & 0x7FFFFFFF) {
            case FillType.SolidColor: {
                fill_color = srgba8_to_float4(reinterpret<SolidColorFill>(u.fills[fill_index]).color);
                break;
            }
            case FillType.Texture: {
                TextureFill fill = reinterpret<TextureFill>(u.fills[fill_index]);
                float2 uv = mul(fill.deviceToTexCoord, float3(pixel_center_screen, 1.0)).xy;
                fill_color = fill.texture.SampleLevel(fill.sampler, uv, 0) * srgba8_to_float4(fill.color);
                break;
            }
            case FillType.LinearGradient: {
                fill_color = eval_linear_gradient(u.gradient_segments, reinterpret<LinearGradientFill>(u.fills[fill_index]), pixel_center_screen);
            }
        }

        //----------------------------------------------------------------------------
        // 4. blend
        //
        // parallel over all pixels in the tile
        //
        float coverage = abs(winding);
        fill_color.rgb *= fill_color.a;
        float4 cover_color = fill_color * coverage;
        color = cover_color + color * (1.0 - cover_color.a);
    }


    //----------------------------------------------------------------------------
    // 5. final blend with render target


    // Load background color & premultiply.
    let coord = tile_coord * tsize + local;
    var cb = u.output.Load(coord);
    cb.rgb *= cb.a;

    // Blend with background color.
    // TODO: make that configurable
    var co = color + cb * (1.0 - color.a);

    // unpremultiply alpha
    if (co.a > 0.0) {
        co.rgb /= co.a;
    }

    // Write back final color.
    u.output.Store(coord, co);
}



//--------------------------------------------------------------------------------------------------

struct CopyToScreenParams {
    RWTexture2D<float4>.Handle texture;
};

[shader("vertex")]
ScreenQuadVSOut copy_to_screen_vertex(uint vertex_id : SV_VertexID)
{
    return screen_quad_vertex_shader(vertex_id);
}

[shader("fragment")]
float4 copy_to_screen_fragment(ScreenQuadVSOut v, float2 frag_coords: SV_Position, uniform CopyToScreenParams* u)
{
    return u.texture.Load(int2(frag_coords));
}











