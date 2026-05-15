import path;

// FIXME
[__AttributeUsage(_AttributeTargets.Function)]
struct passAttribute
{
    string pass;
}


typedef uint8_t4 Srgba8;

float4 srgba8_to_float4(Srgba8 srgba8) {
    return float4(srgba8 / 255.0);
}

// ---------------------------------------------------------------
struct FeatherVertex
{
    float2 position;
    float feather;
};

//---------------------------------------------------------------

struct VertexOutput
{
    float4 pos: SV_Position;
    float2 device_pos;
    float2 uv;
    float feather;
};

struct FragmentOutput
{
    float4 color;
};

struct RootParams
{
    float3x3 device_to_uv_transform;
    float2 screen_size;
    float line_width;
    Srgba8 color;
    Texture2D<float4>.Handle texture;
    SamplerState.Handle sampler;
    uint use_texture;
};

//---------------------------------------------------------------

[shader("vertex")]
VertexOutput paint_vertex_main(FeatherVertex v, uniform RootParams* u)
{
    VertexOutput o;
    o.pos = float4(v.position / u.screen_size * 2.0 - 1.0, 0.0, 1.0);
    o.device_pos = v.position.xy;
    o.feather = v.feather;
    o.uv = mul(u.device_to_uv_transform, float3(v.position.xy, 1.0)).xy;
    return o;
}

[shader("fragment")]
FragmentOutput paint_fragment_main(VertexOutput v, uniform RootParams* u)
{
    float alpha = (1.0 - abs(v.feather)) * min(u.line_width, 1.0);
    float4 tex_color = u.texture.Sample(u.sampler, v.uv);
    FragmentOutput o;
    o.color = srgba8_to_float4(u.color) * tex_color;
    o.color.a *= alpha;
    return o;
}

//----------------------------------------------------------------------------------------
/*
struct FlattenRootParams {
    PathBuf path;

    // output microline buffer
    float2 *microline_buffer;
    uint microline_count;
};

static const uint FLATTEN_WORKGROUP_SIZE = 64;

// Flatten paths into line segments ("microlines") that fit into a tile.
[shader("compute")]
[numthreads(FLATTEN_WORKGROUP_SIZE, 1, 1)]
void flatten_paths(uint3 dispatch_thread_id : SV_DispatchThreadID, uniform FlattenRootParams* u)
{
    uint seg_index = dispatch_thread_id.x;
    if (seg_index >= u.path.segment_count) {
        return;
    }

    let segment = u.path.segments[seg_index];


    switch (segment.kind) {
        case PathSegKind.Line: {
            // Just copy the line segment as-is.
            uint vertex_index = segment.start_vertex;
            float2 p0 = u.path.vertices[vertex_index];
            float2 p1 = u.path.vertices[vertex_index + 1];
            uint out_index = atomicAdd(u.microline_count, 2);
            u.microline_buffer[out_index] = p0;
            u.microline_buffer[out_index + 1] = p1;
            break;
        }
        case PathSegKind.Quad: {
        }
    };
}*/

struct RasterLinesParams {
    float2* vertices;
    uint start_vertex;
    uint vertex_count;
    RWTexture2D<float4>.Handle delta_coverage_output;
}

static const uint RASTER_LINE_WORKGROUP_SIZE = 64;

[shader("compute")]
[pass("raster_lines")]
[numthreads(RASTER_LINE_WORKGROUP_SIZE, 1, 1)]
void raster_lines(uint3 dispatch_thread_id : SV_DispatchThreadID, uniform RasterLinesParams* u)
{
    let line = dispatch_thread_id.x;
    if (line >= max(0, u.vertex_count)) {
        return;
    }

    let v1 = u.vertices[line];
    let v2 = u.vertices[line + 1];

    float2 p = v1;
    float2 q = v2;
    if (p.x > q.x) {
        // swap so that q is always to the right; this simplifies iterations along scanlines
        // this also flips the winding
        p = v2;
        q = v1;
    }

    let d = q - p;

    // calculate first & last edge coordinates
    int j_start;
    int j_end;
    if (p.y < q.y) {
        // line going down (y increasing)
        j_start = int(floor(p.y));
        j_end = int(ceil(q.y));
    } else {
        // line going up (y decreasing)
        j_start = int(ceil(p.y));
        j_end = int(floor(q.y));
    }

    let dyi = p.y < q.y ? 1 : -1;

    // NOTE: I don't expect this to be particularly efficient since there's very little coherence between
    //       threads (segments).
    //       A possible solution to increase coherence would be to sort the lines by their heights, so that
    //       threads in the same wave are likely to have the same number of iterations for the loop in the y axis.
    //       We'd also need to process the coverage of horizontal pixel rows in a separate pass.
    var prev_x = p.x;
    // for each pixel row covered by the line
    for (int j = j_start; j != j_end; j += dyi) {
        let j_1 = j + dyi;
        // solve intersection of the line with the next edge of the tile row
        let t = (d.y == 0) ? 1e6 : (float(j_1) - p.y) / d.y;
        // we know that there must be a crossing and thus t in [0,1]
        // unless this is the last row
        let x_clip = min(p.x + t * d.x, q.x);


        // the horizontal range of pixels covered by the line
        let row_i_start = int(prev_x);
        let row_i_end = int(x_clip);

        var x_cur = prev_x;
        // iteratively compute area and delta coverage for each pixel in row
        // TODO: instead of doing this inline, write the pixel row into shared memory,
        //       then have threads process pixel rows in parallel
        u.delta_coverage_output.Store(uint2(3, j), float4(row_i_start, row_i_end, 0.0, 1.0));
        for (int i = row_i_start; i <= row_i_end; i += 1) {
            let x_right = floor(x_cur) + 1.0;
            let x_next = min(x_right, x_clip);
            let dx = x_next - x_cur;
            let dy = d.y/d.x * dx;
            let area = (x_right - x_cur + x_right - x_next) * dy * 0.5;
            let delta_coverage = dy;
            // write area and delta coverage
            u.delta_coverage_output.Store(uint2(i, j), float4(-area, -delta_coverage, 1.0, 1.0));
            x_cur = x_right;
        }

        prev_x = x_clip;
    }

    // for each scanline: spawn a workgroup that processes the row of pixels
    //    - load groups of 32 (subgroup size) pixels on the row, prefix sum coverage, write the result
    //

    // issue: multiple paths?
    // segments belonging to different paths can't be rasterized in the same texture since
    // there's no way to differentiate their coverage delta.
    // Instead, for each segment:
    // - assign segment to a tile (x,y,path_id)
    // - allocate tile data (pixels) somewhere
    // - rasterize the segment into this tile
    // -> segments at the same tile pos but different path_id are rasterized in different tiles
    //
    // sort produced tiles by (y,x,path_id)
    // for each tile y:
    //   go through tiles from min_x to max_x (extent of tiles on the row):
    //   if tile totally covers the row:
    //      set min_path_id to tile.path_id
    //      subsequent tiles with inferior path_id are skipped
}

//----------------------------------------------------------------------------------------


uint outcode(float2 p) {
    uint code = 0;
    if (p.x < 0.0) code |= 1;
    if (p.x > 1.0) code |= 2;
    if (p.y < 0.0) code |= 4;
    if (p.y > 1.0) code |= 8;
    return code;
}

// side: 1: right, 0: left
// returns whether the line is culled (fully outside the clip boundary)
bool clip_line_x(float xclip, bool side, inout float2 p, inout float2 q)
{
    float2 d = q - p;
    let pclip_l = p.x < xclip;
    let qclip_l = q.x < xclip;
    if (pclip_l == qclip_l) {
        return pclip_l == side;
    }
    float t = (xclip - p.x) / d.x;
    float yclip = p.y + t * d.y;
    float2 pclip = float2(xclip, yclip);
    if ((side & p.x < xclip) || (!side & p.x > xclip)) {
        p = pclip;
    }
    else {
        q = pclip;
    }
    return true;
}

bool clip_line_y(float yclip, bool side, inout float2 p, inout float2 q)
{
    float2 d = q - p;
    let pclip_l = p.y < yclip;
    let qclip_l = q.y < yclip;
    if (pclip_l == qclip_l) {
        return pclip_l == side;
    }
    float t = (yclip - p.y) / d.y;
    float xclip = p.x + t * d.x;
    float2 pclip = float2(xclip, yclip);
    if ((side & p.y < yclip) || (!side & p.y > yclip)) {
        p = pclip;
    }
    else {
        q = pclip;
    }
    return true;
}

struct TileBin {
    int2 coords;
    uint path;
    int winding;
    uint seg_offset;
    uint seg_count;
};

static const uint RASTER_TILE_SIZE = 16;

struct RasterTilesParams {
    float4* segments;
    TileBin* tile_bins;
    uint tile_bin_count;
    uint* bin_offsets;
    uint tile_count;
    RWTexture2D<float4>.Handle output;
};

groupshared float s_coverage[RASTER_TILE_SIZE * RASTER_TILE_SIZE * 2];
groupshared int s_winding_v_area[RASTER_TILE_SIZE];
groupshared int s_winding_v_delta[RASTER_TILE_SIZE];

[shader("compute")]
[numthreads(RASTER_TILE_SIZE, RASTER_TILE_SIZE, 1)]
void raster_tiles(uint3 workgroup_id: SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uniform RasterTilesParams* u)
{
    if (workgroup_id.x >= u.tile_count) {
        return;
    }

    var bin_offset = u.bin_offsets[workgroup_id.x];

    var cur_tile = u.tile_bins[bin_offset];

    let tile_coords = cur_tile.coords;
    let tsize = int2(RASTER_TILE_SIZE, RASTER_TILE_SIZE);

    let local     = int2(group_thread_id.xy);               // coordinates within the tile
    let tile_top_left = tile_coords * tsize;                      // tile top-left corner position in screen coordinates
    let pixel_center  = float2(local) + 0.5;  // pixel center in tile-local coordinates

    while (true) {
        // compute vertical winding numbers at x = 0
        // process segments in parallel

        let local_index = group_thread_id.y * RASTER_TILE_SIZE + group_thread_id.x;
        if (local_index < RASTER_TILE_SIZE) {
            s_winding_v_area[local_index] = 0;
            s_winding_v_delta[local_index] = 0;
        }

        GroupMemoryBarrierWithGroupSync();

        if (local_index < cur_tile.seg_count) {
            float4 seg = u.segments[cur_tile.seg_offset + local_index];
            float2 a = seg.xy;
            float2 b = seg.zw;
            float2 d = b - a;
            // compute intersection at x = 0, i.e. solve a + t*d = (0, y) -> t = -a.x/d.x -> y = a.y + t*d.y
            //float t = -a.x / d.x;

            if (a.x == 0.0 || b.x == 0) {
                let y = a.x == 0.0 ? a.y : b.y;
                let yi = int(floor(y)); // pixel row index of intesection
                let yf = 1 - (y - yi);        // 1 - fractional part
                let yf_quant = int(yf * 255.0); // quantized to 0..255
                let sign = int(sign(d.x));    // crossing left to right: -1, crossing right to left: +1
                InterlockedAdd(s_winding_v_area[yi], sign * yf_quant);
                InterlockedAdd(s_winding_v_delta[yi], sign);
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // prefix sum vertical winding numbers in shared memory
        // this can be done with subgroup ops
        if (local_index < RASTER_TILE_SIZE) {
             int winding = s_winding_v_delta[local_index];
             winding = WavePrefixSum(winding);
             s_winding_v_delta[local_index] = winding;
        }

        GroupMemoryBarrierWithGroupSync();

        float winding = (float) cur_tile.winding + s_winding_v_delta[local.y] + float(s_winding_v_area[local.y]) / 255.0;
        for (uint i = 0; i < cur_tile.seg_count; i++) {
            float4 seg = u.segments[cur_tile.seg_offset + i];
            float2 a = seg.xy - float2(local);
            float2 b = seg.zw - float2(local);
            float2 d = b - a;

            // good luck with that

            if (d.y == 0.0) {
                // horizontal segments don't contribute anything to the winding
                continue;
            }

            if (d.x == 0.0) {
                // vertical segments
                float y_min = saturate(min(a.y, b.y));
                float y_max = saturate(max(a.y, b.y));
                float x = a.x;
                winding += sign(-d.y) * (y_max - y_min) * saturate(1.0 - x);
                continue;
            }

            // FIXME: for near-vertical segments this is super unstable
            float y_0 = a.y - a.x * d.y / d.x;
            float y_1 = a.y + (1.0 - a.x) * d.y / d.x;
            float y_min = saturate(min(a.y, b.y));
            float y_max = saturate(max(a.y, b.y));
            float x_0 = a.x + (y_min - a.y) / d.y * d.x;
            float x_1 = a.x + (y_max - a.y) / d.y * d.x;
            float a_1 = 0.5 * abs(y_min - y_0) * max(0.0, x_0);
            float a_2 = 0.5 * abs(y_max - y_0) * max(0.0, x_1);
            float a_3 = 0.5 * abs(y_min - y_1) * max(0.0, x_0 - 1.0);
            float a_4 = 0.5 * abs(y_max - y_1) * max(0.0, x_1 - 1.0);

            float area = (y_max - y_min) - abs(a_1 - a_2) + abs(a_3 - a_4);
            winding += area * sign(-d.y);

            /*// solve line intersection with horizontal ray from pixel center: a + t*d = (x, 0) -> t = -a.y/d.y -> x = a.x + t*d.x
            float t = -a.y / d.y;
            if (t >= 0.0 && t < 1.0) {
                float x = a.x - d.x * a.y / d.y;
                winding += saturate(-x + 0.5) * sign(-d.y);
            }*/
        }

        u.output.Store(tile_coords * tsize + local, float4((winding).xxx, 1.0));
        //u.output.Store(tile_coords * tsize + local, float4(1.0, 0.0, 0.0, 1.0));

        bin_offset += 1;
        if (bin_offset >= u.tile_bin_count) {
            break;
        }
        cur_tile = u.tile_bins[bin_offset];
        if (cur_tile.coords.x != tile_coords.x || cur_tile.coords.y != tile_coords.y) {
            break;
        }


    }
}


//--------------------------------------------------------------------------------------------------

struct CopyToScreenParams {
    RWTexture2D<float4>.Handle texture;
};

struct ScreenQuadVSOut
{
    float4 pos : SV_Position;
    float2 uv;
};

/// Vertex shader for full-screen quad
ScreenQuadVSOut screen_quad_vertex_shader(uint vertex_id)
{
    float2 positions[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0)
    };

    ScreenQuadVSOut o;
    o.pos = float4(positions[vertex_id], 0.0, 1.0);
    o.uv = o.pos.xy * 0.5 + 0.5;
    return o;
}

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











