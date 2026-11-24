// Manifest: shaders.toml
import common;

struct Line {
    uint32_t base_vertex;
    uint32_t start_index;
    uint32_t index_count;
    float width;
    float filter_width;
}

struct VSOut {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 local_pos;
    [[vk::location(2)]] nointerpolation float width;
    [[vk::location(3)]] nointerpolation float filter_width;
};

struct DrawLinesRootParams {
    float4x4 view_proj;
    float2 screen_size;
    Vertex* vertices;
    uint32_t* indices;
    Line* lines;
};

[shader("vertex")]
VSOut lines_vertex(uint vertex_id : SV_VertexID, uint draw_index : SV_DrawIndex, uniform DrawLinesRootParams* u)
{
    Line line = u.lines[draw_index];

    // Current vertex is at `u.vertices[line.base_vertex + u.indices[line.start_index + vertex_id / 2]]`
    // Index of the current point in u.indices
    let idx = vertex_id / 2;
    bool line_start = idx == 0;
    bool line_end = idx == (line.index_count - 1);

    // Fetch current/prev/next vertices.
    let v = u.vertices[line.base_vertex + u.indices[line.start_index + idx]];
    let v0 = line_start ? v : u.vertices[line.base_vertex + u.indices[line.start_index + idx - 1]];
    let v1 = line_end ? v : u.vertices[line.base_vertex + u.indices[line.start_index + idx + 1]];

    // Project vertex positions in clip space.
    let p = mul(u.view_proj, float4(v.pos, 1.));
    let p0 = mul(u.view_proj, float4(v0.pos, 1.));
    let p1 = mul(u.view_proj, float4(v1.pos, 1.));

    // anti-aliasing margin
    float aa = l.filter_width * sqrt(2.);
    // half-width + anti-aliasing margin
    // clamp the width to 1.0, below that we fade out the line (see fragment shader)
    float hw_aa = max(l.width, 1.) * .5 + aa;

    // pixel size in clip space
    float2 px_size = 2. / u.screen_size;

    VSOut o;
    o.pos = p;
    o.color = float4(v.color) / 255.0;
    o.local_pos = float2(0.0, side * hw_aa);
    o.width = l.width;
    o.filter_width = l.filter_width;

    // Offset the vertex in normal direction (on either side depending on vertex ID).
    let side = (vertex_id % 2 == 0) ? +1.0 : -1.0;
    if (line_start) {
        float2 v = (p1.xy/p1.w - p.xy/p.w) / px_size;
        float2 n = hw_aa * px_size * normalize(float2(-v.y, v.x));
        o.pos.xy += side * n * o.pos.w;
    } else if (line_end) {
        float2 v = (p.xy/p.w - p0.xy/p0.w) / px_size;
        float2 n = hw_aa * px_size * normalize(float2(-v.y, v.x));
        o.pos.xy += side * n * o.pos.w;
    }
    else {
        float2 v0 = normalize((p.xy/p.w - p0.xy/p0.w) / px_size);
        float2 v1 = normalize((p1.xy/p1.w - p.xy/p.w) / px_size);
        float2 vt = 0.5 * (v0 + v1);
        float2 n = float2(-vt.y, vt.x);
        // half-width / sin(theta/2)
        float d = hw_aa / max(cross(float3(v0, 0.0), float3(n, 0.0)).z, 0.05);
        o.pos.xy += side * d * n * px_size * o.pos.w;
    }

    return o;
}

////////////////////////////////////////////////////////////////////////

struct FSOut
{
    [[vk::location(0)]] float4 color;
};


[shader("fragment")]
FSOut lines_fragment(VSOut v, uniform DrawLinesRootParams* u) {
    FSOut o;
    // clamped width
    float width1 = max(v.width, 1.0);
    float h = width1 * 0.5;
    float y = abs(v.local_pos.y);
    float half_filter_width = v.filter_width * 0.5;
    float alpha = (clamp((y + h + half_filter_width), 0., width1) - clamp((y + h - half_filter_width), 0., width1)) / v.filter_width;

    alpha *= min(v.width, 1.0);
    o.color = v.color;
    o.color.a *= alpha;
    //o.color = float4(0.0, 1.0, 0.0, 1.0);
    return o;
}
