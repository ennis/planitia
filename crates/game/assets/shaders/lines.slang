import scene_info;
import common;

struct Line {
    uint32_t start_vertex;
    uint32_t vertex_count;
    float width;
    float filter_width;
}

struct LineVertex {
    float3 position;
    Srgba8 color;
};

struct VertexOutput {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 local_pos;
    [[vk::location(2)]] nointerpolation float width;
    [[vk::location(3)]] nointerpolation float filter_width;
};

struct PushConstants {
    SceneInfo* scene_info;
    LineVertex* vertices;
    Line* lines;
};

[shader("vertex")]
VertexOutput line_drawing_vertex(uint vertex_id : SV_VertexID, uint draw_index : SV_DrawIndex, uniform PushConstants* u)
{
    Line l = u.lines[draw_index];

    let line_vtx_id = vertex_id / 2;

    bool line_start = line_vtx_id == 0;
    bool line_end = line_vtx_id == (l.vertex_count - 1);

    let v = u.vertices[l.start_vertex + line_vtx_id];
    let v0 = line_start ? v : u.vertices[l.start_vertex + line_vtx_id - 1];
    let v1 = line_end ? v : u.vertices[l.start_vertex + line_vtx_id + 1];

    // project vertices in clip space
    let p = mul(u.scene_info.view_proj_matrix, float4(v.position, 1.));
    let p0 = mul(u.scene_info.view_proj_matrix, float4(v0.position, 1.));
    let p1 = mul(u.scene_info.view_proj_matrix, float4(v1.position, 1.));

    // half-width + anti-aliasing margin
    // clamp the width to 1.0, below that we just fade out the line
    float hw_aa = max(l.width, 1.) * .5 + l.filter_width * sqrt(2.);

    float4 a = p;
    float4 b = p;

    float2 px_size = 2. / u.scene_info.screen_size; // pixel size in clip space

    if (line_start) {
        float2 v = (p1.xy/p1.w - p.xy/p.w) / px_size;
        float2 n = hw_aa * px_size * normalize(float2(-v.y, v.x));
        a.xy -= n * a.w;
        b.xy += n * b.w;

    } else if (line_end) {
        float2 v = (p.xy/p.w - p0.xy/p0.w) / px_size;
        float2 n = hw_aa * px_size * normalize(float2(-v.y, v.x));
        a.xy -= n * a.w;
        b.xy += n * b.w;
    }
    else {
        float2 v0 = normalize((p.xy/p.w - p0.xy/p0.w) / px_size);
        float2 v1 = normalize((p1.xy/p1.w - p.xy/p.w) / px_size);
        float2 vt = 0.5 * (v0 + v1);
        float2 n = float2(-vt.y, vt.x);
        // half-width / sin(theta/2)
        float d = hw_aa / max(cross(float3(v0, 0.0), float3(n, 0.0)).z, 0.05);
        // miter points
        a.xy -= d * n * px_size * a.w;
        b.xy += d * n * px_size * b.w;
    }

    VertexOutput o;
    o.pos = (vertex_id % 2 == 0) ? b : a;
    o.color = float4(v.color) / 255.0;
    o.local_pos = (vertex_id % 2 == 0) ? float2(0.0, -hw_aa) : float2(1.0, hw_aa);
    o.width = l.width;
    o.filter_width = l.filter_width;
    return o;
}

////////////////////////////////////////////////////////////////////////

struct FragmentOutput
{
    [[vk::location(0)]] float4 color;
};


[shader("fragment")]
FragmentOutput line_drawing_fragment(VertexOutput v, uniform PushConstants* u) {
    FragmentOutput o;
    // clamped width
    float width1 = max(v.width, 1.0);
    float h = width1 * 0.5;
    float y = abs(v.local_pos.y);
    //float filter_width = 1.5;
    float half_filter_width = v.filter_width * 0.5;
    float alpha = (clamp((y + h + half_filter_width), 0., width1) - clamp((y + h - half_filter_width), 0., width1)) / v.width;
    alpha *= min(v.width, 1.0);
    o.color = v.color;
    o.color.a *= alpha;
    //o.color = float4(0.0, 1.0, 0.0, 1.0);
    return o;
}

////////////////////////////////////////////////////////////////////////

// Converts Normalized Device Coordinates (NDC) to window coordinates.
float3 ndc2win(float3 ndc, float2 screenSize)
{
    return float3(ndc.xy * .5 + .5, ndc.z) * float3(screenSize, 1.);
}

// Converts window coordinates to Normalized Device Coordinates (NDC).
float3 win2ndc(float3 window, float2 screenSize)
{
    return float3(window.xy / float2(screenSize) * 2. - 1., window.z);
}

// Projects a vertex from world space to clip space.
float4 project(float3 pos, float4x4 viewProjectionMatrix)
{
    return mul(viewProjectionMatrix, float4(pos, 1.));
}
