// TODO: that's a sorry path that we need to use
import common;

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv;
};

struct SceneInfo
{
    float4x4 view_matrix;
    float4x4 proj_matrix;
    float2 screen_size;
    float time;
    uint frame;
};

struct PushConstants {
    SceneInfo* scene;
    Srgba8 bottom_color;
    Srgba8 top_color;
};

[shader("vertex")]
VSOut background_vertex_main(uint vertex_id : SV_VertexID, uniform PushConstants u)
{
    float2 positions[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0)
    };

    VSOut o;
    o.pos = float4(positions[vertex_id], 0.0, 1.0);
    o.uv = o.pos.xy * 0.5 + 0.5;
    return o;
}

[shader("fragment")]
float4 background_fragment_main(VSOut v, uint2 frag_pos: SV_Position, uniform PushConstants u) {
    float t = v.uv.y * 0.5 + 0.5;
    float4 color = lerp(
        color32_to_float4(u.bottom_color),
        color32_to_float4(u.top_color),
        t);
    color = dither(color, frag_pos, u.scene.time);
    return color;
}
