import common;
import scene_info;

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv;
};

struct PushConstants {
    SceneInfo* scene;
    Srgba8 bottom_color;
    Srgba8 top_color;
};

[shader("vertex")]
VSOut background_vertex_main(uint vertex_id : SV_VertexID, uniform PushConstants* u)
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
float4 background_fragment_main(VSOut v, uint2 frag_pos: SV_Position, uniform PushConstants* u) {
    float t = v.uv.y * 0.5 + 0.5;
    float4 color = lerp(
        srgba8_to_float4(u.bottom_color),
        srgba8_to_float4(u.top_color),
        t);
    color = dither(color, frag_pos, u.scene.time);
    return color;
}
