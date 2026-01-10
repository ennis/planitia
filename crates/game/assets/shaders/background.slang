import common;
import scene_info;

struct RootParams {
    SceneInfo* scene;
    Srgba8 bottom_color;
    Srgba8 top_color;
};

[shader("vertex")]
[pass("background")]
ScreenQuadVSOut background_vertex_main(uint vertex_id : SV_VertexID, uniform RootParams* u)
{
    return screen_quad_vertex_shader(vertex_id);
}

[shader("fragment")]
[pass("background")]
float4 background_fragment_main(ScreenQuadVSOut v, uint2 frag_pos: SV_Position, uniform RootParams* u) {
    float t = v.uv.y;
    float4 color = lerp(
        srgba8_to_float4(u.top_color),
        srgba8_to_float4(u.bottom_color),
        t);
    color = dither(color, frag_pos, u.scene.time);
    return color;
}
