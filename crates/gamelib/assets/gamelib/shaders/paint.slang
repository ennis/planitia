// ---------------------------------------------------------------
struct FeatherVertex
{
    float2 position;
    float2 uv;
    float4 color;
    float feather;
};

// ---------------------------------------------------------------

struct VertexOutput
{
    float4 pos: SV_Position;
    float2 uv;
    float4 color;
    float feather;
};

struct FragmentOutput
{
    float4 color;
};

struct RootParams
{
    float4x4 matrix;
    float2 screen_size;
    float line_width;
    Texture2D<float4>.Handle texture;
    SamplerState.Handle sampler;
    uint use_texture;
};

// ---------------------------------------------------------------

[shader("vertex")]
VertexOutput paint_vertex_main(FeatherVertex v, uniform RootParams* u)
{
    VertexOutput o;
    o.pos = float4(v.position.xy / u.screen_size * 2.0 - 1.0, 0.0, 1.0);
    o.color = v.color;
    o.feather = v.feather;
    o.uv = v.uv;
    return o;
}

[shader("fragment")]
FragmentOutput paint_fragment_main(VertexOutput v, uniform RootParams* u)
{
    float alpha = (1.0 - abs(v.feather)) * min(u.line_width, 1.0);
    float4 tex_color = u.texture.Sample(u.sampler, v.uv);
    FragmentOutput o;
    o.color = v.color * tex_color;
    o.color.a *= alpha;
    return o;
}
