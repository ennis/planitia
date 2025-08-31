struct FeatherVertex
{
    float2 position;
    float feather;
    float4 color;
};

struct VertexOutput
{
    float4 pos: SV_Position;
    float4 color;
    float feather;
};

struct FragmentOutput
{
    float4 color;
};

struct PushConstants
{
    float4x4 matrix;
    float2 screen_size;
    float line_width;
};

[shader("vertex")]
VertexOutput paint_vertex_main(FeatherVertex v, uniform PushConstants u)
{
    VertexOutput o;
    o.pos = float4(v.position.xy / u.screen_size * 2.0 - 1.0, 0.0, 1.0);
    o.color = v.color;
    o.feather = v.feather;
    return o;
}

[shader("fragment")]
FragmentOutput paint_fragment_main(VertexOutput v, uniform PushConstants u)
{
    float alpha = (1.0 - abs(v.feather)) * min(u.line_width, 1.0);
    FragmentOutput o;
    o.color = v.color;
    o.color.a *= alpha;
    return o;
}
