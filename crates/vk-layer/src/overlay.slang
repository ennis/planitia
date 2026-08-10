struct VertexOutput {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 uv;
}

struct FragmentOutput
{
    [[vk::location(0)]] float4 color;
};

struct RootParams {
    int2 screen_size;
    int2 offset;
    uint8_t4 color;
};

[push_constant]
RootParams u;

[vk::binding(0)]
Texture2D<float4> font_tex;

[vk::binding(1)]
SamplerState font_sampler;

struct VertexInput {
    [[vk::location(0)]] float2 pos;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] float4 color;
};

[shader("vertex")]
VertexOutput text_vertex_main(VertexInput vi) {
    VertexOutput o;
    o.pos = float4(vi.pos, 0.0, 1.0);
    o.color = vi.color;
    o.uv = vi.uv;
    return o;
}

[shader("fragment")]
FragmentOutput text_fragment_main(VertexOutput v) {
    FragmentOutput o;
    float mask = font_tex.Sample(font_sampler, v.uv).r;
    o.color = float4(v.color.rgb, v.color.a * mask);
    return o;
}
