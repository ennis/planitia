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
    float4x4 matrix;
    int2 screen_size;
    int2 offset;
    uint8_t4 shuffle;
    uint8_t4 color;
};

[push_constant]
RootParams u;

[vk::binding(0)]
Texture2D<float4> texture;

[vk::binding(1)]
Texture2D<float4> texture1;

[vk::binding(2)]
SamplerState sampler;

struct VertexInput {
    [[vk::location(0)]] float2 pos;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] float4 color;
};

[shader("vertex")]
VertexOutput overlay_vertex(VertexInput vi) {
    VertexOutput o;
    o.pos = mul(u.matrix, float4(vi.pos, 0.0, 1.0));
    o.color = vi.color;
    o.uv = vi.uv;
    return o;
}

float shuffle(uint shuffle, float4 color0, float4 color1) {
    if (shuffle < 4) {
        return color0[shuffle];
    } else if (shuffle < 8) {
        return color1[shuffle - 4];
    } else {
        return float(shuffle - 8);
    }
}

[shader("fragment")]
FragmentOutput overlay_fragment(VertexOutput v) {
    FragmentOutput o;
    float4 tex_color = texture.Sample(sampler, v.uv);
    float4 tex1_color = texture1.Sample(sampler, v.uv);
    float4 color_shuf = float4(
        shuffle(u.shuffle.x, tex_color, tex1_color),
        shuffle(u.shuffle.y, tex_color, tex1_color),
        shuffle(u.shuffle.z, tex_color, tex1_color),
        shuffle(u.shuffle.w, tex_color, tex1_color)
    );
    o.color = v.color * color_shuf;
    return o;
}
