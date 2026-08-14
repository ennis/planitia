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
    uint8_t4 shuffle;
    uint8_t4 color;
};

[push_constant]
RootParams u;

[vk::binding(0)]
Texture2D<float4> texture;

[vk::binding(1)]
SamplerState sampler;

struct VertexInput {
    [[vk::location(0)]] float2 pos;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] float4 color;
};

[shader("vertex")]
VertexOutput overlay_vertex(VertexInput vi) {
    VertexOutput o;
    o.pos = float4(vi.pos, 0.0, 1.0);
    o.color = vi.color;
    o.uv = vi.uv;
    return o;
}

[shader("fragment")]
FragmentOutput overlay_fragment(VertexOutput v) {
    FragmentOutput o;
    float4 tex_color = texture.Sample(sampler, v.uv);
    float4 tex_color_shuf;
    tex_color_shuf.x = u.shuffle.x < 4 ? tex_color[u.shuffle.x] : float(u.shuffle.x - 4);
    tex_color_shuf.y = u.shuffle.y < 4 ? tex_color[u.shuffle.y] : float(u.shuffle.y - 4);
    tex_color_shuf.z = u.shuffle.z < 4 ? tex_color[u.shuffle.z] : float(u.shuffle.z - 4);
    tex_color_shuf.w = u.shuffle.w < 4 ? tex_color[u.shuffle.w] : float(u.shuffle.w - 4);
    o.color = v.color * tex_color_shuf;
    return o;
}
