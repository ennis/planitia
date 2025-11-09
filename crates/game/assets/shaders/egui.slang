import common;

struct PushConstants {
    float2 screenSize;
};


struct Vertex {
    [[vk::location(0)]] float2 pos;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] uint4 color;
};

struct VertexOutput {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 uv;
}

[shader("vertex")]
VertexOutput eguiVertexMain(Vertex v, uniform PushConstants u) {
    VertexOutput o;
    o.pos = float4(
        2.0 * v.pos.x / u.screenSize.x - 1.0,
        2.0 * v.pos.y / u.screenSize.y - 1.0,
        0.0,
        1.0);
    o.color = float4(v.color) / 255.0;
    o.uv = v.uv;
    return o;
}


struct FragmentOutput
{
    [[vk::location(0)]] float4 color;
};

[[vk::binding(0,0)]] Texture2D tex;
[[vk::binding(1,0)]] SamplerState sampler;

[shader("fragment")]
FragmentOutput eguiFragMain(VertexOutput v) {
    FragmentOutput o;
    o.color = v.color * tex.Sample(sampler, v.uv);
    return o;
}