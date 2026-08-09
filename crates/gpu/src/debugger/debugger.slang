// Manifest: shaders.toml

struct Vertex {
    float2 pos;
    float2 uv;
    uint8_t4 color;
};

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
    Vertex* vertices;
    Texture2D<float4>.Handle font_tex;
    SamplerState.Handle sampler;
    int2 offset;
    uint8_t4 color;
};

[push_constant]
RootParams* u;

VertexOutput vertex_common(uint vertexId)
{
    Vertex v = u->vertices[vertexId];
    VertexOutput o;
    o.pos = float4(v.pos, 0.0, 1.0);
    o.color = float4(v.color) / 255.0;
    o.uv = v.uv;
    return o;
}

[shader("vertex")]
VertexOutput text_vertex_main(uint vertexId : SV_VertexID) {
    return vertex_common(vertexId);
}


[shader("fragment")]
FragmentOutput text_fragment_main(VertexOutput v) {
    FragmentOutput o;
    float mask = u->font_tex.Sample(u->sampler, v.uv).r;
    o.color = float4(v.color.rgb, v.color.a * mask);
    return o;
}


[shader("vertex")]
VertexOutput fill_vertex_main(uint vertexId : SV_VertexID) {
    return vertex_common(vertexId);
}

[shader("fragment")]
FragmentOutput fill_fragment_main(VertexOutput v) {
    FragmentOutput o;
    o.color = v.color;
    return o;
}