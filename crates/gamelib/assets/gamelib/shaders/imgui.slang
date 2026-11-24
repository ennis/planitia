// Manifest: shaders.toml
struct RootParams {
    float2 screenSize;
    Texture2D.Handle tex;
    SamplerState.Handle sampler;
    Vertex* vertices;
    uint16_t* indices;
    uint base_vertex;
};

struct Vertex {
    float2 pos;
    float2 uv;
    uint8_t4 color;
};

struct VertexOutput {
    float4 pos: SV_Position;
    float4 color;
    float2 uv;
}

[shader("vertex")]
VertexOutput imgui_vertex(uint vertex_id: SV_VulkanVertexID , uniform RootParams* u) {
    uint index = u.indices[vertex_id];
    Vertex v = u.vertices[u.base_vertex + index];
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

[shader("fragment")]
FragmentOutput imgui_fragment(VertexOutput v, uniform RootParams* u) {
    FragmentOutput o;
    o.color = v.color * u.tex.Sample(u.sampler, v.uv).r;
    return o;
}