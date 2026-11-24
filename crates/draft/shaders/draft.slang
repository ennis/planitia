// Manifest: shaders.toml
import common;

struct RootParams {
    float4x4 matrix;
    Vertex* vertices;
    uint32_t* indices;
    float2 screen_size;
    Texture2D.Handle texture;
    SamplerState.Handle sampler;
};

struct VSOut {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 uv;
}

[shader("vertex")]
VSOut draft_vertex(uint vertex_id : SV_VertexID, uniform RootParams* u) {
    VSOut o;
    Vertex v = u.vertices[u.indices[vertex_id]];
    o.pos = mul(u.matrix, float4(v.pos, 1.0));
    o.color = float4(v.color) / 255.0;
    o.uv = float2(v.uv) / 65535.0;
    return o;
}

struct FSOut
{
    [[vk::location(0)]] float4 color;
};

[shader("fragment")]
FSOut draft_fragment(VSOut v, uniform RootParams* u) {
    FSOut o;
    o.color = v.color * u.texture.Sample(u.sampler, v.uv);
    return o;
}