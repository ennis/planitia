import bindless;

// ---------------------------------------------------------------
struct GlyphVertex
{
    float2 position;
    float2 tex_coord;
    float4 color;
};

// ---------------------------------------------------------------

struct VertexOutput
{
    float4 pos: SV_Position;
    float2 tex_coord;
    float4 color;
};

struct PushConstants
{
    float2 screen_size;
    Texture2D<float4>.Handle* glyph_atlas;
    SamplerState.Handle glyph_sampler;
};

// ---------------------------------------------------------------


[shader("vertex")]
VertexOutput glyph_vertex_main(GlyphVertex v, uniform PushConstants u)
{
    VertexOutput o;
    o.pos = float4(v.position.xy / u.screen_size * 2.0 - 1.0, 0.0, 1.0);
    o.color = v.color;
    o.tex_coord = v.tex_coord;
    return o;
}

[shader("fragment")]
float4 glyph_fragment_main(
            VertexOutput v,
            uniform PushConstants u)
{
    return v.color * u.glyph_atlas.Sample(u.glyph_sampler, v.tex_coord);
}
