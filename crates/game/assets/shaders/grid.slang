struct VSOut
{
    float4 pos: SV_Position;
};

struct SceneInfo
{
    float4x4 view_matrix;
    float4x4 proj_matrix;
    float2 screen_size;
};

struct PushConstants {
    SceneInfo* scene;
    float grid_scale;
    float padding;
};

[shader("vertex")]
VSOut grid_vertex_main(uint vertexId : SV_VertexID, uniform PushConstants u)
{
    // unscaled XZ plane quad positions
    float2 positions[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0)
    };

    float2 pos = positions[vertexId] * u.grid_scale;
    float4 clip_pos = mul(u.scene.proj_matrix, mul(u.scene.view_matrix, float4(pos.x, 0.0, pos.y, 1.0)));

    VSOut o;
    o.pos = clip_pos;
    return o;
}

[shader("fragment")]
float4 grid_fragment_main(VSOut in, uniform PushConstants u)
{
    return float4(0.5, 1.0, 0.5, 1.0);
}