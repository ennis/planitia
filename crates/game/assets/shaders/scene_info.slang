// SceneInfo struct.

struct SceneInfo
{
    float4x4 view_matrix;
    float4x4 proj_matrix;
    float4x4 view_proj_matrix;
    float2 screen_size;
    float time;
    uint frame;
};
