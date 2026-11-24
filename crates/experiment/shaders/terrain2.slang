
static const uint WORKGROUP_SIZE = 32;

struct Params {
    Texture2D<float>.Handle heightmap;
    SamplerState.Handle sampler;
    RWTexture2D<float4>.Handle output;
    uint2 screen_size;
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 proj_matrix;
    float4x4 inv_proj_matrix;
    float4x4 terrain_matrix;    // terrain to clip space
    float4x4 inv_terrain_matrix;
    float3 eye_pos;
    uint depth_size;
}


[vk::push_constant]
Params* params;


[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID) {
    let x = groupID.x;             // 0..width
    let local_idx = WaveGetLaneIndex();    // 0..32
    let depth_size = params.depth_size;
    float x_ndc = float(x) / float(params.screen_size.x - 1) * 2. - 1.;
    float max_height = 0.0;
    bool g_prev_climb = false;


    for (uint base = 0; base < depth_size; base += WORKGROUP_SIZE) {
        uint depth_idx = min(base + local_idx, depth_size - 1);
        let z_ndc = pow(float(depth_idx) / float(depth_size - 1), 1.0);
        let p_ndc = float3(x_ndc, 0, z_ndc);
        // convert back to view space
        var p_terr = mul(params.inv_terrain_matrix, float4(x_ndc, 0, z_ndc, 1));
        p_terr.xyz /= p_terr.w;
        let uv = p_terr.xz;

        uint2 out_idx = uint2(x, depth_size - 1 - depth_idx);

        //float height = params.heightmap.SampleLevel(params.sampler, float2(out_idx) / float2(1024, 1024), 0).r;
        float height = params.heightmap.SampleLevel(params.sampler, uv, 0).r;
        float height_0 = height * 200.0 ;

        float prefix_max_height = max(max_height, WavePrefixMax(height_0));
        bool climb = height_0 > prefix_max_height;

        bool prev_climb_prev_lane = WaveReadLaneAt(climb, (local_idx == 0) ? 0 : (local_idx - 1));
        bool prev_climb = (local_idx == 0) ? g_prev_climb : prev_climb_prev_lane;

        float local_max_height = max(height_0, prefix_max_height);
        uint2 uv_out_idx = clamp(uv, 0, 1) * uint2(1024, 1024);

        float4 color = (!prev_climb && climb) ? float4(1,0,0,1) : ((prev_climb && !climb) ?  float4(0,1,0,1) : (climb ? float4(local_max_height, local_max_height, local_max_height, 1.) : float4(0,0,0,1)));
        //params.output[out_idx] = color;
        //params.output[out_idx] = prev_climb ? float4(1, 1, 0, 1.) : float4(local_max_height, local_max_height, local_max_height, 1.);
        //params.output[out_idx] = float4(g_prev_climb.xxx * ((local_idx == 0) ? float3(1,0,0) : float3(1,1,1)), 1.);

        // splat the points
        if (climb) {
            var clip = mul(params.terrain_matrix, float4(uv.x, height * 200.0, uv.y, 1));
            let ndc = clip.xyz / clip.w;
            let window_coord = clamp(ndc.xy * 0.5 + 0.5, 0, 1.0) * float2(params.screen_size - 1);
            params.output[window_coord] = float4(z_ndc);
        }
        params.output[uv_out_idx] = float4(0, local_max_height, 0, 1.);

        max_height = WaveActiveMax(local_max_height);
        g_prev_climb = WaveReadLaneAt(climb, WORKGROUP_SIZE - 1);
    }
}

