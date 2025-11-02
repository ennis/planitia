struct VSOut
{
    float4 pos : SV_Position;
    float2 uv;
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
VSOut grid_vertex_main(uint vertex_id : SV_VertexID, uniform PushConstants u)
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

    float2 pos = positions[vertex_id] * u.grid_scale;
    float4 clip_pos = mul(u.scene.proj_matrix, mul(u.scene.view_matrix, float4(pos.x, 0.0, pos.y, 1.0)));
    float2 uv = positions[vertex_id] * 0.5 + 0.5;

    // fill the screen
    //float4 clip_pos = float4(positions[vertex_id], 0.0, 1.0);

    VSOut o;
    o.pos = clip_pos;
    o.uv = uv * u.grid_scale;
    return o;
}

float pulse_train_integral(float x, float period, float duty_cycle)
{
    let n = floor(x / period);
    let pw = period * duty_cycle;
    return n * pw + min(pw, x - n * period);
}

// https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
float grid(float2 uv, float line_width)
{
    let dpdx = ddx(uv);
    let dpdy = ddy(uv);
    let fw = abs(dpdx) + abs(dpdy);
    let effective_line_width = fw;
    //let effective_line_width = min(0.5,max(line_width, fw));
    let line_aa = 1.5 * fw;
    let grid_uv = abs(2 * fract(uv) - 1.0);
    let grid2 = saturate(line_width / effective_line_width) * smoothstep(effective_line_width + line_aa, effective_line_width - line_aa, grid_uv);
    float grid = lerp(grid2.x, 1.0, grid2.y);
    return grid;
}

[shader("fragment")]
float4 grid_fragment_main(VSOut i, uint2 frag_pos: SV_Position, uniform PushConstants u)
{
    float2 frag_uv = frag_pos.xy / u.scene.screen_size;
    return float4(grid(i.uv*10.0, 0.05).xxx,1.0);
}