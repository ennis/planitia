import common;
import scene_info;

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv;
};


struct RootParams {
    SceneInfo* scene;
    float grid_scale;
    float padding;
};

[shader("vertex")]
[pass("grid")]
VSOut grid_vertex_main(uint vertex_id : SV_VertexID, uniform RootParams* u)
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
    float2 uv = positions[vertex_id];

    // fill the screen
    //float4 clip_pos = float4(positions[vertex_id], 0.0, 1.0);
    VSOut o;
    o.pos = clip_pos;
    o.uv = uv * u.grid_scale;
    return o;
}

/*float pulse_train_integral(float x, float period, float duty_cycle)
{
    let n = floor(x / period);
    let pw = period * duty_cycle;
    return n * pw + min(pw, x - n * period);
}*/

// https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
float4 grid(float2 uv, float line_width)
{
    let dpdx = ddx(uv);
    let dpdy = ddy(uv);
    let fw = abs(dpdx) + abs(dpdy);
    let effective_line_width = fw;
    //let effective_line_width = min(0.5,max(line_width, fw));
    let line_aa = 1.5 * fw;
    let grid_uv = 1.0 - abs(2 * fract(uv) - 1.0);
    var grid2 = saturate(line_width / effective_line_width) * smoothstep(effective_line_width + line_aa, effective_line_width - line_aa, grid_uv);

    let coord = floor(uv + 0.5);
    float4 x_color = float4(1., 1., 1., 0.1);   // vertical lines (varying x)
    float4 y_color = float4(1., 1., 1., 0.1);   // horizontal lines (varying y)

    // 5-unit marks
    x_color = coord.x % 5 == 0 ? float4(x_color.xyz, 1.0) : x_color;
    y_color = coord.y % 5 == 0 ? float4(y_color.xyz, 1.0) : y_color;

    // main axis
    x_color = coord.x == 0 ? float4(0., 0., 1., 1.) : x_color;
    y_color = coord.y == 0 ? float4(1., 0., 0., 1.) : y_color;

    // normally horizontal lines are blended over vertical lines, but we want the major axes and subdivisions to be on
    // top, so swap the blending order here
    /*if (coord.y % 5 == 0 && coord.x % 5 != 0)
    {
        let temp = x_color;
        x_color = y_color;
        y_color = temp;
        grid2.xy = grid2.yx;
    }*/

    float4 color = lerp(x_color * grid2.x, y_color, grid2.y);
    color.xyz /= color.w;

    return color;
}

[shader("fragment")]
[pass("grid")]
float4 grid_fragment_main(VSOut i, uint2 frag_pos: SV_Position, uniform RootParams* u)
{
    float2 frag_uv = frag_pos.xy / u.scene.screen_size;
    float2 scaled_uv = i.uv;
    float4 grid_color = grid(scaled_uv, 0.05);

    if (grid_color.w < 0.01) discard;
    return grid_color;
}