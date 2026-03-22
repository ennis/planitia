import common;
import scene_info;
import ridge_detection;

struct Point {
    float3 position;
    uint group_id;
};

struct Vertex {
    uint point;
    float3 normal;
};

struct MeshData {
    Point* points;
    Vertex* vertices;
    uint point_count;
    uint vertex_count;
};

struct Emitter {
    uint2 position;
};

struct RootParams {
    SceneInfo* scene_info;
    MeshData mesh;
    Texture2D<float4>.Handle shading_texture;
    Texture2D<float4>.Handle normal_texture;
    Texture2D<float4>.Handle depth_texture;
    Texture2D<uint16_t4>.Handle aux_texture;
    Texture2D<float4>.Handle contour_target;
    RWTexture2D<float4>.Handle trails_0;
    RWTexture2D<float4>.Handle trails_1;
    Emitter* emitters;
    uint emitter_count;
    float3 light_direction;
    uint sim_step;
    DebugMode debug_mode;
};

enum DebugMode : uint {
    None = 0,
    Shading = 1,
    Normal = 2,
    Depth = 3,
    Aux = 4,
    Contour_MaxCurv = 5,
    Contour_Angle = 6,
    Sim_Trails = 7,
};



///////////////////////////////////////////////////////////////////////////////////////////////////
// Base render + depth pre-pass
///////////////////////////////////////////////////////////////////////////////////////////////////

struct BaseRenderVSOut {
    float4 pos: SV_Position;
    float3 view_dir;
    float4 color;
    float3 normal;
    nointerpolation int group_id;
}

struct BaseRenderFSOut {
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] uint16_t4 aux;
}


[shader("vertex")]
[pass("automaton_base_render")]
BaseRenderVSOut base_render_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform RootParams* u)
{
    BaseRenderVSOut o;
    let vertex = u.mesh.vertices[vertex_id];
    let point = u.mesh.points[vertex.point];
    let obj_pos = float4(point.position,1.0);
    o.pos = mul(u.scene_info.view_proj_matrix, obj_pos);
    o.view_dir = normalize(mul(u.scene_info.view_matrix, obj_pos).xyz);
    o.normal = mul(float3x3(u.scene_info.view_matrix), vertex.normal);
    o.group_id = point.group_id;
    return o;
}

[shader("fragment")]
[pass("automaton_base_render")]
BaseRenderFSOut base_render_fragment(BaseRenderVSOut i, uniform RootParams* u) {
    BaseRenderFSOut o;
    float3 normal = normalize(i.normal);
    //let light_dir = -i.view_dir;
    let light_dir = -normalize(u.light_direction);
    float light_intensity = saturate(dot(normal, light_dir));
    o.color.rgb = float3(light_intensity);
    o.color.a = 1.0;
    o.normal = normal;
    o.aux = uint16_t4(uint16_t(i.group_id), 0, 0, 0);
    // colorize per cluster
    //o.color.rgb = palette(float(i.meshlet_id) * 0.1, float3(0.5,0.5,0.5),float3(0.5,0.5,0.5),float3(1.0,1.0,1.0),float3(0.0,0.33,0.67));
    return o;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Contour detection
///////////////////////////////////////////////////////////////////////////////////////////////////

[pass("contour_detection")]
[shader("vertex")]
ScreenQuadVSOut contour_detection_vertex_shader(uint vertex_id : SV_VertexID)
{
    return screen_quad_vertex_shader(vertex_id);
}

[pass("contour_detection")]
[shader("fragment")]
float4 contour_detection_fragment_shader(
    float3 frag_coord: SV_Position,
    ScreenQuadVSOut vertex_output,
    uniform RootParams* params)
{
    int2 coords = int2(frag_coord.xy);
    int2 size = int2(params.scene_info.screen_size);
    let base = params.shading_texture.Load(int3(coords, 0));

    float max_curv;
    float axis_dist;
    float2 axis;

    compute_ridge_properties(
        params->shading_texture,
        params->depth_texture,
        params->aux_texture,
        coords,
        size,
        max_curv,
        axis_dist,
        axis
    );

    float angle = atan2(axis.y, axis.x);
    return float4(max_curv, angle, 0.0, 1.0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Debug
///////////////////////////////////////////////////////////////////////////////////////////////////

[pass("automaton_debug")]
[shader("vertex")]
ScreenQuadVSOut automaton_debug_vertex_shader(uint vertex_id : SV_VertexID)
{
    return screen_quad_vertex_shader(vertex_id);
}

struct DebugFSOut {
    [[vk::location(0)]] float4 color;
}

[pass("automaton_debug")]
[shader("fragment")]
float4 automaton_debug_fragment_shader(
    float3 frag_coord: SV_Position,
    ScreenQuadVSOut vertex_output,
    uniform RootParams* params)
{
    uint2 coords = uint2(frag_coord.xy);
    int3 coords3 = int3(coords, 0);
    uint2 size = uint2(params.scene_info.screen_size);

    float4 color = float4(0.0, 1.0, 0.0, 1.0);

    switch (params.debug_mode) {
    case DebugMode.None:
        break;
    case DebugMode.Shading: {
        let shading = params.shading_texture.Load(coords3, 0);
        color = shading;
        break;
        }
    case DebugMode.Normal: {
        let normal = params.normal_texture.Load(coords3, 0).rgb;
        color = float4(normal, 1.0);
        break;
        }
    case DebugMode.Depth:
        break;
    case DebugMode.Aux:
        break;
    case DebugMode.Contour_MaxCurv: {
        let contour = params.contour_target.Load(coords3, 0);
        color = float4(colorize_1d(contour.x * 10), 1.0);
        break;
        }
    case DebugMode.Contour_Angle:
        break;
    case DebugMode.Sim_Trails: {
        let input = params.sim_step % 2 == 0 ? params.trails_0 : params.trails_1;
        let data = input.Load(coords);
        color = float4(colorize_1d(data.r / 20.0), 1.0);
        break;
        }
    }

    return color;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Simulation
///////////////////////////////////////////////////////////////////////////////////////////////////


[pass("sim_init_emitters")]
[shader("compute")]
[NumThreads(64, 1, 1)]
void sim_init_emitters(uint3 dispatch_id : SV_DispatchThreadID, uniform RootParams* p) {
    const uint i = dispatch_id.x;
    if (i >= p.emitter_count) {
        return;
    }
    if (p.sim_step == 0) {
        p.emitters[i].position = uint2(p.scene_info.screen_size) / 2;
    }
}

[pass("sim_init_trails")]
[shader("compute")]
[NumThreads(16, 16, 1)]
void sim_init_trails(uint3 dispatch_id : SV_DispatchThreadID, uniform RootParams* params) {
    const uint2 coords = uint2(dispatch_id.xy);
    const uint2 size = uint2(params.scene_info.screen_size);
    if (any(coords >= size)) {
        return;
    }

    if (params.sim_step == 0) {
        let max_curv = params.contour_target.Load(int3(coords, 0)).x;
        let val = max_curv * 20.0;

        params.trails_0.Store(coords, float4(val,0,0,0));
        params.trails_1.Store(coords, float4(0,0,0,0));
    }
}


[pass("sim_step_emitters")]
[shader("compute")]
[NumThreads(64, 1, 1)]
void sim_step_emitters(uint3 dispatch_id : SV_DispatchThreadID, uniform RootParams* params) {
    //let input = params->sim_step % 2 == 0 ? params->trails_0 : params->trails_1;
    //let output = params->sim_step % 2 == 0 ? params->trails_1 : params->trails_0;

}

[pass("sim_step_trails")]
[shader("compute")]
[NumThreads(16, 16, 1)]
void sim_step_trails(uint3 dispatch_id : SV_DispatchThreadID, uniform RootParams* params) {
    let input = params.sim_step % 2 == 0 ? params.trails_0 : params.trails_1;
    let output = params.sim_step % 2 == 0 ? params.trails_1 : params.trails_0;

    let coords = uint2(dispatch_id.xy);
    let size = uint2(params.scene_info.screen_size);
    if (any(coords >= size)) {
        return;
    }

    // load neighbors
    var maxv = 0.0;
    const float beta = -10;
    var avg = 0.0;

    for(int i = -1; i <= 1; i++) { // set 3x3 matrix
        for(int j = -1; j <= 1; j++) {
            var ncoords = clamp(int2(coords) + int2(i, j), int2(0,0), size - int2(1,1));
            var val = input.Load(ncoords).x;
            maxv = max(val - length(float2(i,j)), maxv);
            var z = val - length(float2(i,j));
            avg += z;
        }
    }
    avg /= 9.0;

    for(int i = -1; i <= 1; i++) { // set 3x3 matrix
        for(int j = -1; j <= 1; j++) {
            var ncoords = clamp(int2(coords) + int2(i, j), int2(0,0), size - int2(1,1));
            var val = input.Load(ncoords).x;
            var z = val - length(float2(i,j));
            //maxv += exp(-beta * z) * z;
        }
    }


    output.Store(coords, float4(lerp(maxv, avg, 0.1), 0, 0, 0));
}

[pass("sim_step_next")]
[shader("compute")]
[NumThreads(1, 1, 1)]
void sim_step_next(uint3 dispatch_id : SV_DispatchThreadID, uniform RootParams* params) {
    if (all(dispatch_id == uint3(0,0,0))) {
        params->sim_step += 1;
    }
}

[pass("sim_render")]
[shader("vertex")]
ScreenQuadVSOut sim_render_vertex_main(uint vertex_id : SV_VertexID, uniform RootParams* params) {
    return screen_quad_vertex_shader(vertex_id);
}

struct SimRenderFSOut {
    [[vk::location(0)]] float4 color;
};

[pass("sim_render")]
[shader("fragment")]
SimRenderFSOut sim_render_fragment_main(float3 frag_coord: SV_Position, ScreenQuadVSOut vertex_output, uniform RootParams* params) {
    SimRenderFSOut o;

    let input = params->sim_step % 2 == 0 ? params->trails_0 : params->trails_1;
    let screen_size = uint2(params.scene_info.screen_size);
    let coords = clamp(uint2(frag_coord.xy), uint2(0,0), screen_size - uint2(1,1));
    let data = input.Load(coords);

    o.color = float4(data.rg, 0.0, 1.0);
    return o;
}