import common;
import scene_info;

static const uint16_t CLUSTER_EDGE_FLAG_BORDER = 1 << 0;
static const uint16_t CLUSTER_EDGE_FLAG_CREASE = 1 << 1;

struct ClusterEdge {
    uint16_t normal_a;
    uint16_t normal_b;
    uint16_t vertex_0;
    uint16_t vertex_1;
}

struct ClusterVertex {
    float3 position;
    float3 normal;
}

struct ClusterFace {
    uint16_t3 vertices;
}

struct Meshlet {
    NormalCone normal_cone;
    uint base_vertex;
    uint base_edge;
    uint base_face;
    uint base_face_normal;
    uint16_t vertex_count;
    uint16_t edge_count;
    uint16_t face_count;
    uint16_t face_normal_count;
}

// See documentation in outlines.rs
struct MeshData {
    ClusterVertex* vertices;
    ClusterEdge* edges;
    ClusterFace* faces;
    float3* face_normals;
    Meshlet* meshlets;
}

struct NormalCone {
    float3 axis;
    float angle;
    float3 center;
    float radius;
}

struct OutlineVertex {
    float4 clip_position;
    float angle;
    float reserved0;
    float reserved1;
    float reserved2;
}

static const uint MAX_EDGES = 128;

ClusterVertex fetch_mesh_vertex(MeshData data, uint draw_id, uint vertex_id) {
    let base_face = data.meshlets[draw_id].base_face;
    let base_vertex = data.meshlets[draw_id].base_vertex;
    let face = data.faces[base_face + vertex_id / 3];
    let vertex = data.vertices[base_vertex + face.vertices[vertex_id % 3]];
    return vertex;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Depth pass
///////////////////////////////////////////////////////////////////////////////////////////////////

struct DepthPassVertexOutput {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 local_pos;
};

struct DepthPassRootParams {
    SceneInfo* scene_info;
    MeshData mesh;
};

[shader("vertex")]
[pass("depth_pass")]
DepthPassVertexOutput depth_pass_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform DepthPassRootParams* u)
{
    let vertex = fetch_mesh_vertex(u.mesh, draw_id, vertex_id);
    DepthPassVertexOutput o;
    o.pos = mul(u.scene_info.view_proj_matrix, float4(vertex.position, 1.0));
    return o;
}

[shader("fragment")]
[pass("depth_pass")]
void depth_pass_fragment(DepthPassVertexOutput i) {
    // nothing
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Outline extraction
///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExtractOutlinesRootParams {
    SceneInfo* scene_info;
    MeshData mesh;
    uint cluster_count;
    uint vertex_count;
    OutlineVertex* out_vertices;
    uint* out_indices;
    DrawIndirectCommand* out_draw_command;
}

[shader("compute")]
[numthreads(MAX_EDGES, 1, 1)]
[pass("extract_outlines")]
void extract_outlines(uint3 group_id : SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform ExtractOutlinesRootParams* u)
{
    // we spawn one workgroup per edge cluster, and each thread in the workgroup processes one edge

    // cluster index
    uint cli = group_id.x;
    if (cli >= u.cluster_count) {
        return;
    }

    Meshlet* m = &u.mesh.meshlets[cli];
    let base_edge = m.base_edge;
    let base_vertex = m.base_vertex;
    let base_face_normal = m.base_face_normal;

    uint edge_index = group_thread_id.x;
    if (edge_index >= m.edge_count) {
        return;
    }

    let edge = u.mesh.edges[base_edge + edge_index];
    var is_outside_edge = false;
    var is_contour = false;

    if (edge.normal_a == 0xFFFF || edge.normal_b == 0xFFFF) {
        is_outside_edge = true;
    } else {
        let n0 = u.mesh.face_normals[base_face_normal + edge.normal_a];
        let n1 = u.mesh.face_normals[base_face_normal + edge.normal_b];
        let eye = u.scene_info.eye;
        let view = eye - u.mesh.vertices[base_vertex + edge.vertex_0].position;
        if (dot(n0, view) * dot(n1, view) < 0.0) {
            is_contour = true;
        }
    }

    float2 px_size = 2. / u.scene_info.screen_size;
    float width = 1.0;
    float filter_width = 0.7;
    float hw_aa = max(width, 1.) * .5; //+ filter_width * sqrt(2.);

    if (is_contour || is_outside_edge) {
        var v0 = mul(u.scene_info.view_proj_matrix, float4(u.mesh.vertices[base_vertex + edge.vertex_0].position, 1.));
        var v1 = mul(u.scene_info.view_proj_matrix, float4(u.mesh.vertices[base_vertex + edge.vertex_1].position, 1.));
        var v0offset = v0;
        var v1offset = v1;

        float2 v = (v0.xy/v0.w - v1.xy/v1.w) / px_size;
        float2 n = px_size * normalize(float2(-v.y, v.x));

        let angle = atan(n.y / n.x);

        v0.xy -= hw_aa * n * v0.w;
        v1.xy -= hw_aa * n * v1.w;
        v0offset.xy += hw_aa * n * v0offset.w;
        v1offset.xy += hw_aa * n * v1offset.w;

        uint base_vertex;
        InterlockedAdd(u.vertex_count, 4, base_vertex);
        uint base_index;
        InterlockedAdd(u.out_draw_command.vertex_count, 6, base_index);

        //let v0dbg = float4(2.0 * float2(base_index, cli*7) / u.scene_info.screen_size - 1.0, 0.0, 1.0);
        //let v1dbg = float4(2.0 * float2(base_index + 6, cli*7) / u.scene_info.screen_size - 1.0, 0.0, 1.0);
        //let v2dbg = float4(2.0 * float2(base_index, cli*7 + 6) / u.scene_info.screen_size - 1.0, 0.0, 1.0);
        //let v3dbg = float4(2.0 * float2(base_index + 6, cli*7 + 6) / u.scene_info.screen_size - 1.0, 0.0, 1.0);

        let outside_edge_factor = is_outside_edge ? 1.0 : 0.0;

        u.out_vertices[base_vertex] = OutlineVertex(v0, angle, outside_edge_factor);
        u.out_vertices[base_vertex+1] = OutlineVertex(v1, angle, outside_edge_factor);
        u.out_vertices[base_vertex+2] = OutlineVertex(v0offset, angle, outside_edge_factor);
        u.out_vertices[base_vertex+3] = OutlineVertex(v1offset, angle, outside_edge_factor);

        u.out_indices[base_index] = base_vertex;
        u.out_indices[base_index+1] = base_vertex + 1;
        u.out_indices[base_index+2] = base_vertex + 2;
        u.out_indices[base_index+3] = base_vertex + 1;
        u.out_indices[base_index+4] = base_vertex + 3;
        u.out_indices[base_index+5] = base_vertex + 2;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Render outlines
///////////////////////////////////////////////////////////////////////////////////////////////////

struct RenderOutlinesVSOut {
    float4 pos: SV_Position;
    float angle;
    float4 color;
    float2 local_pos;
};

struct RenderOutlineRootParams {
    SceneInfo* scene_info;
    OutlineVertex* vertices;
    uint* indices;
    Texture2D.Handle depth;
};

[shader("vertex")]
[pass("render_outlines")]
RenderOutlinesVSOut render_outlines_vertex(uint vertex_id : SV_VertexID, uniform RenderOutlineRootParams* u)
{
    let index = u.indices[vertex_id];
    let vertex = u.vertices[index];
    RenderOutlinesVSOut o;
    o.pos = vertex.clip_position;
    o.angle = vertex.angle;
    //if (vertex.reserved0 > 0.5) {
    //    o.color = float4(1, 1, 1, 1);
    //}
    return o;
}

struct RenderOutlinesFSOut {
    [[vk::location(0)]] float4 color;
    float depth: SV_Depth;
};

[shader("fragment")]
[pass("render_outlines")]
RenderOutlinesFSOut render_outlines_fragment(
    float3 frag_coord: SV_Position,
    RenderOutlinesVSOut vertex_output,
    uniform RenderOutlineRootParams* params)
{
    int3 coord = int3(int2(frag_coord.xy), 0);

    int test = 0;
    const float ref_depth = frag_coord.z;
    test += (ref_depth <= params.depth.Load(int3(coord.x - 1, coord.y - 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth.Load(int3(coord.x,     coord.y - 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth.Load(int3(coord.x + 1, coord.y - 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth.Load(int3(coord.x - 1, coord.y,     0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth.Load(int3(coord.x,     coord.y,     0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth.Load(int3(coord.x + 1, coord.y,     0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth.Load(int3(coord.x - 1, coord.y + 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth.Load(int3(coord.x,     coord.y + 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth.Load(int3(coord.x + 1, coord.y + 1, 0), 0).r) ? 1 : 0;

    //float4 color = vertex_output.color;
    if (test < 2) {
        discard;
        //color = float4(0.3, 0.3, 0.6, 1);
    }

    RenderOutlinesFSOut o;
    o.color = float4(vertex_output.angle,1,0,1);
    o.depth = frag_coord.z-0.1;
    return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Corner detection
///////////////////////////////////////////////////////////////////////////////////////////////////

struct CornerDetectionRootParams {
    SceneInfo* scene_info;
    Texture2D.Handle angle_tex;
};

[shader("vertex")]
[pass("corner_detection")]
ScreenQuadVSOut corner_detection_vertex_shader(uint vertex_id : SV_VertexID)
{
    return screen_quad_vertex_shader(vertex_id);
}

struct CornerDetectionFSOut {
    [[vk::location(0)]] float4 color;
    float depth: SV_Depth;
}

[shader("fragment")]
[pass("corner_detection")]
CornerDetectionFSOut corner_detection_fragment_shader(
    float3 frag_coord: SV_Position,
    ScreenQuadVSOut vertex_output,
    uniform CornerDetectionRootParams* params)
{
    int3 coord = int3(int2(frag_coord.xy), 0);

    //int test = 0;
    //const float ref_depth = frag_coord.z;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x - 1, coord.y - 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x,     coord.y - 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x + 1, coord.y - 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x - 1, coord.y,     0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x,     coord.y,     0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x + 1, coord.y,     0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x - 1, coord.y + 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x,     coord.y + 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.angle_tex.Load(int3(coord.x + 1, coord.y + 1, 0), 0).r) ? 1 : 0;

    var angle = params.angle_tex.Load(int3(coord.x, coord.y, 0), 0);
    var color = angle_to_color(angle.r);
    color.a = angle.g;

   //if (test < 2) {
   //    discard;
   //    //color = float4(0.3, 0.3, 0.6, 1);
   //}

    CornerDetectionFSOut o;
    o.color = color;
    o.depth = frag_coord.z-0.1;
    return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Base rendering
///////////////////////////////////////////////////////////////////////////////////////////////////

struct BaseRenderVSOut {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] nointerpolation int meshlet_id;
}

struct BaseRenderFSOut {
    [[vk::location(0)]] float4 color;
}

struct BaseRenderRootParams {
    SceneInfo* scene_info;
    MeshData mesh;
    float3 main_light_direction;    // in world space

    //uint cluster_count;
}

[shader("vertex")]
[pass("base_render")]
BaseRenderVSOut base_render_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform BaseRenderRootParams* u)
{
    let vertex = fetch_mesh_vertex(u.mesh, draw_id, vertex_id);
    BaseRenderVSOut o;
    o.pos = mul(u.scene_info.view_proj_matrix, float4(vertex.position, 1.0));
    o.normal = mul(float3x3(u.scene_info.view_matrix), vertex.normal);
    o.meshlet_id = draw_id;
    return o;
}

[shader("fragment")]
[pass("base_render")]
BaseRenderFSOut base_render_fragment(BaseRenderVSOut i, uniform BaseRenderRootParams* u) {
    BaseRenderFSOut o;
    float3 normal = normalize(i.normal);
    float light_intensity = saturate(dot(normal, -normalize(u.main_light_direction)));
    o.color = float4(light_intensity.xxx, 1.0);

    // colorize per cluster
    //o.color.rgb = palette(float(i.meshlet_id) * 0.1, float3(0.5,0.5,0.5),float3(0.5,0.5,0.5),float3(1.0,1.0,1.0),float3(0.0,0.33,0.67));


    return o;
}