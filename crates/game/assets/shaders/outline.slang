import common;
import scene_info;


struct ClusterEdge {
    uint16_t face_0;
    uint16_t face_1;
    uint16_t vertex_0;
    uint16_t vertex_1;
}

struct ClusterVertex {
    float3 position;
    float3 normal;
}

struct ClusterFace {
    uint16_t3 vertices;
    float3 normal;
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

// See documentation in outlines.rs
struct Cluster {
    ClusterVertex vertices[MAX_EDGES];
    ClusterEdge edges[MAX_EDGES];
    ClusterFace faces[MAX_EDGES*2];
    NormalCone normal_cone;
    uint16_t vertex_count;
    uint16_t edge_count;
    uint16_t face_count;
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
    Cluster* clusters;
    uint cluster_count;
};

[shader("vertex")]
[pass("depth_pass")]
DepthPassVertexOutput depth_pass_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform DepthPassRootParams* u)
{
    let face = u.clusters[draw_id].faces[vertex_id / 3];
    let vertex = u.clusters[draw_id].vertices[face.vertices[vertex_id % 3]];
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

struct OutlineRootParams {
    [bind("scene_info")]
    SceneInfo* scene_info;

    [bind("clusters")]
    Cluster* clusters;

    [bind("cluster_count")]
    uint cluster_count;

    uint vertex_count;

    [bind("outline_vertices")]
    OutlineVertex* out_vertices;

    [bind("outline_indices")]
    uint* out_indices;

    [bind("outline_indirect_command")]
    [len(1)]
    DrawIndirectCommand* out_draw_command;
}

[shader("compute")]
[numthreads(MAX_EDGES, 1, 1)]
[pass("outline")]
void outline_main(uint3 group_id : SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform OutlineRootParams* u)
{
    // we spawn one workgroup per edge cluster, and each thread in the workgroup processes one edge

    // cluster index
    uint cli = group_id.x;
    if (cli >= u.cluster_count) {
        return;
    }

    Cluster* c = &u.clusters[cli];

    uint edge_index = group_thread_id.x;
    if (edge_index >= c.edge_count) {
        return;
    }

    let edge = c.edges[edge_index];
    var is_outside_edge = false;
    var is_contour = false;

    if (edge.face_0 == 0xFFFF || edge.face_1 == 0xFFFF) {
        is_outside_edge = true;
    } else {
        let n0 = c.faces[edge.face_0].normal;
        let n1 = c.faces[edge.face_1].normal;
        let eye = u.scene_info.eye;
        let view = eye - c.vertices[edge.vertex_0].position;
        if (dot(n0, view) * dot(n1, view) < 0.0) {
            is_contour = true;
        }
    }

    float2 px_size = 2. / u.scene_info.screen_size;
    float width = 1.0;
    float filter_width = 0.7;
    float hw_aa = max(width, 1.) * .5; //+ filter_width * sqrt(2.);

    if (is_contour || is_outside_edge) {
        var v0 = mul(u.scene_info.view_proj_matrix, float4(c.vertices[edge.vertex_0].position, 1.));
        var v1 = mul(u.scene_info.view_proj_matrix, float4(c.vertices[edge.vertex_1].position, 1.));
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
};

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