import common;
import scene_info;

static const uint16_t CLUSTER_EDGE_FLAG_BORDER = 1 << 0;
static const uint16_t CLUSTER_EDGE_FLAG_CREASE = 1 << 1;

struct ClusterEdge {
    uint16_t normal_cw;
    uint16_t normal_ccw;
    uint16_t vertex_0;
    uint16_t vertex_1;
}

struct ClusterVertex {
    float3 position;
    uint32_t point_index;
    float3 normal;
    float pointy;
}

struct ClusterFace {
    uint16_t3 vertices;
    uint16_t3 edges;
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
    uint16_t group_id;
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
    // clip-space position
    float4 clip_position;
    // normal orientation in screen space
    float angle;
    // CLUSTER_EDGE_FLAGS*
    uint16_t flags;
    // sub-object ID
    uint16_t group_id;
    // Spikiness
    float pointy;
    // Global point index
    uint32_t rank;
}

struct ContourEdge {
    float3 position_0;
    uint32_t point_0;
    float3 position_1;
    uint32_t point_1;
    uint16_t group_id;
}

struct ContourEdgeBuffer {
    uint32_t count;
    ContourEdge edges[];
}

struct ContourPoint {
    float3 position;
    uint group_id;
    uint next;
}

static const uint MAX_EDGES = 128;

// Common root params for contour extraction and rendering
struct ContourRootParams {
    // input mesh
    SceneInfo* scene_info;
    MeshData mesh;
    uint cluster_count;
    float3 eye;

    // contour extraction output
    ContourEdgeBuffer* contour_edges_old;
    uint contour_point_count;
    ContourPoint* contour_point_list;
    ContourPoint* contour_point_list_subdiv;
    uint global_point_count;
    uint* global_to_contour_index_map;       // global point index -> contour point index

    // contour ranking data
    uint64_t* contours_rank_successors_0;
    uint64_t* contours_rank_successors_1;

    // expanded contour data
    uint expanded_contour_vertex_count;
    OutlineVertex* expanded_contour_vertices;
    uint* expanded_contour_indices;
    DrawIndirectCommand* expanded_contours_draw_command;

    // shading
    float3 main_light_direction;    // in world space
    Srgba8 silhouette_color;

    // textures
    Texture2D.Handle depth_texture;
    Texture2D<uint16_t4>.Handle angle_texture;
    Texture2D<float4>.Handle normal_texture;

    // JFA
    RWTexture2D<int16_t2>.Handle jfa_result;
    RWTexture2D<int16_t2>.Handle jfa_in_texture;
    RWTexture2D<int16_t2>.Handle jfa_out_texture;
    uint jfa_step_size;
}


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

[shader("vertex")]
[pass("depth_pass")]
DepthPassVertexOutput depth_pass_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform ContourRootParams* u)
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
// Base render + depth pre-pass
///////////////////////////////////////////////////////////////////////////////////////////////////

struct BaseRenderVSOut {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] nointerpolation int meshlet_id;
}

struct BaseRenderFSOut {
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float3 normal;
}


[shader("vertex")]
[pass("base_render")]
BaseRenderVSOut base_render_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform ContourRootParams* u)
{
    let vertex = fetch_mesh_vertex(u.mesh, draw_id, vertex_id);
    BaseRenderVSOut o;
    o.pos = mul(u.scene_info.view_proj_matrix, float4(vertex.position, 1.0));
    o.normal = vertex.normal;// mul(float3x3(u.scene_info.view_matrix), vertex.normal);
    o.meshlet_id = draw_id;
    return o;
}

[shader("fragment")]
[pass("base_render")]
BaseRenderFSOut base_render_fragment(BaseRenderVSOut i, uniform ContourRootParams* u) {
    BaseRenderFSOut o;
    float3 normal = normalize(i.normal);
    float light_intensity = saturate(dot(normal, -normalize(u.main_light_direction)));
    o.color = srgba8_to_float4(u.silhouette_color);
    o.normal = normal;
    // colorize per cluster
    //o.color.rgb = palette(float(i.meshlet_id) * 0.1, float3(0.5,0.5,0.5),float3(0.5,0.5,0.5),float3(1.0,1.0,1.0),float3(0.0,0.33,0.67));
    o.color.rgb = normal;
    return o;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Outline extraction
///////////////////////////////////////////////////////////////////////////////////////////////////


[shader("compute")]
[numthreads(MAX_EDGES, 1, 1)]
[pass("extract_contours")]
void extract_contours(uint3 group_id : SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform ContourRootParams* u)
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
    var is_boundary = false;
    var is_contour = false;
    var reverse_edge = false;

    if (edge.normal_cw == 0xFFFF || edge.normal_ccw == 0xFFFF) {
        is_boundary = true;
    } else {
        let n_cw = u.mesh.face_normals[base_face_normal + edge.normal_cw];
        let n_ccw = u.mesh.face_normals[base_face_normal + edge.normal_ccw];
        let view = u.eye - u.mesh.vertices[base_vertex + edge.vertex_0].position;
        if (dot(n_cw, view) * dot(n_ccw, view) < 0.0) {
            // cw face is backfacing, reverse edge direction so that it follows the front-facing polygon
            reverse_edge = dot(n_cw, view) < 0.0;
            is_contour = true;
            let ncross = cross(n_ccw, n_cw);
            let edge_dir = u.mesh.vertices[base_vertex + edge.vertex_1].position - u.mesh.vertices[base_vertex + edge.vertex_0].position;
            //if (dot(edge_dir, ncross) < -0.01) {
            //    // concave edge, not a contour
            //    is_contour = false;
            //}
        }
    }

    if (is_contour || is_boundary) {
        //// write edge
        //uint edge_ptr;
        //InterlockedAdd(u.contour_edges.count, 1, edge_ptr);
        //u.contour_edges.edges[edge_ptr] = ContourEdge(
        //    u.mesh.vertices[base_vertex + edge.vertex_0].position,
        //    u.mesh.vertices[base_vertex + edge.vertex_0].point_index,
        //    u.mesh.vertices[base_vertex + edge.vertex_1].position,
        //    u.mesh.vertices[base_vertex + edge.vertex_1].point_index,
        //    m.group_id
        //);

        // insert points in index map

        uint contour_p0_index;
        InterlockedAdd(u.contour_point_count, 1, contour_p0_index);

        var p0_index = u.mesh.vertices[base_vertex + edge.vertex_0].point_index;
        var p0_pos = u.mesh.vertices[base_vertex + edge.vertex_0].position;
        var p1_index = u.mesh.vertices[base_vertex + edge.vertex_1].point_index;
        var p1_pos = u.mesh.vertices[base_vertex + edge.vertex_1].position;
        if (reverse_edge) {
            let temp = p0_index;
            p0_index = p1_index;
            p1_index = temp;
            let temp_pos = p0_pos;
            p0_pos = p1_pos;
            p1_pos = temp_pos;
        }

        u.global_to_contour_index_map[p0_index] = contour_p0_index;
        u.contour_point_list[contour_p0_index].position = p0_pos;
        u.contour_point_list[contour_p0_index].group_id = m.group_id;
        u.contour_point_list[contour_p0_index].next = p1_index;   // replaced with contour point index in a later pass
    }
}

float point_orientation(float3 point, float3 cam, float3 normal) {
    let view = normalize(point - cam);
    return dot(normalize(normal), view);
}

[shader("compute")]
[numthreads(MAX_EDGES, 1, 1)]
[pass("extract_interpolated_contours")]
void extract_interpolated_contours(uint3 group_id : SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform ContourRootParams* u)
{
    uint cli = group_id.x;
    if (cli >= u.cluster_count) {
        return;
    }

    Meshlet* m = &u.mesh.meshlets[cli];
    let base_face = m.base_face;
    let base_vertex = m.base_vertex;

    uint face_index = group_thread_id.x;
    if (face_index >= m.face_count) {
        return;
    }

    let face = u.mesh.faces[base_face + face_index];
    let v1 = u.mesh.vertices[base_vertex + face.vertices.x];
    let v2 = u.mesh.vertices[base_vertex + face.vertices.y];
    let v3 = u.mesh.vertices[base_vertex + face.vertices.z];
    let o1 = point_orientation(v1.position, u.eye, v1.normal);
    let o2 = point_orientation(v2.position, u.eye, v2.normal);
    let o3 = point_orientation(v3.position, u.eye, v3.normal);
    if (sign(o1) == sign(o2) && sign(o2) == sign(o3)) {
        // no contour crossing
        return;
    }
    float3 interp1;
    float3 interp2;
    uint pkpt_1;
    uint pkpt_2;
    let flip1 = sign(o1) != sign(o2);
    let flip2 = sign(o2) != sign(o3);
    if (flip1 && flip2) {
        // e1, e2
        interp1 = (o2 * v1.position - o1 * v2.position) / (o2 - o1);
        interp2 = (o3 * v2.position - o2 * v3.position) / (o3 - o2);
        pkpt_1 = v1.point_index;
        pkpt_2 = v2.point_index;
    } else if (flip1) {
        // e1, e3
        interp1 = (o2 * v1.position - o1 * v2.position) / (o2 - o1);
        interp2 = (o1 * v3.position - o3 * v1.position) / (o1 - o3);
        pkpt_1 = v1.point_index;
        pkpt_2 = v3.point_index;
    } else {
        // e2, e3
        interp1 = (o3 * v2.position - o2 * v3.position) / (o3 - o2);
        interp2 = (o1 * v3.position - o3 * v1.position) / (o1 - o3);
        pkpt_1 = v2.point_index;
        pkpt_2 = v3.point_index;
    }

    uint e_i;
    InterlockedAdd(u.contour_edges_old.count, 1, e_i);
    u.contour_edges_old.edges[e_i] = ContourEdge(
        interp1,
        pkpt_1,
        interp2,
        pkpt_2,
        m.group_id
    );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Contour list ranking and reordering
///////////////////////////////////////////////////////////////////////////////////////////////////

static const uint SENTINEL = 0xFFFF_FFFF;
static const uint ROOT = 0x8000_0000;
static const uint RANK_BLOCK_SIZE = 256;

struct RankContoursRootParams {
     ContourRootParams* common;
     uint len;
     uint64_t* contours_rank_successors_0;
     uint64_t* contours_rank_successors_1;
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("break_contours_init")]
void break_contours_init(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i < u.len) {
        // extract_contours outputs global point indices as successors; convert to local contour point indices
        let global_next = u.common.contour_point_list[i].next;
        let local_next = global_next == SENTINEL ? SENTINEL : u.common.global_to_contour_index_map[global_next];
        u.common.contour_point_list[i].next = local_next;

        // rank/successors
        u.contours_rank_successors_0[i] = (uint64_t(local_next) << 32) | (local_next == SENTINEL ? 0 : i);
        u.contours_rank_successors_1[i] = (uint64_t(local_next) << 32) | (local_next == SENTINEL ? 0 : i);
    }
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("break_contours_step")]
void break_contours_step(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i >= u.len)
        return;

    let sr = u.contours_rank_successors_0[i];
    let s = uint(sr >> 32);
    let r = uint(sr & 0xFFFFFFFF);
    if (s == SENTINEL || s == i) {
        u.contours_rank_successors_1[i] = sr;
        return;
    }

    let ssr = u.contours_rank_successors_0[s];
    let ss = uint(ssr >> 32);
    let rr = uint(ssr & 0xFFFFFFFF);
    u.contours_rank_successors_1[i] = (uint64_t(ss) << 32) | min(r, rr);
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("rank_contours_init")]
void rank_contours_init(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i < u.len) {
        var successor = u.common.contour_point_list[i].next;
        let root = u.contours_rank_successors_0[i] & 0xFFFFFFFF;
        var init_rank = 1;
        if (root == i || successor == SENTINEL) {
            init_rank = 0;
            successor = SENTINEL;
        }
        u.contours_rank_successors_0[i] = (uint64_t(successor) << 32) | init_rank;
        u.contours_rank_successors_1[i] = (uint64_t(successor) << 32) | init_rank;
    }
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("rank_contours_step")]
void rank_contours_step(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i >= u.len)
       return;

    let sr = u.contours_rank_successors_0[i];
    let s = uint(sr >> 32);
    let r = uint(sr & 0xFFFFFFFF);
    if (s == SENTINEL || s == i) {
        u.contours_rank_successors_1[i] = sr;
        return;
    }

    let ssr = u.contours_rank_successors_0[s];
    let ss = uint(ssr >> 32);
    let rr = uint(ssr & 0xFFFFFFFF);
    u.contours_rank_successors_1[i] = (uint64_t(ss) << 32) | (r + rr);
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("rank_contours_finalize")]
void rank_contours_finalize(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i < u.len) {
        let sr = u.contours_rank_successors_0[i];
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Subdivide contours
///////////////////////////////////////////////////////////////////////////////////////////////////

static const uint SUBDIVIDE_CONTOURS_GROUP_SIZE = 256;

struct SubdivideContoursRootParams {
    ContourRootParams* common;
    uint point_count;
}

[pass("setup_subdivide_contours")]
[shader("compute")]
[numthreads(1, 1, 1)]
void setup_subdivide_contours(uint3 thread_id : SV_DispatchThreadID, uniform SubdivideContoursRootParams* u) {
    u.point_count = u.common.contour_point_count;
}

[pass("subdivide_contours")]
[shader("compute")]
[numthreads(SUBDIVIDE_CONTOURS_GROUP_SIZE, 1, 1)]
void subdivide_contours(uint3 thread_id : SV_DispatchThreadID, uniform SubdivideContoursRootParams* u) {
    let i = thread_id.x;
    if (i >= u.point_count) {
        return;
    }

    let a = u.common.contour_point_list[i];
    let group_id = a.group_id;
    if (a.next == SENTINEL) {
        return;
    }
    let b = u.common.contour_point_list[a.next];

    let c_position = 0.25 * a.position + 0.75 * b.position;
    let a_position = 0.75 * a.position + 0.25 * b.position;

    // Allocate new point
    uint c_index;
    InterlockedAdd(u.common.contour_point_count, 1, c_index);

    // Update positions and links
    u.common.contour_point_list_subdiv[i].position = a_position;
    u.common.contour_point_list_subdiv[i].group_id = group_id;
    u.common.contour_point_list_subdiv[i].next = c_index;
    u.common.contour_point_list_subdiv[c_index].position = c_position;
    u.common.contour_point_list_subdiv[c_index].group_id = group_id;
    u.common.contour_point_list_subdiv[c_index].next = a.next;
}


[pass("finish_subdivide_contours")]
[shader("compute")]
[numthreads(SUBDIVIDE_CONTOURS_GROUP_SIZE, 1, 1)]
void finish_subdivide_contours(uint3 thread_id : SV_DispatchThreadID, uniform SubdivideContoursRootParams* u) {
    let i = thread_id.x;
    if (i >= u.common.contour_point_count) {
        return;
    }
    u.common.contour_point_list[i] = u.common.contour_point_list_subdiv[i];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Expand outlines
///////////////////////////////////////////////////////////////////////////////////////////////////

static const uint EXPAND_CONTOURS_GROUP_SIZE = 256;

[pass("expand_contours")]
[shader("compute")]
[numthreads(EXPAND_CONTOURS_GROUP_SIZE, 1, 1)]
void expand_contours(uint3 thread_id : SV_DispatchThreadID, uniform ContourRootParams* u) {
    let i = thread_id.x;
    if (i >= u.contour_point_count) {
        return;
    }

    let edge_a = u.contour_point_list[i];
    if (edge_a.next == SENTINEL) {
        return;
    }
    let edge_b = u.contour_point_list[edge_a.next];

    let p0 = i;
    let p1 = edge_a.next;

    float2 px_size = 2. / u.scene_info.screen_size;
    float width = 1.0;
    float filter_width = 0.7;
    float hw_aa = max(width, 1.) * .5; //+ filter_width * sqrt(2.);

    var v0 = mul(u.scene_info.view_proj_matrix, float4(edge_a.position, 1.));
    var v1 = mul(u.scene_info.view_proj_matrix, float4(edge_b.position, 1.));
    var v0offset = v0;
    var v1offset = v1;

    float2 v = (v0.xy/v0.w - v1.xy/v1.w) / px_size;
    float2 n = px_size * normalize(float2(-v.y, v.x));

    let angle = atan(n.y / n.x);

    v0.xy -= hw_aa * n * v0.w;
    v1.xy -= hw_aa * n * v1.w;
    v0offset.xy += hw_aa * n * v0offset.w;
    v1offset.xy += hw_aa * n * v1offset.w;

    uint vtx_ptr;
    uint idx_ptr;
    InterlockedAdd(u.expanded_contour_vertex_count, 4, vtx_ptr);
    InterlockedAdd(u.expanded_contours_draw_command.vertex_count, 6, idx_ptr);

    uint16_t flags = /*is_outside_edge ? CLUSTER_EDGE_FLAG_BORDER :*/ 0;

    let pointy_0 = 0.0; //u.mesh.vertices[base_vertex + edge.vertex_0].pointy;
    let pointy_1 = 1.0; //u.mesh.vertices[base_vertex + edge.vertex_1].pointy;
    let rank_0 = u.contours_rank_successors_0[p0] & 0xFFFFFFFF;
    let rank_1 = u.contours_rank_successors_0[p1] & 0xFFFFFFFF;
    let root_0 = u.contours_rank_successors_0[p0] >> 32;
    let root_1 = u.contours_rank_successors_0[p1] >> 32;

    u.expanded_contour_vertices[vtx_ptr] = OutlineVertex(v0, angle, flags, edge_a.group_id, pointy_0, rank_0);
    u.expanded_contour_vertices[vtx_ptr+1] = OutlineVertex(v1, angle, flags, edge_b.group_id, pointy_1, rank_1);
    u.expanded_contour_vertices[vtx_ptr+2] = OutlineVertex(v0offset, angle, flags, edge_a.group_id, pointy_0, rank_0);
    u.expanded_contour_vertices[vtx_ptr+3] = OutlineVertex(v1offset, angle, flags, edge_b.group_id, pointy_1, rank_1);

    u.expanded_contour_indices[idx_ptr] = vtx_ptr;
    u.expanded_contour_indices[idx_ptr+1] = vtx_ptr + 1;
    u.expanded_contour_indices[idx_ptr+2] = vtx_ptr + 2;
    u.expanded_contour_indices[idx_ptr+3] = vtx_ptr + 1;
    u.expanded_contour_indices[idx_ptr+4] = vtx_ptr + 3;
    u.expanded_contour_indices[idx_ptr+5] = vtx_ptr + 2;
}


[pass("expand_interpolated_contours")]
[shader("compute")]
[numthreads(EXPAND_CONTOURS_GROUP_SIZE, 1, 1)]
void expand_interpolated_contours(uint3 thread_id : SV_DispatchThreadID, uniform ContourRootParams* u) {
    let i = thread_id.x;
    if (i >= u.contour_edges_old.count) {
        return;
    }

    let edge = u.contour_edges_old.edges[i];

    float2 px_size = 2. / u.scene_info.screen_size;
    float width = 1.0;
    float filter_width = 0.7;
    float hw_aa = max(width, 1.) * .5; //+ filter_width * sqrt(2.);

    var v0 = mul(u.scene_info.view_proj_matrix, float4(edge.position_0, 1.));
    var v1 = mul(u.scene_info.view_proj_matrix, float4(edge.position_1, 1.));
    var v0offset = v0;
    var v1offset = v1;

    float2 v = (v0.xy/v0.w - v1.xy/v1.w) / px_size;
    float2 n = px_size * normalize(float2(-v.y, v.x));

    let angle = atan(n.y / n.x);

    v0.xy -= hw_aa * n * v0.w;
    v1.xy -= hw_aa * n * v1.w;
    v0offset.xy += hw_aa * n * v0offset.w;
    v1offset.xy += hw_aa * n * v1offset.w;

    uint vtx_ptr;
    uint idx_ptr;
    InterlockedAdd(u.expanded_contour_vertex_count, 4, vtx_ptr);
    InterlockedAdd(u.expanded_contours_draw_command.vertex_count, 6, idx_ptr);

    uint16_t flags = /*is_outside_edge ? CLUSTER_EDGE_FLAG_BORDER :*/ 0;

    let pointy_0 = 0.0; //u.mesh.vertices[base_vertex + edge.vertex_0].pointy;
    let pointy_1 = 1.0; //u.mesh.vertices[base_vertex + edge.vertex_1].pointy;

    u.expanded_contour_vertices[vtx_ptr] = OutlineVertex(v0, angle, flags, edge.group_id, pointy_0, 0);
    u.expanded_contour_vertices[vtx_ptr+1] = OutlineVertex(v1, angle, flags, edge.group_id, pointy_1, 0);
    u.expanded_contour_vertices[vtx_ptr+2] = OutlineVertex(v0offset, angle, flags, edge.group_id, pointy_0, 0);
    u.expanded_contour_vertices[vtx_ptr+3] = OutlineVertex(v1offset, angle, flags, edge.group_id, pointy_1, 0);

    u.expanded_contour_indices[idx_ptr] = vtx_ptr;
    u.expanded_contour_indices[idx_ptr+1] = vtx_ptr + 1;
    u.expanded_contour_indices[idx_ptr+2] = vtx_ptr + 2;
    u.expanded_contour_indices[idx_ptr+3] = vtx_ptr + 1;
    u.expanded_contour_indices[idx_ptr+4] = vtx_ptr + 3;
    u.expanded_contour_indices[idx_ptr+5] = vtx_ptr + 2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Render outlines
///////////////////////////////////////////////////////////////////////////////////////////////////

struct RenderOutlinesVSOut {
    float4 pos: SV_Position;
    float angle;
    float pointy;
    nointerpolation uint group_id;
    uint rank;
};

[pass("render_outlines")]
[shader("vertex")]
RenderOutlinesVSOut render_outlines_vertex(uint vertex_id : SV_VertexID, uniform ContourRootParams* u)
{
    let index = u.expanded_contour_indices[vertex_id];
    let vertex = u.expanded_contour_vertices[index];
    RenderOutlinesVSOut o;
    o.pos = vertex.clip_position;
    o.angle = vertex.angle;
    o.pointy = vertex.pointy;
    o.group_id = vertex.group_id;
    o.rank = vertex.rank;
    return o;
}

struct RenderOutlinesFSOut {
    [[vk::location(0)]] uint4 color;
    float depth: SV_Depth;
};

[pass("render_outlines")]
[shader("fragment")]
RenderOutlinesFSOut render_outlines_fragment(
    float3 frag_coord: SV_Position,
    RenderOutlinesVSOut vsout,
    uniform ContourRootParams* params)
{
    int3 coord = int3(int2(frag_coord.xy), 0);

    int test = 0;
    const float ref_depth = frag_coord.z;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x - 1, coord.y - 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x,     coord.y - 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x + 1, coord.y - 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x - 1, coord.y,     0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x,     coord.y,     0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x + 1, coord.y,     0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x - 1, coord.y + 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x,     coord.y + 1, 0), 0).r) ? 1 : 0;
    test += (ref_depth <= params.depth_texture.Load(int3(coord.x + 1, coord.y + 1, 0), 0).r) ? 1 : 0;

    //float4 color = vsout.color;
    if (test < 2) {
        //discard;
        //color = float4(0.3, 0.3, 0.6, 1);
    }

    RenderOutlinesFSOut o;
    o.color = uint4(vsout.group_id, uint((vsout.angle / TAU) * 65535.), vsout.pointy * 65535, 1);
    o.depth = frag_coord.z-0.1;
    return o;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// Corner detection
///////////////////////////////////////////////////////////////////////////////////////////////////

[pass("corner_detection")]
[shader("vertex")]
ScreenQuadVSOut corner_detection_vertex_shader(uint vertex_id : SV_VertexID)
{
    return screen_quad_vertex_shader(vertex_id);
}

struct CornerDetectionFSOut {
    [[vk::location(0)]] float4 color;
}

[pass("corner_detection")]
[shader("fragment")]
CornerDetectionFSOut corner_detection_fragment_shader(
    float3 frag_coord: SV_Position,
    ScreenQuadVSOut vertex_output,
    uniform ContourRootParams* params)
{
    int3 coord = int3(int2(frag_coord.xy), 0);
    int2 size = int2(params.scene_info.screen_size);

    static const int W = 3;
    var accum = 0.0;
    var weight_sum = 0.0;
    for (int i = -W; i <= W; ++i) {
        for (int j = -W; j <= W; ++j) {
            let coord_ij = clamp(coord.xy + int2(i, j), int2(0,0), size - int2(1,1));
            let data = params.angle_texture.Load(int3(coord_ij, 0));
            let angle = (float(data.y) / 65535.0) * TAU;
            let f = float2(cos(angle), sin(angle));
            let n = normalize(float2(i, j));
            let dp = dot(f, n);
            accum += dp * data.a;
            weight_sum += data.a;
        }
    }

    let div = accum;

    let tex00 = params.angle_texture.Load(int3(coord.x, coord.y, 0));
    //let tex10 = params.angle_tex.Load(int3(coord.x + 1, coord.y, 0));
    //let tex01 = params.angle_tex.Load(int3(coord.x, coord.y + 1, 0));
    //let tex11 = params.angle_tex.Load(int3(coord.x + 1, coord.y + 1, 0));
    //let angle00 = (float(tex00.y) / 65535.0) * TAU;
    //let angle10 = (float(tex10.y) / 65535.0) * TAU;
    //let angle01 = (float(tex01.y) / 65535.0) * TAU;
    //let angle11 = (float(tex11.y) / 65535.0) * TAU;
    //let n00 = float3(cos(angle00), sin(angle00), 0.0);
    //let n10 = float3(cos(angle10), sin(angle10), 0.0);
    //let n01 = float3(cos(angle01), sin(angle01), 0.0);
    //let n11 = float3(cos(angle11), sin(angle11), 0.0);
    //let dnxdx = (n10.x - n00.x) * tex00.a * tex10.a;
    //let dnydy = (n01.y - n00.y) * tex00.a * tex01.a;
    //let div = dnxdx + dnydy;

    let jfa_value = int2(params.jfa_result.Load(coord.xy));
    let ref_outline = params.angle_texture.Load(int3(jfa_value,0));

    let dist = length(float2(jfa_value - coord.xy));
    let width = 1.0;
    let aastep = fwidth(dist);
    let div_factor = smoothstep(width, width - aastep, dist);

    var color = float4(colorize_id(tex00.r*10), tex00.a);
    //color = float4(angle_to_color(float(tex00.b)/10.0).rgb, tex00.a);
    //color = float4(colorize_1d(tex00.b / 65535.0).rgb, tex00.a);

    //color.rgb = colorize_1d(dist * 0.1);

    //color.rgb = colorize_1d(div);

    //let angle = (float(tex00.y) / 65535.0) * TAU;
    //let f = float2(cos(angle), sin(angle));
    //color.rgb = float3(f * 0.5 + 0.5, 0.0);

    //let normal = float3(cos(angle.r), sin(angle.r), 0.0);
    //color.rgb = colorize_1d(dist);
    //var color = float4(float3(div), 1.0);
    //color.rgb = colorize_id(tex00.r);
    //color.rgb = float3(float2(jfa_value) / params.scene_info.screen_size, 0.0);
    //color.a = 1.0;
    //color.a = tex00.a;
    //color.rgb = params.normal_tex.Load(coord, 0).rgb;

    //if (test < 2) {
    //    discard;
    //    //color = float4(0.3, 0.3, 0.6, 1);
    //}

    CornerDetectionFSOut o;
    o.color = color;
    return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// JFA 
///////////////////////////////////////////////////////////////////////////////////////////////////
static const uint JFA_TILE_SIZE = 16;


[shader("compute")]
[pass("jfa_init")]
[numthreads(JFA_TILE_SIZE, JFA_TILE_SIZE, 1)]
void jfa_init(uint3 thread_id : SV_DispatchThreadID, uniform ContourRootParams* params)
{
    int3 coord = int3(thread_id.xy, 0);
    let texel = params.angle_texture.Load(coord, 0);
    let init = texel.a > 0.0 ? int16_t2(coord.x, coord.y) : int16_t2(-1, -1);
    params.jfa_in_texture.Store(coord.xy, init);
}


[shader("compute")]
[pass("jfa_step")]
[numthreads(JFA_TILE_SIZE, JFA_TILE_SIZE, 1)]
void jfa_step(uint3 thread_id : SV_DispatchThreadID, uniform ContourRootParams *params)
{
    int2 coord = thread_id.xy;
    let k = params.jfa_step_size;
    var cur = params.jfa_in_texture.Load(coord);
    var cur_dist = length(float2(cur - coord));

    let size = int2(params.scene_info.screen_size);
    if (coord.x >= size.x || coord.y >= size.y) {
        return;
    }

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            let ncoord = clamp(coord + k * int2(i, j), int2(0,0), size - int2(1,1));
            let nb = params.jfa_in_texture.Load(ncoord);
            let dist = length(float2(nb - coord));
            if (nb.x != -1 && (cur.x == -1 || dist < cur_dist)) {
                cur = nb;
                cur_dist = dist;
            }
        }
    }

    params.jfa_out_texture.Store(coord, cur);
}