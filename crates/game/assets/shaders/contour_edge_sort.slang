import common;
import scene_info;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Chain edges
///////////////////////////////////////////////////////////////////////////////////////////////////

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

static const uint POINT_BUCKET_SIZE = 65536;

struct SortContourEdgesRootParams {
    SceneInfo* scene_info;
    ContourEdgeBuffer* contour_edges;
    // point index (two-level: bucket + index in bucket) -> entry containing next/prev point indices
    uint32_t** neighbor_point_map;

}

[shader("compute")]
[pass("sort_contour_edges")]
[numthreads(32, 1, 1)]
void sort_contour_edges(
    SortContourEdgesRootParams root_params,
    uint3 dispatch_thread_id : SV_DispatchThreadID
)
{
}

// Edge-chaining pass

//
// two level map: point index -> neighbor1, neighbor2
// edges can be directed since they are silhouettes

// B -> C
// G -> H
// F -> G
// E -> F
// D -> E
// C -> D
// A -> B


