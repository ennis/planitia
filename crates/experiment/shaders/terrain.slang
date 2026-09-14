// Manifest: shaders.toml

static const uint TILE_SIZE = 16;

struct MatLayer {
    uint16_t low;
    uint16_t high;
    uint8_t features[8];
}

struct MatStack {
    uint16_t layer_offset; // rel offset in Terrain::layers; total offset is `Tile::base + MatStack::offset`
    uint16_t layer_count;
}

struct Tile {
    uint32_t base_layer; // base offset in Terrain::layers
    uint16_t min_height;
    uint16_t max_height;
    MatStack stacks[TILE_SIZE * TILE_SIZE];
}

struct LOD {
    uint2 dim; // number of tiles
    uint32_t base_tile;    // in tiles (# of tiles = tile_x_count * tile_y_count), tiles stored row-major
};

struct TerrainDesc {
    float4 world_bounds; // x1,y1,x2,y2
    uint32_t tile_count;
    uint32_t lod_count;
    Tile* tiles;
    LOD* lods;
    MatLayer* layers;
}

// - Each tile contains 16x16 material stacks (or just "stacks")
// - Stacks contain variable numbers of layers, so they can't be stored inline
// - Layers are stored in a separate array.

struct TerrainParams {
    TerrainDesc terrain;
    RWTexture2D<float4>.Handle output;
    int screen_width;
    int screen_height;
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 proj_matrix;
    float4x4 inv_proj_matrix;
    float3 eye_pos;
}

static const uint RENDER_TILE_SIZE = 16;

// after view matrix -> view space
// after proj matrix -> clip space (-w, -w, 0) -> (w,w,w)
// after divide -> NDC space (-1, -1, 0) -> (1, 1, 1)

[vk::push_constant]
TerrainParams* params;

void get_tile_and_local_idx(float2 world_pos, out uint2 tile_idx, out uint2 local_idx) {
    float4 bounds = params.terrain.world_bounds * 0.1;
    float2 world_pos_clamped = clamp(world_pos, bounds.xy, bounds.zw);
    LOD lod_0 = params.terrain.lods[0];
    uint2 lod_size = lod_0.dim * TILE_SIZE;
    float2 lod_coord = (world_pos_clamped - bounds.xy) / (bounds.zw - bounds.xy) * float2(lod_size);
    uint2 lod_idx = min(uint2(lod_coord), lod_size - 1);
    tile_idx = lod_idx / uint2(TILE_SIZE);
    local_idx = lod_idx % uint2(TILE_SIZE);
}

float sample_height(float2 world_pos) {
    uint2 tile_idx;
    uint2 local_idx;
    get_tile_and_local_idx(world_pos, tile_idx, local_idx);
    LOD lod_0 = params.terrain.lods[0];
    Tile *tile = &params.terrain.tiles[lod_0.base_tile + tile_idx.y * lod_0.dim.x + tile_idx.x];
    MatStack stack = tile.stacks[local_idx.y * TILE_SIZE + local_idx.x];
    MatLayer top_layer = params.terrain.layers[tile.base_layer + stack.layer_offset + stack.layer_count - 2];
    return float(top_layer.high) / 65535.0 * 4.0;
}

float raycast(float3 origin, float3 world_dir, out float3 hit) {
    float3 dir = normalize(world_dir);
    float t = 0.0;
    for (int i = 0; i < 1000; i += 1) {
        float3 pos = origin + dir * t;
        float height = sample_height(pos.xz);
        if (pos.y < height) {
            hit = pos;
            return t;
        }
        t += 0.53;
    }
    return -1.0;
}

[shader("compute")]
[numthreads(RENDER_TILE_SIZE, RENDER_TILE_SIZE, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (dispatchThreadID.x >= params.screen_width || dispatchThreadID.y >= params.screen_height) {
        return;
    }
    float2 px_ndc = (float2(dispatchThreadID.xy) + 0.5) / float2(params.screen_width, params.screen_height) * 2.0 - 1.0;
    float4 px_view = mul(params->inv_proj_matrix, float4(px_ndc, 0.0, 1.0));
    px_view /= px_view.w;
    float4 world_pos = mul(params->inv_view_matrix, px_view);

    //uint2 tile_idx;
    //uint2 local_idx;
    //get_tile_and_local_idx(float2(dispatchThreadID.xy), tile_idx, local_idx);
    //LOD lod_0 = params.terrain.lods[0];
    //Tile *tile = &params.terrain.tiles[lod_0.base_tile + tile_idx.y * lod_0.dim.x + tile_idx.x];
    //float height = tile.max_height / 65535.0;
    //params.output[dispatchThreadID.xy] = float4(float2(tile_idx) / float2(TILE_SIZE), height, 1.0);
    //params.output[dispatchThreadID.xy] = float4(float3(tile.max_height/ 65535.0), 1.0);
    //params.output[dispatchThreadID.xy] = float4(float3(sample_height(float2(dispatchThreadID.xy))/ 65535.0), 1.0);

    float3 hit;
    float depth = raycast(params.eye_pos, world_pos.xyz - params.eye_pos, hit);
    if (depth < 0) {
        params.output[dispatchThreadID.xy] = float4(1.0, 1.0, 0.0, 1.0);
    } else {
        float shade = depth / 10.0;
        params.output[dispatchThreadID.xy] = float4(shade.xxx/10.0, 1.0);
    }
}


