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

float unpack_height(uint16_t h) {
    return float(h) / 65535.0 * 8.0;
}

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
    return fetch_height_at(0, tile_idx, local_idx);
}

float fetch_height_at(uint lod, uint2 tile_idx, uint2 local_idx) {
    LOD lod_desc = params.terrain.lods[lod];
    Tile *tile = &params.terrain.tiles[lod_desc.base_tile + tile_idx.y * lod_desc.dim.x + tile_idx.x];
    MatStack stack = tile.stacks[local_idx.y * TILE_SIZE + local_idx.x];
    MatLayer top_layer = params.terrain.layers[tile.base_layer + stack.layer_offset + stack.layer_count - 2];
    return unpack_height(top_layer.high);
}

Tile* tile_ptr(uint lod, uint2 tile_idx) {
    return &params.terrain.tiles[params.terrain.lods[lod].base_tile + tile_idx.y * params.terrain.lods[lod].dim.x + tile_idx.x];
}

float raycast(float3 origin, float3 world_dir, out float3 hit, inout float4 color) {
    float4 bounds = params.terrain.world_bounds * 0.1;
    float3 dir = normalize(world_dir);
    float3 pos = origin;
    float4 dbgcolor = float4(0,0,0,1);
    float t = 0.0;
    for (int i = 0; i < 1000; i += 1) {
        //float3 pos = origin + dir * t;
        if (any(pos.xz > bounds.zw) || any(pos.xz < bounds.xy)) {
            color = dbgcolor;
            return -1.0;
        }
        uint2 tile_idx;
        uint2 local_idx;
        get_tile_and_local_idx(pos.xz, tile_idx, local_idx);
        // get y extents of ray in current tile @ LOD 0
        float2 lod_size = params.terrain.lods[0].dim;
        float2 tile_min = float2(tile_idx) / lod_size * (bounds.zw - bounds.xy) + bounds.xy;
        float2 tile_max = float2(tile_idx + 1) / lod_size * (bounds.zw - bounds.xy) + bounds.xy;
        Tile* tile = tile_ptr(0, tile_idx);
        float2 tile_y_extents = float2(unpack_height(tile->min_height), unpack_height(tile->max_height));
        // 2D ray-box intersection in xz
        float t0x = (tile_min.x - pos.x) / dir.x;
        float t0z = (tile_min.y - pos.z) / dir.z;
        float t1x = (tile_max.x - pos.x) / dir.x;
        float t1z = (tile_max.y - pos.z) / dir.z;
        float t_min = max(min(t0x, t1x), min(t0z, t1z));
        float t_max = min(max(t0x, t1x), max(t0z, t1z));
        if (t_min > t_max) {
            hit = pos;
            color = float4(1,0,0,1);
            return -1.0;
        }
        float y0 = pos.y + t_min * dir.y;
        float y1 = pos.y + t_max * dir.y;
        if (y0 > y1) {
            float tmp = y0;
            y0 = y1;
            y1 = tmp;
        }
        // ray does not intersect the tile bbox, skip the whole tile
        if (y0 > tile_y_extents.y || y1 < tile_y_extents.x) {
            pos = pos + (t_max + 0.02) * dir;
            dbgcolor = dbgcolor + float4(0.02,0,0,0);
            continue;
            //color = float4(0,1,0,1);
            //hit = pos;
            //return t;
        }
        float height = fetch_height_at(0, tile_idx, local_idx);
        if (pos.y < height) {
            //pos.y = tile_y_extents.x;
            hit = pos;
            color = dbgcolor;
            return t;
        }
        pos += dir * 0.02;
    }
    //color = float4(1,0,1,0);
    color = dbgcolor;
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
    float4 color;
    float depth = raycast(params.eye_pos, world_pos.xyz - params.eye_pos, hit, color);
    if (depth < 0) {
        params.output[dispatchThreadID.xy] = float4(1.0, 1.0, 1.0, 1.0) * color;
    } else {
        //float shade = distance(hit, params.eye_pos) / 100.0;
        float shade = hit.y/5.0;
        //params.output[dispatchThreadID.xy] = float4(shade.xxx, 1.0);
        params.output[dispatchThreadID.xy] = float4(shade.xxx, 1.0);
    }
}


