// Manifest: shaders.toml

static const uint TILE_SIZE = 16;
static const uint MAX_LODS = 16;

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
    MatLayer* layers;
    LOD lods[MAX_LODS];
}

// - Each tile contains 16x16 material stacks (or just "stacks")
// - Stacks contain variable numbers of layers, so they can't be stored inline
// - Layers are stored in a separate array.

struct TerrainParams {
    RWTexture2D<float4>.Handle output;
    int screen_width;
    int screen_height;
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 proj_matrix;
    float4x4 inv_proj_matrix;
    float3 eye_pos;
    TerrainDesc terrain;
}

static const uint RENDER_TILE_SIZE = 16;

// after view matrix -> view space
// after proj matrix -> clip space (-w, -w, 0) -> (w,w,w)
// after divide -> NDC space (-1, -1, 0) -> (1, 1, 1)

[vk::push_constant]
TerrainParams* params;

static const float WORLD_SCALE = 0.005;
static const float HEIGHT_SCALE = 200.0;

float unpack_height(uint16_t h) {
    return float(h) / 65535.0 * WORLD_SCALE * HEIGHT_SCALE;
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


void get_tile_and_local_idx2(uint lod, float2 pos, out uint2 tile_idx, out uint2 local_idx) {
    uint2 lod_idx = clamp(uint2(pos), uint2(0,0), params.terrain.lods[lod].dim * TILE_SIZE) >> lod;
    tile_idx = lod_idx / uint2(TILE_SIZE);
    local_idx = lod_idx % uint2(TILE_SIZE);
}

void fetch_layer(uint lod, uint2 tile_idx, uint2 local_idx, float y, out MatLayer layer) {
    LOD lod_desc = params.terrain.lods[lod];
    Tile *tile = &params.terrain.tiles[lod_desc.base_tile + tile_idx.y * lod_desc.dim.x + tile_idx.x];
    MatStack stack = tile.stacks[local_idx.y * TILE_SIZE + local_idx.x];
    for (uint i = 0; i < stack.layer_count; ++i) {
        MatLayer l = params.terrain.layers[tile.base_layer + stack.layer_offset + i];
        let low = unpack_height(l.low);
        let high = unpack_height(l.high);
        if (y >= low && y < high) {
            layer = l;
            return;
        }
    }
    layer = params.terrain.layers[tile.base_layer + stack.layer_offset + stack.layer_count - 1];
}

bool is_air(MatLayer layer) {
    return layer.features[0] == 0
        && layer.features[1] == 0
        && layer.features[2] == 0
        && layer.features[3] == 0
        && layer.features[4] == 0
        && layer.features[5] == 0
        && layer.features[6] == 0
        && layer.features[7] == 0;
}

float4 feature_color(MatLayer layer) {
    return float4(layer.features[0]/255.0, layer.features[1]/255.0, layer.features[2]/255.0, 1.0);
}

float raycast2(float3 origin, float3 dir, out float3 hit, out uint steps, out float4 color) {
    let terrain_extents = params.terrain.lods[0].dim * float2(TILE_SIZE);
    float3 pos = origin;
    MatLayer layer;

    for (int i = 0; i < 1000; i += 1) {
        steps = i;
        if (any(pos.xz < 0) || any(pos.xz >= terrain_extents)) {
            color = float4(0,0,0,1);
            return -1.0;
        }
        uint2 tile_idx;
        uint2 local_idx;
        get_tile_and_local_idx2(0, pos.xz, tile_idx, local_idx);
        // get y extents of ray in current tile @ LOD 0
        float2 lod_size = params.terrain.lods[0].dim;
        float2 tile_min = float2(tile_idx) * TILE_SIZE;     // TODO: shift by lod level when not zero
        float2 tile_max = float2(tile_idx + 1) * TILE_SIZE;
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
            pos = pos + (t_max + 0.001) * dir;
            continue;
        }
        //float height = fetch_height_at(0, tile_idx, local_idx);
        fetch_layer(0, tile_idx, local_idx, pos.y, layer);
        if (!is_air(layer)) {
            hit = pos;
            color = feature_color(layer);
            return 0.0;
        }
        pos += 0.001*dir;
    }
    color = float4(0,0,0,1);
    return -1.0;
}

float4 colorize_1d(float x, float min, float max)
{
    float xi = saturate((x - min) / (max - min));
    float r = -0.121 + 0.893 * xi + 0.276 * sin (1.94 - 5.69 * xi);
    float g = 0.07 + 0.947 * xi;
    float b = 0.107 + (1.5 - 1.22 * xi) * xi;
    return float4(r, g, b, 1.0);
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
    float3 world_pos = mul(params->inv_view_matrix, px_view).xyz;

    // TODO factor this out into a 3x3 matrix
    float4 scaled_world_bounds = WORLD_SCALE * params.terrain.world_bounds;
    float2 world_extents = scaled_world_bounds.zw - scaled_world_bounds.xy;
    float3 offset = float3(scaled_world_bounds.x, 0.0, scaled_world_bounds.y);
    let lod_0 = params.terrain.lods[0];
    float3 scale = float3(lod_0.dim.x * TILE_SIZE / world_extents.x, 1.0, lod_0.dim.y * TILE_SIZE / world_extents.y);
    //float3 terrain_pos = (world_pos - offset) * scale;
    float3 ray_dir = normalize(world_pos - params.eye_pos) * scale;
    float3 ray_origin = (params.eye_pos - offset) * scale;

    float3 hit;
    float4 color;
    uint steps;
    //float depth = raycast(params.eye_pos, world_pos.xyz - params.eye_pos, hit, color);
    float depth = raycast2(ray_origin, ray_dir, hit, steps, color);


    uint2 coord = dispatchThreadID.xy;
    if (depth < 0) {
        params.output[coord] = float4(1.0, 1.0, 1.0, 1.0) * color;
    } else {
        float shade = distance(hit, ray_origin) / 80000.0;
        //float shade = hit.y/10.0;
        //params.output[coord] = float4(shade.xxx, 1.0);
        params.output[coord] = float4(color.rgb + float3(shade), color.a);
    }
    //params.output[coord] = colorize_1d(steps, 1, 1000);
}


