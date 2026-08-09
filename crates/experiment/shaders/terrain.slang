// Manifest: shaders.toml

static const uint TILE_SIZE = 16;

struct TerrainSlice {
    uint16_t low;
    uint16_t high;
    uint8_t features[8];
}

struct TerrainTile {
    uint64_t offset;
    uint16_t min_height;
    uint16_t max_height;
    TerrainSlice slices[TILE_SIZE * TILE_SIZE];
}


