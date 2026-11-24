static const uint MAX_INDIRECTIONS = 8;

struct CopyIndirect1DParams {
    uint8_t* base;
    uint8_t* dst;
    uint byte_size;
    uint count;
    uint offset[MAX_INDIRECTIONS];
}

// One thread per byte to read.
[shader("compute")]
[numthreads(32, 1, 1)]
void copy_indirect_1d(uint3 thread_id: SV_DispatchThreadID, uniform CopyIndirect1DParams u)
{
    let index = thread_id.x;
    if (index >= u.byte_size)
        return;

    // Compute the final address by traversing the pointer chain.
    var addr = (uint64_t)u.base;
    for (uint i = 0; i < u.count; ++i) {
        addr = *(uint64_t*)addr;
        addr += u.offset[i];
    }

    *(u.dst + index) = *(uint8_t*)(addr + index);
}

struct CopyImageIndirectParams {
    uint8_t* base;
    uint8_t* dst;
    uint2 extent;
    uint count;
    uint offset[MAX_INDIRECTIONS];
}

// One thread per texel
[shader("compute")]
[numthreads(8, 8, 1)]
void copy_image_indirect_2d(uint3 thread_id: SV_DispatchThreadID, uniform CopyImageIndirectParams u)
{
    let coords = thread_id.xy;
    if (any(coords >= u.extent))
        return;

    // Compute the address of the texture descriptor by traversing the pointer chain.
    var addr = (uint64_t)u.base;
    for (uint i = 0; i < u.count; ++i) {
        addr = *(uint64_t*)addr;
        addr += u.offset[i];
    }

    let image = *(Texture2D<float4>.Handle*)addr;
    let texel = image.Load(int3(int2(coords), 0));

    let dst_index = coords.y * u.extent.x + coords.x;
    ((float4*)u.dst)[dst_index] = texel;
}

