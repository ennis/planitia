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
    var base = (uint64_t)u.base;
    for (uint i = 0; i < u.count; ++i) {
        base = *(uint64_t*)base;
        base += u.offset[i];
    }

    *(u.dst + index) = *(uint8_t*)(base + index);
}