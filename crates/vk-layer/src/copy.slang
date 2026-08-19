// Load and copy arbitrary data from an access chain.

enum AccessKind : uint8_t {
    // Uniform offset
    UniformOffset,
    // Varying offset
    VaryingOffset,
    Indirect,
}

struct Access
{
    AccessKind kind;
    uint32_t offset_or_stride;
}

static const uint MAX_ACCESS_COUNT = 8;

struct CopyFromAccessChain1DParams {
    uint8_t* src;
    uint8_t* dst;

    uint len;
    //uint stride;

    uint src_access_count;
    Access src_access_chain[MAX_ACCESS_COUNT];
}

// One thread per byte to read.
[shader("compute")]
[numthreads(32, 1, 1)]
void copy_from_access_chain_1d(uint3 thread_id: SV_DispatchThreadID, uniform CopyFromAccessChain1DParams u)
{
    let index = thread_id.x;
    if (index >= u.len)
        return;

    var addr = (uint64_t)u.src;
    for (uint i = 0; i < u.src_access_count; ++i) {
        switch (u.src_access_chain[i].kind) {
            case AccessKind::UniformOffset:
                addr += u.src_access_chain[i].offset_or_stride;
                break;
            case AccessKind::VaryingOffset:
                // TODO
                //addr += u.src_access_chain[i].offset_or_stride * index;
                break;
            case AccessKind::Indirect:
                addr = *(uint64_t*)addr;
                break;
        }
    }

    *(u.dst + index) = *(uint8_t*)(addr + index);
}