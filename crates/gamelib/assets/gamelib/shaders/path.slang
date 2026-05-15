module path;

public enum PathSegKind : uint8_t {
    Line,
    Quad,
    Cubic,
};

public struct PathSeg {
    public PathSegKind kind;
    public uint32_t start_vertex;
};

public struct PathBuf {
    public PathSeg* segments;
    public float2* vertices;
    public uint segment_count;
};