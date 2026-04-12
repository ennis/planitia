//! Type descriptors.

use utils::archive::Offset;

/// Represents a constant value.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum Value {
    /// Boolean scalar or vector.
    Bool { count: u8, values: [bool; 4] },
    /// Signed integer scalar or vector.
    I32 { count: u8, values: [i32; 4] },
    /// Unsigned integer scalar or vector.
    U32 { count: u8, values: [u32; 4] },
    /// Floating point scalar or vector.
    F32 { count: u8, values: [f32; 4] },
    /// String
    String { value: Offset<str> },
}

/// Scalar value kinds.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum ScalarType {
    /// Boolean.
    Bool,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 32-bit floating point.
    F32,
}

/// Describes a field of a struct type.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StructField {
    /// Field name.
    pub name: Offset<str>,
    /// Type descriptor of the field.
    pub ty: TypeDesc,
    /// Byte offset of the field within the struct.
    pub offset: u32,
    /// User attributes attached to the field.
    pub attributes: Offset<[Attribute]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StructType {
    pub type_name: Offset<str>,
    pub fields: Offset<[StructField]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VectorType {
    pub scalar: ScalarType,
    pub len: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatrixType {
    pub scalar: ScalarType,
    pub rows: u8,
    pub cols: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SampledType {
    pub scalar: ScalarType,
    pub components: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ImageHandleType {
    pub sampled: SampledType,
    pub read_write: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum TypeDesc {
    Scalar(ScalarType),
    Vector(VectorType),
    Matrix(MatrixType),
    Array { elem: Offset<TypeDesc>, len: u32 },
    RuntimeArray(Offset<TypeDesc>),
    Struct(StructType),
    ImageHandle(ImageHandleType),
    Pointer(Offset<TypeDesc>),
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Attribute {
    pub name: Offset<str>,
    pub value: Value,
}

/// Describes an output variable of a fragment shader.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FragmentShaderOutput {
    /// Color attachment index.
    pub attachment_index: u32,
    /// Scalar type of the output variable.
    pub scalar_type: ScalarType,
    /// Number of components of the output variable (1-4).
    ///
    /// E.g., a `float3` output would have `scalar_type = F32` and `components = 3`.
    pub components: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum Interpolation {
    Flat,
    NoPerspective,
    Perspective,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VertexShaderOutput {
    pub location: u32,
    pub scalar_type: ScalarType,
    pub components: u8,
    pub interpolation: Interpolation,
}

/// Location of a shader parameter value.
///
/// This describes where the value of the parameter is stored,
/// either in a buffer resource, or in push data.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum ParamLocation {
    /// Parameter is in a buffer accessed via a resource binding.
    Binding {
        /// Buffer resource index (in `Signature::resources`).
        index: u32,
        /// Byte offset of the parameter within the buffer.
        offset: u32,
    },
    /// Parameter value is passed in push data.
    PushData {
        /// Byte offset of the parameter value within push data.
        offset: u32,
    },
}

/// Shader uniform parameter.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Param {
    pub name: Offset<str>,
    pub ty: TypeDesc,
    /// Location of the parameter value.
    pub location: ParamLocation,
    pub attributes: Offset<[Attribute]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum ShaderResourceKind {
    /// cbuffer, ConstantBuffer
    UniformBuffer,
    ///
    StorageBuffer,
    /// Texture2D, etc.
    Texture,
    /// RWTexture2D, etc.
    StorageImage,
    Sampler,
}

/// Represents a shader resource.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShaderResource {
    pub binding: u32,
    pub set: u32,
    pub kind: ShaderResourceKind,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub enum ShaderInterface {
    Vertex {
        outputs: Offset<[VertexShaderOutput]>,
    },
    Fragment {
        inputs: Offset<[VertexShaderOutput]>,
        outputs: Offset<[FragmentShaderOutput]>,
    },
}

/// Shader entry point signature.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Signature {
    pub interface: ShaderInterface,
    /// Uniform parameters.
    pub params: Offset<[Param]>,
    pub resources: Offset<[ShaderResource]>,
}
