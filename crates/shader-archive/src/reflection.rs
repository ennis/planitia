//! Type descriptors.

use utils::archive::Offset;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum Value {
    Bool {
        count: u8,
        values: [bool; 4],
    },
    I32 {
        count: u8,
        values: [i32; 4],
    },
    U32 {
        count: u8,
        values: [u32; 4],
    },
    F32 {
        count: u8,
        values: [f32; 4],
    },
    String {
        value: Offset<str>,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum ScalarType {
    Unit,
    Bool,
    I8,
    I16,
    I32,
    U8,
    U16,
    U32,
    F32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StructField {
    pub name: Offset<str>,
    pub ty: TypeDesc,
    pub offset: u32,
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
pub struct  SampledType {
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FragmentShaderOutput {
    pub location: u32,
    pub scalar_type: ScalarType,
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
    pub interpolation: Interpolation
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum ParamLocation {
    SetBinding { set: u32, binding: u32 },
    PushData { offset: u32 },
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Param {
    pub name: Offset<str>,
    pub ty: TypeDesc,
    pub location: ParamLocation,
    pub attributes: Offset<[Attribute]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum ShaderInterface {
    Vertex { outputs: Offset<[VertexShaderOutput]> },
    Fragment { inputs: Offset<[VertexShaderOutput]>, outputs: Offset<[FragmentShaderOutput]> },
}

/// Shader entry point signature.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Signature {
    /// Uniform parameters.
    pub params: Offset<[Param]>,
    pub interface: ShaderInterface,
}