//! Shader reflection structures.

/// Scalar value kinds.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum ScalarType {
    /// Boolean.
    Bool,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// 32-bit floating point.
    F32,
}

/// Describes a field of a struct type.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StructField<'a> {
    /// Field name.
    pub name: &'a str,
    /// Type descriptor of the field.
    pub ty: TypeDesc<'a>,
    /// Byte offset of the field within the struct.
    pub offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StructType<'a> {
    pub name: &'a str,
    pub fields: &'a [StructField<'a>],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VectorType {
    pub scalar: ScalarType,
    pub len: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MatrixType {
    pub scalar: ScalarType,
    pub rows: u8,
    pub cols: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SampledType {
    pub scalar: ScalarType,
    pub components: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ImageHandleType {
    pub sampled: SampledType,
    pub read_write: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum TypeDesc<'a> {
    Void,
    Bool,
    Scalar(ScalarType),
    /// Vector type (scalar type, component count).
    Vector(ScalarType, u8),
    /// Matrix type (scalar type, rows, columns).
    Matrix(ScalarType, u8, u8),
    Array(&'a TypeDesc<'a>, u32),
    RuntimeArray(&'a TypeDesc<'a>),
    Struct(&'a StructType<'a>),
    ImageHandle(ImageHandleType),
    Pointer(&'a TypeDesc<'a>),
    Image,
    SampledImage,
    Sampler,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AccessKind<'a> {
    /// Parameter is in a buffer accessed via a resource binding.
    ///
    /// This should only appear as the first component of an access chain.
    Binding {
        name: &'a str,
        /// Buffer resource index.
        resource_index: u32,
        /// Byte offset of the parameter within the buffer resource.
        offset: u32,
    },

    /// Parameter value is passed in push data.
    PushData {
        /// Byte offset of the parameter value within push data.
        offset: u32,
    },

    /// Field access.
    Field { offset: u32 },

    /// Bounded array access.
    ArrayIndex { stride: u32, count: u32 },

    /// Runtime array index.
    RuntimeArrayIndex { stride: u32 },

    /// Pointer load (dereference)
    Load,
}

#[derive(Copy, Clone, Debug)]
pub struct AccessChain<'a> {
    pub parent: Option<&'a AccessChain<'a>>,
    pub kind: AccessKind<'a>,
    /// Type of the dereferenced data.
    pub ty: TypeDesc<'a>,
    /// Name of the access chain (for debugging)
    pub name: &'a str,
}

pub type ShaderReflection = &'static [&'static AccessChain<'static>];
