use ::spirv as spv;
use std::fmt;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

mod parser;

/// Error parsing a SPIR-V module.
#[derive(Debug, thiserror::Error)]
#[error("SPIR-V parse error")]
pub struct ParseError;

/// Parsed SPIR-V module.
///
/// # Implementation note
///
/// Types are represented by IDs instead of direct references (`&'a TypeInfo`) **solely** because
/// of circular type references (linked lists, for instance). Because of those cases, `TypeInfo`
/// would have to contain interior mutability to allow creating the cycle, which means `Cell` or
/// `UnsafeCell`, which in turn makes the lifetime `'a` invariant, which in turn has a non-negligible
/// ergonomic impact.
pub struct Module {
    r: Vec<InstResult>,
    // This is only used during building
    //ty_map: HashMap<TypeInfo, TypeId>,
    /// The list of all entry points.
    pub entry_points: Vec<EntryPointId>,
}

impl Module {
    pub fn new() -> Module {
        Module { r: vec![], entry_points: vec![] }
    }

    fn reserve(&mut self, result_id: u32) {
        if result_id as usize >= self.r.len() {
            self.r.resize_with((result_id + 1) as usize, || InstResult::None);
        }
    }

    /// Inserts a new type and returns its type id.
    pub fn insert_type(&mut self, result_id: u32, ty: TypeInfo) -> TypeId {
        self.reserve(result_id);
        self.r[result_id as usize] = InstResult::Type(ty.clone());
        TypeId(result_id)
    }

    /// Inserts a new entry point.
    pub fn insert_entry_point(&mut self, result_id: u32, entry_point: EntryPointInfo) -> EntryPointId {
        self.reserve(result_id);
        self.r[result_id as usize] = InstResult::EntryPoint(entry_point.clone());
        let ep_id = EntryPointId(result_id);
        self.entry_points.push(ep_id);
        ep_id
    }

    /// Inserts a new variable.
    pub fn insert_variable(&mut self, result_id: u32, var: VariableInfo) -> VariableId {
        self.reserve(result_id);
        self.r[result_id as usize] = InstResult::Variable(var.clone());
        VariableId(result_id)
    }

    pub fn insert_constant(&mut self, result_id: u32, c: ConstantInfo) -> ConstantId {
        self.reserve(result_id);
        self.r[result_id as usize] = InstResult::Constant(c.clone());
        ConstantId(result_id)
    }

    pub fn insert_string(&mut self, result_id: u32, name: String) -> StringId {
        self.reserve(result_id);
        self.r[result_id as usize] = InstResult::None; // Strings are not stored in the result vector
        StringId(result_id)
    }

    pub fn find_entry_point(&self, name: &str) -> Option<EntryPointId> {
        self.entry_points.iter().find(|&&ep_id| self[ep_id].name == name).copied()
    }
}

impl Index<TypeId> for Module {
    type Output = TypeInfo;

    fn index(&self, index: TypeId) -> &Self::Output {
        let InstResult::Type(info) = &self.r[index.0 as usize] else {
            panic!("invalid TypeId {}", index.0);
        };
        info
    }
}

impl IndexMut<TypeId> for Module {
    fn index_mut(&mut self, index: TypeId) -> &mut Self::Output {
        let InstResult::Type(info) = &mut self.r[index.0 as usize] else {
            panic!("invalid TypeId {}", index.0);
        };
        info
    }
}

impl Index<EntryPointId> for Module {
    type Output = EntryPointInfo;

    fn index(&self, index: EntryPointId) -> &Self::Output {
        let InstResult::EntryPoint(info) = &self.r[index.0 as usize] else {
            panic!("invalid EntryPointId {}", index.0);
        };
        info
    }
}

impl IndexMut<EntryPointId> for Module {
    fn index_mut(&mut self, index: EntryPointId) -> &mut Self::Output {
        let InstResult::EntryPoint(info) = &mut self.r[index.0 as usize] else {
            panic!("invalid EntryPointId {}", index.0);
        };
        info
    }
}

impl Index<VariableId> for Module {
    type Output = VariableInfo;

    fn index(&self, index: VariableId) -> &Self::Output {
        let InstResult::Variable(info) = &self.r[index.0 as usize] else {
            panic!("invalid VariableId {}", index.0);
        };
        info
    }
}

impl IndexMut<VariableId> for Module {
    fn index_mut(&mut self, index: VariableId) -> &mut Self::Output {
        self.reserve(index.0);
        let InstResult::Variable(info) = &mut self.r[index.0 as usize] else {
            panic!("invalid VariableId {}", index.0);
        };
        info
    }
}

impl Index<ConstantId> for Module {
    type Output = ConstantInfo;

    fn index(&self, index: ConstantId) -> &Self::Output {
        let InstResult::Constant(info) = &self.r[index.0 as usize] else {
            panic!("invalid ConstantId {}", index.0);
        };
        info
    }
}

impl IndexMut<ConstantId> for Module {
    fn index_mut(&mut self, index: ConstantId) -> &mut Self::Output {
        self.reserve(index.0);
        let InstResult::Constant(info) = &mut self.r[index.0 as usize] else {
            panic!("invalid ConstantId {}", index.0);
        };
        info
    }
}

/// Entry point identifier.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct EntryPointId(pub u32);

/// Shader entry point.
#[derive(Clone)]
pub struct EntryPointInfo {
    pub stage: spv::ExecutionModel,
    pub name: String,
    pub params: Vec<VariableId>,
}

/// Entry point parameter.
#[derive(Clone, Debug)]
pub struct EntryPointParam {
    pub name: String,
    pub kind: AccessKind,
    pub ty: TypeId,
}

/// Describes where an entry point parameter gets its value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AccessKind {
    /// Parameter is in a buffer accessed via a resource binding.
    ///
    /// This should only appear as the first component of an access chain.
    Binding {
        name: String,
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
}

/// String identifier.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct StringId(pub u32 /* result_id */);


#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum IntegerConstant {
    SpecConstant(u32),
    Constant(u64)
}

/// Type identifier.
///
/// This wraps a SPIR-V `result_id` of a type declaration instruction.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct TypeId(pub u32 /* result_id */);

/// Constant or spec constant result_id.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct ConstantId(pub u32 /* result_id */);

/// Describes a type in a SPIR-V module.
#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TypeInfo {
    /// Void type.
    Void,
    /// Boolean type.
    Bool,
    /// Scalar type.
    Scalar(ScalarType),
    /// Vector type (scalar type, component count).
    Vector(ScalarType, u8),
    /// Matrix type (scalar type, rows, columns).
    Matrix { scalar: ScalarType, rows: u8, cols: u8, stride: Option<u32> },
    /// Array type (element type, length, stride).
    Array { element: TypeId, len: ConstantId, stride: Option<u32> },
    /// Runtime array type (element type, stride).
    RuntimeArray { element: TypeId, stride: Option<u32> },
    /// Structure type.
    Struct(StructType),
    /// Image handle type.
    ImageHandle(ImageHandleType),
    /// Pointer type.
    Pointer(PointerType),
    /// Image type.
    Image,
    /// Sampled image type.
    SampledImage,
    /// Sampler type.
    Sampler,
}

impl TypeInfo {
    /// Returns the [`StructType`] if this represents a structure type, or `None` otherwise.
    pub fn as_struct(&self) -> Option<&StructType> {
        match self {
            TypeInfo::Struct(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the element type and size if this represents an array type, or `None` otherwise.
    pub fn as_array(&self) -> Option<(TypeId, ConstantId)> {
        match *self {
            TypeInfo::Array { element, len, .. } => Some((element, len)),
            _ => None,
        }
    }

    pub fn as_scalar(&self) -> Option<ScalarType> {
        match *self {
            TypeInfo::Scalar(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the offset and type of a field or array element access by field or array index.
    pub fn indexed(&self, index: usize) -> (usize, TypeId) {
        match *self {
            TypeInfo::Array { element, stride, .. } => (index * stride.unwrap() as usize, element),
            TypeInfo::RuntimeArray { element, stride, .. } => (index * stride.unwrap() as usize, element),
            TypeInfo::Struct(ref s) => {
                let field = &s.fields[index];
                (field.offset as usize, field.ty)
            }
            _ => panic!("TypeInfo::indexed called on non-array/struct type"),
        }
    }

    /// Returns the stride of an array or runtime array type.
    pub fn stride(&self) -> Option<usize> {
        match self {
            TypeInfo::Array { stride, .. } => stride.map(|s| s as usize),
            TypeInfo::RuntimeArray { stride, .. } => stride.map(|s| s as usize),
            _ => None,
        }
    }

    /// Returns the offset of a structure field or an array element.
    ///
    /// # Arguments
    ///
    /// - `index` index of the structure field or array element
    ///
    /// # Returns
    ///
    /// The offset of the field or array element, or `None` if `self` isn't a structure or array.
    pub fn field_or_element_offset(&self, index: usize) -> Option<usize> {
        match self {
            TypeInfo::Array { stride, .. } => stride.map(|s| index * s as usize),
            TypeInfo::RuntimeArray { stride, .. } => stride.map(|s| index * s as usize),
            TypeInfo::Struct(s) => Some(s.fields[index].offset as usize),
            _ => None,
        }
    }

    /// Returns the type of a structure field or an array element.
    ///
    /// # Arguments
    ///
    /// - `index` index of the structure field or array element
    ///
    /// # Return value
    ///
    /// The type ID of the field or array element, or `None` if `self` isn't a structure or array.
    pub fn field_or_element_type(&self, index: usize) -> Option<TypeId> {
        match *self {
            TypeInfo::Array { element, .. } => Some(element),
            TypeInfo::RuntimeArray { element, .. } => Some(element),
            TypeInfo::Struct(ref s) => Some(s.fields[index].ty),
            TypeInfo::Pointer(p) => p.pointee,
            _ => None,
        }
    }
}

/// Scalar value type.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

impl ScalarType {
    pub fn byte_size(self) -> usize {
        match self {
            ScalarType::Bool | ScalarType::I8 | ScalarType::U8 => 1,
            ScalarType::I16 | ScalarType::U16 => 2,
            ScalarType::I32 | ScalarType::U32 | ScalarType::F32 => 4,
            ScalarType::I64 | ScalarType::U64 => 8,
        }
    }

    pub fn pretty_name(self) -> &'static str {
        match self {
            ScalarType::Bool => "bool",
            ScalarType::I8 => "i8",
            ScalarType::I16 => "i16",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::U8 => "u8",
            ScalarType::U16 => "u16",
            ScalarType::U32 => "u32",
            ScalarType::U64 => "u64",
            ScalarType::F32 => "f32",
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            ScalarType::Bool => "b",
            ScalarType::I8 => "i8",
            ScalarType::I16 => "i16",
            ScalarType::I32 => "i",
            ScalarType::I64 => "i64",
            ScalarType::U8 => "u8",
            ScalarType::U16 => "u16",
            ScalarType::U32 => "u",
            ScalarType::U64 => "u64",
            ScalarType::F32 => "f",
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SampledType {
    pub scalar: ScalarType,
    pub components: u8,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageHandleType {
    pub sampled: SampledType,
    pub read_write: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PointerType {
    pub storage_class: spv::StorageClass,
    // This is filled after forward declarations
    pub pointee: Option<TypeId>,
}

/// Describes a structure type.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct StructType {
    pub name: String,
    pub fields: Vec<StructField>,
}

/// Describes a field of a structure type.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct StructField {
    /// Field name.
    pub name: String,
    /// Field type.
    pub ty: TypeId,
    /// Byte offset of the field within the struct.
    pub offset: u32,
}

/// Returns the byte size of an instance of the specified type.
pub fn type_byte_size(m: &Module, ty: &TypeInfo) -> Option<usize> {
    match *ty {
        TypeInfo::Void => Some(0),
        TypeInfo::Bool => Some(4),
        TypeInfo::Scalar(s) => Some(s.byte_size()),
        TypeInfo::Vector(s, n) => Some(s.byte_size() * n as usize),
        TypeInfo::Matrix { scalar, rows, cols, stride } => {
            let col_stride = stride.map(|s| s as usize).unwrap_or(scalar.byte_size() * rows as usize);
            Some(col_stride * cols as usize)
        }
        TypeInfo::Array { element, len, stride } => {
            let len = m[len].as_usize().unwrap_or(0);
            let elem_stride = stride.map(|s| s as usize).or_else(|| type_byte_size(m, &m[element]))?;
            Some(elem_stride * len)
        }
        TypeInfo::Struct(ref s) => {
            s.fields.last().and_then(|f| type_byte_size(m, &m[f.ty]).map(|sz| f.offset as usize + sz))
        }
        _ => None,
    }
}

/// Returns a pretty-print wrapper for a [`TypeId`].
pub fn pretty_print_type<'a>(m: &'a Module, ty: &'a TypeInfo) -> PrettyType<'a> {
    PrettyType { m, ty }
}

/// Pretty-printing wrapper for a [`TypeId`].
pub struct PrettyType<'a> {
    m: &'a Module,
    ty: &'a TypeInfo,
}

impl<'a> fmt::Display for PrettyType<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.m;
        match *self.ty {
            TypeInfo::Void => write!(f, "void"),
            TypeInfo::Bool => write!(f, "bool"),
            TypeInfo::Scalar(s) => write!(f, "{}", s.pretty_name()),
            TypeInfo::Vector(s, n) => write!(f, "vec{}{}", n, s.suffix()),
            TypeInfo::Matrix { scalar, rows, cols, .. } => write!(f, "mat{}x{}{}", rows, cols, scalar.suffix()),
            TypeInfo::Array { element, len, .. } => {
                let len = m[len].as_usize().unwrap_or(0);
                write!(f, "{}[{}]", pretty_print_type(m, &m[element]), len)
            },
            TypeInfo::RuntimeArray { element, .. } => write!(f, "{}[]", pretty_print_type(m, &m[element])),
            TypeInfo::Struct(ref s) => write!(f, "{}", s.name),
            TypeInfo::ImageHandle(_) => write!(f, "image_handle"),
            TypeInfo::Pointer(p) => match p.pointee {
                Some(pointee) => write!(f, "*{} {}", pretty_storage_class(p.storage_class), pretty_print_type(m, &m[pointee])),
                None => write!(f, "*{} <unknown>", pretty_storage_class(p.storage_class)),
            },
            TypeInfo::Image => write!(f, "image"),
            TypeInfo::SampledImage => write!(f, "sampled_image"),
            TypeInfo::Sampler => write!(f, "sampler"),
        }
    }
}

fn pretty_storage_class(sc: spv::StorageClass) -> &'static str {
    match sc {
        spv::StorageClass::PhysicalStorageBuffer => "",
        spv::StorageClass::UniformConstant => "[uniform_constant]",
        spv::StorageClass::Input => "[input]",
        spv::StorageClass::Uniform => "[uniform]",
        spv::StorageClass::Output => "[output]",
        spv::StorageClass::Workgroup => "[workgroup]",
        spv::StorageClass::CrossWorkgroup => "[cross_workgroup]",
        spv::StorageClass::Private => "[private]",
        spv::StorageClass::Function => "[function]",
        spv::StorageClass::Generic => "[generic]",
        spv::StorageClass::PushConstant => "[push_constant]",
        spv::StorageClass::AtomicCounter => "[atomic_counter]",
        spv::StorageClass::Image => "[image]",
        _ => "",
    }
}

/// Variable identifier.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct VariableId(pub u32);

/// SPIR-V variable declaration.
#[derive(Clone)]
pub struct VariableInfo {
    pub name: String,
    pub ty: TypeId,
    pub uniform: bool,
    pub sc: spv::StorageClass,
}

#[derive(Clone)]
pub struct ConstantInfo {
    pub ty: ScalarType,
    pub value_bytes: Option<[u8; 8]>,
}

impl ConstantInfo {
    pub fn as_usize(&self) -> Option<usize> {
        let Some(bytes) = self.value_bytes else {
            return None;
        };
        Some(usize::from_le_bytes(bytes))
    }
}

#[derive(Clone)]
enum InstResult {
    None,
    Constant(ConstantInfo),
    Type(TypeInfo),
    Variable(VariableInfo),
    EntryPoint(EntryPointInfo),
}

/*
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VectorType {
    pub scalar: ScalarType,
    pub len: u8,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatrixType {
    pub scalar: ScalarType,
    pub rows: u8,
    pub cols: u8,
}*/
