use num_traits::FromPrimitive;
use spirv_headers as spv;
use spirv_headers::StorageClass;
use std::cell::{Cell, UnsafeCell};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::slice;

pub type Arena = bumpalo::Bump;

#[derive(Debug, thiserror::Error)]
#[error("SPIR-V parse error")]
pub struct ParseError;

pub struct ShaderReflection {
    arena: Arena,
    root: *const Shader<'static>,
}

impl ShaderReflection {
    fn root<'a>(&'a self) -> &'a Shader<'a> {
        unsafe {
            // SAFETY: moving ShaderReflection doesn't invalidate pointers to data inside the arena
            //         so root isn't invalidated.
            //         The lifetime 'a is normally invariant because of interior mutability inside `Shader`,
            //         but the interior mutability is only used during construction, and it's impossible
            //         to mutate anything through the returned reference (outside this module).
            &*(self.root as *const Shader<'a>)
        }
    }

    pub fn entry_points(&self) -> &[EntryPoint<'_>] {
        self.root().entry_points
    }
}

#[derive(Copy, Clone, Default)]
struct Shader<'a> {
    pub entry_points: &'a [EntryPoint<'a>],
}

#[derive(Copy, Clone, Default)]
pub struct EntryPoint<'a> {
    pub name: &'a str,
    pub params: &'a [RootParam<'a>],
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
}

#[derive(Copy, Clone, Debug)]
pub struct RootParam<'a> {
    pub name: &'a str,
    pub kind: AccessKind<'a>,
    pub ty: &'a TypeDesc<'a>,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Instruction<'a> {
    pub opcode: u16,
    pub word_count: u16,
    pub operands: &'a [u32],
}

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
#[derive(Clone, Debug)]
pub struct StructField<'a> {
    /// Field name.
    pub name: &'a str,
    /// Type descriptor of the field.
    pub ty: &'a TypeDesc<'a>,
    /// Byte offset of the field within the struct.
    pub offset: u32,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct StructType<'a> {
    pub name: &'a str,
    pub fields: &'a [StructField<'a>],
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct VectorType {
    pub scalar: ScalarType,
    pub len: u8,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct MatrixType {
    pub scalar: ScalarType,
    pub rows: u8,
    pub cols: u8,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct SampledType {
    pub scalar: ScalarType,
    pub components: u8,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ImageHandleType {
    pub sampled: SampledType,
    pub read_write: bool,
}

#[repr(C)]
pub struct PointerType<'a> {
    pub storage_class: StorageClass,
    // This is filled after forward declarations
    pointee: UnsafeCell<Option<&'a TypeDesc<'a>>>,
}

impl<'a> Debug for PointerType<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointerType")
            .field("storage_class", &self.storage_class)
            .field("pointee", unsafe { &*self.pointee.get() })
            .finish()
    }
}

impl<'a> PointerType<'a> {
    pub fn new(storage_class: StorageClass, pointee: &'a TypeDesc<'a>) -> PointerType<'a> {
        PointerType { storage_class, pointee: UnsafeCell::new(Some(pointee)) }
    }

    fn new_forward(storage_class: StorageClass) -> PointerType<'a> {
        PointerType { storage_class, pointee: UnsafeCell::new(None) }
    }

    pub fn pointee(&self) -> &'a TypeDesc<'a> {
        // SAFETY: this is always safe since a mutable reference to
        //         the value is never
        unsafe { *self.pointee.get() }.unwrap()
    }
}

impl<'a> Clone for PointerType<'a> {
    fn clone(&self) -> Self {
        PointerType {
            storage_class: StorageClass::UniformConstant,
            pointee: unsafe { UnsafeCell::new(*self.pointee.get()) },
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub enum TypeDesc<'a> {
    Void,
    Bool,
    Scalar(ScalarType),
    /// Vector type (scalar type, component count).
    Vector(ScalarType, u8),
    /// Matrix type (scalar type, rows, columns).
    Matrix {
        scalar: ScalarType,
        rows: u8,
        cols: u8,
        stride: Cell<Option<u32>>,
    },
    Array {
        element: &'a TypeDesc<'a>,
        len: u32,
        stride: Cell<Option<u32>>,
    },
    RuntimeArray {
        element: &'a TypeDesc<'a>,
        stride: Cell<Option<u32>>,
    },
    Struct(&'a StructType<'a>),
    ImageHandle(ImageHandleType),
    Pointer(PointerType<'a>),
    Image,
    SampledImage,
    Sampler,
}

pub struct Variable<'a> {
    pub name: &'a str,
    pub ty: &'a TypeDesc<'a>,
    pub uniform: Cell<bool>,
    pub storage_class: spv::StorageClass,
}

struct Decoration<'spv> {
    id: spv::Decoration,
    operands: &'spv [u32],
}

#[derive(Default)]
struct Header {
    magic: u32,
    version: u32,
    generator: u32,
    bound: u32,
    reserved: u32,
}

struct EntryPointIncomplete<'a, 'spv> {
    execution_model: spv::ExecutionModel,
    name: &'a str,
    interface: &'spv [u32],
}

fn decode_instruction<'a>(stream: &mut &'a [u32]) -> Result<Instruction<'a>, ParseError> {
    assert!(stream.len() >= 1);
    let word_count = (stream[0] >> 16) as usize;
    assert!(word_count >= 1);
    let opcode = (stream[0] & 0xffff) as u16;

    if stream.len() < word_count {
        return Err(ParseError);
    }

    let inst = Instruction { opcode, word_count: word_count as u16, operands: &stream[1..word_count] };

    *stream = &stream[word_count..];
    Ok(inst)
}

fn parse_string(stream: &[u32]) -> Result<(&str, &[u32]), ParseError> {
    let bytes = unsafe { slice::from_raw_parts(stream.as_ptr() as *const u8, stream.len() * 4) };
    let Some(nul_pos) = bytes.iter().position(|&b| b == 0) else {
        return Err(ParseError);
    };
    let Ok(str) = str::from_utf8(&bytes[..nul_pos]) else {
        return Err(ParseError);
    };
    let next_word = nul_pos / 4 + 1;
    Ok((str, &stream[next_word..]))
}

fn find_decoration<'spv>(
    decorations: &HashMap<u32, Vec<Decoration<'spv>>>,
    id: u32,
    deco: spv::Decoration,
) -> Option<&'spv [u32]> {
    decorations.get(&id).and_then(|decos| decos.iter().find(|d| d.id == deco).map(|d| d.operands))
}

fn find_member_decoration<'spv>(
    decorations: &HashMap<(u32, u32), Vec<Decoration<'spv>>>,
    struct_ty: u32,
    member: u32,
    deco: spv::Decoration,
) -> Option<&'spv [u32]> {
    decorations.get(&(struct_ty, member)).and_then(|decos| decos.iter().find(|d| d.id == deco).map(|d| d.operands))
}

fn create_shader_reflection_inner<'a>(arena: &'a Arena, spv: &[u32]) -> Result<&'a Shader<'a>, ParseError> {
    let mut header: Header = Default::default();
    let mut tymap: HashMap<u32, &'a TypeDesc<'a>> = Default::default();
    let mut strings: HashMap<u32, &'a str> = Default::default();
    let mut varmap: HashMap<u32, Variable<'a>> = Default::default();
    let mut names: HashMap<u32, &'a str> = Default::default();
    let mut member_names: HashMap<(u32, u32), &str> = Default::default();
    let mut deco: HashMap<u32, Vec<Decoration>> = Default::default();
    let mut mdeco: HashMap<(u32, u32), Vec<Decoration>> = Default::default();
    let mut tmp_entry_points: HashMap<u32, EntryPointIncomplete> = Default::default();

    assert!(spv.len() >= 5);
    header.magic = spv[0];
    header.version = spv[1];
    header.generator = spv[2];
    header.bound = spv[3];
    header.reserved = spv[4];

    assert_eq!(header.magic, 0x07230203);

    let mut stream = &spv[5..];

    // Parse instruction stream
    while !stream.is_empty() {
        let inst = decode_instruction(&mut stream)?;
        let Some(op) = spv::Op::from_u16(inst.opcode) else {
            eprintln!("unknown opcode {}", inst.opcode);
            continue;
        };
        use spv::Op::*;
        match (op, inst.operands) {
            (String, &[result_id, ref data @ ..]) => {
                let s = arena.alloc_str(parse_string(data)?.0);
                strings.insert(result_id, s);
            }
            (EntryPoint, &[execution_model, id, ref name_interface @ ..]) => {
                let (name, interface) = parse_string(name_interface)?;
                let name = arena.alloc_str(name);
                let Some(execution_model) = spv::ExecutionModel::from_u32(execution_model) else {
                    eprintln!("unknown execution model {execution_model}");
                    continue;
                };
                tmp_entry_points.insert(id, EntryPointIncomplete { execution_model, name, interface });
            }
            (Name, &[target, ref name @ ..]) => {
                let str = arena.alloc_str(parse_string(name)?.0);
                names.insert(target, str);
            }
            (MemberName, &[ty, member, ref name @ ..]) => {
                let str = arena.alloc_str(parse_string(name)?.0);
                member_names.insert((ty, member), str);
            }
            (String, &[ref operands @ ..]) => {}
            (Line, &[ref operands @ ..]) => {}
            (Extension, &[ref operands @ ..]) => {}
            (ExecutionMode, &[ref operands @ ..]) => {}
            (Capability, &[ref operands @ ..]) => {}
            (Decorate, &[target, decoration, ref operands @ ..]) => {
                let Some(decoration) = spv::Decoration::from_u32(decoration) else {
                    eprintln!("unknown decoration {decoration}");
                    continue;
                };
                let decoration = Decoration { id: decoration, operands };
                deco.entry(target).or_default().push(decoration);
            }
            (MemberDecorate, &[struct_ty, member, decoration, ref operands @ ..]) => {
                let Some(decoration) = spv::Decoration::from_u32(decoration) else {
                    eprintln!("unknown decoration {decoration}");
                    continue;
                };
                let decoration = Decoration { id: decoration, operands };
                mdeco.entry((struct_ty, member)).or_default().push(decoration);
            }
            (DecorateString, &[ref operands @ ..]) => {}
            (MemberDecorateString, &[ref operands @ ..]) => {}
            //(DecorateId, &[target, decoration, ref operands @ ..]) => {
            //    let decoration = spv::Decoration::from_u32(decoration).ok_or(ParseError)?;
            //    let deco = Decoration { id: decoration, operands };
            //    decorations.entry(target).or_default().push(deco);
            //}
            (TypeVoid, &[result_id]) => {
                let ty = arena.alloc(TypeDesc::Void);
                tymap.insert(result_id, ty);
            }
            (TypeBool, &[result_id]) => {
                let ty = arena.alloc(TypeDesc::Bool);
                tymap.insert(result_id, ty);
            }
            (TypeInt, &[result_id, width, signedness]) => {
                let ty = arena.alloc(match (width, signedness) {
                    (8, 0) => TypeDesc::Scalar(ScalarType::U8),
                    (8, 1) => TypeDesc::Scalar(ScalarType::I8),
                    (16, 0) => TypeDesc::Scalar(ScalarType::U16),
                    (16, 1) => TypeDesc::Scalar(ScalarType::I16),
                    (32, 0) => TypeDesc::Scalar(ScalarType::U32),
                    (32, 1) => TypeDesc::Scalar(ScalarType::I32),
                    (64, 0) => TypeDesc::Scalar(ScalarType::U64),
                    (64, 1) => TypeDesc::Scalar(ScalarType::I64),
                    _ => {
                        eprintln!("unsupported integer type {} {}", width, signedness);
                        return Err(ParseError);
                    },
                });
                tymap.insert(result_id, ty);
            }
            (TypeFloat, &[result_id, width, ref fp_encoding @ ..]) => {
                assert!(fp_encoding.is_empty());
                let ty = arena.alloc(match width {
                    32 => TypeDesc::Scalar(ScalarType::F32),
                    _ => {
                        eprintln!("invalid float type width {}", width);
                        return Err(ParseError);
                    }
                });
                tymap.insert(result_id, ty);
            }
            (TypeVector, &[result_id, component_ty, count]) => {
                let comp_ty = match tymap.get(&component_ty) {
                    Some(TypeDesc::Scalar(scalar)) => *scalar,
                    _ => {
                        eprintln!("invalid vector type");
                        return Err(ParseError);
                    }
                };
                let ty = arena.alloc(TypeDesc::Vector(comp_ty, count as u8));
                tymap.insert(result_id, ty);
            }
            (TypeMatrix, &[result_id, column_ty, col_count]) => {
                let column_ty = match tymap.get(&column_ty) {
                    Some(TypeDesc::Vector(comp_ty, count)) => (*comp_ty, *count),
                    _ => {
                        eprintln!("invalid matrix type");
                        return Err(ParseError);
                    }
                };
                let ty = arena.alloc(TypeDesc::Matrix {
                    scalar: column_ty.0,
                    rows: column_ty.1,
                    cols: col_count as u8,
                    stride: Cell::new(None),
                });
                tymap.insert(result_id, ty);
            }
            (
                TypeImage,
                &[result_id, sampled_type, dim, depth, arrayed, ms, sampled, format, ref access_qualifier @ ..],
            ) => {
                let ty = arena.alloc(TypeDesc::Image);
                tymap.insert(result_id, ty);
            }
            (TypeSampler, &[result_id]) => {
                let ty = arena.alloc(TypeDesc::Sampler);
                tymap.insert(result_id, ty);
            }
            (TypeSampledImage, &[result_id, ref rest @ ..]) => {
                let ty = arena.alloc(TypeDesc::SampledImage);
                tymap.insert(result_id, ty);
            }
            (TypeArray, &[result_id, elem_type, length]) => {
                let Some(element) = tymap.get(&elem_type) else {
                    eprintln!("unknown element type {}", elem_type);
                    continue;
                };
                let ty = arena.alloc(TypeDesc::Array {
                    element,
                    len: length,
                    stride: Cell::new(None),
                });
                tymap.insert(result_id, ty);
            }
            (TypeRuntimeArray, &[result_id, elem_type]) => {
                let Some(element) = tymap.get(&elem_type) else {
                    eprintln!("unknown element type {}", elem_type);
                    continue;
                };
                let ty = arena.alloc(TypeDesc::RuntimeArray {
                    element,
                    stride: Cell::new(None),
                });
                tymap.insert(result_id, ty);
            }
            (TypeStruct, &[result_id, ref member_types @ ..]) => {
                let mut fields = Vec::with_capacity(member_types.len());
                for (i, ty) in member_types.iter().enumerate() {
                    let Some(ty) = tymap.get(ty) else {
                        eprintln!("unknown member type {}", ty);
                        continue;
                    };
                    let name = member_names.get(&(result_id, i as u32)).copied().unwrap_or_default();

                    let offset = match find_member_decoration(&mdeco, result_id, i as u32, spv::Decoration::Offset) {
                        Some(&[offset]) => offset,
                        _ => 0,
                    };

                    fields.push(StructField { name, ty, offset });
                }

                let fields = arena.alloc_slice_clone(&fields);
                let name = names.get(&result_id).copied().unwrap_or_default();
                let ty = arena.alloc(StructType { name, fields });
                let ty = arena.alloc(TypeDesc::Struct(ty));
                tymap.insert(result_id, ty);
            }
            (TypeOpaque, &[ref operands @ ..]) => {
                // TODO
            }
            (TypePointer, &[result_id, storage_class, pointee_ty]) => {
                let Some(storage_class) = spv::StorageClass::from_u32(storage_class) else {
                    eprintln!("unknown storage class {}", storage_class);
                    continue;
                };
                let Some(&pointee_ty) = tymap.get(&pointee_ty) else {
                    eprintln!("unknown pointee type {}", pointee_ty);
                    continue;
                };
                match tymap.entry(result_id) {
                    Entry::Occupied(mut existing) => {
                        match existing.get_mut() {
                            TypeDesc::Pointer(PointerType { pointee: p, .. }) => {
                                // replace forward declaration with final
                                unsafe {
                                    // SAFETY: no other references to the inner value of the cell exist
                                    //         at this time.
                                    p.get().write(Some(pointee_ty))
                                }
                            }
                            _ => {
                                eprintln!("type {} was already defined as a non-pointer type", result_id);
                                return Err(ParseError);
                            },
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(arena.alloc(TypeDesc::Pointer(PointerType::new(storage_class, pointee_ty))));
                    }
                }
            }
            (TypeForwardPointer, &[result_id, storage_class]) => {
                let Some(sc) = spv::StorageClass::from_u32(storage_class) else {
                    eprintln!("unknown storage class {}", storage_class);
                    continue;
                };
                let ty = arena.alloc(TypeDesc::Pointer(PointerType::new_forward(sc)));
                tymap.insert(result_id, ty);
            }
            (TypeFunction, &[ref operands @ ..]) => {}
            (ConstantTrue, &[ref operands @ ..]) => {}
            (ConstantFalse, &[ref operands @ ..]) => {}
            (Constant, &[ref operands @ ..]) => {}
            (ConstantComposite, &[ref operands @ ..]) => {}
            (ConstantSampler, &[ref operands @ ..]) => {}
            (ConstantNull, &[ref operands @ ..]) => {}
            (SpecConstantTrue, &[ref operands @ ..]) => {}
            (SpecConstantFalse, &[ref operands @ ..]) => {}
            (SpecConstant, &[ref operands @ ..]) => {}
            (SpecConstantComposite, &[ref operands @ ..]) => {}
            (SpecConstantOp, &[ref operands @ ..]) => {}
            (Variable, &[result_type, result_id, storage_class, ref initializer @ ..]) => {
                let Some(storage_class) = spv::StorageClass::from_u32(storage_class) else {
                    eprintln!("unknown storage class {}", storage_class);
                    continue;
                };
                let Some(ty) = tymap.get(&result_type) else {
                    eprintln!("unknown variable type {}", result_type);
                    continue;
                };
                let name = names.get(&result_id).copied().unwrap_or_default();
                varmap.insert(result_id, self::Variable { name, storage_class, ty, uniform: Cell::new(false) });
            }
            (SizeOf, &[ref operands @ ..]) => {}
            (_, _) => {}
        }
    }

    // Resolve entry point parameters
    let entry_points: &mut [EntryPoint] = arena.alloc_slice_fill_default(tmp_entry_points.len());

    for (i, (_, ep)) in tmp_entry_points.iter().enumerate() {
        let mut params = vec![];
        for var in ep.interface {
            let Some(var) = varmap.get(var) else {
                eprintln!("unknown variable {}", var);
                continue;
            };
            match var.storage_class {
                spv::StorageClass::Input | spv::StorageClass::Output => {
                    // don't care
                    continue;
                }
                spv::StorageClass::PushConstant => {
                    match var.ty {
                        TypeDesc::Pointer(pt) => {
                            params.push(RootParam {
                                // assume this is at offset zero, I don't see where else I could specify that
                                name: var.name,
                                kind: AccessKind::PushData { offset: 0 },
                                ty: pt.pointee(),
                            })
                        }
                        _ => {
                            // PushConstant variables should always be pointers to a struct
                            // which contains the actual push constants, apparently?
                            // TODO emit warning
                        }
                    }
                }
                _ => {
                    // TODO
                }
            }
        }

        entry_points[i].name = ep.name;
        entry_points[i].params = arena.alloc_slice_clone(&params);
    }

    let shader = arena.alloc(Shader { entry_points });
    Ok(shader)
}

impl ShaderReflection {
    pub fn new(spirv: &[u32]) -> Result<ShaderReflection, ParseError> {
        let arena = Arena::new();
        let root = create_shader_reflection_inner(&arena, spirv)? as *const Shader as *const Shader<'static>;
        Ok(ShaderReflection { arena, root })
    }
}
