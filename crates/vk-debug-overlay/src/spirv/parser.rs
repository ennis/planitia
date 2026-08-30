//! SPIR-V parser
use crate::spirv::{
    ConstantId, ConstantInfo, EntryPointInfo, Module, ParseError, PointerType, ScalarType, StructField, StructType,
    TypeId, TypeInfo, VariableId, VariableInfo,
};
use num_traits::FromPrimitive;
use spirv as spv;
use std::collections::HashMap;
use std::{array, slice};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Instruction<'a> {
    pub opcode: u16,
    pub word_count: u16,
    pub operands: &'a [u32],
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

struct EntryPointIncomplete<'spv> {
    execution_model: spv::ExecutionModel,
    name: String,
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

fn create_shader_reflection_inner(spv: &[u32]) -> Result<Module, ParseError> {
    let mut header: Header = Default::default();
    let mut strings: HashMap<u32, &str> = Default::default();
    let mut names: HashMap<u32, &str> = Default::default();
    let mut member_names: HashMap<(u32, u32), &str> = Default::default();
    let mut deco: HashMap<u32, Vec<Decoration>> = Default::default();
    let mut mdeco: HashMap<(u32, u32), Vec<Decoration>> = Default::default();

    let mut module = Module::new();

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
        let Some(op) = spv::Op::from_u32(inst.opcode as u32) else {
            eprintln!("unknown opcode {}", inst.opcode);
            continue;
        };
        use spv::Op::*;
        match (op, inst.operands) {
            (String, &[result_id, ref data @ ..]) => {
                strings.insert(result_id, parse_string(data)?.0);
            }
            (EntryPoint, &[execution_model, id, ref name_interface @ ..]) => {
                let (name, interface) = parse_string(name_interface)?;
                let Some(execution_model) = spv::ExecutionModel::from_u32(execution_model) else {
                    eprintln!("unknown execution model {execution_model}");
                    continue;
                };

                module.insert_entry_point(
                    id,
                    EntryPointInfo {
                        stage: execution_model,
                        name: name.to_string(),
                        params: interface.iter().map(|id| VariableId(*id)).collect(),
                    },
                );
            }
            (Name, &[target, ref name @ ..]) => {
                names.insert(target, parse_string(name)?.0);
            }
            (MemberName, &[ty, member, ref name @ ..]) => {
                member_names.insert((ty, member), parse_string(name)?.0);
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
                //let ty = bump.alloc(TypeDesc::Void);
                //tymap.insert(result_id, ty);
                module.insert_type(result_id, TypeInfo::Void);
            }
            (TypeBool, &[result_id]) => {
                //let ty = bump.alloc(TypeDesc::Bool);
                //tymap.insert(result_id, ty);
                module.insert_type(result_id, TypeInfo::Bool);
            }
            (TypeInt, &[result_id, width, signedness]) => {
                let ty = match (width, signedness) {
                    (8, 0) => TypeInfo::Scalar(ScalarType::U8),
                    (8, 1) => TypeInfo::Scalar(ScalarType::I8),
                    (16, 0) => TypeInfo::Scalar(ScalarType::U16),
                    (16, 1) => TypeInfo::Scalar(ScalarType::I16),
                    (32, 0) => TypeInfo::Scalar(ScalarType::U32),
                    (32, 1) => TypeInfo::Scalar(ScalarType::I32),
                    (64, 0) => TypeInfo::Scalar(ScalarType::U64),
                    (64, 1) => TypeInfo::Scalar(ScalarType::I64),
                    _ => {
                        eprintln!("unsupported integer type {} {}", width, signedness);
                        return Err(ParseError);
                    }
                };
                module.insert_type(result_id, ty);
            }
            (TypeFloat, &[result_id, width, ref fp_encoding @ ..]) => {
                assert!(fp_encoding.is_empty());
                let ty = match width {
                    32 => TypeInfo::Scalar(ScalarType::F32),
                    _ => {
                        eprintln!("invalid float type width {}", width);
                        return Err(ParseError);
                    }
                };
                module.insert_type(result_id, ty);
            }
            (TypeVector, &[result_id, component_ty, count]) => {
                let comp_ty = match module[TypeId(component_ty)] {
                    TypeInfo::Scalar(scalar) => scalar,
                    _ => {
                        eprintln!("invalid vector type");
                        return Err(ParseError);
                    }
                };
                module.insert_type(result_id, TypeInfo::Vector(comp_ty, count as u8));
            }
            (TypeMatrix, &[result_id, column_ty, col_count]) => {
                let column_ty = match module[TypeId(column_ty)] {
                    TypeInfo::Vector(comp_ty, count) => (comp_ty, count),
                    _ => {
                        eprintln!("invalid matrix type");
                        return Err(ParseError);
                    }
                };
                module.insert_type(
                    result_id,
                    TypeInfo::Matrix { scalar: column_ty.0, rows: column_ty.1, cols: col_count as u8, stride: None },
                );
            }
            (
                TypeImage,
                &[result_id, sampled_type, dim, depth, arrayed, ms, sampled, format, ref access_qualifier @ ..],
            ) => {
                module.insert_type(result_id, TypeInfo::Image);
            }
            (TypeSampler, &[result_id]) => {
                module.insert_type(result_id, TypeInfo::Sampler);
            }
            (TypeSampledImage, &[result_id, ref rest @ ..]) => {
                module.insert_type(result_id, TypeInfo::SampledImage);
            }
            (TypeArray, &[result_id, elem_type, length]) => {
                module.insert_type(
                    result_id,
                    TypeInfo::Array { element: TypeId(elem_type), len: ConstantId(length), stride: None },
                );
            }
            (TypeRuntimeArray, &[result_id, elem_type]) => {
                let ty = TypeInfo::RuntimeArray { element: TypeId(elem_type), stride: None };
                module.insert_type(result_id, ty);
            }
            (TypeStruct, &[result_id, ref member_types @ ..]) => {
                let mut fields = Vec::with_capacity(member_types.len());
                for (i, &ty) in member_types.iter().enumerate() {
                    let name = member_names.get(&(result_id, i as u32)).copied().unwrap_or_default();
                    let offset = match find_member_decoration(&mdeco, result_id, i as u32, spv::Decoration::Offset) {
                        Some(&[offset]) => offset,
                        _ => 0,
                    };

                    fields.push(StructField { name: name.to_owned(), ty: TypeId(ty), offset });
                }

                let name = names.get(&result_id).copied().unwrap_or_default().to_string();
                module.insert_type(result_id, TypeInfo::Struct(StructType { name, fields }));
            }
            (TypeOpaque, &[ref operands @ ..]) => {
                // TODO
            }
            (TypePointer, &[result_id, storage_class, pointee_ty]) => {
                let Some(storage_class) = spv::StorageClass::from_u32(storage_class) else {
                    eprintln!("unknown storage class {}", storage_class);
                    continue;
                };

                module.insert_type(
                    result_id,
                    TypeInfo::Pointer(PointerType { storage_class, pointee: Some(TypeId(pointee_ty)) }),
                );
            }
            (TypeForwardPointer, &[result_id, storage_class]) => {
                let Some(sc) = spv::StorageClass::from_u32(storage_class) else {
                    eprintln!("unknown storage class {}", storage_class);
                    continue;
                };
                module.insert_type(result_id, TypeInfo::Pointer(PointerType { storage_class: sc, pointee: None }));
            }
            (TypeFunction, &[ref operands @ ..]) => {}
            (ConstantTrue, &[ref operands @ ..]) => {}
            (ConstantFalse, &[ref operands @ ..]) => {}
            (Constant, &[result_type, result_id, ref value @ ..]) => {
                let ty = module[TypeId(result_type)].as_scalar().unwrap();
                let value_bytes = unsafe { slice::from_raw_parts(value.as_ptr() as *const u8, value.len() * 4) };
                let value_bytes = array::from_fn(|i| value_bytes.get(i).cloned().unwrap_or(0));
                module.insert_constant(result_id, ConstantInfo { ty, value_bytes: Some(value_bytes) });
            }
            (ConstantComposite, &[ref operands @ ..]) => {}
            (ConstantSampler, &[ref operands @ ..]) => {}
            (ConstantNull, &[ref operands @ ..]) => {}
            (SpecConstantTrue, &[ref operands @ ..]) => {}
            (SpecConstantFalse, &[ref operands @ ..]) => {}
            (SpecConstant, &[ref operands @ ..]) => {}
            (SpecConstantComposite, &[ref operands @ ..]) => {}
            (SpecConstantOp, &[ref operands @ ..]) => {}
            (Variable, &[result_type, result_id, storage_class, ref initializer @ ..]) => {
                let storage_class = spv::StorageClass::from_u32(storage_class).unwrap();
                let name = names.get(&result_id).copied().unwrap_or_default().to_string();
                module.insert_variable(
                    result_id,
                    VariableInfo { name, sc: storage_class, ty: TypeId(result_type), uniform: false },
                );
            }
            (UntypedVariableKHR, &[result_type, result_id, storage_class, ref operands @ ..]) => {
                let storage_class = spv::StorageClass::from_u32(storage_class).unwrap();
                let name = names.get(&result_id).copied().unwrap_or_default().to_string();
                module.insert_variable(
                    result_id,
                    VariableInfo { name, sc: storage_class, ty: TypeId(result_type), uniform: false },
                );
            }
            (SizeOf, &[ref operands @ ..]) => {}
            (_, _) => {}
        }
    }

    /*// Resolve entry point parameters
    let entry_points: &[EntryPointInfo] = tmp_entry_points.iter().map(|(_, ep)| {
        let mut params = vec![];
        for var in ep.interface {
            let var = &module[VariableId(*var)];
            match var.sc {
                spv::StorageClass::Input | spv::StorageClass::Output => {
                    // don't care
                    continue;
                }
                spv::StorageClass::PushConstant => {
                    match &module[var.ty] {
                        TypeInfo::Pointer(pt) => {
                            params.push(EntryPointParam {
                                // assume this is at offset zero, I don't see where else I could specify that
                                name: var.name,
                                kind: AccessKind::PushData { offset: 0 },
                                ty: pt.pointee.unwrap(),
                            })
                        }
                        _ => {
                            // PushConstant variables should always be pointers to a struct
                            // which contains the actual push constants, apparently?
                            eprintln!("PushConstant variable {} is not a pointer type", var.name);
                        }
                    }
                }
                _ => {
                    eprintln!("unsupported entry point param storage class {:?}", var.sc);
                }
            }
        }
        EntryPointInfo { stage: ep.execution_model, name: ep.name, params: bump.alloc_slice_clone(&params) }
    }).collect::<Vec<_>>();*/

    Ok(module)
}

impl Module {
    pub fn parse(spirv: &[u32]) -> Result<Module, ParseError> {
        create_shader_reflection_inner(spirv)
    }
}
