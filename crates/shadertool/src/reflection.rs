//! Extract extended reflection information from a slang shader module.

use crate::Arena;
use color_print::ceprintln;
use gpu_types::reflection as refl;
use log::warn;
use slang::reflection::{TypeLayout, VariableLayout};
use slang::{ParameterCategory, ScalarType, TypeKind};
use std::collections::HashMap;

/// Represents a constant value.
#[repr(C)]
#[derive(Clone)]
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
    String { value: String },
}

#[derive(Clone)]
pub struct UserAttribute {
    pub name: String,
    pub value: Value,
}

/*
/// Describes how a shader parameter is accessed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum ParamLocation {
    /// Parameter is in a buffer accessed via a resource binding.
    Binding {
        /// Buffer resource index (in `Signature::resources`).
        resource_index: u32,
        /// Byte offset of the parameter within the buffer.
        offset: u32,
    },
    /// Parameter value is passed in push data.
    PushData {
        /// Byte offset of the parameter value within push data.
        offset: u32,
    },
    /// Indirect parameter, relative to another pointer parameter.
    Indirect {
        /// Index of the pointer parameter this parameter is relative to (in `Signature::params`).
        rel: u32,
        /// Byte offset of the parameter value relative to the pointer value.
        offset: u32,
    },
}*/
/*
/// Represents a shader parameter.
pub struct Param {
    /// Full name of the parameter.
    ///
    /// If this is part of a struct, the name is the full path to the parameter, e.g., `param0.myField`.
    pub name: String,
    /// Parent parameter index, or `u32::MAX` if this is a root parameter.
    pub parent: u32,
    /// Location of the parameter.
    pub location: ParamLocation,
    /// Size of the parameter in bytes (if applicable).
    ///
    /// This is the number of bytes that should be readable from the parameter location.
    pub byte_size: u32,
    /// User attributes.
    pub attributes: Vec<UserAttribute>,
}*/

fn convert_scalar_type(scalar_type: ScalarType) -> refl::ScalarType {
    match scalar_type {
        ScalarType::Bool => refl::ScalarType::Bool,
        ScalarType::Int8 => refl::ScalarType::I8,
        ScalarType::Uint8 => refl::ScalarType::U8,
        ScalarType::Int16 => refl::ScalarType::I16,
        ScalarType::Uint16 => refl::ScalarType::U16,
        ScalarType::Int32 => refl::ScalarType::I32,
        ScalarType::Uint32 => refl::ScalarType::U32,
        ScalarType::Float32 => refl::ScalarType::F32,
        ScalarType::Uint64 => refl::ScalarType::U64,
        ScalarType::Int64 => refl::ScalarType::I64,
        _ => unimplemented!("Unsupported scalar type: {:?}", scalar_type),
    }
}

pub(crate) struct TypeCollector<'a> {
    arena: &'a Arena,
    pub(crate) struct_types: HashMap<&'a str, &'a refl::StructType<'a>>,
    ty_counter: usize,
}

impl<'a> TypeCollector<'a> {

    pub(crate) fn new(arena: &'a Arena) -> Self {
        TypeCollector {
            arena,
            struct_types: HashMap::new(),
            ty_counter: 0,
        }
    }

    /// Generates a TypeDesc for the given struct TypeLayout.
    fn reflect_struct(&mut self, ty_layout: &TypeLayout) -> refl::TypeDesc<'a> {
        let name = if let Some(name) = ty_layout.name() {
            self.arena.alloc_str(name)
        } else {
            let generated_name = format!("unnamed_{}", self.ty_counter);
            self.ty_counter += 1;
            self.arena.alloc_str(&generated_name)
        };

        if self.struct_types.contains_key(name) {
            return refl::TypeDesc::Struct(self.struct_types.get(name).unwrap());
        }

        // insert a placeholder to avoid infinite recursion on recursive types
        let placeholder_struct = self.arena.alloc(refl::StructType { name, fields: self.arena.alloc_slice_copy(&[]) });
        self.struct_types.insert(name, placeholder_struct);

        let mut fields = vec![];
        for (i, field) in ty_layout.fields().enumerate() {
            let name = if let Some(name) = field.name() {
                self.arena.alloc_str(name)
            } else {
                self.arena.alloc_str(&format!("unnamed_{}", i))
            };

            let ty_desc = self.reflect_type(&field.type_layout().unwrap());
            let offset = field.offset(ParameterCategory::Uniform);
            fields.push(refl::StructField { name, ty: ty_desc, offset: offset as u32 });
        }

        let ty = self.arena.alloc(refl::StructType { name, fields: self.arena.alloc_slice_copy(&fields) });

        self.struct_types.insert(name, ty);
        refl::TypeDesc::Struct(ty)

        /*let mut fields = TokenStream::new();
        let mut offset_checks = TokenStream::new();
        let mut fields_meta = TokenStream::new();

        for (i, field) in ty_layout.fields().enumerate() {
            let field_ident =
                if let Some(name) = field.name() { format_ident!("{name}") } else { format_ident!("unnamed_{}", i) };

            let TypeReflection { ty_ref: field_ty_ref, metadata: field_metadata } = self.reflect_type(&field.type_layout().unwrap());

            fields.append_all(quote! {
                pub #field_ident: #field_ty_ref,
            });

            let field_offset = field.offset(ParameterCategory::Uniform);
            offset_checks.append_all(quote! {
                const _: () = assert!(std::mem::offset_of!(#type_name, #field_ident) == #field_offset);
            });

            fields_meta.append_all(quote! {
                StructField {
                    name: stringify!(#field_ident),
                    ty: #field_metadata,
                    offset: #field_offset as u32,
                },
            });
        }

        let tokens = quote! {
            #[repr(C)]
            #[derive(Copy, Clone)]
            pub struct #type_name {
                #fields
            }
            impl #type_name {

            }
            #offset_checks
        };

        self.struct_declarations.insert(type_name.clone(), tokens);

        TypeReflection {
            ty_ref: quote! { #type_name },
            metadata: quote! { TypeDesc::Struct(#type_name::META) },
        }*/
    }

    /// Generates a TypeDesc for the given TypeLayout.
    fn reflect_type(&mut self, ty_layout: &TypeLayout) -> refl::TypeDesc<'a> {
        match ty_layout.kind() {
            TypeKind::Scalar => {
                let scalar_ty = convert_scalar_type(ty_layout.scalar_type().unwrap());
                refl::TypeDesc::Scalar(scalar_ty)
            }
            TypeKind::Vector => {
                let scalar_ty = convert_scalar_type(ty_layout.scalar_type().unwrap());
                let vector_size = ty_layout.column_count().unwrap();
                refl::TypeDesc::Vector(scalar_ty, vector_size as u8)
            }
            TypeKind::Matrix => {
                let scalar_ty = convert_scalar_type(ty_layout.scalar_type().unwrap());
                let column_count = ty_layout.column_count().unwrap();
                let row_count = ty_layout.row_count().unwrap();
                refl::TypeDesc::Matrix(scalar_ty, row_count as u8, column_count as u8)
            }
            TypeKind::Pointer => {
                let elem_ty_desc = self.reflect_type(&ty_layout.element_type_layout().unwrap());
                refl::TypeDesc::Pointer(self.arena.alloc(elem_ty_desc))
            }
            TypeKind::Array => {
                let elem_ty_desc = self.reflect_type(&ty_layout.element_type_layout().unwrap());
                let array_size = ty_layout.total_array_element_count();
                refl::TypeDesc::Array(self.arena.alloc(elem_ty_desc), array_size as u32)
            }
            TypeKind::Struct => self.reflect_struct(ty_layout),
            _ => {
                ceprintln!(
                    "<y>warning</>: unknown type kind {:?} for {}",
                    ty_layout.kind(),
                    ty_layout.name().unwrap_or("unnamed")
                );
                refl::TypeDesc::Scalar(refl::ScalarType::U8)
            }
        }
        /*  let ScalarTypeInfo { metadata, .. } = ScalarTypeInfo::from(ty_layout.scalar_type().unwrap());
                let scalar_ty = format_ident!("{}", ty_layout.name().unwrap());
                TypeReflection {
                    ty_ref: quote! { #scalar_ty },
                    metadata: quote! { TypeDesc::Scalar(#metadata) },
                }
            }
            TypeKind::Vector => {
                // TODO Handle special case of texture and image descriptors.
                //      Blocked on https://github.com/shader-slang/slang/issues/8845
                //let name = ty_layout.name().unwrap_or("unnamed");

                //let uniform_size = ty_layout.size(ParameterCategory::Uniform);
                //let descriptor_slot_size = ty_layout.size(ParameterCategory::DescriptorTableSlot);

                let ScalarTypeInfo { suffix, metadata } = ScalarTypeInfo::from(ty_layout.scalar_type().unwrap());
                let vector_size = ty_layout.column_count().unwrap();
                let vector_type = format_ident!("vec{vector_size}{suffix}");

                TypeReflection {
                    ty_ref: quote! { #vector_type },
                    metadata: quote! { TypeDesc::Vector(VectorType { scalar: #metadata, len: #vector_size }) },
                }
            }
            TypeKind::Matrix => {
                let ScalarTypeInfo { suffix, metadata } = ScalarTypeInfo::from(ty_layout.scalar_type().unwrap());
                let column_count = ty_layout.column_count().unwrap();
                let row_count = ty_layout.row_count().unwrap();
                let matrix_type = format_ident!("mat{column_count}x{row_count}{suffix}");
                TypeReflection {
                    ty_ref: quote! { #matrix_type },
                    metadata: quote! { TypeDesc::Matrix(MatrixType { scalar: #metadata, rows: #row_count, cols: #column_count }) },
                }
            }
            TypeKind::Pointer => {
                let TypeReflection { ty_ref, metadata } = self.reflect_type(&ty_layout.element_type_layout().unwrap());
                TypeReflection {
                    ty_ref: quote! { ::gpu::Ptr<#ty_ref> },
                    metadata: quote! { TypeDesc::Pointer(const { &#metadata }) },
                }
            }
            TypeKind::Array => {
                let TypeReflection { ty_ref, metadata } = self.reflect_type(&ty_layout.element_type_layout().unwrap());
                let array_size = ty_layout.total_array_element_count();
                TypeReflection {
                    ty_ref: quote! { [#ty_ref; #array_size] },
                    metadata: quote! { TypeDesc::Array { elem: const { &#metadata }, len: #array_size } },
                }
            }
            TypeKind::Struct => self.reflect_struct(ty_layout),
            _ => {
                ceprintln!(
                    "<y>warning</>: unknown type kind {:?} for {}",
                    ty_layout.kind(),
                    ty_layout.name().unwrap_or("unnamed")
                );
                TypeReflection {
                    ty_ref: quote! { () },
                    metadata: quote! { TypeDesc::Scalar(ScalarType::U8) },
                }
            }
        }*/
    }

}

/// Collects reflection information from modules and entry points.
pub(crate) struct ParamCollector<'a, 'ty> {
    tys: &'ty mut TypeCollector<'a>,
    pub(crate) access_chains: Vec<&'a refl::AccessChain<'a>>,
}

impl<'a, 'ty> ParamCollector<'a, 'ty> {
    pub(crate) fn new(tys: &'ty mut TypeCollector<'a>) -> Self {
        ParamCollector { tys, access_chains: vec![] }
    }

    fn insert_access_chain(
        &mut self,
        parent: Option<&'a refl::AccessChain<'a>>,
        kind: refl::AccessKind<'a>,
        ty: refl::TypeDesc<'a>,
        suffix: &'a str,
    ) -> &'a refl::AccessChain<'a> {
        let name = match parent {
            Some(parent) => format!("{}.{}", parent.name, suffix),
            None => suffix.to_string(),
        };
        if let Some(ac) = self.access_chains.iter().find(|&ac| ac.name == name) {
            if ac.kind != kind {
                warn!("duplicate access chain with different kinds: {}", name);
            }
            return ac;
        }
        let access_chain = self.tys.arena.alloc(refl::AccessChain { parent, name: self.tys.arena.alloc_str(&name), kind, ty });
        self.access_chains.push(access_chain);
        access_chain
    }

    fn reflect_child_access_chain<'b>(
        &mut self,
        access_chain: &'a refl::AccessChain<'a>,
        ty_layout: &'b TypeLayout,
        type_path: &mut Vec<&'b TypeLayout>,
    ) {
        // avoid infinite recursion
        if type_path.iter().any(|&t| t as *const _ == ty_layout as *const _) {
            //eprintln!("skipping recursive type for {}", full_name);
            return;
        }

        // slang's struct types are weird in the sense that they can hold simultaneously
        // ordinary data and resource slots (textures, buffers), when wrapped in ParameterBlocks.
        // Which means that the layout of structs are "multidimensional" instead of being
        // just offsets and sizes in bytes.
        // For instance, consider the following slang struct type:
        //
        // struct S {
        //     float4 field;            // offset=0(mem), size=16 bytes (mem)
        //     Texture2D texture;       // offset=0(tex), size=1 slot (tex)
        // }
        //
        // The "size" of this struct is: 16 bytes of ordinary data + 1 texture slot.
        //
        // We explicitly don't support non-ordinary struct types.
        // Those "non-ordinary" struct types have a ParameterCategory different from Uniform,
        // so we can detect them here and bail out.
        for category in ty_layout.categories() {
            if category != ParameterCategory::Uniform {
                //ceprintln!("<r>error</>: unsupported parameter category {:?} in type of {}", category, full_name);
                return;
            }
        }

        let ty_desc = self.tys.reflect_type(ty_layout);

        match ty_desc {
            refl::TypeDesc::Struct(struct_type) => {
                // emit an access chain for all fields
                for (i, field) in struct_type.fields.iter().enumerate() {
                    let field_access = self.insert_access_chain(
                        Some(access_chain),
                        refl::AccessKind::Field { offset: field.offset },
                        field.ty,
                        field.name,
                    );

                    type_path.push(ty_layout);
                    let field_ty_layout = ty_layout.fields().nth(i).unwrap().type_layout().unwrap();
                    self.reflect_child_access_chain(field_access, field_ty_layout, type_path);
                    type_path.pop();
                }
            }
            refl::TypeDesc::Pointer(pointee) => {
                // emit an access chain for the dereferenced type
                //let name = self.arena.alloc_str(&format!("{full_name}.$"));
                let deref_access = self.insert_access_chain(Some(access_chain), refl::AccessKind::Load, *pointee, "$");

                //type_path.push(ty_layout);
                let pointee_ty_layout = ty_layout.element_type_layout().unwrap();
                self.reflect_child_access_chain(deref_access, pointee_ty_layout, type_path);
                //type_path.pop();
            }
            refl::TypeDesc::Array(elem, len) => {
                // emit an access chain for array indexing
                let index_access = self.insert_access_chain(
                    Some(access_chain),
                    refl::AccessKind::ArrayIndex {
                        count: len,
                        stride: ty_layout.element_stride(ParameterCategory::Uniform) as u32,
                    },
                    *elem,
                    "[@]",
                );

                self.reflect_child_access_chain(index_access, ty_layout.element_type_layout().unwrap(), type_path);
            }
            refl::TypeDesc::Scalar(_) => {}
            refl::TypeDesc::Vector(..) => {}
            refl::TypeDesc::Matrix(..) => {}
            refl::TypeDesc::RuntimeArray(_) => {}
            refl::TypeDesc::ImageHandle(_) => {}
        }
    }

    fn reflect_param(&mut self, param: &VariableLayout) {
        // NOTE: this has turned into an unwrap fest since the last slang-rs update;
        //       at some point we should fork and make our own bindings to slang
        //       (at least panic on null pointers instead of turning *everything* into Option)

        //let set = param.binding_space();
        //let binding = param.binding_index();
        let category = param.category().unwrap();
        let name = self.tys.arena.alloc_str(param.variable().unwrap().name().unwrap_or("unnamed"));

        match category {
            ParameterCategory::Uniform => {
                let type_layout = param.type_layout().unwrap();
                let ty_desc = self.tys.reflect_type(type_layout);
                let offset = param.offset(slang::ParameterCategory::Uniform);
                let root_access_chain =
                    self.insert_access_chain(None, refl::AccessKind::PushData { offset: offset as u32 }, ty_desc, name);
                self.reflect_child_access_chain(root_access_chain, type_layout, &mut vec![]);
            }
            ParameterCategory::PushConstantBuffer => {
                // sanity check
                let type_layout = param.type_layout().unwrap();
                assert!(type_layout.kind() == TypeKind::ConstantBuffer);
                // The type of push constant buffers is `ConstantBuffer<T>`, which conceptually
                // represents a constant buffer slot. This is meaningless for push constants,
                // and we don't want this type to end up in the reflected type hierarchy,
                // so pass through it.
                let cbuffer_content_layout = type_layout.element_type_layout().unwrap();
                let ty_desc = self.tys.reflect_type(cbuffer_content_layout);
                let root_access_chain =
                    self.insert_access_chain(None, refl::AccessKind::PushData { offset: 0 }, ty_desc, name);
                self.reflect_child_access_chain(root_access_chain, cbuffer_content_layout, &mut vec![]);
            }
            ParameterCategory::None => {
                //eprintln!("resource {}: {:?}", name, category);
            }
            _ => {
                //ceprintln!("<r>error</>: unsupported parameter category {category:?} for {name}");
            }
        }
    }

    pub(crate) fn reflect_entry_point(&mut self, entry_point: &slang::reflection::EntryPoint) {
        //eprintln!("entry point: {}", entry_point.name().unwrap_or("unnamed"));
        for param in entry_point.parameters() {
            self.reflect_param(param);
        }
    }

    pub(crate) fn reflect_global_params(&mut self, shader: &slang::reflection::Shader) {
        //let global_params_layout = shader.global_params_var_layout();
        let ty_layout = shader.global_params_type_layout().unwrap();
        if ty_layout.kind() != TypeKind::Struct {
            panic!("expected global params to be a struct");
        }
        //eprintln!("ty_layout: kind={:?} size={} bytes", ty_layout.kind(), ty_layout.size(ParameterCategory::Uniform));
        for field in ty_layout.fields() {
            self.reflect_param(field);
        }
    }
}