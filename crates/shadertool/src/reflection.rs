//! Extract extended reflection information from a slang shader module.

use crate::Module;
use color_print::ceprintln;
use proc_macro2::{Ident, TokenStream};
use quote::{TokenStreamExt, format_ident, quote};
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
}

#[derive(Clone)]
pub struct UserAttribute {
    pub name: String,
    pub value: Value,
}

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
}

/// Rust code for a type defined in slang.
struct TypeReflection {
    rust: TokenStream,
}

pub(crate) struct CollectedReflectionData {
    pub(crate) params: Vec<Param>,
    types: HashMap<Ident, TypeReflection>,
    ty_counter: usize,
}

type ParamIndex = u32;

fn scalar_type_suffix(scalar_type: ScalarType) -> &'static str {
    match scalar_type {
        ScalarType::Bool => "b",
        ScalarType::Int32 => "i",
        ScalarType::Uint32 => "u",
        ScalarType::Int16 => "i16",
        ScalarType::Uint16 => "u16",
        ScalarType::Int8 => "i8",
        ScalarType::Uint8 => "u8",
        ScalarType::Float32 => "f",
        ScalarType::Float16 => "h",
        _ => panic!("unsupported scalar type {:?}", scalar_type),
    }
}

impl CollectedReflectionData {
    pub(crate) fn new() -> Self {
        CollectedReflectionData { params: vec![], types: Default::default(), ty_counter: 0 }
    }

    /// Generates Rust code for a struct type described by `ty_layout`.
    ///
    /// Returns the Rust path of the struct type.
    fn reflect_struct(&mut self, ty_layout: &TypeLayout) -> TokenStream {
        let type_name = if let Some(name) = ty_layout.name() {
            format_ident!("{name}")
        } else {
            let generated_name = format_ident!("unnamed_{}", self.ty_counter);
            self.ty_counter += 1;
            generated_name
        };

        if self.types.contains_key(&type_name) {
            return quote! { #type_name };
        }

        // insert a placeholder to avoid infinite recursion on recursive types
        self.types.insert(type_name.clone(), TypeReflection { rust: quote! {} });

        let mut fields = TokenStream::new();
        let mut offset_checks = TokenStream::new();

        for (i, field) in ty_layout.fields().enumerate() {
            let field_ident =
                if let Some(name) = field.name() { format_ident!("{name}") } else { format_ident!("unnamed_{}", i) };
            let field_type_name = self.reflect_type(&field.type_layout().unwrap());
            fields.append_all(quote! {
                pub #field_ident: #field_type_name,
            });

            let field_offset = field.offset(ParameterCategory::Uniform);
            offset_checks.append_all(quote! {
                const _: () = assert!(std::mem::offset_of!(#type_name, #field_ident) == #field_offset);
            });
        }

        let rust_struct = quote! {
            #[repr(C)]
            #[derive(Copy, Clone)]
            pub struct #type_name {
                #fields
            }
            #offset_checks
        };

        //eprintln!("reflected struct {}", rust_struct);

        self.types.insert(type_name.clone(), TypeReflection { rust: rust_struct });

        quote! { #type_name }
    }

    /// Emits reflection information for a struct type, recursively.
    ///
    /// Returns the reflected type name.
    fn reflect_type(&mut self, ty_layout: &TypeLayout) -> TokenStream {
        match ty_layout.kind() {
            TypeKind::Scalar => {
                // copy verbatim
                let scalar_ty = format_ident!("{}", ty_layout.name().unwrap());
                quote! { #scalar_ty }
            }
            TypeKind::Vector => {
                // TODO Handle special case of texture and image descriptors.
                //      Blocked on https://github.com/shader-slang/slang/issues/8845
                //let name = ty_layout.name().unwrap_or("unnamed");

                //let uniform_size = ty_layout.size(ParameterCategory::Uniform);
                //let descriptor_slot_size = ty_layout.size(ParameterCategory::DescriptorTableSlot);
                let vector_size = ty_layout.column_count().unwrap();
                let scalar_type = ty_layout.scalar_type().unwrap();
                let scalar_ty_suffix = scalar_type_suffix(scalar_type);
                let vector_type = format_ident!("vec{vector_size}{scalar_ty_suffix}");
                quote! { #vector_type }
            }
            TypeKind::Matrix => {
                let scalar_ty_suffix = scalar_type_suffix(ty_layout.scalar_type().unwrap());
                let column_count = ty_layout.column_count().unwrap();
                let row_count = ty_layout.row_count().unwrap();
                let matrix_type = format_ident!("mat{column_count}x{row_count}{scalar_ty_suffix}");
                quote! { #matrix_type }
            }
            TypeKind::Pointer => {
                // pointers
                let element_ty = self.reflect_type(&ty_layout.element_type_layout().unwrap());
                quote! { ::gpu::Ptr<#element_ty> }
            }
            TypeKind::Array => {
                let element_ty = self.reflect_type(&ty_layout.element_type_layout().unwrap());
                let array_size = ty_layout.total_array_element_count();
                quote! { [#element_ty; #array_size] }
            }
            TypeKind::Struct => self.reflect_struct(ty_layout),
            _ => {
                ceprintln!(
                    "<y>warning</>: unknown type kind {:?} for {}",
                    ty_layout.kind(),
                    ty_layout.name().unwrap_or("unnamed")
                );
                quote! { () }
            }
        }
    }

    fn add_param(
        &mut self,
        name: &str,
        location: ParamLocation,
        byte_size: u32,
        parent: Option<ParamIndex>,
    ) -> ParamIndex {
        let param_index = self.params.len() as ParamIndex;
        self.params.push(Param {
            name: name.to_string(),
            parent: parent.unwrap_or(ParamIndex::MAX),
            location,
            byte_size,
            attributes: vec![],
        });
        param_index
    }

    /// Recursively reflects the type structure of a parameter, emitting new param entries for struct fields and pointer dereferences.
    ///
    /// Concretely:
    /// - for structs: emits a new param reflection for each field, then recursively calls `reflect_variable_type_layout` on each field
    /// - for pointers: emits a new param reflection for the dereferenced type, then recursively calls `reflect_variable_type_layout` on the pointee type
    /// - for other types: does nothing
    fn reflect_variable_type_layout<'b>(
        &mut self,
        param_index: ParamIndex,
        full_name: &str,
        location: ParamLocation,
        ty_layout: &'b TypeLayout,
        type_path: &mut Vec<&'b TypeLayout>,
    ) {
        // avoid infinite recursion
        if type_path.iter().any(|&t| t as *const _ == ty_layout as *const _) {
            //eprintln!("skipping recursive type for {}", full_name);
            return;
        }

        for category in ty_layout.categories() {
            if category != ParameterCategory::Uniform {
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
                //ceprintln!("<r>error</>: unsupported parameter category {:?} in type of {}", category, full_name);
                return;
            }
        }

        let kind = ty_layout.kind();

        match kind {
            TypeKind::Struct => {
                for field in ty_layout.fields() {
                    let field_name = field.variable().unwrap().name().unwrap_or("unnamed");
                    let field_type = field.type_layout().unwrap();

                    //eprintln!("     struct field {}: {:?}", field_name, field_type.kind());

                    let offset = field.offset(ParameterCategory::Uniform) as u32;
                    let field_location = match location {
                        ParamLocation::Binding { resource_index, offset: base_offset } => {
                            ParamLocation::Binding { resource_index, offset: base_offset + offset }
                        }
                        ParamLocation::Indirect { rel, offset: base_offset } => {
                            ParamLocation::Indirect { rel, offset: base_offset + offset }
                        }
                        ParamLocation::PushData { offset: base_offset } => {
                            ParamLocation::PushData { offset: base_offset + offset }
                        }
                    };

                    let field_full_name = format!("{}.{}", full_name, field_name);
                    //eprintln!("field {field_full_name} @ {:?} kind={:?}", field_location, field_type.kind());

                    let index = self.add_param(&field_full_name, field_location, 0, Some(param_index));
                    type_path.push(ty_layout);
                    self.reflect_variable_type_layout(index, &field_full_name, field_location, &field_type, type_path);
                    type_path.pop();
                }
            }
            TypeKind::Pointer => {
                let deref_location = ParamLocation::Indirect { rel: param_index, offset: 0 };
                let deref_name = format!("{}.$", full_name);
                let index = self.add_param(&deref_name, deref_location, 0, Some(param_index));
                //eprintln!("deref {}.$ @ {:?} kind={:?}", full_name, deref_location, ty_layout.kind());
                type_path.push(ty_layout);
                self.reflect_variable_type_layout(
                    index,
                    &deref_name,
                    deref_location,
                    ty_layout.element_type_layout().unwrap(),
                    type_path,
                );
                type_path.pop();
            }
            TypeKind::Matrix | TypeKind::Scalar | TypeKind::Vector => {
                // nothing to do
            }
            other => {
                ceprintln!("<y>warning</>: unknown type kind {other:?} for {full_name}");
                return;
            }
        }
    }

    fn reflect_param(&mut self, param: &VariableLayout) {
        // NOTE: this has turned into an unwrap fest since the last slang-rs update;
        //       at some point we should fork and make our own bindings to slang
        //       (at least panic on null pointers instead of turning *everything* into Option)

        //let set = param.binding_space();
        //let binding = param.binding_index();
        let category = param.category().unwrap();
        let name = param.variable().unwrap().name().unwrap_or("unnamed");

        match category {
            ParameterCategory::Uniform => {
                let type_layout = param.type_layout().unwrap();
                let offset = param.offset(slang::ParameterCategory::Uniform);
                let location = ParamLocation::PushData { offset: offset as u32 };
                //eprintln!("push data {}: {:?} offset {}", name, category, offset);
                let size = type_layout.size(ParameterCategory::Uniform) as u32;
                let index = self.add_param(name, location, size, None);
                self.reflect_variable_type_layout(index, name, location, type_layout, &mut vec![]);
                self.reflect_type(type_layout);
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
                let location = ParamLocation::PushData { offset: 0 };
                let size = cbuffer_content_layout.size(ParameterCategory::Uniform) as u32;
                let index = self.add_param(name, location, size, None);
                self.reflect_variable_type_layout(index, name, location, cbuffer_content_layout, &mut vec![]);
                self.reflect_type(cbuffer_content_layout);
            }
            ParameterCategory::None => {
                //eprintln!("resource {}: {:?}", name, category);
                return;
            }
            _ => {
                //ceprintln!("<r>error</>: unsupported parameter category {category:?} for {name}");
            }
        };
    }

    pub(crate) fn reflect_entry_point(&mut self, entry_point: &slang::reflection::EntryPoint) {
        //eprintln!("entry point: {}", entry_point.name().unwrap_or("unnamed"));
        for param in entry_point.parameters() {
            self.reflect_param(param);
        }
    }

    pub(crate) fn reflect_shader(&mut self, shader: &slang::reflection::Shader) {
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

impl Module {
    pub fn generate_reflection(&self) -> TokenStream {
        let mut collector = CollectedReflectionData::new();
        let program_layout = self.program.layout(0).unwrap();
        collector.reflect_shader(&program_layout);
        let entry_point_count = program_layout.entry_point_count();
        for i in 0..entry_point_count {
            let entry_point = program_layout.entry_point_by_index(i).unwrap();
            collector.reflect_entry_point(&entry_point);
        }
        let types = collector.types.values().map(|t| &t.rust);
        quote! {
            #(#types)*
        }
    }
}
