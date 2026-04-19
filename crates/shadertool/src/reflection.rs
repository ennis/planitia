//! Extract extended reflection information from a slang shader module.

use crate::BuildOptions;
use color_print::{ceprintln, cprintln};
use sharc::archive::{ArchiveWriter, Offset};
use sharc::reflection::ParamLocation;
use sharc::{ShaderArchiveRoot, reflection};
use slang::reflection::{TypeLayout, VariableLayout};
use slang::{ParameterCategory, TypeKind};

pub(crate) struct CollectedReflectionData<'a> {
    archive: &'a mut ArchiveWriter<ShaderArchiveRoot>,
    options: &'a BuildOptions,
    pub(crate) params: Vec<reflection::Param>,
}

type ParamIndex = u32;

impl<'a> CollectedReflectionData<'a> {
    pub(crate) fn new(archive: &'a mut ArchiveWriter<ShaderArchiveRoot>, options: &'a BuildOptions) -> Self {
        CollectedReflectionData {
            archive,
            options,
            params: vec![],
        }
    }

    fn add_param(&mut self, name: &str, location: ParamLocation, byte_size: u32) -> ParamIndex {
        let param_index = self.params.len() as ParamIndex;
        self.params.push(reflection::Param {
            name: self.archive.write_str(name),
            location,
            byte_size,
            attributes: Offset::INVALID,
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
            eprintln!("skipping recursive type for {}", full_name);
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
                ceprintln!(
                    "<r>error</>: unsupported parameter category {:?} in type of {}",
                    category,
                    full_name
                );
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
                        ParamLocation::Binding {
                            resource_index,
                            offset: base_offset,
                        } => ParamLocation::Binding {
                            resource_index,
                            offset: base_offset + offset,
                        },
                        ParamLocation::Indirect {
                            rel,
                            offset: base_offset,
                        } => ParamLocation::Indirect {
                            rel,
                            offset: base_offset + offset,
                        },
                        ParamLocation::PushData { offset: base_offset } => ParamLocation::PushData {
                            offset: base_offset + offset,
                        },
                    };

                    let field_full_name = format!("{}.{}", full_name, field_name);
                    eprintln!(
                        "field {field_full_name} @ {:?} kind={:?}",
                        field_location,
                        field_type.kind()
                    );

                    let index = self.add_param(&field_full_name, field_location, 0);
                    type_path.push(ty_layout);
                    self.reflect_variable_type_layout(index, &field_full_name, field_location, &field_type, type_path);
                    type_path.pop();
                }
            }
            TypeKind::Pointer => {
                let deref_location = ParamLocation::Indirect {
                    rel: param_index,
                    offset: 0,
                };
                let deref_name = format!("{}.$", full_name);
                let index = self.add_param(&deref_name, deref_location, 0);
                eprintln!(
                    "deref {}.$ @ {:?} kind={:?}",
                    full_name,
                    deref_location,
                    ty_layout.kind()
                );
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

        let set = param.binding_space();
        let binding = param.binding_index();
        let category = param.category().unwrap();
        let name = param.variable().unwrap().name().unwrap_or("unnamed");

        match category {
            ParameterCategory::Uniform => {
                let offset = param.offset(slang::ParameterCategory::Uniform);
                let location = ParamLocation::PushData { offset: offset as u32 };
                eprintln!(
                    "push data {}: {:?} set {}, binding {}, offset {}",
                    name, category, set, binding, offset
                );
                let size = param.type_layout().unwrap().size(ParameterCategory::Uniform) as u32;
                let index = self.add_param(name, location, size);
                self.reflect_variable_type_layout(index, name, location, param.type_layout().unwrap(), &mut vec![]);
            }
            ParameterCategory::PushConstantBuffer => {
                // sanity check
                assert!(param.type_layout().unwrap().kind() == TypeKind::ConstantBuffer);
                // The type of push constant buffers is `ConstantBuffer<T>`, which conceptually
                // represents a constant buffer slot. This is meaningless for push constants,
                // and we don't want this type to end up in the reflected type hierarchy,
                // so pass through it.
                let cbuffer_content_layout = param.type_layout().unwrap().element_type_layout().unwrap();
                let location = ParamLocation::PushData { offset: 0 };
                let size = cbuffer_content_layout.size(ParameterCategory::Uniform) as u32;
                let index = self.add_param(name, location, size);
                self.reflect_variable_type_layout(index, name, location, cbuffer_content_layout, &mut vec![]);
            }
            ParameterCategory::None => {
                //eprintln!(
                //    "resource {}: {:?} set {}, binding {}",
                //    name, category, set, binding
                //);
                return;
            }
            _ => panic!("unsupported parameter category: {:?}", category),
        };
    }

    pub(crate) fn reflect_shader(&mut self, shader: &slang::reflection::Shader) {
        let global_params_layout = shader.global_params_var_layout();
        let ty_layout = shader.global_params_type_layout().unwrap();
        if ty_layout.kind() != TypeKind::Struct {
            panic!("expected global params to be a struct");
        }
        for field in ty_layout.fields() {
            self.reflect_param(field);
        }
    }
}
