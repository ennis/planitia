//! Extract extended reflection information from a slang shader module.

use slang::reflection::VariableLayout;
use shader_archive::archive::ArchiveWriter;
use crate::BuildOptions;
use shader_archive::{reflection, ShaderArchiveRoot};

struct StructTypeInfo {
    name: String,
    fields: Vec<StructFieldInfo>,
}

struct StructFieldInfo {
    name: String,
    ty: TypeDesc,
    offset: u32,
   // attributes: Vec<Attribute>,
}

enum TypeDesc {
    Scalar(reflection::ScalarType),
    Vector(reflection::VectorType),
    Matrix(reflection::MatrixType),
    Sampled(reflection::SampledType),
    Struct(StructTypeInfo),
}

pub(crate) struct CollectedReflectionData<'a> {
    archive: &'a mut ArchiveWriter<ShaderArchiveRoot>,
    options: &'a BuildOptions,
}

impl<'a> CollectedReflectionData<'a> {
    pub(crate) fn new(archive: &'a mut ArchiveWriter<ShaderArchiveRoot>, options: &'a BuildOptions) -> Self {
        CollectedReflectionData {
            archive,
            options,
        }
    }

    fn reflect_param(&mut self, param: &VariableLayout) {

        // in shader:
        // - all global uniforms put in a single cbuffer
        // - textures are declared the usual way (no need for handles)
        //
        // with VK_EXT_descriptor_heap:
        // - indexed bindings mapped to descriptor heap indices in push data for textures and samplers
        // - indexed bindings mapped to buffer pointers for uniforms and shader storage
    }

    pub(crate) fn reflect_entry_point(&mut self, entry_point: &slang::reflection::EntryPoint) {
        for p in entry_point.parameters() {
            self.reflect_param(p);
        }
    }

    pub(crate) fn reflect_shader(&mut self, shader: &slang::reflection::Shader) {
        if let Some(entry_point) = shader.entry_point_by_index(0) {
            self.reflect_entry_point(&entry_point);
        }
    }
}