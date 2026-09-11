#![allow(non_upper_case_globals)]
use gpu_types::reflection as refl;
use gpu_types::reflection::{AccessKind, ScalarType};
use gpu_types::vulkan::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote, TokenStreamExt};
use std::slice;

macro_rules! return_error {
    ($span:expr, $msg:literal) => {
        return Err(syn::Error::new($span, format!($msg)));
    };
    ($span:expr, $fmt:literal, $($arg:tt)*) => {
        return Err(syn::Error::new($span, format!($fmt, $($arg)*)));
    };
}

fn quote_scalar_type(scalar: refl::ScalarType) -> TokenStream {
    let scalar_str = format!("{scalar:?}");
    let scalar_ident = format_ident!("{scalar_str}");
    quote!(refl::ScalarType::#scalar_ident)
}

fn generate_scalar_type(scalar: refl::ScalarType) -> TokenStream {
    match scalar {
        refl::ScalarType::Bool => quote!(bool),
        refl::ScalarType::I8 => quote!(i8),
        refl::ScalarType::I16 => quote!(i16),
        refl::ScalarType::I32 => quote!(i32),
        refl::ScalarType::I64 => quote!(i64),
        refl::ScalarType::U8 => quote!(u8),
        refl::ScalarType::U16 => quote!(u16),
        refl::ScalarType::U32 => quote!(u32),
        refl::ScalarType::U64 => quote!(u64),
        refl::ScalarType::F32 => quote!(f32),
    }
}

fn scalar_type_suffix(scalar_type: ScalarType) -> &'static str {
    match scalar_type {
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

fn generate_vector_type(scalar_type: ScalarType, len: usize) -> Ident {
    let suffix = scalar_type_suffix(scalar_type);
    format_ident!("vec{len}{suffix}")
}

fn generate_matrix_type(scalar_type: ScalarType, rows: usize, cols: usize) -> Ident {
    let suffix = scalar_type_suffix(scalar_type);
    format_ident!("mat{cols}x{rows}{suffix}")
}

fn generate_type(ty: &refl::TypeDesc) -> TokenStream {
    match ty {
        refl::TypeDesc::Scalar(scalar_type) => {
            let scalar_type_tokens = generate_scalar_type(*scalar_type);
            quote!(#scalar_type_tokens)
        }
        refl::TypeDesc::Vector(scalar, len) => {
            let vector_type_ident = generate_vector_type(*scalar, *len as usize);
            quote!(#vector_type_ident)
        }
        refl::TypeDesc::Matrix(scalar, rows, cols) => {
            let matrix_type_ident = generate_matrix_type(*scalar, *rows as usize, *cols as usize);
            quote!(#matrix_type_ident)
        }
        refl::TypeDesc::Struct(struct_ty) => {
            let struct_name = format_ident!("{}", struct_ty.name);
            quote!(#struct_name)
        }
        refl::TypeDesc::Array(elem, len) => {
            let elem = generate_type(elem);
            let len = *len as usize;
            quote!([#elem; #len])
        }
        refl::TypeDesc::RuntimeArray(_) => {
            todo!("Runtime arrays are not yet supported")
        }
        refl::TypeDesc::ImageHandle(_) => {
            todo!("Image handles are not yet supported")
        }
        refl::TypeDesc::Pointer(pointee) => {
            let pointee = generate_type(pointee);
            quote!(gpu::Ptr<#pointee>)
        }
        _ => {
            todo!("unsupported type in reflection information")
        }
    }
}

fn quote_type(ty: &refl::TypeDesc) -> TokenStream {
    match ty {
        refl::TypeDesc::Scalar(scalar_type) => {
            let scalar_type = quote_scalar_type(*scalar_type);
            quote! { refl::TypeDesc::Scalar(#scalar_type) }
        }
        refl::TypeDesc::Vector(scalar, len) => {
            let scalar = quote_scalar_type(*scalar);
            let len = *len;
            quote! {
                refl::TypeDesc::Vector(#scalar, #len)
            }
        }
        refl::TypeDesc::Matrix(scalar, rows, cols) => {
            let scalar = quote_scalar_type(*scalar);
            let rows = *rows;
            let cols = *cols;
            quote! {
                refl::TypeDesc::Matrix(#scalar,#rows, #cols)
            }
        }
        refl::TypeDesc::Struct(struct_ty) => {
            let struct_name = format_ident!("{}", struct_ty.name);
            quote! {
                refl::TypeDesc::Struct(&#struct_name::TYPE_DESC)
            }
        }
        refl::TypeDesc::Array(elem, len) => {
            let elem_quoted = quote_type(elem);
            let len = *len;
            quote! {
                refl::TypeDesc::Array(&const { #elem_quoted }, #len)
            }
        }
        refl::TypeDesc::RuntimeArray(elem) => {
            let elem_quoted = quote_type(elem);
            quote! {
                refl::TypeDesc::RuntimeArray(&const { #elem_quoted })
            }
        }
        refl::TypeDesc::ImageHandle(ih) => {
            let scalar = quote_scalar_type(ih.sampled.scalar);
            let components = ih.sampled.components;
            let read_write = ih.read_write;
            quote! {
                refl::TypeDesc::ImageHandle(refl::ImageHandleType {
                    sampled: refl::SampledType {
                        scalar: #scalar,
                        components: #components,
                    },
                    read_write: #read_write,
                })
            }
        }
        refl::TypeDesc::Pointer(pointee) => {
            let pointee_quoted = quote_type(pointee);
            quote! {
                refl::TypeDesc::Pointer(&const { #pointee_quoted })
            }
        }
        _ => {
            todo!("unsupported type in reflection information")
        }
    }
}

fn generate_struct_decl(s: &refl::StructType) -> TokenStream {
    let struct_name = format_ident!("{}", s.name);
    let struct_name_str = s.name;
    let fields = s.fields.iter().map(|f| {
        let field_name = format_ident!("{}", f.name);
        let field_type = generate_type(&f.ty);
        quote!(pub #field_name: #field_type)
    });
    let fields_meta = s.fields.iter().map(|f| {
        let field_name = f.name;
        let field_type = quote_type(&f.ty);
        let offset = f.offset;
        quote! {
            refl::StructField {
                name: #field_name,
                ty: #field_type,
                offset: #offset,
            }
        }
    });

    quote! {
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct #struct_name {
            #(#fields),*
        }
        impl #struct_name {
            pub const TYPE_DESC: refl::StructType<'static> = refl::StructType {
                name: #struct_name_str,
                fields: &[
                    #(#fields_meta),*
                ],
            };
        }
    }
}

fn mangle_access_chain_name(entry_point: Option<&str>, name: &str) -> Ident {
    let components = name.split('.').collect::<Vec<_>>();
    let mut mangled = if let Some(entry_point) = entry_point {
        format!("E{}{}", entry_point.len(), entry_point)
    } else {
        String::new()
    };
    for c in components {
        if c.starts_with('$') {
            mangled.push('D');
        } else if c.starts_with('[') {
            mangled.push('I');
        } else {
            mangled.push_str(&format!("F{}{}", c.len(), c));
        }
    }

    format_ident!("{mangled}")
}

fn generate_access_chain(entry_point: Option<&str>, s: &refl::AccessChain) -> TokenStream {
    let name = s.name;
    let ident = mangle_access_chain_name(entry_point, name);
    let ty = quote_type(&s.ty);

    let kind = match s.kind {
        AccessKind::Binding { name, resource_index, offset } => {
            quote! {
                refl::AccessKind::Binding {
                    name: #name,
                    resource_index: #resource_index,
                    offset: #offset,
                }
            }
        }
        AccessKind::PushData { offset } => {
            quote! {
                refl::AccessKind::PushData {
                    offset: #offset,
                }
            }
        }
        AccessKind::Field { offset } => {
            quote! {
                refl::AccessKind::Field {
                    offset: #offset,
                }
            }
        }
        AccessKind::ArrayIndex { count, stride } => {
            quote! {
                refl::AccessKind::ArrayIndex { count: #count, stride: #stride }
            }
        }
        AccessKind::RuntimeArrayIndex { stride } => {
            quote! {
                refl::AccessKind::RuntimeArrayIndex { stride: #stride }
            }
        }
        AccessKind::Load => {
            quote! {
                refl::AccessKind::Load
            }
        }
    };
    let parent = match s.parent {
        Some(s) => {
            let parent_name = mangle_access_chain_name(entry_point, s.name);
            quote!(Some(&#parent_name))
        }
        None => quote!(None),
    };

    quote! {
        static #ident: refl::AccessChain = refl::AccessChain {
            parent: #parent,
            kind: #kind,
            name: #name,
            ty: #ty
        };
    }
}

pub(crate) fn shader_module_impl(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> syn::Result<TokenStream> {
    let shader_path_lit: syn::LitStr = syn::parse(attr)?;
    let mut shader_path = shader_path_lit.value();

    // HACK: we allow (and ignore) a `#<number>` suffix on the shader path to work around
    //       a bug in RustRover where the macro is not re-evaluated when the shader file changes.
    //       Changing the suffix forces a re-evaluation. The suffix is stripped and ignored here.
    if let Some(pos) = shader_path.rfind('#') {
        shader_path.truncate(pos);
    }

    // Parse the module declaration
    let item_mod: syn::ItemMod = syn::parse(item)?;

    let mod_name = &item_mod.ident;
    let vis = &item_mod.vis;
    let attrs = &item_mod.attrs;

    // Compile the shader module at build time using shadertool
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let shader_file = std::path::Path::new(&manifest_dir).join(&shader_path);

    if !shader_file.exists() {
        return Err(syn::Error::new(
            shader_path_lit.span(),
            format!("shader file not found: {}", shader_file.display()),
        ));
    }

    // Compile the shader module
    let options = shadertool::BuildOptions {
        emit_cargo_deps: false,
        emit_debug_information: true, // TODO configure?
        emit_spirv_binaries: true,
        emit_reflection: false,
        include_paths: vec![],
        output_directory: None,
    };

    let arena = shadertool::Arena::new();
    let module = shadertool::compile(&shader_file, &arena, &options)
        .map_err(|err| syn::Error::new(shader_path_lit.span(), format!("failed to compile shader:\n{err}")))?;

    // Convert SPIR-V bytecode to u32 array
    let bytecode_str = {
        let spirv_bytes = unsafe { slice::from_raw_parts(module.spirv.as_ptr() as *const u8, module.spirv.len() * 4) };
        syn::LitByteStr::new(&spirv_bytes, shader_path_lit.span())
    };

    let bytecode = quote! {
        static __BYTECODE: &[u32] = ::gpu::bytes_as_u32!(#bytecode_str);
    };

    // Write SPIR-V binary
    //let spirv_output = shader_file
    //    .parent()
    //    .unwrap()
    //    .join(format!("spirv/{}.spv", shader_file.file_stem().unwrap().to_string_lossy()))
    //    .to_string_lossy()
    //    .to_string();

    // Include all dependencies.
    let dependencies = {
        let paths = module.dependencies.iter().map(|dep| dep.path.to_string_lossy());
        quote! {
            static __DEPENDENCIES: &[&str] = &[
                #(include_str!(#paths)),*
            ];
        }
    };

    // Write ShaderEntryPoints
    let mut entry_points = vec![];
    for (_i, ep) in module.entry_points.iter().enumerate() {
        let ep_name = &ep.name;
        let ep_name_ident = format_ident!("{}", ep_name);
        let push_constants_size = ep.push_constants_size;
        let workgroup_size = ep.workgroup_size;

        let stage = match ep.stage {
            VK_SHADER_STAGE_VERTEX_BIT => format_ident!("Vertex"),
            VK_SHADER_STAGE_FRAGMENT_BIT => format_ident!("Fragment"),
            VK_SHADER_STAGE_MESH_BIT_EXT => format_ident!("Mesh"),
            VK_SHADER_STAGE_TASK_BIT_EXT => format_ident!("Task"),
            VK_SHADER_STAGE_COMPUTE_BIT => format_ident!("Compute"),
            _ => continue,
        };

        // generate entry point parameter reflection
        let mut param_access_chains = TokenStream::new();
        param_access_chains.append_all(ep.refl_params.iter().map(|&s| generate_access_chain(Some(ep_name), s)));

        let param_access_chain_names = {
            let mut names = vec![];
            for gp in module.refl_global_params.iter() {
                names.push(mangle_access_chain_name(None, gp.name));
            }
            for epp in ep.refl_params.iter() {
                names.push(mangle_access_chain_name(Some(ep_name), epp.name));
            }
            names
        };

        //let param_reflection = generate_entry_point_param_reflection(&module, &reflection, i);

        entry_points.push(quote! {
            #param_access_chains

            #[allow(non_upper_case_globals)]
            pub static #ep_name_ident: ::gpu::ShaderEntryPoint<'static> = ::gpu::ShaderEntryPoint {
                stage: ::gpu::ShaderStage::#stage,
                code: super::__BYTECODE,
                entry_point: #ep_name,
                push_constants_size: #push_constants_size,
                source_path: Some(#shader_path),
                workgroup_size: [#(#workgroup_size),*],
                refl_params: &[
                    #(&#param_access_chain_names),*
                ]
            };
        });
    }

    let mut pipelines = vec![];

    // Write PipelineCreateInfos
    for pipeline in module.pipelines.iter() {
        let pipeline_name = &pipeline.name;
        let mut vertex = None;
        let mut fragment = None;
        let mut mesh = None;
        let mut task = None;
        let mut compute = None;
        for &ep_index in pipeline.stages.iter() {
            let entry_point = &module.entry_points[ep_index];
            eprintln!("entry_point {} stage {:?}", entry_point.name, entry_point.stage);
            let ep_name_ident = format_ident!("{}", entry_point.name);
            match entry_point.stage {
                VK_SHADER_STAGE_VERTEX_BIT => vertex = Some(ep_name_ident),
                VK_SHADER_STAGE_FRAGMENT_BIT => fragment = Some(ep_name_ident),
                VK_SHADER_STAGE_MESH_BIT_EXT => mesh = Some(ep_name_ident),
                VK_SHADER_STAGE_TASK_BIT_EXT => task = Some(ep_name_ident),
                VK_SHADER_STAGE_COMPUTE_BIT => compute = Some(ep_name_ident),
                _ => {}
            }
        }

        let has_vertex = vertex.is_some();
        let has_fragment = fragment.is_some();
        let has_mesh = mesh.is_some();
        let has_task = task.is_some();
        let has_compute = compute.is_some();

        #[derive(Copy, Clone)]
        enum PipelineType {
            Primitive,
            //Mesh,
            Compute,
        }
        let pipeline_type;
        let create_fn;

        let push_constants_size = pipeline.push_constants_size;

        let mut color_targets = vec![];
        for color_target in pipeline.graphics_state.color_targets.iter() {
            let format = vk_format_tokens(color_target.format);
            let blend_equation = match color_target.blend_equation {
                Some(equation) => {
                    let src_color = vk_blend_factor_tokens(equation.src_color_blend_factor);
                    let dst_color = vk_blend_factor_tokens(equation.dst_color_blend_factor);
                    let color_op = vk_blend_op_tokens(equation.color_blend_op);
                    let src_alpha = vk_blend_factor_tokens(equation.src_alpha_blend_factor);
                    let dst_alpha = vk_blend_factor_tokens(equation.dst_alpha_blend_factor);
                    let alpha_op = vk_blend_op_tokens(equation.alpha_blend_op);
                    quote!(Some(gpu::ColorBlendEquation {
                        src_color_blend_factor: #src_color,
                        dst_color_blend_factor: #dst_color,
                        color_blend_op: #color_op,
                        src_alpha_blend_factor: #src_alpha,
                        dst_alpha_blend_factor: #dst_alpha,
                        alpha_blend_op: #alpha_op,
                    }))
                }
                None => quote!(None),
            };

            color_targets.push(quote! {
                gpu::ColorTargetState {
                    format: #format,
                    blend_equation: #blend_equation,
                    ..
                }
            })
        }

        let depth_stencil = if let Some(ref depth_stencil) = pipeline.graphics_state.depth_stencil {
            let format = vk_format_tokens(depth_stencil.format);
            let depth_write_enable = depth_stencil.depth_write_enable;
            let depth_compare_op = vk_compare_op_tokens(depth_stencil.depth_compare_op);
            quote!(Some(gpu::DepthStencilState {
                format: #format,
                depth_write_enable: #depth_write_enable,
                depth_compare_op: #depth_compare_op,
                ..
            }))
        } else {
            quote!(None)
        };

        let polygon_mode = vk_polygon_mode_tokens(pipeline.graphics_state.rasterizer.polygon_mode);
        let cull_mode = vk_cull_mode_tokens(pipeline.graphics_state.rasterizer.cull_mode);
        let front_face = vk_front_face_tokens(pipeline.graphics_state.rasterizer.front_face);
        let rasterization = quote!(gpu::RasterizationState {
            polygon_mode: #polygon_mode,
            cull_mode: #cull_mode,
            front_face: #front_face,
            ..
        });

        match (has_vertex, has_fragment, has_mesh, has_task, has_compute) {
            (true, true, false, false, false) => {
                let vertex = vertex.unwrap();
                let fragment = fragment.unwrap();

                // Primitive shading
                pipeline_type = PipelineType::Primitive;
                create_fn = quote! {
                    fn create_pipeline() -> Result<gpu::GraphicsPipeline, gpu::Error>  {
                        static CREATE_INFO : gpu::GraphicsPipelineCreateInfo = gpu::GraphicsPipelineCreateInfo {
                            push_constants_size: #push_constants_size,
                            pre_rasterization_shaders: gpu::PreRasterizationShaders::PrimitiveShading {
                                vertex: self::entry_points::#vertex,
                            },
                            rasterization: #rasterization,
                            depth_stencil: #depth_stencil,
                            fragment: gpu::FragmentState {
                                shader: self::entry_points::#fragment,
                                color_targets: &[#(#color_targets)*],
                                ..
                            },
                            ..
                        };
                        let pipeline = gpu::GraphicsPipeline::new(CREATE_INFO)?;
                        unsafe {
                            gpu::set_debug_name(&pipeline, #pipeline_name);
                        }
                        Ok(pipeline)
                    }
                };
            }
            (false, true, true, true, false) | (false, true, true, false, false) => {
                // Mesh shading
                todo!("mesh shading pipelines")
            }
            (false, false, false, false, true) => {
                pipeline_type = PipelineType::Compute;
                create_fn = quote! {
                    fn create_pipeline() -> Result<gpu::ComputePipeline, gpu::Error>  {
                        static CREATE_INFO : gpu::ComputePipelineCreateInfo = gpu::ComputePipelineCreateInfo {
                            push_constants_size: #push_constants_size,
                            shader: self::entry_points::#compute,
                            ..
                        };
                        let pipeline = gpu::ComputePipeline::new(CREATE_INFO)?;
                        unsafe {
                            gpu::set_debug_name(&pipeline, #pipeline_name);
                        }
                        Ok(pipeline)
                    }
                };
            }
            _ => {
                let mut pipeline_stages_str = String::new();
                if has_vertex {
                    pipeline_stages_str.push_str("vertex+");
                }
                if has_fragment {
                    pipeline_stages_str.push_str("fragment+");
                }
                if has_compute {
                    pipeline_stages_str.push_str("compute+");
                }
                if has_mesh {
                    pipeline_stages_str.push_str("mesh+");
                }
                if has_task {
                    pipeline_stages_str.push_str("task+");
                }
                if !pipeline_stages_str.is_empty() {
                    // remove trailing "+"
                    pipeline_stages_str.truncate(pipeline_stages_str.len() - 1);
                }
                return_error!(shader_path_lit.span(), "inconsistent pipeline stages: {pipeline_stages_str} (expected vertex+fragment, mesh+fragment, task+mesh+fragment, or compute)");
            }
        }

        let gpu_pipeline_type = match pipeline_type {
            PipelineType::Primitive /*| PipelineType::Mesh*/ => {
                quote!(gpu::GraphicsPipeline)
            }
            PipelineType::Compute => {
                quote!(gpu::ComputePipeline)
            }
        };

        let pipeline_name_ident = format_ident!("{}", pipeline.name);

        pipelines.push(quote! {
            #[allow(non_upper_case_globals)]
            pub static #pipeline_name_ident: std::sync::LazyLock<#gpu_pipeline_type> = std::sync::LazyLock::new(|| {
                #create_fn
                create_pipeline().expect("failed to create pipeline")
            });
        });
    }

    // reflect types
    let reflection = {
        let mut tokens = TokenStream::new();
        tokens.append_all(module.refl_struct_types.values().map(|&s| generate_struct_decl(s)));
        tokens.append_all(module.refl_global_params.iter().map(|&s| generate_access_chain(None, s)));
        tokens
    };

    let output = quote! {
        #(#attrs)*
        #vis mod #mod_name {
            use ::gpu::shader_types::*;
            use ::gpu::reflection as refl;
            #dependencies
            #bytecode
            #reflection
            pub mod entry_points {
                use super::*;
                #(#entry_points)*
            }
            #(#pipelines)*
        }
    };

    // dump expansion to .txt file
    // if let Ok(out_dir) = std::env::var("CARGO_MANIFEST_DIR") {
    //     let dump_dir = std::path::Path::new(&out_dir).join("target").join("macro-expansions");
    //     let _ = std::fs::create_dir_all(&dump_dir);
    //     let dump_path = dump_dir.join(format!("{}.txt", mod_name));
    //     let _ = std::fs::write(&dump_path, output.to_string());
    //     eprintln!("macro expansion written to {}", dump_path.display());
    // }

    Ok(output)
}

macro_rules! enum_to_ident {
    ($value:expr; $($variants:ident),* $(,)?) => {
        match $value {
            $($variants => Ident::new(stringify!($variants), Span::call_site()),)*
            _ => panic!("invalid enum value: {}", $value),
        }
    }
}

fn vk_polygon_mode_tokens(polygon_mode: VkPolygonMode) -> TokenStream {
    let v = enum_to_ident! {polygon_mode; VK_POLYGON_MODE_FILL, VK_POLYGON_MODE_LINE, VK_POLYGON_MODE_POINT};
    quote!(gpu::vulkan::#v)
}

fn vk_cull_mode_tokens(cull_mode: VkCullModeFlags) -> TokenStream {
    let mut flags = vec![];
    if cull_mode & VK_CULL_MODE_FRONT_BIT != 0 {
        flags.push(format_ident!("VK_CULL_MODE_FRONT_BIT"));
    }
    if cull_mode & VK_CULL_MODE_BACK_BIT != 0 {
        flags.push(format_ident!("VK_CULL_MODE_BACK_BIT"));
    }
    quote!(gpu::vulkan::VK_CULL_MODE_NONE #( | gpu::vulkan::#flags)*)
}

fn vk_front_face_tokens(front_face: VkFrontFace) -> TokenStream {
    let v = enum_to_ident! {
        front_face; VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FRONT_FACE_CLOCKWISE
    };
    quote!(gpu::vulkan::#v)
}

fn vk_format_tokens(format: VkFormat) -> TokenStream {
    let v = enum_to_ident! {format;
        VK_FORMAT_UNDEFINED,
        VK_FORMAT_R4G4_UNORM_PACK8,
        VK_FORMAT_R4G4B4A4_UNORM_PACK16,
        VK_FORMAT_B4G4R4A4_UNORM_PACK16,
        VK_FORMAT_R5G6B5_UNORM_PACK16,
        VK_FORMAT_B5G6R5_UNORM_PACK16,
        VK_FORMAT_R5G5B5A1_UNORM_PACK16,
        VK_FORMAT_B5G5R5A1_UNORM_PACK16,
        VK_FORMAT_A1R5G5B5_UNORM_PACK16,
        VK_FORMAT_R8_UNORM,
        VK_FORMAT_R8_SNORM,
        VK_FORMAT_R8_USCALED,
        VK_FORMAT_R8_SSCALED,
        VK_FORMAT_R8_UINT,
        VK_FORMAT_R8_SINT,
        VK_FORMAT_R8_SRGB,
        VK_FORMAT_R8G8_UNORM,
        VK_FORMAT_R8G8_SNORM,
        VK_FORMAT_R8G8_USCALED,
        VK_FORMAT_R8G8_SSCALED,
        VK_FORMAT_R8G8_UINT,
        VK_FORMAT_R8G8_SINT,
        VK_FORMAT_R8G8_SRGB,
        VK_FORMAT_R8G8B8_UNORM,
        VK_FORMAT_R8G8B8_SNORM,
        VK_FORMAT_R8G8B8_USCALED,
        VK_FORMAT_R8G8B8_SSCALED,
        VK_FORMAT_R8G8B8_UINT,
        VK_FORMAT_R8G8B8_SINT,
        VK_FORMAT_R8G8B8_SRGB,
        VK_FORMAT_B8G8R8_UNORM,
        VK_FORMAT_B8G8R8_SNORM,
        VK_FORMAT_B8G8R8_USCALED,
        VK_FORMAT_B8G8R8_SSCALED,
        VK_FORMAT_B8G8R8_UINT,
        VK_FORMAT_B8G8R8_SINT,
        VK_FORMAT_B8G8R8_SRGB,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R8G8B8A8_SNORM,
        VK_FORMAT_R8G8B8A8_USCALED,
        VK_FORMAT_R8G8B8A8_SSCALED,
        VK_FORMAT_R8G8B8A8_UINT,
        VK_FORMAT_R8G8B8A8_SINT,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_B8G8R8A8_SNORM,
        VK_FORMAT_B8G8R8A8_USCALED,
        VK_FORMAT_B8G8R8A8_SSCALED,
        VK_FORMAT_B8G8R8A8_UINT,
        VK_FORMAT_B8G8R8A8_SINT,
        VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_A8B8G8R8_UNORM_PACK32,
        VK_FORMAT_A8B8G8R8_SNORM_PACK32,
        VK_FORMAT_A8B8G8R8_USCALED_PACK32,
        VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
        VK_FORMAT_A8B8G8R8_UINT_PACK32,
        VK_FORMAT_A8B8G8R8_SINT_PACK32,
        VK_FORMAT_A8B8G8R8_SRGB_PACK32,
        VK_FORMAT_A2R10G10B10_UNORM_PACK32,
        VK_FORMAT_A2R10G10B10_SNORM_PACK32,
        VK_FORMAT_A2R10G10B10_USCALED_PACK32,
        VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
        VK_FORMAT_A2R10G10B10_UINT_PACK32,
        VK_FORMAT_A2R10G10B10_SINT_PACK32,
        VK_FORMAT_A2B10G10R10_UNORM_PACK32,
        VK_FORMAT_A2B10G10R10_SNORM_PACK32,
        VK_FORMAT_A2B10G10R10_USCALED_PACK32,
        VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
        VK_FORMAT_A2B10G10R10_UINT_PACK32,
        VK_FORMAT_A2B10G10R10_SINT_PACK32,
        VK_FORMAT_R16_UNORM,
        VK_FORMAT_R16_SNORM,
        VK_FORMAT_R16_USCALED,
        VK_FORMAT_R16_SSCALED,
        VK_FORMAT_R16_UINT,
        VK_FORMAT_R16_SINT,
        VK_FORMAT_R16_SFLOAT,
        VK_FORMAT_R16G16_UNORM,
        VK_FORMAT_R16G16_SNORM,
        VK_FORMAT_R16G16_USCALED,
        VK_FORMAT_R16G16_SSCALED,
        VK_FORMAT_R16G16_UINT,
        VK_FORMAT_R16G16_SINT,
        VK_FORMAT_R16G16_SFLOAT,
        VK_FORMAT_R16G16B16_UNORM,
        VK_FORMAT_R16G16B16_SNORM,
        VK_FORMAT_R16G16B16_USCALED,
        VK_FORMAT_R16G16B16_SSCALED,
        VK_FORMAT_R16G16B16_UINT,
        VK_FORMAT_R16G16B16_SINT,
        VK_FORMAT_R16G16B16_SFLOAT,
        VK_FORMAT_R16G16B16A16_UNORM,
        VK_FORMAT_R16G16B16A16_SNORM,
        VK_FORMAT_R16G16B16A16_USCALED,
        VK_FORMAT_R16G16B16A16_SSCALED,
        VK_FORMAT_R16G16B16A16_UINT,
        VK_FORMAT_R16G16B16A16_SINT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32_UINT,
        VK_FORMAT_R32_SINT,
        VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R32G32_UINT,
        VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R32G32_SFLOAT,
        VK_FORMAT_R32G32B32_UINT,
        VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R32G32B32_SFLOAT,
        VK_FORMAT_R32G32B32A32_UINT,
        VK_FORMAT_R32G32B32A32_SINT,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_FORMAT_R64_UINT,
        VK_FORMAT_R64_SINT,
        VK_FORMAT_R64_SFLOAT,
        VK_FORMAT_R64G64_UINT,
        VK_FORMAT_R64G64_SINT,
        VK_FORMAT_R64G64_SFLOAT,
        VK_FORMAT_R64G64B64_UINT,
        VK_FORMAT_R64G64B64_SINT,
        VK_FORMAT_R64G64B64_SFLOAT,
        VK_FORMAT_R64G64B64A64_UINT,
        VK_FORMAT_R64G64B64A64_SINT,
        VK_FORMAT_R64G64B64A64_SFLOAT,
        VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
        VK_FORMAT_D16_UNORM,
        VK_FORMAT_X8_D24_UNORM_PACK32,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_S8_UINT,
        VK_FORMAT_D16_UNORM_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_BC1_RGB_UNORM_BLOCK,
        VK_FORMAT_BC1_RGB_SRGB_BLOCK,
        VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
        VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
        VK_FORMAT_BC2_UNORM_BLOCK,
        VK_FORMAT_BC2_SRGB_BLOCK,
        VK_FORMAT_BC3_UNORM_BLOCK,
        VK_FORMAT_BC3_SRGB_BLOCK,
        VK_FORMAT_BC4_UNORM_BLOCK,
        VK_FORMAT_BC4_SNORM_BLOCK,
        VK_FORMAT_BC5_UNORM_BLOCK,
        VK_FORMAT_BC5_SNORM_BLOCK,
        VK_FORMAT_BC6H_UFLOAT_BLOCK,
        VK_FORMAT_BC6H_SFLOAT_BLOCK,
        VK_FORMAT_BC7_UNORM_BLOCK,
        VK_FORMAT_BC7_SRGB_BLOCK,
        VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
        VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
        VK_FORMAT_EAC_R11_UNORM_BLOCK,
        VK_FORMAT_EAC_R11_SNORM_BLOCK,
        VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
        VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
        VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
        VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
        VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
        VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
        VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
        VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
        VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
        VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
        VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
        VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
        VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
        VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
        VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
        VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
        VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
        VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
        VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
        VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
        VK_FORMAT_ASTC_12x12_SRGB_BLOCK
    };
    quote!(gpu::vulkan::#v)
}

fn vk_blend_factor_tokens(blend_factor: VkBlendFactor) -> TokenStream {
    let v = enum_to_ident! { blend_factor;
        VK_BLEND_FACTOR_ZERO,
        VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_SRC_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
        VK_BLEND_FACTOR_DST_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
        VK_BLEND_FACTOR_SRC_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_FACTOR_DST_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
        VK_BLEND_FACTOR_CONSTANT_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
        VK_BLEND_FACTOR_CONSTANT_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
        VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
        VK_BLEND_FACTOR_SRC1_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
        VK_BLEND_FACTOR_SRC1_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA
    };
    quote!(gpu::vulkan::#v)
}

fn vk_blend_op_tokens(blend_op: VkBlendOp) -> TokenStream {
    let v = enum_to_ident! {blend_op;
        VK_BLEND_OP_ADD,
        VK_BLEND_OP_SUBTRACT,
        VK_BLEND_OP_REVERSE_SUBTRACT,
        VK_BLEND_OP_MIN,
        VK_BLEND_OP_MAX
    };
    quote!(gpu::vulkan::#v)
}

fn vk_compare_op_tokens(compare_op: VkCompareOp) -> TokenStream {
    let v = enum_to_ident! {compare_op;
        VK_COMPARE_OP_NEVER,
        VK_COMPARE_OP_LESS,
        VK_COMPARE_OP_EQUAL,
        VK_COMPARE_OP_LESS_OR_EQUAL,
        VK_COMPARE_OP_GREATER,
        VK_COMPARE_OP_NOT_EQUAL,
        VK_COMPARE_OP_GREATER_OR_EQUAL,
        VK_COMPARE_OP_ALWAYS
    };
    quote!(gpu::vulkan::#v)
}
