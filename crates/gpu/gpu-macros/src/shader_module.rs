use proc_macro2::TokenStream;
use quote::{format_ident, quote};

macro_rules! return_error {
    ($span:expr, $msg:literal) => {
        return Err(syn::Error::new($span, format!($msg)));
    };
    ($span:expr, $fmt:literal, $($arg:tt)*) => {
        return Err(syn::Error::new($span, format!($fmt, $($arg)*)));
    };
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
        emit_debug_information: true,   // TODO configure?
        emit_spirv_binaries: true,
        emit_reflection: false,
        include_paths: vec![],
        output_directory: None,
    };

    let module = shadertool::compile(&shader_file, &options)
        .map_err(|err| syn::Error::new(shader_path_lit.span(), format!("failed to compile shader:\n{err}")))?;

    // Write SPIR-V binary
    let spirv_output = shader_file
        .parent()
        .unwrap()
        .join(format!("spirv/{}.spv", shader_file.file_stem().unwrap().to_string_lossy()))
        .to_string_lossy()
        .to_string();
    let bytecode = quote! {
        static __BYTECODE: &[u32] = ::gpu::include_bytes_as_u32!(#spirv_output);
    };

    // Include all dependencies.
    let dependencies = {
        let paths = module.dependencies.iter().map(|dep| {
            dep.path.to_string_lossy()
        });
        quote! {
            static __DEPENDENCIES: &[&str] = &[
                #(include_str!(#paths)),*
            ];
        }
    };

    // Write ShaderEntryPoints
    let mut entry_points = vec![];
    for entry_point in module.entry_points.iter() {
        let ep_name = &entry_point.name;
        let ep_name_ident = format_ident!("{}", ep_name);
        let push_constants_size = entry_point.push_constants_size;
        let workgroup_size = entry_point.workgroup_size;

        let stage = match entry_point.stage {
            gpu_types::vk::ShaderStageFlags::VERTEX => format_ident!("Vertex"),
            gpu_types::vk::ShaderStageFlags::FRAGMENT => format_ident!("Fragment"),
            gpu_types::vk::ShaderStageFlags::MESH_EXT => format_ident!("Mesh"),
            gpu_types::vk::ShaderStageFlags::TASK_EXT => format_ident!("Task"),
            gpu_types::vk::ShaderStageFlags::COMPUTE => format_ident!("Compute"),
            _ => continue,
        };

        entry_points.push(quote! {
            #[allow(non_upper_case_globals)]
            pub static #ep_name_ident: ::gpu::ShaderEntryPoint<'static> = ::gpu::ShaderEntryPoint {
                stage: ::gpu::ShaderStage::#stage,
                code: super::__BYTECODE,
                entry_point: #ep_name,
                push_constants_size: #push_constants_size,
                source_path: Some(#shader_path),
                workgroup_size: [#(#workgroup_size),*],
            };
        });
    }

    let mut pipelines = vec![];

    // Write PipelineCreateInfos
    for pipeline in module.pipelines.iter() {
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
                gpu_types::vk::ShaderStageFlags::VERTEX => vertex = Some(ep_name_ident),
                gpu_types::vk::ShaderStageFlags::FRAGMENT => fragment = Some(ep_name_ident),
                gpu_types::vk::ShaderStageFlags::MESH_EXT => mesh = Some(ep_name_ident),
                gpu_types::vk::ShaderStageFlags::TASK_EXT => task = Some(ep_name_ident),
                gpu_types::vk::ShaderStageFlags::COMPUTE => compute = Some(ep_name_ident),
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
    let reflection = module.generate_reflection();

    let output = quote! {
        #(#attrs)*
        #vis mod #mod_name {
            use ::gpu::shader_types::*;
            #dependencies
            #bytecode
            #reflection
            pub mod entry_points {
                #(#entry_points)*
            }
            #(#pipelines)*
        }
    };

    Ok(output)
}

fn vk_polygon_mode_tokens(polygon_mode: gpu_types::vk::PolygonMode) -> TokenStream {
    let polygon_mode_str = format!("{polygon_mode:?}");
    let polygon_mode_ident = format_ident!("{polygon_mode_str}");
    quote!(gpu::vk::PolygonMode::#polygon_mode_ident)
}

fn vk_cull_mode_tokens(cull_mode: gpu_types::vk::CullModeFlags) -> TokenStream {
    let mut flags = vec![];
    if cull_mode.contains(gpu_types::vk::CullModeFlags::FRONT) {
        flags.push(format_ident!("FRONT"));
    }
    if cull_mode.contains(gpu_types::vk::CullModeFlags::BACK) {
        flags.push(format_ident!("BACK"));
    }
    quote!(gpu::vk::CullModeFlags::NONE #( | gpu::vk::CullModeFlags::#flags)*)
}

fn vk_front_face_tokens(front_face: gpu_types::vk::FrontFace) -> TokenStream {
    let front_face_str = format!("{front_face:?}");
    let front_face_ident = format_ident!("{front_face_str}");
    quote!(gpu::vk::FrontFace::#front_face_ident)
}

fn vk_format_tokens(format: gpu_types::vk::Format) -> TokenStream {
    let format_str = format!("{format:?}");
    let format_ident = format_ident!("{format_str}");
    quote!(gpu::vk::Format::#format_ident)
}

fn vk_blend_factor_tokens(blend_factor: gpu_types::vk::BlendFactor) -> TokenStream {
    let blend_factor_str = format!("{blend_factor:?}");
    let blend_factor_ident = format_ident!("{blend_factor_str}");
    quote!(gpu::vk::BlendFactor::#blend_factor_ident)
}

fn vk_blend_op_tokens(blend_op: gpu_types::vk::BlendOp) -> TokenStream {
    let blend_op_str = format!("{blend_op:?}");
    let blend_op_ident = format_ident!("{blend_op_str}");
    quote!(gpu::vk::BlendOp::#blend_op_ident)
}

fn vk_compare_op_tokens(compare_op: gpu_types::vk::CompareOp) -> TokenStream {
    let compare_op_str = format!("{compare_op:?}");
    let compare_op_ident = format_ident!("{compare_op_str}");
    quote!(gpu::vk::CompareOp::#compare_op_ident)
}
