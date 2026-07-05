
//! Dumps the contents of a `.sharc` shader archive to stdout.

use chrono::{DateTime, Utc};
use color_print::{cformat, cprintln};
use sharc::reflection::{Param, ParamLocation, ShaderResource, ShaderResourceKind, Signature};
use sharc::{
    FileDependency, GraphicsPipeline, ImageResourceDesc, ImageResourceSize, Module, Pass,
    PipelineKind, ShaderArchive,
};
use std::path::Path;

////////////////////////////////////////////////////////////////////////////////////////////////////

fn format_unix_time(secs: u64) -> String {
    if secs == 0 {
        return "<unknown>".to_string();
    }
    let dt = DateTime::<Utc>::from_timestamp(secs as i64, 0)
        .unwrap_or_default();
    dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

fn print_file_dep(archive: &ShaderArchive, dep: &FileDependency, indent: &str) {
    if dep.path.is_valid() {
        let path = &archive[dep.path];
        cprintln!(
            "{indent}<bold>{path}</> <dim>(mtime: {})</>",
            format_unix_time(dep.mtime)
        );
    } else {
        cprintln!("{indent}<dim>(none)</>");
    }
}

fn print_params(archive: &ShaderArchive, params: &[Param], indent: &str) {
    for (i, param) in params.iter().enumerate() {
        let name = if param.name.is_valid() { &archive[param.name] } else { "<unnamed>" };
        let loc = match param.location {
            ParamLocation::Binding { resource_index, offset } => {
                format!("Binding(resource={resource_index}, offset={offset})")
            }
            ParamLocation::PushData { offset } => format!("PushData(offset={offset})"),
            ParamLocation::Indirect { rel, offset } => {
                format!("Indirect(rel={rel}, offset={offset})")
            }
        };
        cprintln!(
            "{indent}<dim>[{i}]</> <green>{name}</>  <dim>{loc}</>  size={}",
            param.byte_size
        );
    }
}

fn print_resources(resources: &[ShaderResource], indent: &str) {
    for (i, res) in resources.iter().enumerate() {
        let kind = match res.kind {
            ShaderResourceKind::UniformBuffer => "UniformBuffer",
            ShaderResourceKind::StorageBuffer => "StorageBuffer",
            ShaderResourceKind::Texture => "Texture",
            ShaderResourceKind::StorageImage => "StorageImage",
            ShaderResourceKind::Sampler => "Sampler",
        };
        cprintln!(
            "{indent}<dim>[{i}]</> <cyan>{kind}</>  set={}, binding={}",
            res.set, res.binding
        );
    }
}

fn print_signature(archive: &ShaderArchive, sig: &Signature, indent: &str) {
    let child = format!("{indent}  ");
    let params = if sig.params.is_valid() { &archive[sig.params] } else { &[] };
    let resources = if sig.resources.is_valid() { &archive[sig.resources] } else { &[] };

    cprintln!("{indent}<bold>Signature:</>  params={}, resources={}", params.len(), resources.len());
    if !params.is_empty() {
        cprintln!("{indent}  <bold>Params:</>");
        print_params(archive, params, &format!("{child}  "));
    }
    if !resources.is_empty() {
        cprintln!("{indent}  <bold>Resources:</>");
        print_resources(resources, &format!("{child}  "));
    }
}

fn print_graphics_pipeline(archive: &ShaderArchive, gfx: &GraphicsPipeline, indent: &str) {
    cprintln!("{indent}<bold>Push constants:</> {} bytes", gfx.push_constants_size);

    // Rasterization
    cprintln!(
        "{indent}<bold>Rasterization:</>  polygon_mode={:?}  cull_mode={:?}",
        gfx.rasterization.polygon_mode,
        gfx.rasterization.cull_mode,
    );

    // Depth/stencil
    let ds = &gfx.depth_stencil;
    if ds.enable {
        let write_str = if ds.depth_write_enable {
            cformat!("<green>true</>")
        } else {
            cformat!("<red>false</>")
        };
        cprintln!(
            "{indent}<bold>Depth/stencil:</>  enabled=<green>true</>  format={:?}  compare_op={:?}  write={write_str}",
            ds.format,
            ds.depth_compare_op,
        );
    } else {
        cprintln!("{indent}<bold>Depth/stencil:</>  enabled=<dim>false</>");
    }

    // Shaders
    let shaders = if gfx.shaders.is_valid() { &archive[gfx.shaders] } else { &[] };
    if !shaders.is_empty() {
        let shader_descs: Vec<String> = shaders
            .iter()
            .map(|s| cformat!("<magenta>{:?}</>(\"{}\")", s.stage, s.entry_point))
            .collect();
        cprintln!("{indent}<bold>Shaders:</>  {}", shader_descs.join("  "));
    }

    // Color attachments
    let color_attachments =
        if gfx.color_attachments.is_valid() { &archive[gfx.color_attachments] } else { &[] };
    if !color_attachments.is_empty() {
        cprintln!("{indent}<bold>Color attachments:</> {}", color_attachments.len());
        for (i, ca) in color_attachments.iter().enumerate() {
            let clear = if let Some(c) = ca.clear_color {
                format!(" clear=({:.2},{:.2},{:.2},{:.2})", c[0], c[1], c[2], c[3])
            } else {
                String::new()
            };
            cprintln!("{indent}  <dim>[{i}]</> <cyan>{}</>{clear}", ca.resource_name);
        }
    }

    // Depth/stencil attachment
    if let Some(dsa) = &gfx.depth_stencil_attachment {
        let cd = if let Some(d) = dsa.clear_depth { format!(" clear_depth={d:.2}") } else { String::new() };
        let cs = if let Some(s) = dsa.clear_stencil { format!(" clear_stencil={s}") } else { String::new() };
        cprintln!("{indent}<bold>Depth/stencil attachment:</>  <cyan>{}</>{cd}{cs}", dsa.resource_name);
    }

    // Color targets (pipeline format info)
    let color_targets =
        if gfx.color_targets.is_valid() { &archive[gfx.color_targets] } else { &[] };
    if !color_targets.is_empty() {
        cprintln!("{indent}<bold>Color targets:</> {}", color_targets.len());
        for (i, ct_offset) in color_targets.iter().enumerate() {
            if ct_offset.is_valid() {
                let ct = &archive[*ct_offset];
                let blend = if let Some(b) = ct.blend {
                    format!(
                        "  blend={:?}/{:?}/{:?} α={:?}/{:?}/{:?}",
                        b.src_color_blend_factor,
                        b.dst_color_blend_factor,
                        b.color_blend_op,
                        b.src_alpha_blend_factor,
                        b.dst_alpha_blend_factor,
                        b.alpha_blend_op,
                    )
                } else {
                    "  blend=<dim>off</>".to_string()
                };
                cprintln!("{indent}  <dim>[{i}]</> <cyan>{:?}</>{blend}", ct.format);
            } else {
                cprintln!("{indent}  <dim>[{i}]</> <dim>(invalid)</>");
            }
        }
    }
}

fn print_pass(archive: &ShaderArchive, pass: &Pass, indent: &str) {
    let kind_label = match &pass.kind {
        PipelineKind::Graphics(_) => "<bold,blue>Graphics</>",
        PipelineKind::Compute(_) => "<bold,yellow>Compute</>",
    };
    cprintln!(
        "{indent}<bold,white>Pass:</> <bold,green>\"{}\"</>  [{kind_label}]",
        pass.name
    );

    let child = format!("{indent}  ");

    match &pass.kind {
        PipelineKind::Graphics(gfx) => {
            print_graphics_pipeline(archive, gfx, &child);
        }
        PipelineKind::Compute(cmp) => {
            cprintln!("{child}<bold>Push constants:</> {} bytes", cmp.push_constants_size);
            cprintln!(
                "{child}<bold>Workgroup size:</>  {}×{}×{}",
                cmp.workgroup_size[0],
                cmp.workgroup_size[1],
                cmp.workgroup_size[2],
            );
            cprintln!(
                "{child}<bold>Shader:</> <magenta>{:?}</>(\"{}\")",
                cmp.compute_shader.stage,
                cmp.compute_shader.entry_point,
            );
        }
    }

    // Root params
    let rpl = &pass.root_params;
    let rp_params =
        if rpl.parameters.is_valid() { &archive[rpl.parameters] } else { &[] };
    cprintln!("{child}<bold>Root params layout:</>  {} bytes, {} entries", rpl.byte_size, rp_params.len());
    for (i, rp) in rp_params.iter().enumerate() {
        cprintln!(
            "{child}  <dim>[{i}]</> <green>{}</> \"{}\": offset={} size={} format={:?}",
            rp.name,
            rp.render_world_binding,
            rp.offset,
            rp.size,
            rp.format,
        );
    }

    // Signature
    if pass.signature.is_valid() {
        let sig = &archive[pass.signature];
        print_signature(archive, sig, &child);
    }
}

fn print_module(archive: &ShaderArchive, idx: usize, module: &Module, indent: &str) {
    let name = if module.name.is_valid() { &archive[module.name] } else { "<unnamed>" };
    cprintln!("{indent}<bold,cyan>Module [{}]:</> <bold,green>\"{}\"</>", idx, name);

    let child = format!("{indent}  ");

    // Source file
    cprintln!("{child}<bold>Source:</>  ");
    print_file_dep(archive, &module.file, &format!("{child}  "));

    // Include paths
    let include_paths =
        if module.include_paths.is_valid() { &archive[module.include_paths] } else { &[] };
    if !include_paths.is_empty() {
        cprintln!("{child}<bold>Include paths:</> {}", include_paths.len());
        for ip in include_paths {
            if ip.is_valid() {
                cprintln!("{child}  {}", &archive[*ip]);
            }
        }
    }

    // Debug info flag
    if module.debug_info {
        cprintln!("{child}<bold>Debug info:</>  <green>true</>");
    } else {
        cprintln!("{child}<bold>Debug info:</>  <dim>false</>");
    }

    // SPIR-V
    let spirv = if module.spirv.is_valid() { &archive[module.spirv] } else { &[] };
    let spirv_bytes = spirv.len() * 4;
    cprintln!(
        "{child}<bold>SPIR-V:</>  {} words ({} bytes)",
        spirv.len(),
        spirv_bytes
    );

    // Module-level params
    let params = if module.params.is_valid() { &archive[module.params] } else { &[] };
    if !params.is_empty() {
        cprintln!("{child}<bold>Module params:</> {}", params.len());
        print_params(archive, params, &format!("{child}  "));
    }

    // Passes
    let passes = if module.passes.is_valid() { &archive[module.passes] } else { &[] };
    cprintln!("{child}<bold>Passes:</> {}", passes.len());
    for pass in passes.iter() {
        print_pass(archive, pass, &format!("{child}  "));
        println!();
    }
}

fn print_image_resource(img: &ImageResourceDesc, idx: usize, indent: &str) {
    let size_desc = match img.size {
        ImageResourceSize::Dynamic => "dynamic".to_string(),
        ImageResourceSize::RenderTarget => "render-target".to_string(),
        ImageResourceSize::Fixed { width, height } => format!("{width}×{height}"),
    };
    cprintln!(
        "{indent}<dim>[{idx}]</> <bold,green>\"{}\"</>  format={:?}  size=<cyan>{size_desc}</>  usage={:?}",
        img.name,
        img.format,
        img.usage,
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Loads a `.sharc` archive from `path` and prints its contents to stdout.
pub fn dump(path: &Path) -> anyhow::Result<()> {
    let archive = ShaderArchive::load(path)
        .map_err(|e| anyhow::anyhow!("failed to load archive: {e}"))?;
    let root = archive.root();

    cprintln!("<bold,cyan>Shader Archive:</> <bold>{}</>", path.display());
    println!();

    // Manifest
    cprintln!("  <bold>Manifest:</>");
    print_file_dep(&archive, &root.manifest, "    ");
    println!();

    // Image resources
    let images = if root.images.is_valid() { &archive[root.images] } else { &[] };
    cprintln!("  <bold>Image Resources:</> {}", images.len());
    if !images.is_empty() {
        for (i, img) in images.iter().enumerate() {
            print_image_resource(img, i, "    ");
        }
        println!();
    }

    // Modules
    let modules = if root.modules.is_valid() { &archive[root.modules] } else { &[] };
    cprintln!("  <bold>Modules:</> {}", modules.len());
    println!();
    for (i, module) in modules.iter().enumerate() {
        print_module(&archive, i, module, "  ");
        println!();
    }

    Ok(())
}






