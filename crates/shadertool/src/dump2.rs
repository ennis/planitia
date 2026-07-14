use chrono::{DateTime, Utc};
use color_print::{cprint, cprintln, cwrite};
use sharc::archive::Offset;
use sharc::gpu_types::vk;
use sharc::reflection::{ParamLocation, Signature};
use std::path::Path;

fn format_unix_time(secs: u64) -> String {
    if secs == 0 {
        return "<unknown>".to_string();
    }
    let dt = DateTime::<Utc>::from_timestamp(secs as i64, 0).unwrap_or_default();
    dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

struct Indent(usize);

impl std::fmt::Display for Indent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for _ in 0..self.0 {
            write!(f, "  ")?; // 2 spaces per indent level
        }
        Ok(())
    }
}

struct OnOff(bool);

impl std::fmt::Display for OnOff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 { cwrite!(f, "<green>ON</>") } else { cwrite!(f, "OFF") }
    }
}

struct Printer<'a> {
    a: &'a sharc::ShaderArchive,
    indent: usize,
}

impl<'a> Printer<'a> {
    fn new(a: &'a sharc::ShaderArchive) -> Self {
        Printer { a, indent: 0 }
    }

    fn get_indent(&self) -> Indent {
        Indent(self.indent)
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        if self.indent > 0 {
            self.indent -= 1;
        }
    }

    fn print_reflection_param_location(&mut self, param: &sharc::reflection::ParamLocation) {
        match param {
            ParamLocation::Binding { resource_index, offset } => {
                cprint!("Binding(resource={resource_index}, offset={offset})");
            }
            ParamLocation::PushData { offset } => {
                cprint!("PushData(offset={offset})");
            }
            ParamLocation::Indirect { rel, offset } => {
                cprint!("Indirect(base=<dim>[{rel}]</>, offset={offset})");
            }
        }
    }

    fn print_reflection_param(&mut self, index: usize, param: &sharc::reflection::Param) {
        let indent = self.get_indent();
        let name = &self.a[param.name];
        let byte_size = param.byte_size;
        cprint!("{indent}<dim>[{index}]</> <bold>{name}</>: {byte_size} bytes  ");
        self.print_reflection_param_location(&param.location);
        cprintln!();
    }

    fn print_signature(&mut self, signature: &sharc::reflection::Signature) {
        let indent = self.get_indent();
        let params = &self.a[signature.params];
        let resources = &self.a[signature.resources];

        cprintln!("{indent}<bold>Signature</>:");
        self.indent();
        cprintln!("{indent}<bold>Parameters</>: {}", params.len());
        self.indent();
        for (index, param) in params.iter().enumerate() {
            self.print_reflection_param(index, param);
        }
        self.dedent();

        self.indent();
        cprintln!("{indent}<bold>Resources</>: {}", resources.len());
        for (index, resource) in resources.iter().enumerate() {
            let indent = self.get_indent();
            let set = resource.set;
            let binding = resource.binding;
            let kind = resource.kind;
            cprintln!("{indent}<dim>[{index}]</> {kind:?} <dim>set={set} binding={binding}</dim>");
        }

        self.dedent();
        self.dedent();
    }

    fn print_signature_opt(&mut self, signature: Offset<Signature>) {
        if signature.is_valid() {
            let signature = &self.a[signature];
            self.print_signature(signature);
        }
    }

    fn print_color_target(&mut self, index: usize, target: &sharc::ColorTarget) {
        let format = target.format;
        if let Some(blend) = target.blend {
            let src_color = blend.src_color_blend_factor;
            let dst_color = blend.dst_color_blend_factor;
            let color_op = blend.color_blend_op;
            let src_alpha = blend.src_alpha_blend_factor;
            let dst_alpha = blend.dst_alpha_blend_factor;
            let alpha_op = blend.alpha_blend_op;
            cprintln!(
                "{}<dim>[{index}]</> <cyan>{format:?}</> Blend Enabled color=<cyan>{src_color:?}/{dst_color:?}/{color_op:?}</> alpha=<cyan>{src_alpha:?}/{dst_alpha:?}/{alpha_op:?}</>",
                Indent(self.indent)
            );
        } else {
            cprintln!("{}<dim>[{index}]</> <cyan>{format:?}</> Blend Disabled", Indent(self.indent));
        }
    }

    fn print_color_targets(&mut self, targets: &[Offset<sharc::ColorTarget>]) {
        cprintln!("{}<bold>Color Targets</>: {}", Indent(self.indent), targets.len());
        self.indent();
        for (index, target) in targets.iter().enumerate() {
            let color_target = &self.a[*target];
            self.print_color_target(index, color_target);
        }
        self.dedent();
    }

    fn print_shader(&mut self, shader: &sharc::Shader) {
        let stage = shader.stage;
        let name = shader.entry_point.as_str();
        cprint!("{stage:?}(\"{name}\")");
    }

    fn print_shaders(&mut self, shaders: &[sharc::Shader]) {
        cprint!("{}<bold>Shaders</>:  ", Indent(self.indent));
        for shader in shaders {
            self.print_shader(shader);
            cprint!("  ");
        }
        cprintln!();
    }

    fn print_rasterization_state(&mut self, rs: &sharc::RasterizationState) {
        cprint!("{}<bold>Rasterization</>:  ", Indent(self.indent));
        let cull_mode = match rs.cull_mode {
            vk::CullModeFlags::FRONT => "FRONT",
            vk::CullModeFlags::BACK => "BACK",
            vk::CullModeFlags::FRONT_AND_BACK => "FRONT_AND_BACK",
            vk::CullModeFlags::NONE => "NONE",
            _ => "INVALID",
        };
        cprintln!("Cull Mode:<cyan>{}</>  Polygon Mode:<cyan>{:?}</>", cull_mode, rs.polygon_mode);
    }

    fn print_depth_stencil_state(&mut self, ds: &sharc::DepthStencilState) {
        cprintln!(
            "{}<bold>Depth/Stencil</>:  <cyan>{:?}</>   Depth Test:{}  Compare:<cyan>{:?}</>  Depth Write:{}",
            Indent(self.indent),
            ds.format,
            OnOff(ds.enable),
            ds.depth_compare_op,
            OnOff(ds.depth_write_enable),
        );
    }

    fn print_push_constants(&mut self, push_constants_size: u16) {
        cprintln!("{}<bold>Push Constants</>:  {} bytes", Indent(self.indent), push_constants_size);
    }

    fn print_pass(&mut self, index: usize, pass: &sharc::Pass) {
        let name = &pass.name;
        match pass.kind {
            sharc::PipelineKind::Graphics(ref data) => {
                cprintln!(
                    "{}<dim>[{index}]</> <bold>Pass</> <bold,blue>{name}</> [Graphics/Primitive Shading]",
                    Indent(self.indent)
                );
                self.indent();
                self.print_push_constants(data.push_constants_size);
                self.print_rasterization_state(&data.rasterization);
                self.print_depth_stencil_state(&data.depth_stencil);
                self.print_shaders(&self.a[data.shaders]);
                self.print_color_targets(&self.a[data.color_targets]);
                self.print_signature_opt(pass.signature);
                self.dedent();
                cprintln!();
            }
            sharc::PipelineKind::Compute(ref data) => {
                cprintln!(
                    "{}<dim>[{index}]</> <bold>Pass</> <bold,yellow>{name}</> [Compute]",
                    Indent(self.indent)
                );
                self.indent();
                self.print_push_constants(data.push_constants_size);
                cprintln!(
                    "{}<bold>Workgroup Size</>: {}×{}×{}",
                    Indent(self.indent),
                    data.workgroup_size[0],
                    data.workgroup_size[1],
                    data.workgroup_size[2]
                );
                self.print_shaders(&[data.compute_shader]);
                self.print_signature_opt(pass.signature);
                self.dedent();
                cprintln!();
            }
        }
    }

    fn print_file_dependency(&mut self, dep: &sharc::FileDependency) {
        let path = &self.a[dep.path];
        let mtime = format_unix_time(dep.mtime);
        cprint!("{path} <dim>(mtime: {mtime})</>");
    }

    fn print_module(&mut self, _index: usize, module: &sharc::Module) {
        let name = &self.a[module.name];

        cprintln!("{}<bold>Module</> <bold,green>{name}</>", Indent(self.indent));

        self.indent();

        cprint!("{}<bold>Source</>:  ", Indent(self.indent));
        self.print_file_dependency(&module.file);
        cprintln!();

        cprintln!("{}<bold>Debug Info</>:  {}", Indent(self.indent), module.debug_info);

        let spirv = &self.a[module.spirv];
        cprintln!("{}<bold>SPIR-V</>:  {} bytes", Indent(self.indent), spirv.len() * size_of::<u32>());

        let passes = &self.a[module.passes];
        cprintln!("{}<bold>Passes</>:  {}\n", Indent(self.indent), passes.len());

        for (index, pass) in passes.iter().enumerate() {
            self.print_pass(index, pass);
        }

        let params = &self.a[module.params];
        cprintln!("{}<bold>Parameters</>:  {}", Indent(self.indent), params.len());
        self.indent();
        for (index, param) in params.iter().enumerate() {
            self.print_reflection_param(index, param);
        }
        self.dedent();

        self.dedent();

        cprintln!();
    }

    fn print(&mut self, path: &Path) {
        let root = self.a.root();
        let modules = &self.a[root.modules];

        cprintln!("{}<bold>Shader Archive</>: {}", Indent(self.indent), path.display());

        self.indent();

        cprint!("{}<bold>Manifest</>:", Indent(self.indent));
        self.print_file_dependency(&root.manifest);
        cprintln!();

        cprintln!("{}<bold>Modules</>: {}\n", Indent(self.indent), modules.len());

        for (i, module) in modules.iter().enumerate() {
            self.print_module(i, module);
        }
        self.dedent();
    }
}

pub fn dump_archive_file(path: &Path) -> anyhow::Result<()> {
    let archive = sharc::ShaderArchive::load(path)?;
    let mut printer = Printer::new(&archive);
    printer.print(path);
    Ok(())
}
