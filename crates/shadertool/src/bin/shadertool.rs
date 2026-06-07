use clap::Parser;
use color_print::ceprintln;

///////////////////////////////////////////////////////////////////////////////////

#[derive(Parser, Debug)]
struct Args {
    /// Path to build manifest.
    manifest_path: Option<String>,
    /// Don't print logs to stdout.
    #[clap(short, long)]
    quiet: bool,
    /// Print cargo dependency directives.
    #[clap(long)]
    emit_cargo_deps: bool,
    /// Emit shader debug information.
    #[clap(short, long)]
    debug: bool,
    /// Dump SPIR-V binaries to disk alongside the archive.
    #[clap(long)]
    dump_spirv: bool,
    /// Open graphical editor.
    #[clap(long)]
    editor: bool,
    /// Verbosity level (0-3).
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
    )]
    verbose: u8,
}

fn main() {
    env_logger::builder().parse_default_env().format_target(false).format_timestamp(None).init();

    let args = Args::parse();

    if args.editor {
        //run_editor();
        return;
    } else if let Some(manifest_path) = args.manifest_path {
        let build_options = shadertool::BuildOptions {
            quiet: args.quiet,
            emit_cargo_deps: args.emit_cargo_deps,
            emit_debug_information: args.debug,
            emit_spirv_binaries: args.dump_spirv,
            verbosity: args.verbose,
        };
        match shadertool::build(&manifest_path, &build_options) {
            Ok(()) => {}
            Err(err) => {
                ceprintln!("<r,bold>error:</> {err:#}");
                std::process::exit(1);
            }
        }
    }
}
