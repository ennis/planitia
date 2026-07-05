use std::path::PathBuf;
use clap::{Parser, Subcommand};
use color_print::ceprintln;

///////////////////////////////////////////////////////////////////////////////////

#[derive(Parser, Debug)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compile shader source files into a .sharc archive.
    Build(BuildArgs),
    /// Dump information about a .sharc archive to stdout.
    Dump(DumpArgs),
}

#[derive(clap::Args, Debug)]
struct BuildArgs {
    /// Input files (glob patterns supported, e.g. `shaders/**/*.slang`).
    input_files: Vec<String>,
    /// Include paths.
    #[arg(long, short = 'I')]
    include: Vec<PathBuf>,
    /// Don't print logs to stdout.
    #[arg(short, long)]
    quiet: bool,
    /// Print cargo dependency directives.
    #[arg(long)]
    emit_cargo_deps: bool,
    /// Emit shader debug information.
    #[arg(short, long)]
    debug: bool,
    /// Dump SPIR-V binaries to disk alongside the archive.
    #[arg(long)]
    dump_spirv: bool,
    /// Verbosity level (0-3).
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
    )]
    verbose: u8,
}

#[derive(clap::Args, Debug)]
struct DumpArgs {
    /// Path to the .sharc archive file to inspect.
    archive: PathBuf,
}

///////////////////////////////////////////////////////////////////////////////////

fn main() {
    env_logger::builder().parse_default_env().format_target(false).format_timestamp(None).init();

    let args = Args::parse();

    match args.command {
        Command::Build(build_args) => cmd_build(build_args),
        Command::Dump(dump_args) => cmd_dump(dump_args),
    }
}

fn cmd_build(args: BuildArgs) {
    if args.input_files.is_empty() {
        ceprintln!("<r,bold>error:</> No input files specified.");
        std::process::exit(1);
    }

    let build_options = shadertool::BuildOptions {
        emit_cargo_deps: args.emit_cargo_deps,
        emit_debug_information: args.debug,
        emit_spirv_binaries: args.dump_spirv,
        include_paths: args.include,
        output_directory: None,
    };
    let log_options = shadertool::LogOptions {
        quiet: args.quiet,
        verbosity: args.verbose,
    };

    let input_files = args.input_files.iter().map(|s| s.as_str());
    match shadertool::build(input_files, &build_options, &log_options) {
        Ok(()) => {}
        Err(err) => {
            ceprintln!("<r,bold>error:</> {err:#}");
            std::process::exit(1);
        }
    }
}

fn cmd_dump(args: DumpArgs) {
    match shadertool::dump_archive_file(&args.archive) {
        Ok(()) => {}
        Err(err) => {
            ceprintln!("<r,bold>error:</> {err:#}");
            std::process::exit(1);
        }
    }
}
