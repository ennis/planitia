mod editor;

use crate::editor::run_editor;
use include_dir::{include_dir, Dir};
use clap::Parser;

static PROJECT_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/html");

///////////////////////////////////////////////////////////////////////////////////

#[derive(Parser, Debug)]
struct Args {
    /// Path to build manifest.
    manifest_path: Option<String>,
    /// Don't print anything to stdout.
    #[clap(short, long)]
    quiet: bool,
    /// Open graphical editor.
    #[clap(long)]
    editor: bool,
}

fn main() {
    let args = Args::parse();

    if args.editor {
        run_editor();
        return;
    } else if let Some(manifest_path) = args.manifest_path {
        let build_options = pipeline_build_lib::BuildOptions { quiet: args.quiet };
        match pipeline_build_lib::build_pipeline(&manifest_path, &build_options) {
            Ok(()) => {}
            Err(_err) => {
                std::process::exit(1);
            }
        }
    }
}

