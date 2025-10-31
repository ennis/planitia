use crate::asset::{Provider, VfsPath};
use log::warn;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use utils::aligned_vec::AVec;

/// Implements [`Provider`] for a directory in the local file system.
pub struct LocalProvider {
    root_directory: PathBuf,
}

impl LocalProvider {
    /// Creates a new file system provider rooted at the given directory.
    pub fn new(root_directory: String) -> Self {
        Self {
            root_directory: PathBuf::from(root_directory),
        }
    }

    /// Constructs the full path in the file system for the given VFS path.
    ///
    /// This expects the VFS path to be absolute.
    fn full_path(&self, path: &VfsPath) -> PathBuf {
        let file_part = path.strip_source().path_without_fragment();
        // we shouldn't receive relative paths here
        if file_part.is_relative() {
            warn!(
                "A relative path was passed to FileSystemProvider: {}. Paths passed to providers should be absolute.",
                path.as_str()
            );
        }
        let path_without_first_slash = if file_part.0.starts_with('/') {
            &file_part.0[1..]
        } else {
            &file_part.0
        };

        self.root_directory.join(path_without_first_slash)
    }
}

fn get_file_size(path: &Path) -> io::Result<u64> {
    let metadata = std::fs::metadata(path)?;
    Ok(metadata.len())
}

impl Provider for LocalProvider {
    fn exists(&self, path: &VfsPath) -> bool {
        let p = self.full_path(path);
        p.exists()
    }

    fn load(&self, path: &VfsPath) -> Result<AVec<u8>, io::Error> {
        let p = self.full_path(path);
        let file_size = get_file_size(&p)?;
        let mut buffer = AVec::with_capacity(0, file_size as usize);
        let mut file = std::fs::File::open(&p)?;
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }
}
