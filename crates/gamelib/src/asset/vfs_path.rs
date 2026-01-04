use std::ops::Deref;

/// Represents a VFS path.
///
/// # Examples
///
/// let path: &VfsPath = VfsPath::new("textures/texture.png");
/// let path2: &VfsPath = VfsPath::new("shaders/pipeline.parc#my_pipeline");
/// let path_with_source: &VfsPath = VfsPath::new("embedded:/shaders/pipeline.parc#my_pipeline");
#[repr(transparent)]
pub struct VfsPath(pub str);

impl VfsPath {
    /// Creates a new VfsPath from a string slice.
    pub fn new(path: &str) -> &VfsPath {
        // SAFETY: VfsPath is #[repr(transparent)] over str
        unsafe { &*(path as *const str as *const VfsPath) }
    }

    /// Returns the path as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Creates an owned VfsPathBuf from this VfsPath.
    pub fn to_path_buf(&self) -> VfsPathBuf {
        VfsPathBuf(self.0.to_string())
    }

    /// Returns whether this represents a relative path.
    ///
    /// A path is relative if the path without the source prefix does not start with '/'.
    pub fn is_relative(&self) -> bool {
        !self.strip_source().0.starts_with('/')
    }

    /// Returns the source prefix of the path, if any.
    ///
    /// E.g. "embedded" in "embedded:/shaders/pipeline.parc#my_pipeline"
    pub fn source(&self) -> Option<&str> {
        let s = &self.0;

        let mut pos = 0;

        let mut chars = s.chars();
        loop {
            match chars.next() {
                Some(ch) => {
                    if ch == ':' {
                        break;
                    } else if ch == '/' || ch == '#' {
                        return None;
                    }
                    pos += ch.len_utf8();
                }
                None => return None,
            }
        }
        Some(&s[..pos])
    }

    /// Strips the source prefix from the path, if any.
    pub fn strip_source(&self) -> &VfsPath {
        let s = &self.0;
        match self.source() {
            Some(source) => {
                let start = source.len() + 1; // +1 for ':'
                VfsPath::new(&s[start..])
            }
            None => self,
        }
    }

    /// Strips the fragment part from the path, if any.
    pub fn path_without_fragment(&self) -> &VfsPath {
        let s = &self.0;
        match s.rfind('#') {
            Some(pos) => VfsPath::new(&s[..pos]),
            None => self,
        }
    }

    /// Returns the fragment part of the path, if any.
    pub fn fragment(&self) -> Option<&str> {
        let s = &self.0;
        match s.rfind('#') {
            Some(pos) => Some(&s[pos + 1..]),
            None => None,
        }
    }

    /// Returns the directory part of the path (including the trailing slash).
    ///
    /// # Examples
    ///
    ///  - returns "shaders/" for "shaders/pipeline.parc#my_pipeline"
    ///  - returns "/shaders/" for "/shaders/pipeline.parc#my_pipeline"
    ///  - returns "" for "pipeline.parc#my_pipeline"
    ///  - returns "/" for "embedded:/pipeline.parc#my_pipeline"
    pub fn directory(&self) -> &str {
        let s = &self.0;
        match s.rfind('/') {
            Some(pos) => &s[..=pos],
            None => "",
        }
    }
}

impl AsRef<str> for VfsPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl AsRef<VfsPath> for VfsPath {
    fn as_ref(&self) -> &VfsPath {
        self
    }
}

impl AsRef<VfsPath> for str {
    fn as_ref(&self) -> &VfsPath {
        VfsPath::new(self)
    }
}

/// Owned version of VfsPath.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VfsPathBuf(String);

impl Deref for VfsPathBuf {
    type Target = VfsPath;

    fn deref(&self) -> &Self::Target {
        VfsPath::new(&self.0)
    }
}

impl AsRef<VfsPath> for VfsPathBuf {
    fn as_ref(&self) -> &VfsPath {
        self.deref()
    }
}

impl PartialEq for VfsPath {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for VfsPath {}

#[cfg(test)]
mod tests {
    use super::VfsPath;

    #[test]
    fn test_vfs_path_source() {
        let path1 = VfsPath::new("embedded:/shaders/pipeline.parc#my_pipeline");
        assert_eq!(path1.source(), Some("embedded"));

        let path2 = VfsPath::new("shaders/pipeline.parc#my_pipeline");
        assert_eq!(path2.source(), None);

        let path3 = VfsPath::new("local:/textures/texture.png");
        assert_eq!(path3.source(), Some("local"));

        let path4 = VfsPath::new("invalid/source:path");
        assert_eq!(path4.source(), None);
    }
}
