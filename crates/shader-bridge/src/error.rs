use std::fmt;

/// Compilation error.
#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    EntryPointNotFound(String),
    CompilationError(slang::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            //CompilationError::CompileError(err) => write!(f, "compilation errors: {}", err),
            Error::IoError(err) => write!(f, "I/O error: {}", err),
            Error::EntryPointNotFound(name) => write!(f, "entry point not found: {}", name),
            Error::CompilationError(err) => write!(f, "compilation errors:\n{}", err),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<slang::Error> for Error {
    fn from(err: slang::Error) -> Self {
        Error::CompilationError(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}
