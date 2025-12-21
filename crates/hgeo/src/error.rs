use std::str::Utf8Error;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("malformed .geo file: {0}")]
    Malformed(&'static str),
    #[error("unsupported .geo file")]
    Unsupported,
    #[error("end of file")]
    Eof,
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("Invalid binary json magic number")]
    InvalidBjsonMagic,
    #[error("invalid binary length encoding")]
    InvalidLengthEncoding,
    #[error("invalid utf-8 string")]
    InvalidUtf8String(#[from] Utf8Error),
    #[error("invalid token index")]
    InvalidTokenIndex,
}
