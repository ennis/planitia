#[cfg(windows)]
pub mod windows;

#[cfg(windows)]
pub(crate) use windows::PlatformExtensions;
