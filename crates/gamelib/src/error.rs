//! Context-capturing error type.

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::panic::Location;

struct ExcInner {
    location: &'static Location<'static>,
    this: Box<dyn std::error::Error + Send + Sync>,
    children: Vec<ExcInner>,
}

impl ExcInner {
    fn print_error_tree(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        for _ in 0..indent {
            write!(f, "  ")?;
        }
        writeln!(f, "{} (at {}:{})", self.this, self.location.file(), self.location.line())?;
        for child in &self.children {
            child.print_error_tree(f, indent + 1)?;
        }
        Ok(())
    }
}

impl fmt::Debug for ExcInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print_error_tree(f, 0)
    }
}

impl fmt::Display for ExcInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print_error_tree(f, 0)
    }
}

impl std::error::Error for ExcInner {}

pub struct Exc<E> {
    inner: Box<ExcInner>,
    _phantom: PhantomData<E>,
}

impl<E> fmt::Debug for Exc<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.this.fmt(f)
    }
}

impl<E> Exc<E> {
    #[track_caller]
    pub fn new(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            inner: Box::new(ExcInner { location: Location::caller(), this: Box::new(err), children: vec![] }),
            _phantom: PhantomData,
        }
    }

    pub fn add_child(&mut self, child: impl Into<Exc<E>>) {
        let child = child.into();
        self.inner.children.push(*child.inner);
    }

    pub fn raise<F>(self, outer: F) -> Exc<F>
    where
        F: std::error::Error + Send + Sync + 'static,
    {
        let mut new_exc = Exc::new(outer);
        new_exc.inner.children.push(*self.inner);
        new_exc
    }
}

impl<E> From<E> for Exc<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[track_caller]
    fn from(e: E) -> Self {
        Exc::new(e)
    }
}

impl<E> From<Exc<E>> for Box<dyn std::error::Error + Send + Sync + 'static>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(exc: Exc<E>) -> Self {
        exc.inner
    }
}

pub type Result<T, E> = std::result::Result<T, Exc<E>>;
