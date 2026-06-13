//! Context-capturing error type.

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::panic::Location;
use color_print::{ceprintln, cwriteln};

pub struct ExcFrame {
    location: &'static Location<'static>,
    this: Box<dyn std::error::Error + Send + Sync>,
    children: Vec<ExcFrame>,
}

impl ExcFrame {
    fn print_error_tree_rec(&self, f: &mut fmt::Formatter<'_>, prefix: &mut String) -> fmt::Result {
        // write vertical lines for previous levels
        let space = if !prefix.is_empty() { " " } else { "" };
        cwriteln!(f, "{prefix}{space}{} <dim>(at {}:{})</dim>", self.this, self.location.file(), self.location.line())?;

        match prefix.pop() {
            Some('├') => prefix.push('│'),
            None => {},
            _ => prefix.push(' '),
        }

        prefix.push(' ');
        for (i,child) in self.children.iter().enumerate() {
            prefix.push(if i == self.children.len() - 1 { '╰' } else { '├' });
            child.print_error_tree_rec(f, prefix)?;
            prefix.pop();
        }
        prefix.pop();
        Ok(())
    }

    fn print_error_tree(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print_error_tree_rec(f, &mut String::new())
    }

    /// Logs the error tree to stderr.
    fn log_error(&self) {
        ceprintln!("<red>error:</red> {}", self);
    }
}

impl fmt::Debug for ExcFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print_error_tree(f)
    }
}

impl fmt::Display for ExcFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print_error_tree(f)
    }
}

impl std::error::Error for ExcFrame {}

//--------------------------------------------------------------------------------------------------

pub struct Exc<E> {
    frame: Box<ExcFrame>,
    _phantom: PhantomData<E>,
}

impl<E> fmt::Debug for Exc<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.frame.fmt(f)
    }
}

impl<E> fmt::Display for Exc<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.frame.fmt(f)
    }
}

impl<E> Exc<E> {
    #[track_caller]
    pub fn new(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            frame: Box::new(ExcFrame { location: Location::caller(), this: Box::new(err), children: vec![] }),
            _phantom: PhantomData,
        }
    }

    pub fn add_child<F>(&mut self, child: impl Into<Exc<F>>) {
        let child = child.into();
        self.frame.children.push(*child.frame);
    }

    #[track_caller]
    pub fn raise<F>(self, outer: F) -> Exc<F>
    where
        F: std::error::Error + Send + Sync + 'static,
    {
        let mut new_exc = Exc::new(outer);
        new_exc.frame.children.push(*self.frame);
        new_exc
    }

    /// Logs the error tree to stderr.
    pub fn log_error(&self) {
        self.frame.log_error();
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
        exc.frame
    }
}

//--------------------------------------------------------------------------------------------------
pub struct ExcAny {
    frame: Box<ExcFrame>,
}

impl fmt::Debug for ExcAny {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.frame.fmt(f)
    }
}

impl fmt::Display for ExcAny {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.frame.fmt(f)
    }
}

impl ExcAny {
    #[track_caller]
    pub fn new(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self { frame: Box::new(ExcFrame { location: Location::caller(), this: Box::new(err), children: vec![] }) }
    }

    pub fn add_child(&mut self, child: ExcAny) {
        self.frame.children.push(*child.frame);
    }

    #[track_caller]
    pub fn raise(self, outer: impl std::error::Error + Send + Sync + 'static) -> ExcAny {
        let mut new_exc = ExcAny::new(outer);
        new_exc.frame.children.push(*self.frame);
        new_exc
    }
}

impl<E> From<E> for ExcAny
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[track_caller]
    fn from(err: E) -> Self {
        ExcAny::new(err)
    }
}

//--------------------------------------------------------------------------------------------------

pub type ExcResult<T, E> = std::result::Result<T, Exc<E>>;
pub type ExcResultAny<T> = std::result::Result<T, ExcAny>;

pub trait ResultExt<T, E> {
    #[track_caller]
    fn raise<G>(self, outer: G) -> ExcResult<T, G>
    where
        G: std::error::Error + Send + Sync + 'static;

    #[track_caller]
    fn or_raise<F, G>(self, outer: F) -> ExcResult<T, G>
    where
        Self: Sized,
        F: FnOnce() -> G,
        G: std::error::Error + Send + Sync + 'static;
}

impl<T, E> ResultExt<T, E> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[track_caller]
    fn raise<G>(self, outer: G) -> ExcResult<T, G>
    where
        G: Error + Send + Sync + 'static,
    {
        match self {
            Ok(value) => Ok(value),
            Err(err) => {
                let mut exc = Exc::new(outer);
                exc.add_child(err);
                Err(exc)
            }
        }
    }

    #[track_caller]
    fn or_raise<F, G>(self, outer: F) -> ExcResult<T, G>
    where
        Self: Sized,
        F: FnOnce() -> G,
        G: Error + Send + Sync + 'static
    {
        match self {
            Ok(value) => Ok(value),
            Err(err) => {
                let mut exc = Exc::new(outer());
                exc.add_child(err);
                Err(exc)
            }
        }
    }
}

impl<T, E> ResultExt<T, E> for Result<T, Exc<E>>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[track_caller]
    fn raise<G>(self, outer: G) -> ExcResult<T, G>
    where
        G: Error + Send + Sync + 'static,
    {
        match self {
            Ok(value) => Ok(value),
            Err(exc) => Err(exc.raise(outer)),
        }
    }

    #[track_caller]
    fn or_raise<F, G>(self, outer: F) -> ExcResult<T, G>
    where
        Self: Sized,
        F: FnOnce() -> G,
        G: Error + Send + Sync + 'static
    {
        match self {
            Ok(value) => Ok(value),
            Err(exc) => Err(exc.raise(outer())),
        }
    }
}

pub trait OptionExt<T> {
    #[track_caller]
    fn ok_or_raise<F, G>(self, outer: F) -> ExcResult<T, G>
    where
        F: FnOnce() -> G,
        G: std::error::Error + Send + Sync + 'static;

    #[track_caller]
    fn ok_or_raise_value<G>(self, outer: G) -> ExcResult<T, G>
    where
        G: std::error::Error + Send + Sync + 'static;
}

impl <T> OptionExt<T> for Option<T> {
    #[track_caller]
    fn ok_or_raise<F, G>(self, outer: F) -> ExcResult<T, G>
    where
        F: FnOnce() -> G,
        G: std::error::Error + Send + Sync + 'static,
    {
        match self {
            Some(value) => Ok(value),
            None => Err(Exc::new(outer())),
        }
    }

    #[track_caller]
    fn ok_or_raise_value<G>(self, outer: G) -> ExcResult<T, G>
    where
        G: std::error::Error + Send + Sync + 'static,
    {
        match self {
            Some(value) => Ok(value),
            None => Err(Exc::new(outer)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(thiserror::Error, Debug)]
    #[error("IO error")]
    struct IOError;

    #[derive(thiserror::Error, Debug)]
    #[error("Parse error")]
    struct ParseError;

    #[derive(thiserror::Error, Debug)]
    #[error("Compilation error: {0}")]
    struct CompilationError(String);

    #[derive(thiserror::Error, Debug)]
    #[error("Execution error")]
    struct ExecutionError;

    #[derive(thiserror::Error, Debug)]
    #[error("Application error")]
    struct AppError;

    fn try_read() -> ExcResult<(), IOError> {
        Err(IOError)?
    }

    fn try_parse() -> ExcResult<(), ParseError> {
        Err(ParseError)?
    }

    fn try_compile() -> ExcResult<(), CompilationError> {
        try_parse().raise(CompilationError("Failed to compile".to_string()))
    }

    fn try_stuff() -> ExcResult<(), ExecutionError> {
        let r1 = try_read().raise(ExecutionError)?;
        Ok(())
    }

    fn try_app() -> ExcResult<(), AppError> {
        let _ = try_stuff().raise(AppError)?;
        Ok(())
    }

    fn try_compile_multiple() -> ExcResult<(), ExecutionError> {
        let mut compile_errors = Vec::new();
        if let Err(err) = try_compile() {
            compile_errors.push(err);
        }
        if let Err(err) = try_compile() {
            compile_errors.push(err);
        }
        if let Err(err) = try_compile() {
            compile_errors.push(err);
        }
        if compile_errors.is_empty() {
            Ok(())
        } else {
            let mut exc = Exc::new(ExecutionError);
            for err in compile_errors {
                exc.add_child(err);
            }
            Err(exc)
        }
    }

    fn try_app_2() -> ExcResult<(), AppError> {
        let _ = try_compile_multiple().raise(AppError)?;
        Ok(())
    }

    //------

    /*fn try_read_any() -> ExcResultAny<()> {
        Err(IOError)?
    }

    fn try_parse_any() -> ExcResultAny<()> {
        Err(ParseError)?
    }

    fn try_compile_any() -> ExcResultAny<()> {
        Err(CompilationError("Failed to compile".to_string()))?
    }

    fn try_stuff_any() -> ExcResultAny<()> {
        let _ = try_read_any()?;
        let _ = try_parse_any()?;
        let _ = try_compile_any()?;
        Ok(())
    }

    fn try_app_any() -> ExcResultAny<()> {
        let _ = try_stuff_any()?;
        Ok(())
    }*/


    #[test]
    fn test_exc() {
        let result = try_app();
        match result {
            Ok(_) => panic!("Expected error"),
            Err(err) => {
                println!("{err}");
            }
        }

        let result_2 = try_app_2();
        match result_2 {
            Ok(_) => panic!("Expected error"),
            Err(err) => {
                println!("{err}");
            }
        }
    }
}
