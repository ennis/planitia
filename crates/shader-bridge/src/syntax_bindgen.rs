//! Generate Rust types & constants from a slang source file.
//!
//! This is a purely syntactical transformation that doesn't use the compiler or the reflection API.
//! It simply parses declarations in the slang source and translates them to Rust code.
//! As a result, the layout of generated structs cannot be verified against the shader.
//! (they are `repr(C)` so their layout should mostly match if the shaders are compiled with -fvk-use-scalar-layout)
//!
//! Furthermore, it doesn't attempt to map slang primitive types to Rust types, and expects the Rust module to
//! define appropriate typedefs for slang types.
//! Most types are translated verbatim to rust types (including generic types), except:
//! - associated type references like `Texture2D<float>.Handle`, translated to `Texture2D_Handle<float>`
//! - pointer types like `int32_t*` , translated to `Pointer<int32_t>`
//! - array types like `int32_t[10]`, translated to `[int32_t; 10]`
//!
//! NOTE: using reflection would be better for this, but the slang API doesn't support getting
//! the value of constants (`static const`s) in shaders, so this is a temporary solution
//! until this is possible.

use heck::{ToShoutySnakeCase, ToSnakeCase};
use logos::{Logos, Span};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, TokenStreamExt};
use std::iter::Peekable;
use std::ops::Range;
use std::panic::Location;
use std::path::Path;
use std::str::FromStr;
use std::{fs, io};
use Token::*;

/// Slang source file tokens.
///
/// We also define a `logos` lexer for the token nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Logos)]
#[allow(non_camel_case_types)]
#[repr(u16)]
enum Token<'a> {
    //
    // Tokens
    //
    #[token("(")]
    L_PAREN = 0,
    #[token(")")]
    R_PAREN,
    #[token("{")]
    L_CURLY,
    #[token("}")]
    R_CURLY,
    #[token(",")]
    COMMA,
    #[token("[")]
    L_BRACK,
    #[token("]")]
    R_BRACK,
    #[token(".")]
    DOT,
    #[token("++")]
    PLUSPLUS,
    #[token("--")]
    MINUSMINUS,
    #[token("+")]
    PLUS,
    #[token("-")]
    MINUS,
    #[token("!")]
    BANG,
    #[token("~")]
    TILDE,
    #[token("*")]
    STAR,
    #[token("/")]
    SLASH,
    #[token("%")]
    PERCENT,
    #[token("<<")]
    SHL,
    #[token(">>")]
    SHR,
    #[token("<")]
    L_ANGLE,
    #[token(">")]
    R_ANGLE,
    #[token("<=")]
    LTEQ,
    #[token(">=")]
    GTEQ,
    #[token("==")]
    EQ2,
    #[token("!=")]
    NEQ,
    #[token("&")]
    AMP,
    #[token("^")]
    CARET,
    #[token("|")]
    PIPE,
    #[token("&&")]
    AMP2,
    #[token("^^")]
    CARET2,
    #[token("||")]
    PIPE2,
    #[token("?")]
    QUESTION,
    #[token(":")]
    COLON,
    #[token(";")]
    SEMICOLON,
    #[token("=")]
    EQ,
    #[token("*=")]
    STAREQ,
    #[token("/=")]
    SLASHEQ,
    #[token("%=")]
    PERCENTEQ,
    #[token("+=")]
    PLUSEQ,
    #[token("-=")]
    MINUSEQ,
    #[token("<<=")]
    SHLEQ,
    #[token(">>=")]
    SHREQ,
    #[token("&=")]
    AMPEQ,
    #[token("^=")]
    CARETEQ,
    #[token("|=")]
    PIPEEQ,
    #[token("@")]
    AT,

    #[token("static")]
    STATIC_KW,
    #[token("const")]
    CONST_KW,
    #[token("public")]
    PUBLIC_KW,
    #[token("extern")]
    EXTERN_KW,
    #[token("struct")]
    STRUCT_KW,

    #[regex(r#"[a-zA-Z_][a-zA-Z0-9_]*"#)]
    IDENT(&'a str),

    #[regex(r"0b[0-1]+")]
    #[regex(r"0o[0-7]+")]
    #[regex(r"[0-9]+")]
    #[regex(r"0[xX][0-9A-Fa-f]+")]
    INT_NUMBER(&'a str),
    #[regex(r#""([^\\"]*)""#)]
    STRING(&'a str),
    #[regex("[0-9]+[.]")]
    #[regex("[0-9]+(?:[eE][+-]?[0-9]+)")]
    #[regex("[0-9]*[.][0-9]+(?:[eE][+-]?[0-9]+)?")]
    //#[regex("[+-]?[0-9][0-9_]*([.][0-9][0-9_]*)?(?:[eE][+-]?[0-9_]*[0-9][0-9_]*)?(f32|f64)")]
    FLOAT_NUMBER(&'a str),

    #[regex("//.*", logos::skip)]
    LINE_COMMENT,
    #[regex(r"/\*([^*]|\*[^/])+\*/", logos::skip)]
    BLOCK_COMMENT,
    #[regex("[ \t\r\n]*", logos::skip)]
    WHITESPACE,

    #[error]
    LEXER_ERROR,
}

type Lexer<'a> = Peekable<logos::SpannedIter<'a, Token<'a>>>;
type SpannedToken<'a> = (Token<'a>, Span);

trait LexerExt<'a> {
    #[track_caller]
    fn expect(&mut self, token: Token<'a>) -> Result<SpannedToken<'a>, SyntaxError>;
    #[track_caller]
    fn expect_ident(&mut self) -> Result<&'a str, SyntaxError>;
    fn maybe_token(&mut self, token: Token<'a>) -> bool;
    fn copy_verbatim_until(&mut self, source: &str, token: Token<'a>, out: &mut TokenStream);
}

impl<'a> LexerExt<'a> for Lexer<'a> {
    #[track_caller]
    fn expect(&mut self, token: Token<'a>) -> Result<SpannedToken<'a>, SyntaxError> {
        match self.next() {
            Some((kind, range)) if kind == token => Ok((kind, range)),
            Some((kind, range)) => Err(SyntaxError::new(
                range,
                format!("expected {:?}, found {:?} (at {:?})", token, kind, Location::caller()),
            )),
            None => Err(SyntaxError::eof()),
        }
    }

    #[track_caller]
    fn expect_ident(&mut self) -> Result<&'a str, SyntaxError> {
        match self.next() {
            Some((Token::IDENT(ident), _)) => Ok(ident),
            Some((kind, range)) => Err(SyntaxError::new(
                range,
                format!("expected identifier, found {:?}  (at {:?})", kind, Location::caller()),
            )),
            None => Err(SyntaxError::eof()),
        }
    }

    fn maybe_token(&mut self, token: Token<'a>) -> bool {
        match self.peek() {
            Some((kind, _)) if *kind == token => {
                self.next();
                true
            }
            _ => false,
        }
    }

    fn copy_verbatim_until(&mut self, source: &str, token: Token<'a>, out: &mut TokenStream) {
        while let Some((kind, _)) = self.peek() {
            if kind == &token {
                break;
            }
            let (_, span) = self.next().unwrap();
            out.extend(TokenStream::from_str(&source[span]).unwrap());
        }
    }
}

#[derive(Debug)]
struct SyntaxError {
    range: Range<usize>,
    message: String,
}

impl SyntaxError {
    fn new(range: Range<usize>, message: impl Into<String>) -> Self {
        Self {
            range,
            message: message.into(),
        }
    }

    fn eof() -> Self {
        Self {
            range: 0..0,
            message: "unexpected end of file".to_string(),
        }
    }
}

/// Parses an optional array declarator, like `[]` or `[10]`.
///
/// Returns a tuple of `(is_array, array_len)`, where `is_array` is true if an array declarator was found,
/// and `array_len` is the length of the array if it was specified.
fn parse_array_declarator(source: &str, lexer: &mut Lexer) -> Result<(bool, Option<TokenStream>), SyntaxError> {
    if let Some((L_BRACK, _)) = lexer.peek() {
        lexer.next();
        let mut len = TokenStream::new();
        lexer.copy_verbatim_until(source, R_BRACK, &mut len);
        lexer.expect(R_BRACK)?; // consume ']'
        if len.is_empty() {
            Ok((true, None))
        } else {
            Ok((true, Some(len)))
        }
    } else {
        Ok((false, None))
    }
}

/// Translates a type.
///
/// Examples of supported types:
/// - simple type references: `int32_t`
/// - Texture types with one generic parameter: `Texture2D<float>`
/// - Bindless handle types: `Texture2D<float>.Handle`
/// - Pointer types: `int32_t*`
/// - Array types: `int32_t[10]`
fn translate_type(source: &str, lexer: &mut Lexer) -> Result<TokenStream, SyntaxError> {
    let ident = lexer.expect_ident()?;
    let generic = if let Some((L_ANGLE, _)) = lexer.peek() {
        lexer.next();
        let ty = translate_type(source, lexer)?;
        let (next, range) = lexer.next().ok_or(SyntaxError::eof())?;
        if next != R_ANGLE {
            return Err(SyntaxError::new(range, "expected '>'"));
        }
        Some(ty)
    } else {
        None
    };

    let associated = if let Some((DOT, _)) = lexer.peek() {
        lexer.next();
        let associated = lexer.expect_ident()?;
        Some(associated)
    } else {
        None
    };

    let pointer = if let Some((STAR, _)) = lexer.peek() {
        lexer.next();
        true
    } else {
        false
    };

    let (is_array, array_len) = parse_array_declarator(source, lexer)?;

    let ident = if let Some(assoc) = associated {
        format_ident!("{ident}_{assoc}")
    } else {
        format_ident!("{ident}")
    };

    let mut t = if let Some(generic) = generic {
        quote! { #ident<#generic> }
    } else {
        quote! { #ident }
    };

    if pointer {
        t = quote! { Pointer<#t> }
    };

    if is_array {
        if let Some(array_len) = array_len {
            t = quote! { [ #t ; #array_len as usize ] }
        } else {
            t = quote! { [ #t ] }
        };
    }

    Ok(t)
}

/// Translates a variable declaration or a field declaration.
fn translate_variable(source: &str, lexer: &mut Lexer, is_field: bool) -> Result<TokenStream, SyntaxError> {
    let mut ty = translate_type(source, lexer)?;
    let ident = lexer.expect_ident()?;
    let (is_array, array_len) = parse_array_declarator(source, lexer)?;
    if is_array {
        if let Some(array_len) = array_len {
            ty = quote! { [ #ty ; #array_len as usize ] }
        } else {
            ty = quote! { [ #ty ] }
        }
    }

    let initializer = if lexer.maybe_token(EQ) {
        let mut tokens = TokenStream::new();
        lexer.copy_verbatim_until(source, SEMICOLON, &mut tokens);
        Some(tokens)
    } else {
        None
    };

    lexer.expect(SEMICOLON)?;

    let ident_snake = if is_field {
        ident.to_snake_case()
    } else {
        ident.to_shouty_snake_case()
    };
    let ident_snake = format_ident!("{ident_snake}");

    let initializer = initializer.iter();
    Ok(quote! {
        #ident_snake: #ty #(= #initializer)*
    })
}

/// Translates a struct declaration.
fn translate_struct(source: &str, lexer: &mut Lexer) -> Result<TokenStream, SyntaxError> {
    let _struct = lexer.expect(STRUCT_KW)?;
    let ident = lexer.expect_ident()?;
    let (_l_curly, _) = lexer.expect(L_CURLY)?;
    let mut tokens = TokenStream::new();
    loop {
        match lexer.peek() {
            Some((R_CURLY, _)) => {
                lexer.next();
                break;
            }
            Some(_) => {
                let field = translate_variable(source, lexer, true)?;
                tokens.append_all(quote! {pub #field,});
            }
            None => return Err(SyntaxError::eof()),
        }
    }
    // finish with semicolon
    lexer.expect(SEMICOLON)?;

    let ident = format_ident!("{ident}");
    Ok(quote! {
        #[repr(C)]
        #[derive(Copy,Clone)]
        pub struct #ident {
            #tokens
        }
    })
}

fn write_sanity_checks(out: &mut TokenStream) {
    let q = quote! {
        const _LAYOUT_CHECKS: () = {
            const fn same_layout<A, B>() -> bool {
                let a = std::alloc::Layout::new::<A>();
                let b = std::alloc::Layout::new::<B>();
                (a.size() == b.size()) && (a.align() == b.align())
            }
            assert!(same_layout::<uint, u32>());
            assert!(same_layout::<int, i32>());
            assert!(same_layout::<float, f32>());
            assert!(same_layout::<float2, [f32; 2]>());
            assert!(same_layout::<float3, [f32; 3]>());
            assert!(same_layout::<float4, [f32; 4]>());
            assert!(same_layout::<float4x4, [[f32; 4]; 4]>());
            assert!(same_layout::<uint2, [u32; 2]>());
            assert!(same_layout::<uint3, [u32; 3]>());
            assert!(same_layout::<uint4, [u32; 4]>());
            assert!(same_layout::<int2, [i32; 2]>());
            assert!(same_layout::<int3, [i32; 3]>());
            assert!(same_layout::<int4, [i32; 4]>());
            assert!(same_layout::<Texture2D_Handle<float4>, [u32; 2]>());
            assert!(same_layout::<RWTexture2D_Handle<float4>, [u32; 2]>());
            assert!(same_layout::<SamplerState_Handle, [u32; 2]>());
            assert!(same_layout::<Pointer<float4>, u64>());
        };
    };
    out.extend(q);
}

fn translate_inner(source: &str, lexer: &mut Lexer) -> Result<TokenStream, SyntaxError> {
    let mut tokens = TokenStream::new();
    write_sanity_checks(&mut tokens);
    loop {
        match lexer.peek() {
            Some((STRUCT_KW, _)) => {
                let struct_decl = translate_struct(source, lexer)?;
                tokens.append_all(struct_decl);
            }
            Some(_) => {
                let _is_static = lexer.maybe_token(STATIC_KW);
                let _is_const = lexer.maybe_token(CONST_KW);
                let var = translate_variable(source, lexer, false)?;
                tokens.append_all(quote! {pub const #var;});
            }
            None => break,
        }
    }
    Ok(tokens)
}

struct LineMap {
    lines: Vec<usize>,
}

impl LineMap {
    fn new(source: &str) -> Self {
        let mut lines = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                lines.push(i + 1);
            }
        }
        Self { lines }
    }

    fn line_col(&self, offset: usize) -> (usize, usize) {
        let line = self.lines.binary_search(&offset).unwrap_or_else(|i| i - 1).max(0);
        let col = offset - self.lines[line] + 1;
        (line + 1, col)
    }
}

/// Translates slang declarations to Rust code.
///
/// # Arguments
/// * `file` - file path for error messages
/// * `source` - the contents of the slang source file
///
/// See the module documentation for more details and limitations.
pub fn translate_slang_shared_decls(source_path: &Path, output: &mut dyn io::Write) {
    let source = fs::read_to_string(source_path).unwrap();
    let line_map = LineMap::new(&source);
    let lexer = Token::lexer(&source);
    let mut lexer = lexer.spanned().peekable();
    let source_path_display = source_path.display();
    let tokens = match translate_inner(&source, &mut lexer) {
        Ok(tokens) => tokens,
        Err(err) => {
            let (line, col) = line_map.line_col(err.range.start);
            let message = &err.message;
            let message_fmt = format!("error: {source_path_display}:{line}:{col}: {message}");
            eprintln!("{message_fmt}");
            quote! {
                ::std::compile_error!(#message_fmt);
            }
        }
    };

    output.write(&tokens.to_string().as_bytes()).unwrap();
}
