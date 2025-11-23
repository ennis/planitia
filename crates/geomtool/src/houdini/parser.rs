mod binary;
mod json;

use super::Error::Malformed;
use super::{Attribute, AttributeStorage, BezierBasis, BezierRun, Error, Geo, PrimVar, Primitive, StorageKind};
use json::ParserImpl;
use smol_str::SmolStr;
use std::borrow::Cow;
use std::marker::PhantomData;
use std::path::Path;
use std::{fs, ptr};

#[derive(Clone, Debug)]
struct PackedArray<'a, T: Copy> {
    data: &'a [u8],
    _phantom: PhantomData<T>,
}

impl<'a, T: Copy> PackedArray<'a, T> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }

    fn iter(&self) -> impl Iterator<Item = T> + 'a {
        let size = size_of::<T>();
        self.data
            .chunks_exact(size)
            .map(|chunk| unsafe { ptr::read_unaligned(chunk.as_ptr() as *const T) })
    }
}

/// Events produced by the parser implementation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum Event<'a> {
    Integer(i64),
    Float(f64),
    String(Cow<'a, str>),
    Boolean(bool),
    BeginArray,
    EndArray,
    BeginMap,
    EndMap,
    Eof,
    // binary JSON doesn't impose alignment restrictions on arrays, so we can't use slices directly
    //UABool(PackedArray<'a, u8>),
    //UAInt8(PackedArray<'a, u8>),
    //UAInt16(PackedArray<'a, u16>),
    //UAInt32(PackedArray<'a, i32>),
    //UAInt64(PackedArray<'a, i64>),
    //UAReal32(PackedArray<'a, f32>),
    //UAReal64(PackedArray<'a, f64>),
}

impl<'a> Event<'a> {
    fn as_integer(&self) -> Option<i64> {
        match self {
            Event::Integer(i) => Some(*i),
            Event::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    fn as_float(&self) -> Option<f64> {
        match self {
            Event::Integer(i) => Some(*i as f64),
            Event::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Event::String(s) => Some(s),
            _ => None,
        }
    }
}

macro_rules! expect {
    ($p:expr, $pat:pat) => {
        let $pat = $p.next()? else {
            return Err(Error::Malformed(""));
        };
    };
}

trait Parser<'a> {
    fn next(&mut self) -> Result<Event<'a>, Error>;

    fn next_no_eof(&mut self) -> Result<Event<'a>, Error> {
        let e = self.next()?;
        if let Event::Eof = e {
            return Err(Error::Eof);
        }
        Ok(e)
    }

    /// Returns whether the next token is an end-of-structure (end of the current array or map).
    fn eos(&mut self) -> bool;

    /// Skips the next value. If the next event is the start of an array or map, skips the entire
    /// array or map.
    fn skip(&mut self) {
        let mut depth = 0;
        while let Ok(e) = self.next() {
            match e {
                Event::BeginArray | Event::BeginMap => {
                    depth += 1;
                    //eprintln!("skip: begin array/map {depth}");
                }
                Event::EndArray | Event::EndMap => {
                    //eprintln!("skip: end array/map {depth}");
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                Event::Eof => break,
                _ => {
                    if depth == 0 {
                        return;
                    }
                }
            }
        }
    }

    /// Reads a boolean value.
    fn boolean(&mut self) -> Result<bool, Error> {
        expect!(self, Event::Boolean(b));
        Ok(b)
    }

    /// Reads a string value.
    fn str(&mut self) -> Result<Cow<'a, str>, Error> {
        expect!(self, Event::String(s));
        Ok(s.clone())
    }

    /// Reads an integer value.
    fn integer(&mut self) -> Result<i64, Error> {
        expect!(self, e);
        e.as_integer().ok_or(Error::Malformed("expected integer"))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl StorageKind {
    fn parse(s: &str) -> Result<StorageKind, Error> {
        match s {
            "fpreal32" => Ok(StorageKind::FpReal32),
            "fpreal64" => Ok(StorageKind::FpReal64),
            "int32" => Ok(StorageKind::Int32),
            "int64" => Ok(StorageKind::Int64),
            _ => Err(Error::Malformed("unknown storage kind")),
        }
    }
}

impl AttributeStorage {
    fn new(storage_kind: StorageKind) -> AttributeStorage {
        match storage_kind {
            StorageKind::FpReal32 => AttributeStorage::FpReal32(Vec::new()),
            StorageKind::FpReal64 => AttributeStorage::FpReal64(Vec::new()),
            StorageKind::Int32 => AttributeStorage::Int32(Vec::new()),
            StorageKind::Int64 => AttributeStorage::Int64(Vec::new()),
        }
    }

    fn read_element(&mut self, e: &mut dyn Parser<'_>) -> Result<(), Error> {
        //eprintln!("read_element");
        match e.next()? {
            Event::Float(f) => match self {
                AttributeStorage::FpReal32(v) => v.push(f as f32),
                AttributeStorage::FpReal64(v) => v.push(f as f64),
                AttributeStorage::Int32(v) => v.push(f as i32),
                AttributeStorage::Int64(v) => v.push(f as i64),
            },
            Event::Integer(i) => match self {
                AttributeStorage::FpReal32(v) => v.push(i as f32),
                AttributeStorage::FpReal64(v) => v.push(i as f64),
                AttributeStorage::Int32(v) => v.push(i as i32),
                AttributeStorage::Int64(v) => v.push(i),
            },
            _ => {
                return Err(Malformed("invalid attribute element"));
            }
        }
        Ok(())
    }
}

pub(crate) fn read_int32_array(p: &mut dyn Parser) -> Result<Vec<i32>, Error> {
    let mut v = Vec::new();
    expect!(p, Event::BeginArray);
    loop {
        match p.next_no_eof()? {
            Event::EndArray => break,
            e => {
                v.push(e.as_integer().ok_or(Error::Malformed("expected integer"))? as i32);
            }
        }
    }
    Ok(v)
}

pub(crate) fn read_fp32_array(p: &mut dyn Parser) -> Result<Vec<f32>, Error> {
    let mut v = Vec::new();
    expect!(p, Event::BeginArray);
    loop {
        match p.next_no_eof()? {
            Event::EndArray => break,
            e => {
                v.push(e.as_float().ok_or(Error::Malformed("expected float"))? as f32);
            }
        }
    }
    Ok(v)
}

macro_rules! read_kv_array {
    ($p:ident, $($key:pat => $b:block)*) => {
        {
            expect!($p, Event::BeginArray);
            loop {
                match $p.next_no_eof()? {
                    Event::EndArray => break,
                    Event::String(key) => {
                        match key.as_ref() {
                            $($key => $b)*
                            _ => {$p.skip();}
                        }
                    }
                    _ => return Err(Error::Malformed("expected string key")),
                }
            }
        }
    };
}

macro_rules! read_map {
    ($p:ident, $($key:pat => $b:block)*) => {
        {
            expect!($p, Event::BeginMap);
            loop {
                match $p.next_no_eof()? {
                    Event::EndMap => break,
                    Event::String(key) => {
                        match key.as_ref() {
                            $($key => $b)*
                            _ => {$p.skip();}
                        }
                    }
                    _ => return Err(Error::Malformed("expected string key")),
                }
            }
        }
    };
}

macro_rules! read_array {
    ($p:ident => $b:expr) => {{
        expect!($p, Event::BeginArray);
        while !$p.eos() {
            $b;
        }
        expect!($p, Event::EndArray);
    }};
}

fn read_topology(p: &mut dyn Parser, geo: &mut Geo) -> Result<(), Error> {
    read_kv_array! {p,
        "pointref" => {
            read_kv_array! {p,
                "indices" => {
                    read_array!(p => {
                        geo.topology.push(p.integer()? as u32);
                    })
                }
            }
        }
    }
    Ok(())
}

fn read_point_attribute(p: &mut dyn Parser) -> Result<Attribute, Error> {
    let mut name = SmolStr::default();
    let mut storage = None;
    let mut size = 0;
    let mut storage_kind = StorageKind::Int32;

    //eprintln!("read_point_attribute metadata");
    //eprintln!("read_point_attribute data");

    expect!(p, Event::BeginArray);

    read_kv_array! {p,
        "name" => {
            name = p.str()?.into();
        }
    }

    read_kv_array! {p,
        "name" => {
            name = p.str()?.into();
        }
        "values" => {
            read_kv_array!(p,
                "size" => {
                    size = p.integer()? as usize;
                }
                "storage" => {
                    storage_kind = StorageKind::parse(&p.str()?)?;
                }
                "arrays" => {
                    storage = Some(AttributeStorage::new(storage_kind));
                    read_array! {p =>
                        read_array! {p =>
                            storage.as_mut().unwrap().read_element(p)?
                        }
                    }
                }
                "tuples" => {
                    storage = Some(AttributeStorage::new(storage_kind));
                    //eprintln!("read tuple data");
                    read_array! { p =>
                        read_array! { p =>
                            storage.as_mut().unwrap().read_element(p)?
                        }
                    }
                }
            );
        }
    }

    expect!(p, Event::EndArray);

    let Some(storage) = storage else {
        return Err(Malformed("no storage data"));
    };
    Ok(Attribute { name, size, storage })
}

enum PrimType {
    Run,
}

fn read_bezier_basis(p: &mut dyn Parser) -> Result<BezierBasis, Error> {
    let mut ty = None;
    let mut order = 3;
    let mut knots = Vec::new();
    read_kv_array! {p,
        "type" => {
            ty = Some(p.str()?.to_string());
        }
        "order" => {
            order = p.integer()? as u32;
        }
        "knots" => {
            knots = read_fp32_array(p)?;
        }
    }
    Ok(BezierBasis { order, knots })
}

enum PrimitiveRun {
    BezierRun(BezierRun),
}

impl PrimitiveRun {
    fn read_uniform_fields(&mut self, p: &mut dyn Parser) -> Result<(), Error> {
        match self {
            PrimitiveRun::BezierRun(r) => read_map! {p,
                "vertex" => {
                    r.vertices = PrimVar::Uniform(read_int32_array(p)?);
                }
                "closed" => {
                    r.closed = PrimVar::Uniform(p.boolean()?);
                }
                "basis" => {
                    r.basis = PrimVar::Uniform(read_bezier_basis(p)?);
                }
            },
        }
        Ok(())
    }

    fn read_varying_fields(&mut self, fields: &[String], p: &mut dyn Parser) -> Result<(), Error> {
        match self {
            PrimitiveRun::BezierRun(r) => {
                let mut vertices = vec![];
                let mut basis = vec![];
                let mut closed = vec![];

                read_array! {p =>
                    // array of primitives
                    {
                        r.count += 1;
                        read_array!{p =>
                            // array of fields in the primitive
                            for f in fields {
                                match f.as_str() {
                                    "vertex" => {
                                        vertices.push(read_int32_array(p)?);
                                    }
                                    "closed" => {
                                        closed.push(p.boolean()?);
                                    }
                                    "basis" => {
                                        basis.push(read_bezier_basis(p)?);
                                    }
                                    _ => {
                                        p.skip();
                                    }
                                }
                            }
                        }
                    }
                }

                if !vertices.is_empty() {
                    r.vertices = PrimVar::Varying(vertices);
                }
                if !closed.is_empty() {
                    r.closed = PrimVar::Varying(closed);
                }
                if !basis.is_empty() {
                    r.basis = PrimVar::Varying(basis);
                }
            }
        }
        Ok(())
    }
}

fn read_primitives(p: &mut dyn Parser, geo: &mut Geo) -> Result<(), Error> {
    read_array! {p => {
        expect!(p, Event::BeginArray);
        let mut prim_type = None;
        let mut varying_fields = Vec::new();
        let mut primitive_run = None;

        read_kv_array! {p,
            "type" => {
                match p.str()?.as_ref() {
                    "run" => prim_type = Some(PrimType::Run),
                    _ => {}
                }
            }
            "runtype" => {
                match p.str()?.as_ref() {
                    "BezierCurve" => primitive_run = Some(PrimitiveRun::BezierRun(BezierRun::default())),
                    _ => {}
                }
            }
            "varyingfields" => {
                read_array!(p => varying_fields.push(p.str()?.to_string()));
            }
            "uniformfields" => {
                let primitive_run = primitive_run.as_mut().ok_or(Malformed(""))?;
                primitive_run.read_uniform_fields(p)?;
            }
        }

        {
            let primitive_run = primitive_run.as_mut().ok_or(Malformed(""))?;
            primitive_run.read_varying_fields(&varying_fields, p)?;
        }

        match primitive_run {
            Some(PrimitiveRun::BezierRun(r)) => {
                geo.primitives.push(Primitive::BezierRun(r));
            }
            _ => {}
        }

        expect!(p, Event::EndArray);
    }}
    Ok(())
}

fn read_attributes(p: &mut dyn Parser, geo: &mut Geo) -> Result<(), Error> {
    read_kv_array! {p,
        "pointattributes" => {
            read_array!(p => {
                geo.point_attributes.push(read_point_attribute(p)?);
            })
        }
        "primitiveattributes" => {
            read_array!(p => {
                geo.primitive_attributes.push(read_point_attribute(p)?);
            })
        }
    }
    Ok(())
}

fn read_file(p: &mut dyn Parser) -> Result<Geo, Error> {
    let mut geo = Geo::default();

    read_kv_array! {p,
        "pointcount" => { geo.point_count = p.integer()? as usize}
        "vertexcount" => {geo.vertex_count = p.integer()? as usize}
        "primitivecount" =>{ geo.primitive_count = p.integer()? as usize}
        "topology" => {read_topology(p, &mut geo)?}
        "attributes" => {read_attributes(p, &mut geo)?}
        "primitives" => {read_primitives(p, &mut geo)?}
    }

    // Sanity checks for the position attribute.
    // TODO make this errors instead of panics
    assert!(
        geo.point_attributes.len() > 0,
        "the geometry should contain at least one point attribute"
    );
    let positions = &geo.point_attributes[0];
    assert_eq!(
        positions.name, "P",
        "the first point attribute should be the point position"
    );
    assert_eq!(positions.size, 3, "the position attribute should have 3 components");
    assert!(
        positions.as_f32_slice().is_some(),
        "the position attribute should be fpreal32"
    );
    let positions_fp32 = positions.as_f32_slice().unwrap();
    assert!(
        positions_fp32.len() % 3 == 0,
        "the number of positions should be a multiple of 3"
    );
    assert_eq!(positions_fp32.len(), geo.point_count * 3);

    Ok(geo)
}

impl Geo {
    /// Loads houdini geometry data from a byte slice.
    ///
    /// This supports both the JSON-based format (".geo"), or the binary format (".bgeo").
    pub fn from_bytes(data: &[u8]) -> Result<Geo, Error> {
        let mut _binary_parser;
        let mut _json_parser;
        let parser: &mut dyn Parser;

        // check for binary marker (0x7F)
        if data.get(0) == Some(&0x7F) {
            _binary_parser = binary::ParserImpl::new(data);
            parser = &mut _binary_parser;
        } else {
            let str = std::str::from_utf8(data).map_err(|_| Malformed("invalid UTF-8"))?;
            _json_parser = ParserImpl::new(str);
            parser = &mut _json_parser;
        }

        let geo = read_file(parser)?;
        Ok(geo)
    }

    /// Loads a houdini geometry file (`.geo` or `.bgeo`).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Geo, Error> {
        let data = fs::read(path)?;
        Geo::from_bytes(&data)
    }
}
