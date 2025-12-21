mod binary;
mod json;

use super::Error::Malformed;
use super::{
    Attribute, AttributeStorage, BezierBasis, BezierRun, Error, Geo, Group, PolygonRun, PrimRun, PrimRunKind,
    StorageKind, Var,
};
use fixedbitset::FixedBitSet;
use json::ParserImpl;
use log::warn;
use smol_str::SmolStr;
use std::borrow::Cow;
use std::collections::HashMap;
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

    fn as_byte(&self) -> Option<u8> {
        match self {
            Event::Integer(i) => {
                if *i >= 0 && *i <= u8::MAX as i64 {
                    Some(*i as u8)
                } else {
                    None
                }
            }
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
            return Err(Error::Malformed(concat!("expected ", stringify!($pat))));
        };
    };
    ($p:expr, $pat:pat, $msg:literal) => {
        let $pat = $p.next()? else {
            return Err(Error::Malformed($msg));
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
        match self.next()? {
            Event::Integer(i) => Ok(i != 0),
            Event::Boolean(b) => Ok(b),
            _ => Err(Error::Malformed("expected boolean")),
        }
    }

    /// Reads a string value.
    fn str(&mut self) -> Result<Cow<'a, str>, Error> {
        expect!(self, Event::String(s));
        Ok(s.clone())
    }

    /// Reads an owned string value.
    fn owned_str(&mut self) -> Result<String, Error> {
        expect!(self, Event::String(s));
        Ok(s.into_owned())
    }

    /// Reads an integer value.
    fn integer(&mut self) -> Result<i64, Error> {
        expect!(self, e);
        e.as_integer().ok_or(Error::Malformed("expected integer"))
    }

    /// Reads an integer value, with a maximum value of i32::MAX.
    fn int32(&mut self) -> Result<i32, Error> {
        let i = self.integer()?;
        if i > i32::MAX as i64 {
            return Err(Error::Malformed("integer out of range for i32"));
        }
        Ok(i as i32)
    }

    /// Reads an unsigned integer value, with a maximum value of u32::MAX.
    fn uint32(&mut self) -> Result<u32, Error> {
        let i = self.integer()?;
        if i < 0 || i > u32::MAX as i64 {
            return Err(Error::Malformed("integer out of range for u32"));
        }
        Ok(i as u32)
    }

    fn uint8(&mut self) -> Result<u8, Error> {
        let i = self.integer()?;
        if i < 0 || i > u8::MAX as i64 {
            return Err(Error::Malformed("integer out of range for u8"));
        }
        Ok(i as u8)
    }
}

fn read_array<T>(
    p: &mut dyn Parser,
    mut elem_parser: impl FnMut(&mut dyn Parser) -> Result<T, Error>,
) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    expect!(p, Event::BeginArray);
    while !p.eos() {
        v.push(elem_parser(p)?);
    }
    expect!(p, Event::EndArray);
    Ok(v)
}

fn read_byte_array(p: &mut dyn Parser) -> Result<Vec<u8>, Error> {
    read_array(p, |p| p.uint8())
}

fn read_int32_array(p: &mut dyn Parser) -> Result<Vec<i32>, Error> {
    read_array(p, |p| p.int32())
}

fn read_uint32_array(p: &mut dyn Parser) -> Result<Vec<u32>, Error> {
    read_array(p, |p| p.uint32())
}

fn read_float32_array(p: &mut dyn Parser) -> Result<Vec<f32>, Error> {
    read_array(p, |p| match p.next_no_eof()? {
        Event::Float(f) => Ok(f as f32),
        Event::Integer(i) => Ok(i as f32),
        _ => Err(Error::Malformed("expected float")),
    })
}

////////////////////////////////////////////////////////////////////////////////////////////////////

trait AttribValue: Copy + 'static {
    const STORAGE_KIND: StorageKind;
    const ZERO: Self;
    fn read(p: &mut dyn Parser) -> Result<Self, Error>;
}

impl AttribValue for f32 {
    const STORAGE_KIND: StorageKind = StorageKind::FpReal32;
    const ZERO: Self = 0.0;
    fn read(p: &mut dyn Parser) -> Result<Self, Error> {
        match p.next_no_eof()? {
            Event::Float(f) => Ok(f as f32),
            Event::Integer(i) => Ok(i as f32),
            _ => Err(Error::Malformed("expected float")),
        }
    }
}

impl AttribValue for f64 {
    const STORAGE_KIND: StorageKind = StorageKind::FpReal64;
    const ZERO: Self = 0.0;
    fn read(p: &mut dyn Parser) -> Result<Self, Error> {
        match p.next_no_eof()? {
            Event::Float(f) => Ok(f),
            Event::Integer(i) => Ok(i as f64),
            _ => Err(Error::Malformed("expected float")),
        }
    }
}

impl AttribValue for i32 {
    const STORAGE_KIND: StorageKind = StorageKind::Int32;
    const ZERO: Self = 0;
    fn read(p: &mut dyn Parser) -> Result<Self, Error> {
        match p.next_no_eof()? {
            Event::Integer(i) => Ok(i as i32),
            Event::Float(f) => Ok(f as i32),
            _ => Err(Error::Malformed("expected integer")),
        }
    }
}

impl AttribValue for i64 {
    const STORAGE_KIND: StorageKind = StorageKind::Int64;
    const ZERO: Self = 0;
    fn read(p: &mut dyn Parser) -> Result<Self, Error> {
        match p.next_no_eof()? {
            Event::Integer(i) => Ok(i),
            Event::Float(f) => Ok(f as i64),
            _ => Err(Error::Malformed("expected integer")),
        }
    }
}

impl StorageKind {
    fn parse(s: &str) -> Result<StorageKind, Error> {
        match s {
            "fpreal32" => Ok(StorageKind::FpReal32),
            "fpreal64" => Ok(StorageKind::FpReal64),
            "int32" => Ok(StorageKind::Int32),
            "int64" => Ok(StorageKind::Int64),
            _ => Err(Malformed("unknown storage kind")),
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
            StorageKind::String => AttributeStorage::Strings {
                values: Vec::new(),
                indices: Vec::new(),
            }
        }
    }
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

fn read_rawpagedata_generic<T: AttribValue>(
    p: &mut dyn Parser,
    storage: &mut Vec<T>,
    pagesize: usize,
    packing: &[i32],
    constantpageflags: &[Vec<i32>],
) -> Result<(), Error> {
    // Packing = [3,1] => PackOffsets = [0,3]
    // Packing = [4]   => PackOffsets = [0]
    // Packing = [1,1,1,1] => PackOffsets = [0,1,2,3]
    let packoffsets = packing
        .iter()
        .scan(0, |acc, &size| {
            let offset = *acc;
            *acc += size as usize;
            Some(offset)
        })
        .collect::<Vec<_>>();

    // (X, Y, Z) => 3
    // (X, Y, Z, W) => 4
    let tuplesize: usize = packing.iter().map(|&s| s as usize).sum();

    // unfortunately, we have to read the whole raw data into storage first: due to the way it
    // is packed, we need to know in advance the size of the raw data to unpack it correctly,
    // so we can't unpack it on the fly.
    let mut raw = Vec::new();
    expect!(p, Event::BeginArray);
    while !p.eos() {
        raw.push(T::read(p)?);
    }
    expect!(p, Event::EndArray);

    // ... and we can't determine the size of storage in advance either, because of constant pages.
    // Assume no constant pages for now, we'll resize later.
    storage.resize(raw.len(), T::ZERO);

    // base offset of current page in target storage
    let mut base = 0;
    // raw data read pointer
    let mut ptr = 0;
    let len = raw.len();

    eprintln!("constantpageflags {:?}", constantpageflags);

    let mut pageindex = 0;
    loop {
        if ptr >= len {
            break;
        }

        // cur_pagesize == number of tuples in the current page
        // this depends on whether we're at the last page and constant flags for the page

        let mut a = 0; // size in elements of constant packs
        let mut b = 0; // size in elements of varying packs; number of elements in page == a + cur_pagesize * b
        for (packindex, packsize) in packing.iter().enumerate() {
            let constant = !constantpageflags.is_empty()
                && !constantpageflags[packindex].is_empty()
                && constantpageflags[packindex][pageindex] == 1;
            if constant {
                a += *packsize as usize;
            } else {
                b += *packsize as usize;
            }
        }

        // remaining elements in raw data
        let remaining = len - ptr;
        let tuples_in_page = if remaining < a + b * pagesize {
            // not enough data for a full page, compute cur_pagesize accordingly
            (remaining - a) / b
        } else {
            pagesize
        };
        eprintln!("tuples_in_page: {tuples_in_page}, ptr={ptr}, len={len}");

        // number of elements represented by the current page
        let num_elements_in_page = tuples_in_page * tuplesize;

        // ensure that the storage is large enough to receive the current page
        if storage.len() < base + num_elements_in_page {
            storage.resize(base + num_elements_in_page, T::ZERO);
        }

        // read each packed subvector in the page
        for (packindex, packsize) in packing.iter().enumerate() {
            let packsize = *packsize as usize;
            let mut off = packoffsets[packindex];

            let constant = !constantpageflags.is_empty()
                && !constantpageflags[packindex].is_empty()
                && constantpageflags[packindex][pageindex] == 1;

            for _ in 0..tuples_in_page {
                storage[base + off..base + off + packsize].copy_from_slice(&raw[ptr..ptr + packsize]);

                off += tuplesize;
                if !constant {
                    ptr += packsize;
                }
            }
            if constant {
                ptr += packsize;
            }
        }

        pageindex += 1;
        base += num_elements_in_page;
    }

    Ok(())
}

fn read_rawpagedata(
    p: &mut dyn Parser,
    storage: &mut AttributeStorage,
    size: usize,
    pagesize: usize,
    packing: &[i32],
    constantpageflags: &[Vec<i32>],
) -> Result<(), Error> {
    match storage {
        AttributeStorage::FpReal32(v) => {
            v.resize(size, 0.0);
            read_rawpagedata_generic::<f32>(p, v, pagesize, packing, constantpageflags)?;
        }
        AttributeStorage::FpReal64(v) => {
            v.resize(size, 0.0);
            read_rawpagedata_generic::<f64>(p, v, pagesize, packing, constantpageflags)?;
        }
        AttributeStorage::Int32(v) => {
            v.resize(size, 0);
            read_rawpagedata_generic::<i32>(p, v, pagesize, packing, constantpageflags)?;
        }
        AttributeStorage::Int64(v) => {
            v.resize(size, 0);
            read_rawpagedata_generic::<i64>(p, v, pagesize, packing, constantpageflags)?;
        }
        AttributeStorage::Strings { .. } => {
            panic!("string attributes cannot use rawpagedata");
        }
    }
    Ok(())
}

fn read_arrays_generic<T: AttribValue>(p: &mut dyn Parser, storage: &mut Vec<T>) -> Result<(), Error> {
    read_array! {p =>
        read_array! {p =>
            storage.push(T::read(p)?)
        }
    };
    Ok(())
}

fn read_arrays(p: &mut dyn Parser, storage: &mut AttributeStorage) -> Result<(), Error> {
    match storage {
        AttributeStorage::FpReal32(v) => {
            read_arrays_generic::<f32>(p, v)?;
        }
        AttributeStorage::FpReal64(v) => {
            read_arrays_generic::<f64>(p, v)?;
        }
        AttributeStorage::Int32(v) => {
            read_arrays_generic::<i32>(p, v)?;
        }
        AttributeStorage::Int64(v) => {
            read_arrays_generic::<i64>(p, v)?;
        }
        AttributeStorage::Strings { .. } => {
            panic!("string attributes cannot use arrays");
        }
    }
    Ok(())
}

fn read_point_attribute(p: &mut dyn Parser) -> Result<Attribute, Error> {
    let mut name = SmolStr::default();
    let mut storage = None;
    let mut size = 0;
    let mut storage_kind = StorageKind::Int32;
    let mut packing = vec![];
    let mut pagesize = 0;
    let mut constantpageflags: Vec<Vec<i32>> = vec![];
    let mut strings = vec![];

    #[derive(Copy, Clone, Eq, PartialEq)]
    enum AttributeType {
        Numeric,
        String,
    }
    let mut attribute_type = AttributeType::Numeric;

    //eprintln!("read_point_attribute metadata");
    //eprintln!("read_point_attribute data");

    expect!(p, Event::BeginArray);

    read_kv_array! {p,
        "name" => {
            name = p.str()?.into();
        }
        "type" => {
            let ty = p.str()?;
            match ty.as_ref() {
                "string" => {
                    attribute_type = AttributeType::String;
                }
                "numeric" => {
                    attribute_type = AttributeType::Numeric;
                }
                _ => {
                    return Err(Malformed("unknown attribute type"));
                }
            }
        }
    }

    read_kv_array! {p,
        "name" => {
            name = p.str()?.into();
        }
        "strings" => {
            strings = read_array(p, |p| p.str().map(SmolStr::new))?;
        }
        "values" | "indices" => {
            read_kv_array!(p,
                "size" => {
                    size = p.integer()? as usize;
                }
                "storage" => {
                    storage_kind = StorageKind::parse(&p.str()?)?;
                }
                "packing" => {
                    packing = read_int32_array(p)?;
                }
                "pagesize" => {
                    pagesize = p.integer()? as usize;
                }
                "arrays" => {
                    storage = Some(AttributeStorage::new(storage_kind));
                    read_arrays(p, storage.as_mut().ok_or(Malformed("expected arrays"))?)?;
                }
                "tuples" => {
                    storage = Some(AttributeStorage::new(storage_kind));
                    read_arrays(p, storage.as_mut().ok_or(Malformed("expected tuples"))?)?;
                }
                "constantpageflags" => {
                    read_array!(p => {
                        let mut subvector_flags = Vec::new();
                        read_array!(p => {
                            subvector_flags.push(p.boolean()? as i32);
                        });
                        constantpageflags.push(subvector_flags);
                    });
                }
                "rawpagedata" => {
                    storage = Some(AttributeStorage::new(storage_kind));
                    if packing.is_empty() {
                        packing = vec![size as i32];
                    }
                    read_rawpagedata(
                        p,
                        storage.as_mut().ok_or(Malformed("expected rawpagedata"))?,
                        size,
                        pagesize,
                        &packing,
                        &constantpageflags,
                    )?;
                }
            );
        }
    }

    expect!(p, Event::EndArray, "expected end of attribute array");

    let Some(mut storage) = storage else {
        return Err(Malformed("no storage data"));
    };

    // handle string attributes, in which case storage contains indices into the strings array
    if attribute_type == AttributeType::String {
        let indices = match storage {
            AttributeStorage::Int32(v) => v,
            AttributeStorage::Int64(v) => v.iter().map(|&x| x as i32).collect(),
            _ => return Err(Malformed("invalid storage type for string attribute")),
        };
        storage = AttributeStorage::Strings {
            values: strings,
            indices,
        };
    }

    Ok(Attribute { name, size, storage })
}

enum PrimType {
    Run,
    PolygonCurveRun,
    PolygonRun,
    BezierRun,
    Unknown(String),
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
            knots = read_float32_array(p)?;
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
                    r.vertices = Var::Uniform(read_int32_array(p)?);
                }
                "closed" => {
                    r.closed = Var::Uniform(p.boolean()?);
                }
                "basis" => {
                    r.basis = Var::Uniform(read_bezier_basis(p)?);
                }
            },
        }
        Ok(())
    }

    fn read_varying_fields(&mut self, fields: &[String], p: &mut dyn Parser) -> Result<(), Error> {
        match self {
            PrimitiveRun::BezierRun(r) => {}
        }
        Ok(())
    }
}

fn read_bezier_run_data(
    p: &mut dyn Parser,
    fields: &[String],
    uniform_fields: &UniformFields,
) -> Result<BezierRun, Error> {
    let mut r = BezierRun::default();

    if let Some(vertices) = &uniform_fields.vertices {
        r.vertices = Var::Uniform(vertices.clone());
    }
    if let Some(closed) = uniform_fields.closed {
        r.closed = Var::Uniform(closed);
    }
    if let Some(basis) = &uniform_fields.basis {
        r.basis = Var::Uniform(basis.clone());
    }

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
        r.vertices = Var::Varying(vertices);
    }
    if !closed.is_empty() {
        r.closed = Var::Varying(closed);
    }
    if !basis.is_empty() {
        r.basis = Var::Varying(basis);
    }
    Ok(r)
}

fn decode_rle_array(p: &mut dyn Parser) -> Result<Var<i32>, Error> {
    let mut pairs = read_int32_array(p)?;
    if pairs.len() % 2 != 0 {
        return Err(Malformed("expected even number of RLE values"));
    }
    if pairs.len() == 2 {
        return Ok(Var::Uniform(pairs[0]));
    } else {
        let mut values = Vec::new();
        let mut i = 0;
        while i < pairs.len() {
            let value = pairs[i];
            let count = pairs[i + 1] as usize;
            for _ in 0..count {
                values.push(value);
            }
            i += 2;
        }
        Ok(Var::Varying(values))
    }
}

fn read_group(p: &mut dyn Parser, _prim_count_hint: usize) -> Result<(SmolStr, Group), Error> {
    expect!(p, Event::BeginArray);

    let mut name = SmolStr::default();
    let mut group = Group::default();

    read_kv_array! {p,
        "name" => {
            name = SmolStr::new(p.str()?);
        }
    }

    read_kv_array! {p,
        "selection" => {
            read_kv_array! {p,
                "unordered" => {
                    read_kv_array!{p,
                        "boolRLE" => {
                            let mut index = 0;
                            let mut bool_rle = Vec::new();
                            read_array!{p =>
                                {
                                    let run_length = p.uint32()?;
                                    let value = p.boolean()?;
                                    if value {
                                        bool_rle.push(index..(index+run_length));
                                    }
                                    index += run_length;
                                }
                            }
                            group = Group::RunLengthEncoded(bool_rle);
                        }
                        "i8" => {
                            let bitmap = read_byte_array(p)?;
                            let mut bitset = FixedBitSet::with_capacity(bitmap.len() * 8);
                            for (i, byte) in bitmap.iter().enumerate() {
                                for bit in 0..8 {
                                    let prim_index = i * 8 + bit;
                                    if (byte & (1 << bit)) != 0 {
                                        bitset.insert(prim_index);
                                    }
                                }
                            }
                            group = Group::Bitset(bitset);
                        }
                    }
                }
            }
        }
    }

    expect!(p, Event::EndArray);

    Ok((name, group))
}

#[derive(Default)]
struct PolyRunData {
    start_vertex: u32,
    count: u32,
    vertex_counts: Var<i32>,
}

fn read_poly_run_data(p: &mut dyn Parser) -> Result<PolyRunData, Error> {
    let mut r = PolyRunData::default();
    read_kv_array! {p,
        "startvertex" | "s_v" => {
            r.start_vertex = p.integer()? as u32;
        }
        "nprimitives" | "n_p" => {
            r.count = p.uint32()?;
        }
        "nvertices" | "n_v" => {
            r.vertex_counts = Var::Varying(read_int32_array(p)?);
        }
        "nvertices_rle" | "r_v" => {
            r.vertex_counts = decode_rle_array(p)?;
        }
    }
    Ok(r)
}

#[derive(Default)]
struct UniformFields {
    /// Curve vertices
    vertices: Option<Vec<i32>>,
    /// Curve basis
    basis: Option<BezierBasis>,
    /// Whether the curve is closed
    closed: Option<bool>,
}

fn read_uniform_fields(p: &mut dyn Parser) -> Result<UniformFields, Error> {
    let mut r = UniformFields::default();
    read_map! {p,
        "vertex" => {
            r.vertices = Some(read_int32_array(p)?);
        }
        "closed" => {
            r.closed = Some(p.boolean()?);
        }
        "basis" => {
            r.basis = Some(read_bezier_basis(p)?);
        }
    }
    Ok(r)
}

fn read_primitives(p: &mut dyn Parser, geo: &mut Geo) -> Result<(), Error> {
    let mut base_prim_index: u32 = 0;

    read_array! {p => {
        expect!(p, Event::BeginArray);
        let mut prim_type = None;

        let mut varyingfields = Vec::new();
        let mut uniformfields = UniformFields::default();

        read_kv_array! {p,
            "type" => {
                match p.str()?.as_ref() {
                    "run" => prim_type = Some(PrimType::Run),
                    "PolygonCurve_run" | "c_r" => prim_type = Some(PrimType::PolygonCurveRun),
                    "Polygon_run" | "p_r" => prim_type = Some(PrimType::PolygonRun),
                    other => {
                        prim_type = Some(PrimType::Unknown(other.to_string()));
                    }
                }
            }
            "runtype" => {
                match p.str()?.as_ref() {
                    "BezierCurve" => prim_type = Some(PrimType::BezierRun),
                    _ => {}
                }
            }
            "varyingfields" => {
                read_array!(p => varyingfields.push(p.str()?.to_string()));
            }
            "uniformfields" => {
                uniformfields = read_uniform_fields(p)?;
            }
        }

        match prim_type {
            Some(PrimType::PolygonRun) => {
                let rundata = read_poly_run_data(p)?;
                geo.prim_runs.push(PrimRun {
                    count: rundata.count,
                    base_index: base_prim_index,
                    kind: PrimRunKind::PolygonRun(PolygonRun {
                        start_vertex: rundata.start_vertex,
                        vertex_counts: rundata.vertex_counts,
                        count: rundata.count,
                        base_prim_index,
                        closed: true,
                    })
                });
            }
            Some(PrimType::PolygonCurveRun) => {
                let rundata = read_poly_run_data(p)?;
                geo.prim_runs.push(PrimRun {
                    count: rundata.count,
                    base_index: base_prim_index,
                    kind: PrimRunKind::PolygonRun(PolygonRun {
                        start_vertex: rundata.start_vertex,
                        vertex_counts: rundata.vertex_counts,
                        count: rundata.count,
                        base_prim_index,
                        closed: false,
                    })
                });
            }
            Some(PrimType::BezierRun) => {
                let rundata = read_bezier_run_data(p, &varyingfields, &uniformfields)?;
                geo.prim_runs.push(PrimRun {
                    count: rundata.count,
                    base_index: base_prim_index,
                    kind: PrimRunKind::BezierRun(rundata),
                });
            }
            Some(PrimType::Unknown(ty)) => {
                warn!("not loading unknown primitive type: {ty}");
                p.skip();
            }
            _ => {
                return Err(Malformed("primitive type not specified"));
            }
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
        "vertexattributes" => {
            read_array!(p => {
                geo.vertex_attributes.push(read_point_attribute(p)?);
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
        "pointgroups" => {
            read_array!(p => {
                let (name, group) = read_group(p, geo.primitive_count)?;
                geo.point_groups.insert(name, group);
            })
        }
        "primitivegroups" => {
            read_array!(p => {
                let (name, group) = read_group(p, geo.primitive_count)?;
                geo.primitive_groups.insert(name, group);
            })
        }
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
    let positions_fp32 = positions.cast::<f32>();
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
