//! Binary geometry format

use crate::houdini::error::Error;
use crate::houdini::parser::binary::State::{Complete, UniformArray};
use crate::houdini::parser::{Event, PackedArray, Parser};
use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum State {
    Start = 0,
    Complete = 1,
    MapStart = 2,
    //MapSep = 3,
    MapNeedValue = 4,
    //MapGotValue = 5,
    MapNeedKey = 6,
    ArrayStart = 7,
    ArrayNeedValue = 8,
    //ArrayGotValue = 9,
    UniformArray,
    //FinishUniformArray,
}

const JID_NULL: i8 = 0x00;
const JID_MAP_BEGIN: i8 = 0x7b;
const JID_MAP_END: i8 = 0x7d;
const JID_ARRAY_BEGIN: i8 = 0x5b;
const JID_ARRAY_END: i8 = 0x5d;
const JID_BOOL: i8 = 0x10;
const JID_INT8: i8 = 0x11;
const JID_INT16: i8 = 0x12;
const JID_INT32: i8 = 0x13;
const JID_INT64: i8 = 0x14;
const JID_REAL16: i8 = 0x18;
const JID_REAL32: i8 = 0x19;
const JID_REAL64: i8 = 0x1a;
const JID_UINT8: i8 = 0x21;
const JID_UINT16: i8 = 0x22;
const JID_STRING: i8 = 0x27;
const JID_FALSE: i8 = 0x30;
const JID_TRUE: i8 = 0x31;
const JID_TOKENDEF: i8 = 0x2b;
const JID_TOKENREF: i8 = 0x26;
const JID_TOKENUNDEF: i8 = 0x2d;
const JID_UNIFORM_ARRAY: i8 = 0x40;
const JID_KEY_SEPARATOR: i8 = 0x3a;
const JID_VALUE_SEPARATOR: i8 = 0x2c;
const JID_MAGIC: i8 = 0x7f;

// 0x4e534a62
const BINARY_MAGIC: [u8; 4] = [b'N', b'S', b'J', b'b'];
// 0x624a534e
const BINARY_MAGIC_SWAP: [u8; 4] = [b'b', b'J', b'S', b'N'];

enum Token {}

pub(crate) struct ParserImpl<'a> {
    data: &'a [u8],
    state: Vec<State>,
    tokens: HashMap<usize, &'a str>,
    depth: usize,
    uniform_type: i8,
    uniform_count_remaining: usize,
    /// For UniformArray(JID_BOOL), current entry of the bitmap
    cur_bits: u32,
    /// How many bits are remaining in `bool_bits`
    rem_bits: u8,
}

impl<'a> ParserImpl<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            state: vec![State::Start],
            tokens: HashMap::new(),
            depth: 0,
            uniform_type: 0,
            uniform_count_remaining: 0,
            cur_bits: 0,
            rem_bits: 0,
        }
    }

    fn read_fixed_bytes<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        if self.data.len() < N {
            return Err(Error::Eof);
        }
        let value = self.data[0..N].try_into().unwrap();
        self.data = &self.data[N..];
        Ok(value)
    }

    fn read_items_as_byte_array<T: Sized + Copy>(&mut self, count: usize) -> Result<&[u8], Error> {
        let byte_count = count * size_of::<T>();
        if self.data.len() < byte_count {
            return Err(Error::Eof);
        }
        let bytes = &self.data[0..byte_count];
        self.data = &self.data[byte_count..];
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, Error> {
        self.read_fixed_bytes::<1>().map(|bytes| bytes[0])
    }

    fn read_i8(&mut self) -> Result<i8, Error> {
        self.read_fixed_bytes::<1>().map(|bytes| bytes[0] as i8)
    }

    fn read_u16(&mut self) -> Result<u16, Error> {
        self.read_fixed_bytes::<2>().map(|bytes| u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32, Error> {
        self.read_fixed_bytes::<4>().map(|bytes| u32::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> Result<f32, Error> {
        self.read_fixed_bytes::<4>()
            .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes)))
    }

    fn read_u64(&mut self) -> Result<u64, Error> {
        self.read_fixed_bytes::<8>().map(|bytes| u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64, Error> {
        self.read_fixed_bytes::<8>().map(|bytes| i64::from_le_bytes(bytes))
    }

    fn read_f64(&mut self) -> Result<f64, Error> {
        self.read_fixed_bytes::<8>()
            .map(|bytes| f64::from_bits(u64::from_le_bytes(bytes)))
    }

    fn read_len(&mut self) -> Result<usize, Error> {
        let n = self.read_u8()?;
        let len = match n {
            0..0xf1 => n as usize,
            0xf2 => self.read_u16().map(|v| v as usize)?,
            0xf4 => self.read_u32().map(|v| v as usize)?,
            0xf8 => self.read_i64().map(|v| v as usize)?,
            _ => {
                return Err(Error::InvalidLengthEncoding);
            }
        };

        Ok(len)
    }

    fn read_str(&mut self) -> Result<&'a str, Error> {
        let len = self.read_len()?;
        if self.data.len() < len {
            return Err(Error::Eof);
        }
        let s = std::str::from_utf8(&self.data[0..len])?;
        self.data = &self.data[len..];
        Ok(s)
    }

    fn read_string_token(&mut self) -> Result<&'a str, Error> {
        let token_index = self.read_len()?;
        self.tokens.get(&token_index).copied().ok_or(Error::InvalidTokenIndex)
    }

    fn define_token(&mut self) -> Result<(), Error> {
        let token_index = self.read_len()?;
        let token_str = self.read_str()?;
        self.tokens.insert(token_index, token_str);
        Ok(())
    }

    fn undefine_token(&mut self) -> Result<(), Error> {
        let token_index = self.read_len()?;
        self.tokens.remove(&token_index);
        Ok(())
    }

    fn read_jid(&mut self) -> Result<i8, Error> {
        let mut token = self.read_i8()?;

        if token == JID_MAGIC {
            let magic = self.read_u32()?;
            if magic != u32::from_le_bytes(BINARY_MAGIC) {
                return Err(Error::InvalidBjsonMagic);
            }
            return self.read_jid();
        }

        loop {
            match token {
                JID_TOKENDEF => {
                    self.define_token()?;
                }
                JID_TOKENUNDEF => {
                    self.undefine_token()?;
                }
                _ => break,
            }
            token = self.read_i8()?;
        }
        Ok(token)
    }

    fn set_state(&mut self, state: State) {
        *self.state.last_mut().unwrap() = state;
    }

    fn read_value(&mut self, jid: i8) -> Result<Event<'a>, Error> {
        let event;
        match jid {
            JID_STRING => {
                eprintln!("JID_STRING");
                let s = self.read_str()?;
                event = Event::String(Cow::Borrowed(s));
            }
            JID_TOKENREF => {
                let s = self.read_string_token()?;
                eprintln!("JID_TOKENREF `{s}`");
                event = Event::String(Cow::Borrowed(s));
            }
            JID_INT8 => {
                eprintln!("JID_INT8");
                let i = self.read_i8()?;
                event = Event::Integer(i as i64);
            }
            JID_INT16 => {
                //eprintln!("JID_INT16");
                let i = self.read_u16()?;
                event = Event::Integer(i as i64);
            }
            JID_INT32 => {
                eprintln!("JID_INT32");
                let i = self.read_u32()?;
                event = Event::Integer(i as i64);
            }
            JID_INT64 => {
                eprintln!("JID_INT64");
                let i = self.read_i64()?;
                event = Event::Integer(i);
            }
            JID_UINT8 => {
                eprintln!("JID_UINT8");
                let u = self.read_u8()?;
                event = Event::Integer(u as i64);
            }
            JID_UINT16 => {
                //eprintln!("JID_UINT16");
                let u = self.read_u16()?;
                event = Event::Integer(u as i64);
            }
            JID_REAL32 => {
                //eprintln!("JID_REAL32");
                let f = self.read_f32()?;
                event = Event::Float(f as f64);
            }
            JID_REAL64 => {
                eprintln!("JID_REAL64");
                let f = self.read_f64()?;
                event = Event::Float(f);
            }
            JID_TRUE => {
                eprintln!("JID_TRUE");
                event = Event::Boolean(true);
            }
            JID_FALSE => {
                eprintln!("JID_FALSE");
                event = Event::Boolean(false);
            }
            JID_BOOL => {
                eprintln!("JID_BOOL");
                let b = self.read_u8()?;
                event = match b {
                    0 => Event::Boolean(false),
                    _ => Event::Boolean(true),
                };
            }
            _ => {
                eprintln!("unexpected jid value: {:0x}", jid);
                return Err(Error::Malformed("unexpected value"))
            },
        }

        Ok(event)
    }

    fn next_event(&mut self) -> Result<Event<'a>, Error> {
        let next_event;

        let state = *self.state.last().unwrap();
        //eprintln!("{:?}", self.state);

        if state == UniformArray {
            if self.uniform_count_remaining == 0 {
                self.state.pop();
                return Ok(Event::EndArray);
            }
            self.uniform_count_remaining -= 1;

            if self.uniform_type == JID_BOOL {
                // read next byte if necessary
                if self.rem_bits == 0 {
                    self.cur_bits = self.read_u32()?;
                    self.rem_bits = 32;
                }
                // read next bit
                let bit = self.cur_bits & 1 != 0;
                self.cur_bits >>= 1;
                self.rem_bits -= 1;
                return Ok(Event::Boolean(bit));
            } else {
                let event = self.read_value(self.uniform_type)?;
                return Ok(event);
            }
        }

        let token = match self.read_jid() {
            Ok(t) => t,
            Err(Error::Eof) => {
                return Ok(Event::Eof);
            }
            Err(e) => {
                return Err(e);
            }
        };

        use State::*;
        match state {
            Start | MapNeedValue | ArrayNeedValue | ArrayStart => match token {

                JID_ARRAY_END => {
                    eprintln!("JID_ARRAY_END");
                    if !matches!(state, ArrayStart | ArrayNeedValue) {
                        return Err(Error::Malformed("unexpected array end"));
                    }
                    self.state
                        .pop()
                        .ok_or(Error::Malformed("unbalanced array delimiters"))?;
                    next_event = Event::EndArray;
                }
                JID_MAP_END => {
                    eprintln!("JID_MAP_END");
                    if state != MapStart {
                        return Err(Error::Malformed("unexpected map end"));
                    }
                    self.state.pop().ok_or(Error::Malformed("unbalanced map delimiters"))?;
                    next_event = Event::EndMap;
                }
                //JID_KEY_SEPARATOR | JID_VALUE_SEPARATOR => {
                //    eprintln!("JID_KEY_SEPARATOR or JID_VALUE_SEPARATOR");
                //    return Err(Error::Malformed("invalid separator token"));
                //}
                JID_STRING | JID_BOOL | JID_NULL | JID_INT8 | JID_INT16 | JID_INT32 | JID_INT64 | JID_UINT8
                | JID_UINT16 | JID_REAL64 | JID_UNIFORM_ARRAY | JID_TRUE | JID_FALSE | JID_TOKENREF | JID_MAP_BEGIN | JID_ARRAY_BEGIN => {
                    match state {
                        Start => {
                            self.set_state(Complete);
                        }
                        MapNeedValue => {
                            self.set_state(MapNeedKey);
                        }
                        ArrayNeedValue | ArrayStart => {
                            self.set_state(ArrayNeedValue);
                        }
                        _ => {}
                    }
                    match token {
                        JID_MAP_BEGIN => {
                            eprintln!("JID_MAP_BEGIN");
                            self.state.push(MapStart);
                            next_event = Event::BeginMap;
                        }
                        JID_ARRAY_BEGIN => {
                            eprintln!("JID_ARRAY_BEGIN");
                            self.state.push(ArrayStart);
                            next_event = Event::BeginArray;
                        }
                        JID_UNIFORM_ARRAY => {
                            self.uniform_type = self.read_i8()?;
                            eprintln!("JID_UNIFORM_ARRAY {:0x}", self.uniform_type);
                            self.uniform_count_remaining = self.read_len()?;
                            self.cur_bits = 0;
                            self.rem_bits = 0;
                            self.state.push(UniformArray);
                            next_event = Event::BeginArray;
                        }
                        _ => {
                            next_event = self.read_value(token)?;
                        }
                    }
                }
                _ => {
                    eprintln!("unexpected token: {:0x}", token);
                    return Err(Error::Malformed("unexpected token"));
                }
            },
            MapStart | MapNeedKey => match token {
                JID_TOKENREF => {
                    let key = self.read_string_token()?;
                    eprintln!("JID_TOKENREF `{key}`");
                    self.set_state(MapNeedValue);
                    next_event = Event::String(Cow::Borrowed(key));
                }
                JID_STRING => {
                    self.set_state(MapNeedValue);
                    let key = self.read_str()?;
                    eprintln!("JID_STRING `{key}`");
                    next_event = Event::String(Cow::Borrowed(self.read_str()?));
                }
                JID_MAP_END => {
                    eprintln!("JID_MAP_END");
                    self.state.pop().ok_or(Error::Malformed("unbalanced map delimiters"))?;
                    next_event = Event::EndMap;
                }
                _ => {
                    eprintln!("unexpected token in map key position: {:0x}", token);
                    return Err(Error::Malformed("expected string key or map end"));
                }
            },
            // after key
            //MapSep => match token {
            //    JID_KEY_SEPARATOR => {
            //        eprintln!("JID_KEY_SEPARATOR");
            //        self.set_state(MapNeedValue);
            //        return self.next_event();
            //    }
            //    _ => {
            //        return Err(Error::Malformed("expected key separator"));
            //    }
            //},
            // after map value, expect either `,` or `}`
            //MapGotValue => match token {
            //    JID_VALUE_SEPARATOR => {
            //        eprintln!("JID_VALUE_SEPARATOR");
            //        self.set_state(MapNeedKey);
            //        return self.next_event();
            //    }
            //    JID_MAP_END => {
            //        eprintln!("JID_MAP_END");
            //        self.state.pop().ok_or(Error::Malformed("unbalanced map delimiters"))?;
            //        next_event = Event::EndMap;
            //    }
            //    _ => {
            //        return Err(Error::Malformed("expected value separator or map end"));
            //    }
            //},
            //ArrayGotValue => match token {
            //    JID_VALUE_SEPARATOR => {
            //        eprintln!("JID_VALUE_SEPARATOR");
            //        self.set_state(ArrayNeedValue);
            //        return self.next_event();
            //    }
            //    JID_ARRAY_END => {
            //        eprintln!("JID_ARRAY_END");
            //        self.state
            //            .pop()
            //            .ok_or(Error::Malformed("unbalanced array delimiters"))?;
            //        next_event = Event::EndArray;
            //    }
            //    _ => {
            //        eprintln!("unexpected token after array value: {:0x}", token);
            //        return Err(Error::Malformed("expected value separator or array end"));
            //    }
            //},
            UniformArray => {
                // handled above
                unreachable!()
            }
            Complete => {
                return Err(Error::Malformed("data after complete value"));
            }
        }

        Ok(next_event)
    }
}

impl<'a> Parser<'a> for ParserImpl<'a> {
    fn next(&mut self) -> Result<Event<'a>, Error> {
        self.next_event()
    }

    fn eos(&mut self) -> bool {
        let state = *self.state.last().unwrap();

        (state == UniformArray && self.uniform_count_remaining == 0)
            ||
        (state != UniformArray
            && match self.data.get(0) {
                Some(x) if *x as i8 == JID_ARRAY_END || *x as i8 == JID_MAP_END => true,
                None => true,
                _ => false,
            })
    }
}
