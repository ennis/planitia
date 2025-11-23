//! JSON geometry format

use std::borrow::Cow;
use crate::houdini::error::Error;
use crate::houdini::parser::{Event, Parser};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ParserState {
    Array,
    Map,
}

pub(crate) struct ParserImpl<'a> {
    data: &'a str,
    state: Vec<ParserState>,
    depth: usize,
}

impl<'a> Parser<'a> for ParserImpl<'a> {
    fn next(&mut self) -> Result<Event<'a>, Error> {
        self.next_event()
    }

    fn eos(&mut self) -> bool {
        self.skip_ws();
        match self.data.chars().next() {
            Some(']' | '}') => true,
            None => true,
            _ => false,
        }
    }

}


impl<'a> ParserImpl<'a> {
    pub(crate) fn new(data: &'a str) -> Self {
        Self {
            data,
            state: Vec::new(),
            depth: 0,
        }
    }

    fn skip_ws(&mut self) {
        self.data = self.data.trim_start_matches(|c: char| c.is_ascii_whitespace());
    }

    fn next_event(&mut self) -> Result<Event<'a>, Error> {
        self.skip_ws();
        let n = match self.data.chars().next() {
            Some('[') => {
                self.data = &self.data[1..];
                self.state.push(ParserState::Array);
                Event::BeginArray
            }
            Some('{') => {
                self.data = &self.data[1..];
                self.state.push(ParserState::Map);
                Event::BeginMap
            }
            Some(']') => {
                self.state.pop().ok_or(Error::Malformed("unbalanced array or map"))?;
                self.data = &self.data[1..];
                Event::EndArray
            }
            Some('}') => {
                self.state.pop().ok_or(Error::Malformed("unbalanced array or map"))?;
                self.data = &self.data[1..];
                Event::EndMap
            }
            Some(',') => {
                // TODO maybe do some basic syntax checking here,
                // but in general we assume that the input is well-formed.
                self.data = &self.data[1..];
                return self.next_event();
            }
            Some(':') => {
                // TODO same as above
                self.data = &self.data[1..];
                return self.next_event();
            }
            Some(_) => {
                let mut des = serde_json::Deserializer::from_str(self.data).into_iter();
                let event = match des.next() {
                    Some(Ok(serde_json::Value::String(value))) => Event::String(Cow::Owned(value)),
                    Some(Ok(serde_json::Value::Number(value))) => {
                        Event::Float(value.as_f64().expect("invalid number in json repr"))
                    }
                    Some(Ok(serde_json::Value::Bool(value))) => Event::Boolean(value),
                    Some(Ok(serde_json::Value::Null)) => {
                        panic!("null");
                    }
                    _ => {
                        panic!("unexpected json value");
                    }
                };
                self.data = &self.data[des.byte_offset()..];
                event
            }
            None => Event::Eof,
        };
        //eprintln!("next: {:?}", n);
        Ok(n)
    }
}
