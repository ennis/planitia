use std::cell::RefCell;
use std::io::Write;

struct ProfilerScope {
    name: &'static str,
    location: &'static std::panic::Location<'static>,
}

struct ProfilerState {
    scopes: Vec<ProfilerScope>,
}

thread_local! {
    static PROF_STREAM: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}


const PROF_TAG_SCOPE_ENTER: u8 = '(' as u8;
const PROF_TAG_SCOPE_EXIT: u8 = ')' as u8;
const PROF_TAG_STRING_DATA: u8 = '#' as u8;
const PROF_TAG_FRAME_MARKER: u8 = 'F' as u8;

fn prof_write_string_data(stream: &mut Vec<u8>, s: &str) {
    stream.write(&[PROF_TAG_STRING_DATA]).unwrap();
    stream.write(&(s.len() as u32).to_le_bytes()).unwrap();
    stream.write(s.as_bytes()).unwrap();
}

#[doc(hidden)]
pub fn profiler_scope_enter(name: &'static str) {
    let time_ns = std::time::Instant::now().elapsed().as_nanos();
    PROF_STREAM.with(|stream| {
        let mut stream = &mut *stream.borrow_mut();
        stream.write(&[PROF_TAG_SCOPE_ENTER]).unwrap();
        stream.write(&time_ns.to_le_bytes()).unwrap();
        prof_write_string_data(stream, name);
    });
}

#[doc(hidden)]
pub fn profiler_scope_exit() {
    PROF_STREAM.with(|stream| {
        let mut stream = stream.borrow_mut();
        stream.write(&[PROF_TAG_SCOPE_EXIT]).unwrap();
        let time_ns = std::time::Instant::now().elapsed().as_nanos();
        stream.write(&time_ns.to_le_bytes()).unwrap();
    });
}

//--------------------------------------------------------------
pub enum ProfilerEvent<'a> {
    ScopeEnter { name: &'a str, time_ns: u128 },
    ScopeExit { time_ns: u128 },
    FrameMarker { time_ns: u128 },
}

fn read_u128(stream: &mut &[u8]) -> Option<u128> {
    if stream.len() < 16 {
        return None;
    }
    let value = u128::from_le_bytes(stream[..16].try_into().unwrap());
    *stream = &stream[16..];
    Some(value)
}

fn read_tag(stream: &mut &[u8]) -> Option<u8> {
    if stream.is_empty() {
        return None;
    }
    let tag = stream[0];
    *stream = &stream[1..];
    Some(tag)
}

fn read_str<'a>(stream: &mut &'a [u8]) -> Option<&'a str> {
    if stream.len() < 4 {
        return None;
    }
    let len = u32::from_le_bytes(stream[..4].try_into().unwrap()) as usize;
    *stream = &stream[4..];
    if stream.len() < len {
        return None;
    }
    let s = std::str::from_utf8(&stream[..len]).ok()?;
    *stream = &stream[len..];
    Some(s)
}

pub struct ProfilerStream {
    data: Vec<u8>,
}

impl ProfilerStream {
    pub fn iter(&self) -> impl Iterator<Item = ProfilerEvent<'_> > + '_ {
        struct Iter<'a> {
            data: &'a [u8],
        }

        impl<'a> Iterator for Iter<'a> {
            type Item = ProfilerEvent<'a>;
            fn next(&mut self) -> Option<Self::Item> {
                if self.data.is_empty() {
                    return None;
                }

                let tag = read_tag(&mut self.data)?;
                match tag {
                    PROF_TAG_SCOPE_ENTER => {
                        let time_ns = read_u128(&mut self.data)?;
                        let str_tag = read_tag(&mut self.data)?;
                        if str_tag != PROF_TAG_STRING_DATA {
                            return None; // Expected string data tag
                        }
                        let name = read_str(&mut self.data)?;
                        Some(ProfilerEvent::ScopeEnter { name, time_ns })
                    }
                    PROF_TAG_SCOPE_EXIT => {
                        let time_ns = read_u128(&mut self.data)?;
                        Some(ProfilerEvent::ScopeExit { time_ns })
                    }
                    PROF_TAG_FRAME_MARKER => {
                        let time_ns = read_u128(&mut self.data)?;
                        Some(ProfilerEvent::FrameMarker { time_ns })
                    }
                    _ => None, // Unknown tag
                }
            }
        }
        Iter { data: &self.data }
    }
}

pub struct RangeStats {
    pub total_time_ns: u128,
    pub count: u32,
}

/// Per-scope statistics.
pub struct ProfilerStats {
    /// How many iterations are accumulated.
    pub iter_count: u32,
}

impl ProfilerStats {
    /// Iterates over all root scopes.
    pub fn root_scopes(&self) {
        // TODO
    }
}