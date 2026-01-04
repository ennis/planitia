/// Null-terminated UTF-8 string with fixed maximum length.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct ZString<const N: usize>([u8; N]);

impl<const N: usize> ZString<N> {
    pub const MAX_LENGTH: usize = N - 1;

    pub fn len(&self) -> usize {
        self.0.iter().position(|&c| c == 0).unwrap()
    }

    pub fn as_str(&self) -> &str {
        let len = self.len();
        unsafe { std::str::from_utf8_unchecked(&self.0[..len]) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        let len = self.len();
        &self.0[..len]
    }

    pub fn as_bytes_with_nul(&self) -> &[u8] {
        let len = self.len();
        &self.0[..=len]
    }

    pub fn as_bytes_full(&self) -> &[u8; N] {
        &self.0
    }

    pub fn new(s: &str) -> Self {
        let bytes = s.as_bytes();
        assert!(bytes.len() <= Self::MAX_LENGTH, "string too long");
        let mut arr = [0u8; N];
        arr[0..bytes.len()].copy_from_slice(bytes);
        ZString(arr)
    }
}

impl<const N: usize> From<&str> for ZString<N> {
    fn from(s: &str) -> Self {
        ZString::new(s)
    }
}

impl<const N: usize> From<String> for ZString<N> {
    fn from(s: String) -> Self {
        ZString::new(&s)
    }
}

impl<const N: usize> std::fmt::Debug for ZString<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

impl<const N: usize> std::fmt::Display for ZString<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl<const N: usize> Default for ZString<N> {
    fn default() -> Self {
        ZString([0u8; N])
    }
}

impl<const N: usize> PartialEq for ZString<N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl<const N: usize> Eq for ZString<N> {}

impl<const N: usize> std::hash::Hash for ZString<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

pub type ZString16 = ZString<16>;
pub type ZString32 = ZString<32>;
pub type ZString64 = ZString<64>;
