use aligned_vec::{ABox, AVec};
use std::alloc::Layout;
use std::borrow::Borrow;
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Deref, Index, IndexMut};
use std::path::Path;
use std::{io, ptr, slice};

/// Boxed byte slice aligned to cache lines.
type AlignedData = ABox<[u8]>;

/// Vec<u8> aligned to cache lines.
type AlignedVec = AVec<u8>;

/// Errors while reading archive data.
#[derive(thiserror::Error, Debug)]
pub enum ArchiveError {
    #[error("unexpected end of file")]
    UnexpectedEof,
    #[error("invalid signature: expected {expected:?}, found {found:?}")]
    InvalidSignature { expected: [u8; 4], found: [u8; 4] },
    #[error("unsupported version: expected {expected}, found {found}")]
    UnsupportedVersion { expected: u32, found: u32 },
    #[error("offset out of range")]
    OffsetOutOfRange,
    #[error("invalid data")]
    InvalidData,
    #[error("misaligned data")]
    MisalignedData,
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

///////////////////////////////////////////////////////////

/*
fn length_prefixed_array_offset<Element>(count: usize) -> usize {
    Layout::new::<u32>()
        .extend(Layout::array::<Element>(count).unwrap())
        .unwrap()
        .1
}*/

/// Returns the (layout, array_offset)
fn length_prefixed_array_layout<Element>(count: usize) -> (Layout, usize) {
    let (layout, array_offset) = Layout::new::<u32>()
        .extend(Layout::array::<Element>(count).unwrap())
        .unwrap();
    (layout.pad_to_align(), array_offset)
}

#[derive(Copy, Clone, Debug)]
struct ArchiveHeader {
    signature: [u8; 4],
    version: u32,
    root: OffsetUntyped,
}

///////////////////////////////////////////////////////////

pub trait ArchiveRoot: Copy + 'static {
    const SIGNATURE: [u8; 4];
    const VERSION: u32;
}

pub trait ArchiveData {
    /// Tries to cast a byte buffer to a reference of this type.
    unsafe fn cast(buffer: *const u8, len: usize) -> Result<*const Self, ArchiveError>;
}

impl<T: Copy + 'static> ArchiveData for T {
    unsafe fn cast(buffer: *const u8, len: usize) -> Result<*const Self, ArchiveError> {
        // check alignment
        let align = align_of::<T>();
        if (buffer as usize) & (align - 1) != 0 {
            return Err(ArchiveError::MisalignedData);
        }
        // check size
        if len < size_of::<T>() {
            return Err(ArchiveError::UnexpectedEof);
        }
        // SAFETY: size checked, alignment checked.
        //         It's possible that the byte pattern is not a valid instance of T,
        //         but that's not something we can check here.
        Ok(buffer as *const T)
    }
}

impl<T: Copy + 'static> ArchiveData for [T] {
    unsafe fn cast(buffer: *const u8, len: usize) -> Result<*const Self, ArchiveError> {
        // check alignment
        let align = length_prefixed_array_layout::<T>(1).0.align();
        if (buffer as usize) & (align - 1) != 0 {
            return Err(ArchiveError::MisalignedData);
        }
        // check if there's enough data for the length prefix
        if len < size_of::<u32>() {
            return Err(ArchiveError::UnexpectedEof);
        }
        // read length prefix
        let count = unsafe { *(buffer as *const u32) as usize };
        // final layout
        let (layout, start_offset) = length_prefixed_array_layout::<T>(count);
        //dbg!(count, layout, start_offset);
        // check if there's enough data for the whole prefix+array (alignment shouldn't change)
        if len < layout.size() {
            return Err(ArchiveError::UnexpectedEof);
        }
        // SAFETY: size checked, alignment checked
        unsafe { Ok(slice::from_raw_parts(buffer.add(start_offset) as *const T, count)) }
    }
}

impl ArchiveData for str {
    unsafe fn cast(buffer: *const u8, len: usize) -> Result<*const Self, ArchiveError> {
        // check if there's enough data for the length prefix
        if len < size_of::<u32>() {
            return Err(ArchiveError::UnexpectedEof);
        }
        // read length prefix
        let count = unsafe { *(buffer as *const u32) as usize };
        let total_size = size_of::<u32>() + count;
        // check if there's enough data for the whole prefix+string
        if len < total_size {
            return Err(ArchiveError::UnexpectedEof);
        }
        // SAFETY: size checked
        let str_slice = unsafe { slice::from_raw_parts(buffer.add(size_of::<u32>()), count) };
        match std::str::from_utf8(str_slice) {
            Ok(s) => Ok(s),
            Err(_) => Err(ArchiveError::InvalidData),
        }
    }
}

///////////////////////////////////////////////////////////

/// Represents an offset to some data of an unspecified type within an archive file.
pub type OffsetUntyped = u32;

/// Represents an offset to some data within an archive file.
#[repr(transparent)]
pub struct Offset<T: ?Sized>(pub(crate) u32, PhantomData<*const T>);

impl<T: ?Sized> Clone for Offset<T> {
    fn clone(&self) -> Self {
        Offset(self.0, PhantomData)
    }
}

impl<T: ?Sized> Copy for Offset<T> {}

impl<T: ?Sized + ArchiveData> Offset<T> {
    /// Invalid offset (represents empty / no data).
    pub const INVALID: Self = Offset(u32::MAX, PhantomData);

    fn new(offset: u32) -> Self {
        Self(offset, PhantomData)
    }

    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }

    /*/// Returns a reference to the data at this offset within the provided archive reader.
    pub fn get<'a, H>(&self, reader: &'a ArchiveReader<H>) -> Result<&'a T, ReadError> {
        self.as_ptr(reader.1.as_ptr(), reader.1.len())
            .map(|ptr| unsafe { &*ptr })
    }*/

    pub fn as_ptr(&self, base: *const u8, len: usize) -> Result<*const T, ArchiveError> {
        assert!((self.0 as usize) < len);
        unsafe { T::cast(base.add(self.0 as usize), len - self.0 as usize) }
    }

    pub fn as_mut_ptr(&self, base: *mut u8, len: usize) -> Result<*mut T, ArchiveError> {
        self.as_ptr(base, len).map(|ptr| ptr as *mut T)
    }
}

/*
#[macro_export]
macro_rules! field_offset {
    ($offset:expr, $t:ty, $field:ident) => {{
        // helper to infer the field type as U since we can't name it directly
        fn field_offset_impl<T, U>(offset: Offset<T>, field_offset: u32, _access: impl FnOnce(&T) -> &U) -> Offset<U> {
            Offset::new(offset.0 + field_offset)
        }
        let field_offset = ::core::mem::offset_of!($t, $field);
        field_offset_impl($offset, field_offset as u32, |x: &$t| &x.$field)
    }};
}*/

///////////////////////////////////////////////////////////

/// Wrapper around a byte slice with convenience methods to read or reinterpret data.
#[repr(transparent)]
pub struct ArchiveReader<Root>(PhantomData<fn() -> Root>, [u8]);

impl<Root: ArchiveRoot> ArchiveReader<Root> {
    /// FIXME check alignment requirements
    pub fn new(data: &[u8]) -> Result<&Self, ArchiveError> {
        // SAFETY: same repr.
        let this = unsafe { &*(data as *const [u8] as *const ArchiveReader<Root>) };
        let header = this.header()?;

        if header.signature != Root::SIGNATURE {
            return Err(ArchiveError::InvalidSignature {
                expected: Root::SIGNATURE,
                found: header.signature,
            });
        }
        if header.version != Root::VERSION {
            return Err(ArchiveError::UnsupportedVersion {
                expected: Root::VERSION,
                found: header.version,
            });
        }
        Ok(this)
    }

    /// Reads the archive header.
    fn header(&self) -> Result<&ArchiveHeader, ArchiveError> {
        unsafe {
            let ptr = Offset::<ArchiveHeader>::new(0).as_ptr(self.1.as_ptr(), self.1.len())?;
            Ok(&*ptr)
        }
    }

    /// Reads the root data.
    pub fn root(&self) -> &Root {
        let header = self.header().expect("invalid archive header");
        let root_offset = Offset::<Root>::new(header.root);
        &self[root_offset]
    }
}

impl<T: ?Sized + ArchiveData, Root: ArchiveRoot> Index<Offset<T>> for ArchiveReader<Root> {
    type Output = T;

    fn index(&self, index: Offset<T>) -> &Self::Output {
        unsafe { &*index.as_ptr(self.1.as_ptr(), self.1.len()).unwrap() }
    }
}

impl<Root: ArchiveRoot> ToOwned for ArchiveReader<Root> {
    type Owned = ArchiveReaderOwned<Root>;

    fn to_owned(&self) -> Self::Owned {
        let mut storage = AlignedVec::with_capacity(0, self.1.len());
        storage.extend_from_slice(&self.1);
        ArchiveReaderOwned::new(storage.into_boxed_slice()).unwrap()
    }
}

///////////////////////////////////////////////////////////

/// Owned version of ArchiveReader that keeps the data alive.
pub struct ArchiveReaderOwned<Root: 'static> {
    data: *mut [u8],
    align: usize,
    /// Self-references the contents of `data`.
    reader: &'static ArchiveReader<Root>,
}

// SAFETY: the data inside an ArchiveReader is immutable, no interior mutability.
unsafe impl<Root: ArchiveRoot> Send for ArchiveReaderOwned<Root> {}
unsafe impl<Root: ArchiveRoot> Sync for ArchiveReaderOwned<Root> {}

impl<Root> Drop for ArchiveReaderOwned<Root> {
    fn drop(&mut self) {
        // reconstruct the box and drop it
        unsafe {
            let vec = AlignedData::from_raw_parts(self.align, self.data);
            drop(vec);
        }
    }
}

impl<Root: ArchiveRoot> ArchiveReaderOwned<Root> {
    fn new(data: AlignedData) -> Result<Self, ArchiveError> {
        // convert to raw pointer because moving the box would invalidate the self-reference in ArchiveReader
        let (data, align) = AlignedData::into_raw_parts(data);
        /*dbg!(align, data.len());*/
        let reader = unsafe { ArchiveReader::new(slice::from_raw_parts(data as *const u8, data.len()))? };
        Ok(ArchiveReaderOwned { data, align, reader })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ArchiveError> {
        fn load_inner(path: &Path) -> std::io::Result<AlignedData> {
            let mut file = File::open(path)?;
            let file_size = file.metadata()?.len() as usize;
            let mut storage = AVec::new(0);
            storage.resize(file_size, 0);
            let read = file.read(&mut storage)?;
            if read != file_size {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "failed to read whole file",
                ));
            }
            Ok(storage.into_boxed_slice())
        }

        let storage = load_inner(path.as_ref())?;
        Ok(ArchiveReaderOwned::new(storage)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArchiveError> {
        let mut storage = AVec::new(0);
        storage.extend_from_slice(bytes);
        ArchiveReaderOwned::new(storage.into_boxed_slice())
    }
}

impl<Root: ArchiveRoot> Deref for ArchiveReaderOwned<Root> {
    type Target = ArchiveReader<Root>;

    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

impl<Root: ArchiveRoot> Borrow<ArchiveReader<Root>> for ArchiveReaderOwned<Root> {
    fn borrow(&self) -> &ArchiveReader<Root> {
        &self.reader
    }
}

///////////////////////////////////////////////////////////

pub struct ArchiveWriter<Root: ArchiveRoot> {
    storage: AlignedVec,
    _phantom: PhantomData<fn() -> Root>,
}

impl<Root: ArchiveRoot> ArchiveWriter<Root> {
    pub fn new() -> Self {
        let mut this = ArchiveWriter {
            storage: AlignedVec::new(0),
            _phantom: PhantomData,
        };
        // reserve space for archive header
        let _ = this.write(&ArchiveHeader {
            signature: Root::SIGNATURE,
            version: Root::VERSION,
            root: OffsetUntyped::MAX,
        });
        this
    }

    fn alloc_layout(&mut self, layout: Layout) -> (OffsetUntyped, *mut u8) {
        let aligned_len = (self.storage.len() + layout.align() - 1) & !(layout.align() - 1);
        let new_len = aligned_len + layout.size();
        self.storage.resize(new_len, 0);
        let ptr = unsafe { self.storage.as_mut_ptr().add(aligned_len) };
        // TODO check for overflow
        let offset = aligned_len as u32;
        (offset, ptr)
    }

    /// Allocates aligned storage for `count` values of type T and returns a pointer to it.
    pub fn alloc_slice<T: Copy + 'static>(&mut self, count: usize) -> (Offset<[T]>, *mut T) {
        let layout = Layout::array::<T>(count).unwrap().pad_to_align();
        let (offset, ptr) = self.alloc_layout(layout);
        (Offset::new(offset), ptr as *mut T)
    }

    pub fn alloc<T: Copy + 'static>(&mut self) -> (Offset<T>, *mut T) {
        let (offset, ptr) = self.alloc_slice::<T>(1);
        (Offset::new(offset.0), ptr)
    }

    /// Starts a length-prefixed array of `count` elements of type T and returns a pointer to the first element.
    pub fn alloc_length_prefixed<T: Copy + 'static>(&mut self, count: usize) -> (Offset<[T]>, *mut T) {
        let (layout, array_offset) = length_prefixed_array_layout::<T>(count);
        let (offset, ptr) = self.alloc_layout(layout);
        unsafe {
            // write length prefix
            *(ptr as *mut u32) = count as u32;
            // return pointer to first element
            let array_ptr = ptr.add(array_offset) as *mut T;
            (Offset::new(offset), array_ptr)
        }
    }

    pub fn write<T: Copy + 'static>(&mut self, value: &T) -> Offset<T> {
        let (offset, ptr) = self.alloc::<T>();
        unsafe {
            ptr.write(*value);
        }
        offset
    }

    pub fn write_slice<T: Copy + 'static>(&mut self, slice: &[T]) -> Offset<[T]> {
        let (offset, ptr) = self.alloc_length_prefixed::<T>(slice.len());
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
        }
        offset
    }

    pub fn write_str(&mut self, s: &str) -> Offset<str> {
        let bytes = s.as_bytes();
        let offset = self.write_slice(bytes);
        Offset::new(offset.0)
    }

    pub fn write_iter<T: Copy + 'static, I>(&mut self, count: usize, values: I) -> Offset<[T]>
    where
        I: IntoIterator<Item = T>,
    {
        let (offset, mut ptr) = self.alloc_length_prefixed::<T>(count);
        for value in values.into_iter().take(count) {
            unsafe {
                ptr.write(value);
                ptr = ptr.add(1);
            }
        }
        offset
    }

    pub fn as_slice(&self) -> &[u8] {
        self.storage.as_slice()
    }

    /// Writes the root object and sets the root offset in the archive header.
    pub fn write_root(&mut self, root: &Root) {
        let root_offset = self.write(root);
        let header = self.storage.as_ptr() as *mut ArchiveHeader;
        unsafe {
            (*header).root = root_offset.0;
        }
    }

    pub fn write_to_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.write_to_file_inner(path.as_ref())
    }

    fn write_to_file_inner(&self, path: &Path) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        file.write_all(self.as_slice())?;
        Ok(())
    }
}

impl<T: ?Sized + ArchiveData, Root: ArchiveRoot> Index<Offset<T>> for ArchiveWriter<Root> {
    type Output = T;

    fn index(&self, index: Offset<T>) -> &Self::Output {
        unsafe { &*index.as_ptr(self.storage.as_ptr(), self.storage.len()).unwrap() }
    }
}

impl<T: ?Sized + ArchiveData, Root: ArchiveRoot> IndexMut<Offset<T>> for ArchiveWriter<Root> {
    fn index_mut(&mut self, index: Offset<T>) -> &mut Self::Output {
        unsafe { &mut *index.as_mut_ptr(self.storage.as_mut_ptr(), self.storage.len()).unwrap() }
    }
}

////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::archive::{ArchiveReader, ArchiveRoot, Offset};

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct TestData {
        a: u32,
        b: u16,
        c: u16,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct Root {
        test_data: Offset<[TestData]>,
    }

    impl ArchiveRoot for Root {
        const SIGNATURE: [u8; 4] = *b"TEST";
        const VERSION: u32 = 1;
    }

    #[repr(align(16))]
    struct Wrapper([u8; 36]);

    #[test]
    fn read() {
        let bytes = Wrapper([
            b'T', b'E', b'S', b'T', // magic
            0x01, 0x00, 0x00, 0x00, // version = 1
            0x0C, 0x00, 0x00, 0x00, // offset to root = 12
            0x10, 0x00, 0x00, 0x00, // offset to test_data = 16
            // test_data array (2 elements)
            0x02, 0x00, 0x00, 0x00, // length prefix = 2
            0x01, 0x00, 0x00, 0x00, // first TestData.a = 1
            0x02, 0x00, // first TestData.b = 2
            0x03, 0x00, // first TestData.c = 3
            0x04, 0x00, 0x00, 0x00, // second TestData.a = 4
            0x05, 0x00, // second TestData.b = 5
            0x06, 0x00, // second TestData.c = 6
        ]);

        let reader = ArchiveReader::<Root>::new(&bytes.0).unwrap();
        let root = reader.root();
        let array = &reader[root.test_data];

        assert_eq!(array.len(), 2);
        assert_eq!(array[0].a, 1);
        assert_eq!(array[0].b, 2);
        assert_eq!(array[0].c, 3);
        assert_eq!(array[1].a, 4);
        assert_eq!(array[1].b, 5);
        assert_eq!(array[1].c, 6);
    }
}
