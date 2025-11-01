// Asset = a type representing a resource which has a backing file (asset file), stored on disk, or maybe embedded in the executable, or stored as static data in the crate.
//       Asset file: the file associated to an asset
// Provider = something that can provide the bytes for an asset file, given a path
// AssetID = unique identifier for an asset, stable across file renaming/moving
// Cache = keeps references to loaded assets, so they can be shared

// Loading an asset: turning the raw bytes of the asset file into a usable object in memory (e.g. PNG file -> Image data in memory)
//      may involve decompressing, parsing, etc.

/*

A single asset file can be associated to produce multiple assets in memory. For example,
a PNG file can be associated to both the CPU-side image data structure and the GPU-side texture.
One asset can have multiple representations.

E.g.
    - Asset file: "texture.png"
        - repr 1: Handle<Image>         // managed by ImageLoader
        - repr 2: Handle<gpu::Image>    // managed by GpuResources

Issue: graphics pipelines

The asset file is the "pipeline archive", which contains multiple pipelines.
The graphics pipeline is made from data from the pipeline archive, so there's a dependency
between the graphics pipeline asset and the pipeline archive asset.

The graphics pipeline object is a "derived" asset, that depends on the archive. Ideally it should
also be stored in the cache and hot-reloaded.

 */
mod local_provider;
mod vfs_path;

use std::any::{Any, TypeId};
pub use vfs_path::*;

use std::collections::{HashMap, HashSet};
use std::io;
use std::ops::Deref;
use std::sync::{Arc, OnceLock, RwLock};
use log::debug;
use utils::aligned_vec::AVec;

/*
path resolution:
- if there's an explicit source, query the registered provider for that source
- otherwise, use the default resolver:
*/

/// File system providers provide file data from VFS paths.
pub trait Provider {
    /// Returns whether the provider can provide the given path.
    fn exists(&self, path: &VfsPath) -> bool;

    /// Loads the file as an aligned (to the cache line size) byte vector.
    fn load(&self, path: &VfsPath) -> Result<AVec<u8>, io::Error>;

    /// Loads the file as a static byte slice.
    ///
    /// For embedded assets, this will return a pointer to the static data.
    /// For other providers, this may allocate memory and leak it.
    ///
    /// The default implementation calls `load` and leaks the data.
    ///
    /// The returned slice is aligned to the cache line size.
    fn load_static(&self, path: &VfsPath) -> Result<&'static [u8], io::Error> {
        let (ptr, _alignment, length, _capacity) = self.load(path)?.into_raw_parts();
        // SAFETY: we leak the memory, so the pointer is valid for 'static
        unsafe { Ok(std::slice::from_raw_parts(ptr, length)) }
    }

    /// Returns the name of this provider.
    fn name(&self) -> &str;

    // Sets up an event listener for changes to the given path.
    //fn watch(&self, path: &VfsPath) -> EventToken;
}

/// Global registry of file system providers.
pub(crate) struct Providers {
    by_source: HashMap<String, Vec<Box<dyn Provider + Send + Sync>>>,
}

impl Providers {
    /// Registers a provider for a given source.
    ///
    /// # Panics
    ///
    /// Panics if a provider is already registered for the given source.
    pub(crate) fn register_provider(&mut self, source: &str, provider: Box<dyn Provider + Send + Sync>) {
        if self.by_source.contains_key(source) {
            panic!("provider already registered for source: {}", source);
        }
        self.by_source.insert(source.to_string(), vec![provider]);
    }

    /// Registers a provider for the default source.
    ///
    /// Equivalent to `register_provider("", provider)`.
    pub(crate) fn register_default_provider(&mut self, provider: Box<dyn Provider + Send + Sync>) {
        self.register_provider("", provider);
    }

    /// Registers an overlay.
    ///
    /// Overlays are providers for the default source that are queried before the default provider.
    /// If it fails to provide the asset, the next overlay is queried, and so on,
    /// until the default provider is queried.
    ///
    /// The last registered overlay is queried first.
    pub(crate) fn register_overlay(&mut self, provider: Box<dyn Provider + Send + Sync>) {
        self.by_source.entry("".to_string()).or_default().push(provider);
    }

    /// Finds the appropriate provider for the given VFS path.
    fn find_provider(&self, path: &VfsPath) -> Result<&dyn Provider, io::Error> {
        let source = path.source().unwrap_or("");
        debug!("find_provider: looking for provider for path `{}`, source = {}", path.as_str(), source);
        if let Some(providers) = self.by_source.get(source) {
            for provider in providers.iter().rev() {
                if provider.exists(path) {
                    return Ok(provider.as_ref());
                } else {
                    debug!("find_provider: {} did not have `{}`", source, path.as_str());
                }
            }
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no provider found for path: {}", path.as_str()),
        ))
    }

    /// Loads an asset file from the given VFS path.
    ///
    /// This will select the appropriate provider based on the source prefix of the path.
    pub(crate) fn load(&self, path: &VfsPath) -> Result<AVec<u8>, io::Error> {
        let provider = self.find_provider(path)?;
        provider.load(path)
    }

    /// Loads an asset file as a static byte slice from the given VFS path.
    pub(crate) fn load_static(&self, path: &VfsPath) -> Result<&'static [u8], io::Error> {
        let provider = self.find_provider(path)?;
        provider.load_static(path)
    }

    /// Returns the global instance of this registry.
    pub(crate) fn get() -> &'static RwLock<Providers> {
        static PROVIDERS: OnceLock<RwLock<Providers>> = OnceLock::new();
        PROVIDERS.get_or_init(|| {
            RwLock::new(Providers {
                by_source: HashMap::new(),
            })
        })
    }
}

/// Trait for types that can be inserted in the asset cache. A marker trait, synonymous with `'static + Send + Sync`.
pub trait Asset: 'static + Send + Sync {}
impl<T: 'static + Send + Sync> Asset for T {}

/// A reference to a loaded asset.
pub struct Handle<T: Asset>(Arc<Entry<T>>);

impl<T: Asset> Handle<T> {
    fn new(entry: Arc<Entry<T>>) -> Self {
        Self(entry)
    }
}

impl<T: Asset> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

// reference equality
impl<T: Asset> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl<T: Asset> Eq for Handle<T> {}

// hash based on pointer
impl<T: Asset> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

// ord based on pointer
impl<T: Asset> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Asset> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (Arc::as_ptr(&self.0) as usize).cmp(&(Arc::as_ptr(&other.0) as usize))
    }
}

// derefs to the asset object
impl<T: Asset> Deref for Handle<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.asset
    }
}

////////////////////////////////////////////////////////////////////

enum Loader<T> {
    FromSlice(fn(&[u8], &mut Dependencies) -> T),
    FromStaticSlice(fn(&'static [u8], &mut Dependencies) -> T),
    WithDependencies(fn(&VfsPath, &mut Dependencies) -> T),
}

struct Entry<T> {
    loader: Loader<T>,
    path: VfsPathBuf,
    dependencies: Dependencies,
    asset: T,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    path: VfsPathBuf,
    type_id: TypeId,
}

/// Asset cache proxy that tracks dependencies during asset loading.
pub struct Dependencies {
    dependencies: HashSet<CacheKey>,
}

impl Dependencies {
    fn new() -> Self {
        Self {
            dependencies: HashSet::new(),
        }
    }

    fn add_path<T: Asset>(&mut self, path: &VfsPath) {
        let key = CacheKey {
            path: path.to_path_buf(),
            type_id: TypeId::of::<T>(),
        };
        self.dependencies.insert(key);
    }

    pub fn add<T: Asset>(&mut self, handle: &Handle<T>) {
        self.add_path::<T>(&handle.0.path);
    }
}

/// Holds cached assets indexed by their VFS path and type.
pub struct AssetCache {
    by_path: RwLock<HashMap<CacheKey, Arc<dyn Any + Send + Sync>>>,
}

impl AssetCache {
    fn new() -> Self {
        Self {
            by_path: RwLock::new(HashMap::new()),
        }
    }

    fn insert_inner<T: Asset>(&self, path: &VfsPath, loader: Loader<T>) -> Handle<T> {
        let key = CacheKey {
            path: path.to_path_buf(),
            type_id: TypeId::of::<T>(),
        };

        // Check if an entry already exists.
        // The cache is locked only for the duration of the check.
        if let Some(existing) = self.by_path.read().unwrap().get(&key) {
            return Handle::new(existing.clone().downcast().expect("invalid asset type stored in cache"));
        }

        // Cache unlocked here.

        // Read the asset file, and load the asset.
        let entry = {
            let mut deps = Dependencies::new();
            let providers = Providers::get().read().unwrap();
            let asset = match loader {
                Loader::FromSlice(f) => {
                    let bytes = providers.load(path).unwrap();
                    f(&bytes, &mut deps)
                }
                Loader::FromStaticSlice(f) => {
                    // Note that we shouldn't reload assets loaded from static slices, since the data
                    // isn't supposed to change
                    let bytes = providers.load_static(path).unwrap();
                    f(bytes, &mut deps)
                }
                Loader::WithDependencies(f) => f(path, &mut deps),
            };

            Arc::new(Entry {
                loader,
                path: path.to_path_buf(),
                dependencies: deps,
                asset,
            })
        };

        // Insert the entry into the cache.
        // Note that another thread may have inserted the same entry in the meantime, but
        // there's nothing we can do about it.
        self.by_path.write().unwrap().insert(key, entry.clone());
        Handle::new(entry)
    }

    pub fn load_and_insert<T: Asset>(&self, path: &VfsPath, loader: fn(&[u8], &mut Dependencies) -> T) -> Handle<T> {
        self.insert_inner(path, Loader::FromSlice(loader))
    }

    pub fn load_and_insert_static<T: Asset>(
        &self,
        path: &VfsPath,
        loader: fn(&'static [u8], &mut Dependencies) -> T,
    ) -> Handle<T> {
        self.insert_inner(path, Loader::FromStaticSlice(loader))
    }

    pub fn insert_with_dependencies<T: Asset>(&self, path: &VfsPath, loader: fn(&VfsPath, &mut Dependencies) -> T) -> Handle<T> {
        self.insert_inner(path, Loader::WithDependencies(loader))
    }

    /// Returns the global instance of the asset cache.
    pub fn instance() -> &'static AssetCache {
        static ASSET_CACHE: OnceLock<AssetCache> = OnceLock::new();
        ASSET_CACHE.get_or_init(|| AssetCache::new())
    }

    pub fn register_filesystem_path(path: impl AsRef<std::path::Path>) {
        let mut providers = Providers::get().write().unwrap();
        providers.register_provider(
            "",
            Box::new(local_provider::LocalProvider::new(path.as_ref().to_path_buf())),
        );
    }
}
