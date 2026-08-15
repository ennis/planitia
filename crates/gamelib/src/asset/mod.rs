// Asset = a type representing a resource which has a backing file (asset file), stored on disk, or maybe embedded in the executable, or stored as static data in the crate.
//       Asset file: the file associated to an asset
// Provider = something that can provide the bytes for an asset file, given a path
// AssetID = unique identifier for an asset, stable across file renaming/moving
// Cache = keeps references to loaded assets, so they can be shared

// Loading an asset: turning the raw bytes of the asset file into a usable object in memory (e.g. PNG file -> Image data in memory)
//      may involve decompressing, parsing, etc.

/*
NOTE
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
also be stored in the cache and hot-reloaded.*/
mod local_provider;
mod vfs_path;

use std::any::{Any, TypeId};
pub use vfs_path::*;

use crate::error::{Exc, ExcResult, ResultExt};
use crate::platform::wake_event_loop;
use log::{debug, error, info, trace};
use notify_debouncer_mini::notify::{RecommendedWatcher, RecursiveMode, Watcher};
use notify_debouncer_mini::{DebounceEventHandler, DebounceEventResult, Debouncer, new_debouncer};
use slotmap::SlotMap;
use std::cell::UnsafeCell;
use std::cmp::PartialEq;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::time::{Duration, SystemTime};
use std::{io, mem};
use utils::aligned_vec::AVec;

pub type LoadResult<T> = Result<T, Box<dyn Error + Send + Sync + 'static>>;

#[derive(thiserror::Error, Debug, Clone)]
#[error("asset error")]
pub struct AssetError;

#[derive(thiserror::Error, Debug, Clone)]
#[error("failed to watch asset file for changes")]
pub struct WatchLocalFileError;

#[derive(thiserror::Error, Debug, Clone)]
#[error("asset not loaded")]
pub struct AssetNotLoadedError;

////////////////////////////////////////////////////////////////////

/// Metadata about a file in the VFS.
pub struct FileMetadata {
    /// If this is a file on the local file system, the absolute path to the file.
    pub local_path: Option<PathBuf>,
    /// Last modification time.
    pub modified: SystemTime,
}

/// Trait that combines `io::Read` and `io::Seek`.
pub trait ReadSeek: io::Read + io::Seek {}

// Blanket impl
impl<T: io::Read + io::Seek> ReadSeek for T {}

/// File system providers provide file data from VFS paths.
pub trait Provider {
    /// Returns whether the provider can provide the given path.
    fn exists(&self, path: &VfsPath) -> Result<FileMetadata, io::Error>;

    /// Loads the file as an aligned (to the cache line size) byte vector.
    fn load(&self, path: &VfsPath) -> Result<AVec<u8>, io::Error>;

    /// Returns a reader for the file.
    ///
    /// The default implementation loads the whole data and returns a `io::Cursor`.
    fn open(&self, path: &VfsPath) -> Result<Box<dyn ReadSeek>, io::Error> {
        let data = self.load(path)?;
        Ok(Box::new(io::Cursor::new(data)))
    }

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
    fn find_provider(&self, path: &VfsPath) -> Result<(&dyn Provider, FileMetadata), io::Error> {
        let source = path.source().unwrap_or("");
        trace!("find_provider: looking for provider for path `{}`, source = {}", path.as_str(), source);
        if let Some(providers) = self.by_source.get(source) {
            for provider in providers.iter().rev() {
                if let Ok(metadata) = provider.exists(path) {
                    return Ok((provider.as_ref(), metadata));
                } else {
                    trace!("find_provider: {} did not have `{}`", source, path.as_str());
                }
            }
        }
        Err(io::Error::new(io::ErrorKind::NotFound, "no provider could resolve path"))
    }

    /// Returns the global instance of this registry.
    pub(crate) fn get() -> &'static RwLock<Providers> {
        static PROVIDERS: OnceLock<RwLock<Providers>> = OnceLock::new();
        PROVIDERS.get_or_init(|| RwLock::new(Providers { by_source: HashMap::new() }))
    }
}

/// Trait for types that can be inserted in the asset cache. A marker trait, synonymous with `'static + Send + Sync`.
pub trait Asset: 'static + Any + Send + Sync {}
impl<T: 'static + Send + Sync> Asset for T {}

/// Asset read guard.
pub struct AssetReadGuard<'a, T: Asset> {
    guard: RwLockReadGuard<'a, LoadResult<T>>,
}

impl<'a, T: Asset> Deref for AssetReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &*self.guard {
            Ok(asset) => asset,
            Err(err) => panic!("attempted to read an asset that failed to load: {}", err),
        }
    }
}

/// Assets that have default loader functions.
pub trait DefaultLoader: Asset + Sized {
    /// Loads the asset.
    ///
    /// # Arguments
    ///
    /// * `path` - the VFS path of the asset file
    /// * `metadata` - metadata about the asset file, provided by the provider
    /// * `provider` - the file system [`Provider`] that can be used to read asset file data
    /// * `dependencies` - tracks dependencies on other assets and local files
    fn load(
        path: &VfsPath,
        metadata: &FileMetadata,
        provider: &dyn Provider,
        dependencies: &mut Dependencies,
    ) -> LoadResult<Self>;
}

#[macro_export]
macro_rules! static_assets {
    (
        $($(#[$attr:meta])* $v:vis static $name:ident : $ty:ty = $path:expr;)*
    ) => {
        $(
            $(#[$attr])*
            $v static $name: std::sync::LazyLock<$crate::asset::Handle<$ty>> = std::sync::LazyLock::new(|| {
                $crate::asset::AssetCache::instance().load(
                    &$crate::asset::VfsPath::new($path),
                    // TODO: support other load strategies
                    <$ty as $crate::asset::DefaultLoader>::load,
                )
            });
        )*
    };
}

pub use static_assets;

/// A reference to an asset.
pub struct Handle<T: Asset>(Arc<Entry<AssetStorage<T>>>);

impl<T: Asset> Handle<T> {
    fn new(entry: Arc<Entry<AssetStorage<T>>>) -> Self {
        Self(entry)
    }

    pub fn read(&self) -> ExcResult<AssetReadGuard<'_, T>, AssetNotLoadedError> {
        let guard = self.0.asset.read().unwrap();
        match guard.as_ref() {
            Ok(_asset) => Ok(AssetReadGuard { guard }),
            Err(_err) => Err(Exc::new(AssetNotLoadedError)),
        }
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

/*
// derefs to the asset object
impl<T: Asset> Deref for Handle<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.asset_and_loader.asset
    }
}*/

////////////////////////////////////////////////////////////////////

type LoadFn<T> = fn(&VfsPath, &FileMetadata, &dyn Provider, &mut Dependencies) -> LoadResult<T>;

type AssetStorage<T> = RwLock<LoadResult<T>>;

/// An entry in the asset cache.
///
/// `T` should always be `AssetStorage<U>` (a.k.a. `RwLock<LoadResult<U>>`) for some concrete `U: Asset`.
/// The `asset` field is typed as `T` rather than `RwLock<LoadResult<U>>` directly to support type erasure to `Entry<dyn Any>`:
/// since `Result<T>` requires `T: Sized`, storing `asset: RwLock<LoadResult<T>>` directly would prevent
/// unsized coercion from `Entry<T>` to `Entry<dyn Any>`.
struct Entry<T: ?Sized = dyn Any + Send + Sync> {
    path: VfsPathBuf,
    dirty: AtomicBool,
    type_id: TypeId,
    load: *const (),
    reload: fn(&Entry),
    #[cfg(feature = "hot_reload")]
    dependencies: Mutex<Dependencies>,
    asset: T,
}

// SAFETY: Entry only contains function pointers, so it's safe to send/sync it
unsafe impl<T: ?Sized + Send + Sync> Send for Entry<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for Entry<T> {}

impl<T: Asset> Entry<AssetStorage<T>> {
    /// Loads or reloads the asset.
    fn reload(&self) {
        // Mark as clean before reloading, because some loaders may immediately modify/rebuild
        // the underlying asset file, triggering another reload. This is the case, for example,
        // with shader archives in hot-reload mode, which are automatically rebuilt if their
        // source files have a later modification time.
        self.dirty.store(false, Relaxed);

        let mut deps = Dependencies::new(&self.path);

        // SAFETY: it is impossible to build a `Entry<AssetStorage<T>>` with `self.load` not of type `LoadFn<T>`.
        //         This is enforced when creating the Entry.
        let load: LoadFn<T> = unsafe { std::mem::transmute(self.load) };

        // Resolve the VFS path and call the load function.
        let providers = Providers::get().read().unwrap();
        let result = match providers.find_provider(&self.path) {
            Ok((provider, metadata)) => {
                // Watch asset file if hot-reload enabled and the file is on this file system.
                #[cfg(feature = "hot_reload")]
                if let Some(ref local_path) = metadata.local_path {
                    if let Err(err) = deps.watch_local_file(local_path) {
                        err.log_to_stderr();
                    }
                }

                // Invoke the load function.
                load(&self.path, &metadata, provider, &mut deps)
            }
            Err(err) => {
                // No valid provider for this path.
                Err(Box::new(err) as Box<dyn Error + Send + Sync + 'static>)
            }
        };

        // Log load result to stderr.
        match result {
            Ok(_) => debug!("loaded asset `{}`", self.path.as_str()),
            Err(ref err) => {
                error!("failed to load asset `{}`: {}", self.path.as_str(), err)
            }
        }

        let load_successful = result.is_ok();

        // Update the asset object.
        let mut asset = self.asset.write().unwrap();
        *asset = result;

        // Update the dependencies, but only if the reload was successful.
        // When unsuccessful, the computed dependencies may be incomplete or invalid, and it might
        // leave the asset in a state where it will never be reloaded again,
        // even if the source file is repaired.
        #[cfg(feature = "hot_reload")]
        {
            if load_successful {
                debug!("asset `{}` depends on {:?}", self.path.as_str(), deps.assets);
                *self.dependencies.lock().unwrap() = deps;
            }
        }
    }
}

impl Entry {
    fn downcast_ref<T: Asset>(&self) -> Option<&Entry<AssetStorage<T>>> {
        if self.type_id == TypeId::of::<T>() {
            // SAFETY: we checked the type id
            Some(unsafe { &*(self as *const _ as *const Entry<AssetStorage<T>>) })
        } else {
            None
        }
    }

    fn downcast<T: Asset>(self: Arc<Self>) -> Option<Arc<Entry<AssetStorage<T>>>> {
        if self.type_id == TypeId::of::<T>() {
            // SAFETY: we checked the type id
            Some(unsafe { Arc::from_raw(Arc::into_raw(self) as *const Entry<AssetStorage<T>>) })
        } else {
            None
        }
    }

    fn reload_dyn(&self) {
        (self.reload)(self);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    path: VfsPathBuf,
    type_id: TypeId,
}

/// Asset cache proxy that tracks dependencies during asset loading.
///
/// This is a no-op if hot reloading is disabled.
pub struct Dependencies {
    #[cfg(feature = "hot_reload")]
    assets: HashSet<CacheKey>,
    #[cfg(feature = "hot_reload")]
    watcher: Debouncer<RecommendedWatcher>,
}

impl Dependencies {
    fn new(path: &VfsPath) -> Self {
        #[cfg(feature = "hot_reload")]
        {
            Self {
                assets: HashSet::new(),
                watcher: new_debouncer(std::time::Duration::from_millis(500), {
                    let path = path.to_path_buf();
                    move |event: DebounceEventResult| {
                        match event {
                            Ok(ref events) => {
                                for ev in events {
                                    debug!("asset file changed: {:?}, dependency of {}", ev.path, path.as_str());
                                }
                            }
                            Err(err) => {
                                error!("asset file watcher error: {err}");
                            }
                        }
                        AssetCache::instance().asset_changed(&path);
                    }
                })
                .unwrap(),
            }
        }
        #[cfg(not(feature = "hot_reload"))]
        {
            Self {}
        }
    }

    fn add_path<T: Asset>(&mut self, path: &VfsPath) {
        #[cfg(feature = "hot_reload")]
        {
            let key = CacheKey { path: path.to_path_buf(), type_id: TypeId::of::<T>() };
            self.assets.insert(key);
        }
    }

    /// Adds a dependency on another asset.
    pub fn add<T: Asset>(&mut self, handle: &Handle<T>) {
        self.add_path::<T>(&handle.0.path);
    }

    /// Adds a dependency on a file on the local file system.
    ///
    /// Changes to the file will mark the asset as requiring a reload,
    /// which will be done in the next call to [`AssetCache::do_reload`].
    #[cfg(feature = "hot_reload")]
    pub fn watch_local_file<P: AsRef<Path>>(&mut self, path: P) -> ExcResult<(), WatchLocalFileError> {
        let path = path.as_ref();

        assert!(!path.as_os_str().is_empty());

        debug!("watch_local_file: `{}`", path.display());
        self.watcher.watcher().watch(path, RecursiveMode::NonRecursive).raise(WatchLocalFileError)?;
        Ok(())
    }
}

struct Inner {
    by_path: HashMap<CacheKey, Arc<Entry>>,
    /// For each asset key, the set of assets that depend on it (i.e. that should be reloaded
    /// when one changes).
    dependency_graph: HashMap<CacheKey, HashSet<CacheKey>>,
}

impl Inner {
    fn get_entry(&self, key: &CacheKey) -> Option<Arc<Entry>> {
        self.by_path.get(key).cloned()
    }
}

/// Holds cached assets indexed by their VFS path and type.
pub struct AssetCache {
    inner: RwLock<Inner>,
    dirty_paths: Mutex<HashSet<VfsPathBuf>>,
}

impl AssetCache {
    fn new() -> Self {
        Self {
            inner: RwLock::new(Inner { by_path: HashMap::new(), dependency_graph: HashMap::new() }),
            dirty_paths: Mutex::new(Default::default()),
        }
    }

    unsafe fn insert_inner<T: Asset>(&self, path: &VfsPath, loader: LoadFn<T>) -> Handle<T> {
        let key = CacheKey { path: path.to_path_buf(), type_id: TypeId::of::<T>() };

        // Check if an entry already exists and is clean.
        // The cache is locked only for the duration of the check.
        if let Some(existing) = self.inner.read().unwrap().by_path.get(&key)
            && !existing.dirty.load(Relaxed)
        {
            return Handle::new(existing.clone().downcast().expect("invalid asset type stored in cache"));
        }

        fn reload_thunk<T: Asset>(entry: &Entry) {
            entry.downcast_ref::<T>().unwrap().reload()
        }

        // Create entry.
        let entry = Arc::new(Entry {
            path: path.to_path_buf(),
            #[cfg(feature = "hot_reload")]
            dependencies: Mutex::new(Dependencies::new(path)),
            dirty: Default::default(),
            type_id: TypeId::of::<T>(),
            load: loader as *const (),
            reload: reload_thunk::<T>,
            asset: RwLock::new(Err(Box::new(AssetNotLoadedError) as Box<dyn Error + Send + Sync + 'static>)),
        });

        // Initial load.
        entry.reload();

        // Insert the entry into the cache.
        // Note that another thread may have inserted the same entry in the meantime, but
        // there's nothing we can do about it.
        let mut inner = self.inner.write().unwrap();

        #[cfg(feature = "hot_reload")]
        {
            // Update dependencies for hot reload.
            let dependencies = entry.dependencies.lock().unwrap().assets.clone();
            for dep in dependencies.iter() {
                debug!("asset `{}` depends on `{}`", path.as_str(), dep.path.as_str());
                // TODO we track only one level of dependencies for now
                inner.dependency_graph.entry(dep.clone()).or_default().insert(key.clone());
            }
        }

        inner.by_path.insert(key, entry.clone());
        Handle::new(entry)
    }

    /// Loads the asset file at the given path and invokes the given loader function to create the asset,
    /// then inserts the asset into the cache and returns a handle to it.
    pub fn load<T: Asset>(&self, path: &VfsPath, load: LoadFn<T>) -> Handle<T> {
        unsafe { self.insert_inner(path, load) }
    }

    /// Reloads assets which have changed on the file system.
    ///
    /// Should be called at regular intervals (typically, once per frame).
    /// Reloads all assets that have been marked as dirty due to changes to the asset files on disk,
    /// as well as dependent assets.
    #[cfg(feature = "hot_reload")]
    pub fn do_reload(&self) {
        let dirty_paths = mem::take(&mut *self.dirty_paths.lock().unwrap());

        if dirty_paths.is_empty() {
            return;
        }

        debug!("--- AssetCache: reloading assets ---");

        // Mark all affected entries as dirty and collect the keys of dirty entries.
        let mut keys_to_reload = HashSet::new();
        {
            let inner = self.inner.read().unwrap();
            for path in dirty_paths.iter() {
                for (key, entry) in inner.by_path.iter() {
                    if key.path.path_without_fragment() == &**path {
                        entry.dirty.store(true, Relaxed);
                        keys_to_reload.insert(key.clone());
                    }
                }
            }
        }

        loop {
            for k in mem::take(&mut keys_to_reload) {
                let inner = self.inner.read().unwrap();
                // Skip assets that no longer exist (removed from cache, or last handle dropped).
                let Some(entry) = inner.get_entry(&k) else { continue };

                // Check if the dependencies are up-to-date.
                let mut any_dependency_dirty = false;
                for dep_key in entry.dependencies.lock().unwrap().assets.iter() {
                    let Some(dep_entry) = inner.get_entry(dep_key) else {
                        continue;
                    };
                    if dep_entry.dirty.load(Relaxed) {
                        any_dependency_dirty = true;
                        break;
                    }
                }

                // If the asset is up-to-date there's nothing to do.
                if !entry.dirty.load(Relaxed) {
                    // If the asset is up-to-date but its dependencies are dirty, this is a bug.
                    debug_assert!(
                        !any_dependency_dirty,
                        "asset `{}` is marked clean but has dirty dependencies; this should not happen",
                        k.path.as_str()
                    );
                    continue;
                }

                if any_dependency_dirty {
                    // If the asset has dirty dependencies, they should be reloaded first,
                    // in which case the asset is not ready to be reloaded, so put it back in the queue.
                    keys_to_reload.insert(k);
                    continue;
                }

                // Queue dependents of this asset for reload.
                for dep in inner.dependency_graph.get(&k).into_iter().flatten() {
                    keys_to_reload.insert(dep.clone());
                }

                // Unlock before reloading since it may create new entries in the cache.
                drop(inner);

                // Reload the asset.
                entry.reload_dyn();
            }

            // Loop until there are no more dirty assets to reload.
            if keys_to_reload.is_empty() {
                break;
            }
        }

        debug!("--- AssetCache: finished ---");
    }

    /// Called by providers to notify that a file has changed.
    pub fn asset_changed(&self, path: &VfsPath) {
        debug!("asset file changed: {}", path.as_str());
        #[cfg(feature = "hot_reload")]
        self.dirty_paths.lock().unwrap().insert(path.path_without_fragment().to_path_buf());
    }

    /// Returns the global instance of the asset cache.
    pub fn instance() -> &'static AssetCache {
        static ASSET_CACHE: OnceLock<AssetCache> = OnceLock::new();
        let cache = ASSET_CACHE.get_or_init(|| AssetCache::new());
        cache
    }

    pub fn register_directory(path: impl AsRef<std::path::Path>) {
        let mut providers = Providers::get().write().unwrap();
        providers.register_overlay(Box::new(local_provider::LocalProvider::new(path.as_ref().to_path_buf())));
    }
}

/// Opens an asset file.
pub fn open_asset(path: impl AsRef<VfsPath>) -> ExcResult<Box<dyn ReadSeek>, AssetError> {
    let path = path.as_ref();
    let providers = Providers::get().read().unwrap();
    let (provider, _metadata) = providers.find_provider(path).raise(AssetError)?;
    provider.open(path).raise(AssetError)
}

/// Loads an asset file into a byte vector.
pub fn load_asset(path: impl AsRef<VfsPath>) -> ExcResult<AVec<u8>, AssetError> {
    let path = path.as_ref();
    let providers = Providers::get().read().unwrap();
    let (provider, _metadata) = providers.find_provider(path).raise(AssetError)?;
    provider.load(path).raise(AssetError)
}
