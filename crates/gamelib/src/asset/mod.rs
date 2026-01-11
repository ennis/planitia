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

use crate::platform::{UserEvent, wake_event_loop};
use log::{debug, error, info};
use notify_debouncer_mini::notify::{RecommendedWatcher, RecursiveMode, Watcher};
use notify_debouncer_mini::{DebounceEventHandler, DebounceEventResult, Debouncer, new_debouncer};
use slotmap::SlotMap;
use std::cell::UnsafeCell;
use std::cmp::PartialEq;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::time::{Duration, SystemTime};
use std::{io, mem};
use utils::aligned_vec::AVec;

pub type LoadResult<T> = Result<T, anyhow::Error>;

#[derive(thiserror::Error, Debug)]
pub enum AssetLoadError {
    #[error("no provider found for path")]
    NoProviderFound,
    #[error("I/O error while loading asset: {0}")]
    IoError(#[source] io::Error),
    /// Error from loader.
    #[error("asset not loaded")]
    NotLoaded,
}

////////////////////////////////////////////////////////////////////

type LocalFileWatcher = Debouncer<RecommendedWatcher>;

/// Events emitted to the event loop by `FileWatcher`.
pub struct FileSystemEvent {
    pub paths: Vec<PathBuf>,
}

/// Generic file watcher.
///
/// This will emit `file_changed` events to the main event loop when a file changes.
pub struct FileWatcher {
    watcher: LocalFileWatcher,
}

impl FileWatcher {
    pub fn new(callback: fn()) -> Result<Self, io::Error> {
        #[derive(Clone)]
        struct Handler(fn());

        impl DebounceEventHandler for Handler {
            fn handle_event(&mut self, event_result: DebounceEventResult) {
                match event_result {
                    Ok(ref events) => {
                        let paths = events.iter().map(|event| event.path.to_path_buf()).collect::<Vec<_>>();
                        if !paths.is_empty() {
                            debug!("FileWatcher: files changed: {paths:?}");
                            let callback = self.0;
                            wake_event_loop(move || callback());
                        }
                    }
                    Err(err) => {
                        error!("FileWatcher error: {err}");
                    }
                }
            }
        }

        const DEBOUNCE_TIMEOUT_MS: u64 = 500;

        let watcher = new_debouncer(Duration::from_millis(DEBOUNCE_TIMEOUT_MS), Handler(callback)).unwrap();
        Ok(Self { watcher })
    }

    pub fn watch_file<P: AsRef<Path>>(&mut self, path: P) {
        self.watcher
            .watcher()
            .watch(path.as_ref(), RecursiveMode::NonRecursive)
            .unwrap();
    }
}

////////////////////////////////////////////////////////////////////

/// Metadata about a file in the VFS.
pub struct FileMetadata {
    /// If this is a file on the local file system, the absolute path to the file.
    pub local_path: Option<PathBuf>,
    /// Last modification time.
    pub modified: SystemTime,
}

/// File system providers provide file data from VFS paths.
pub trait Provider {
    /// Returns whether the provider can provide the given path.
    fn exists(&self, path: &VfsPath) -> Result<FileMetadata, io::Error>;

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
        debug!(
            "find_provider: looking for provider for path `{}`, source = {}",
            path.as_str(),
            source
        );
        if let Some(providers) = self.by_source.get(source) {
            for provider in providers.iter().rev() {
                if let Ok(metadata) = provider.exists(path) {
                    return Ok((provider.as_ref(), metadata));
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

    /*/// Loads an asset file from the given VFS path.
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
    }*/

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
pub trait Asset: 'static + Any + Send + Sync {}
impl<T: 'static + Send + Sync> Asset for T {}

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

/*
impl<'a, T: Asset> AssetReadGuard<'a, T> {
    pub fn try_get(&self) -> Result<&T, AssetLoadError> {
        match self.guard.as_ref() {
            Ok(asset) => Ok(asset),
            Err(_err) => Err(AssetLoadError::NotLoaded),
        }
    }
}*/

/// Assets that have default loader functions.
pub trait DefaultLoader: Asset + Sized {
    /// Loads the asset.
    fn load(path: &VfsPath, metadata: &FileMetadata, provider: &dyn Provider, dependencies: &mut Dependencies) -> LoadResult<Self>;
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

    pub fn read(&self) -> Result<AssetReadGuard<'_, T>, AssetLoadError> {
        let guard = self.0.asset.read().unwrap();
        match guard.as_ref() {
            Ok(_asset) => Ok(AssetReadGuard { guard }),
            Err(_err) => Err(AssetLoadError::NotLoaded),
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

// FIXME: there should be only one kind of loader function, taking a VfsPath and Dependencies,
//        and the loader should be responsible for loading the file data itself if needed.
enum LoaderKind {
    /// Loads the asset from a byte slice.
    ///
    /// Loader function signature: `fn(&[u8], &FileMetadata, &mut Dependencies) -> LoadResult<T>`
    FromSlice,
    /// Loads the asset from a static byte slice.
    ///
    /// This can be more efficient if the target asset type can reference the static data directly,
    /// without copying it.
    ///
    /// Loader function signature: `fn(&'static [u8], &FileMetadata, &mut Dependencies) -> LoadResult<T>`
    FromStaticSlice,
    /// Loads the asset from a file specified by a VFS path.
    ///
    /// Loader function signature: `fn(&VfsPath, &mut Dependencies) -> LoadResult<T>`
    FromPath,
}

type LoadFromSliceFn<T> = fn(&[u8], &FileMetadata, &mut Dependencies) -> LoadResult<T>;
type LoadFromStaticSliceFn<T> = fn(&'static [u8], &FileMetadata, &mut Dependencies) -> LoadResult<T>;
type LoadFromPathFn<T> = fn(&VfsPath, &mut Dependencies) -> LoadResult<T>;

type LoadFn<T> = fn(&VfsPath, &FileMetadata, &dyn Provider, &mut Dependencies) -> LoadResult<T>;

struct Loader {
    type_id: TypeId,
    //kind: LoaderKind,
    func: *const (),
    reload: fn(&Entry),
}

// SAFETY: Loader only contains function pointers, so it's safe to send/sync it
unsafe impl Send for Loader {}
unsafe impl Sync for Loader {}

fn reload_thunk<T: Asset>(entry: &Entry) {
    entry.downcast_ref::<T>().unwrap().reload()
}

impl Loader {

    fn new<T: Asset>(load_fn: LoadFn<T>) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            //kind: LoaderKind::FromPath,
            func: load_fn as *const (),
            reload: reload_thunk::<T>,
        }
    }

    /*fn from_slice<T: Asset>(f: LoadFromSliceFn<T>) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            kind: LoaderKind::FromSlice,
            func: f as *const (),
            reload: reload_thunk::<T>,
        }
    }

    fn from_static_slice<T: Asset>(f: LoadFromStaticSliceFn<T>) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            kind: LoaderKind::FromStaticSlice,
            func: f as *const (),
            reload: reload_thunk::<T>,
        }
    }

    fn from_path<T: Asset>(f: LoadFromPathFn<T>) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            kind: LoaderKind::FromPath,
            func: f as *const (),
            reload: reload_thunk::<T>,
        }
    }*/

    fn load<T: Asset>(&self, path: &VfsPath, providers: &Providers, deps: &mut Dependencies) -> LoadResult<T> {
        let f: LoadFn<T> = unsafe { std::mem::transmute(self.func) };
        let (provider, metadata) = providers.find_provider(path)?;
        // track local file dependencies for hot reloading
        if let Some(ref local_path) = metadata.local_path {
            deps.add_local_file(local_path);
        }
        let result = f(path, &metadata, provider, deps);

        match result {
            Err(err) => {
                error!("failed to load asset `{}`: {}", path.as_str(), err);
                Err(err)
            }
            Ok(asset) => Ok(asset),
        }

        //let bytes = provider.load(path)?;
        /*let result = match self.kind {
            LoaderKind::FromSlice => {
                let f: LoadFromSliceFn<T> = unsafe { std::mem::transmute(self.func) };
                // TODO error handling
                let (provider, metadata) = providers.find_provider(path)?;
                let bytes = provider.load(path)?;
                // track local file dependencies for hot reloading
                if let Some(ref local_path) = metadata.local_path {
                    deps.add_local_file(local_path);
                }
                f(&bytes, &metadata, deps)
            }
            LoaderKind::FromStaticSlice => {
                // Note that we shouldn't reload assets loaded from static slices, since the data
                // isn't supposed to change
                let f: LoadFromStaticSliceFn<T> = unsafe { std::mem::transmute(self.func) };
                let (provider, metadata) = providers.find_provider(path)?;
                let bytes = provider.load_static(path)?;
                f(bytes, &metadata, deps)
            }
            LoaderKind::FromPath => {
                let f: LoadFromPathFn<T> = unsafe { std::mem::transmute(self.func) };
                f(path, deps)
            }
        };*/

    }
}

type AssetStorage<T> = RwLock<LoadResult<T>>;

struct Entry<T: ?Sized = dyn Any + Send + Sync> {
    path: VfsPathBuf,
    dirty: AtomicBool,
    loader: Loader,
    #[cfg(feature = "hot_reload")]
    dependencies: Dependencies,
    asset: T,
}

impl<T: Asset> Entry<AssetStorage<T>> {
    fn reload(&self) {
        let mut deps = Dependencies::new(&self.path);
        let providers = Providers::get().read().unwrap();
        let result = self.loader.load(&self.path, &providers, &mut deps);
        // Mark as clean before reloading, because some loaders may immediately modify/rebuild
        // the underlying asset file, triggering another reload. This is the case, for example,
        // with shader archives in hot-reload mode, which are automatically rebuilt if their
        // source files have a later modification time.
        self.dirty.store(false, Relaxed);

        // swap the asset
        // FIXME: we only update the asset;
        //        we assume that the dependencies don't change but that may not be true
        let mut asset = self.asset.write().unwrap();
        *asset = result;
    }
}

/*
// separate struct so that we can unsize Arc<Entry<WithLoader<T>>> to Arc<Entry<dyn Any + Send + Sync>>
// (the function pointers block unsized coercion)
struct WithLoader<T> {
    loader: Loader<T>,
    asset: T,
}

trait Reload {
    fn reload(&mut self, path: &VfsPath) -> Dependencies;
}

impl<T: Asset> Reload for WithLoader<T> {
    fn reload(&mut self, path: &VfsPath) -> Dependencies {
        // Issue: the asset may be in use (locked for reading).
        //
        // We have several options:
        // - on reload, remove the entry from the cache

        let mut deps = Dependencies::new();
        let providers = Providers::get().read().unwrap();
        let new_asset = load_asset(&self.loader, path, &providers, &mut deps);
        self.asset = new_asset;
        deps
    }
}*/

impl Entry {
    fn downcast_ref<T: Asset>(&self) -> Option<&Entry<AssetStorage<T>>> {
        if self.loader.type_id == TypeId::of::<T>() {
            // SAFETY: we checked the type id
            Some(unsafe { &*(self as *const _ as *const Entry<AssetStorage<T>>) })
        } else {
            None
        }
    }

    fn downcast<T: Asset>(self: Arc<Self>) -> Option<Arc<Entry<AssetStorage<T>>>> {
        if self.loader.type_id == TypeId::of::<T>() {
            // SAFETY: we checked the type id
            Some(unsafe { Arc::from_raw(Arc::into_raw(self) as *const Entry<AssetStorage<T>>) })
        } else {
            None
        }
    }

    fn reload_dyn(&self) {
        (self.loader.reload)(self);
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
    dependencies: HashSet<CacheKey>,
    #[cfg(feature = "hot_reload")]
    local_files: Debouncer<RecommendedWatcher>,
}

impl Dependencies {
    fn new(path: &VfsPath) -> Self {
        #[cfg(feature = "hot_reload")]
        {
            Self {
                dependencies: HashSet::new(),
                local_files: new_debouncer(std::time::Duration::from_millis(500), {
                    let path = path.to_path_buf();
                    move |event| {
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
            let key = CacheKey {
                path: path.to_path_buf(),
                type_id: TypeId::of::<T>(),
            };
            self.dependencies.insert(key);
        }
    }

    pub fn add<T: Asset>(&mut self, handle: &Handle<T>) {
        self.add_path::<T>(&handle.0.path);
    }

    /// Adds a dependency on a file on the local file system.
    ///
    /// If hot reloading is enabled, changes to the file will trigger asset reloads.
    pub fn add_local_file<P: AsRef<Path>>(&mut self, path: P) {
        #[cfg(feature = "hot_reload")]
        {
            let path = path.as_ref();
            debug!("watching for changes: `{}`", path.display());
            self.local_files
                .watcher()
                .watch(path, RecursiveMode::NonRecursive)
                .unwrap()
        }
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
            inner: RwLock::new(Inner {
                by_path: HashMap::new(),
                dependency_graph: HashMap::new(),
            }),
            dirty_paths: Mutex::new(Default::default()),
        }
    }

    unsafe fn insert_inner<T: Asset>(&self, path: &VfsPath, loader: Loader) -> Handle<T> {
        let key = CacheKey {
            path: path.to_path_buf(),
            type_id: TypeId::of::<T>(),
        };

        // Check if an entry already exists.
        // The cache is locked only for the duration of the check.
        if let Some(existing) = self.inner.read().unwrap().by_path.get(&key) {
            return Handle::new(existing.clone().downcast().expect("invalid asset type stored in cache"));
        }

        // Cache unlocked here.

        // Read the asset file, and load the asset.
        #[cfg(feature = "hot_reload")]
        let dependencies;

        let entry = {
            let mut deps = Dependencies::new(path);
            let providers = Providers::get().read().unwrap();
            let asset = loader.load(path, &providers, &mut deps);

            #[cfg(feature = "hot_reload")]
            {
                dependencies = deps.dependencies.clone();
            }

            Arc::new(Entry {
                path: path.to_path_buf(),
                #[cfg(feature = "hot_reload")]
                dependencies: deps,
                dirty: Default::default(),
                loader,
                asset: RwLock::new(asset),
            })
        };

        // Insert the entry into the cache.
        // Note that another thread may have inserted the same entry in the meantime, but
        // there's nothing we can do about it.
        let mut inner = self.inner.write().unwrap();

        #[cfg(feature = "hot_reload")]
        {
            // update dependencies for hot reload
            for dep in dependencies.iter() {
                debug!("asset `{}` depends on `{}`", path.as_str(), dep.path.as_str());
                // TODO we track only one level of dependencies for now
                inner
                    .dependency_graph
                    .entry(dep.clone())
                    .or_default()
                    .insert(key.clone());
            }
        }

        inner.by_path.insert(key, entry.clone());
        Handle::new(entry)
    }

    /// Loads the asset file at the given path and invokes the given loader function to create the asset,
    /// then inserts the asset into the cache and returns a handle to it.
    pub fn load<T: Asset>(&self, path: &VfsPath, loader: LoadFn<T>) -> Handle<T> {
        unsafe { self.insert_inner(path, Loader::new(loader)) }
    }

    pub fn do_reload(&self) {
        #[cfg(feature = "hot_reload")]
        {
            let dirty_paths = mem::take(&mut *self.dirty_paths.lock().unwrap());

            let mut keys_to_reload: HashSet<CacheKey> = {
                let inner = self.inner.read().unwrap();
                dirty_paths
                    .iter()
                    .flat_map(|path| {
                        inner
                            .by_path
                            .keys()
                            .filter(|key| key.path.path_without_fragment() == &**path)
                            .cloned()
                    })
                    .collect()
            };

            loop {
                for k in mem::take(&mut keys_to_reload) {
                    let inner = self.inner.read().unwrap();
                    // skip entries that no longer exist (removed from cache, or last handle dropped)
                    let Some(entry) = inner.get_entry(&k) else { continue };

                    // skip if the entry is not ready to be reloaded: i.e. if any of its dependencies are dirty
                    let mut can_reload = true;
                    for dep_key in entry.dependencies.dependencies.iter() {
                        let Some(dep_entry) = inner.get_entry(dep_key) else {
                            continue;
                        };
                        if dep_entry.dirty.load(Relaxed) {
                            can_reload = false;
                            break;
                        }
                    }

                    if !can_reload {
                        // put back in the queue
                        keys_to_reload.insert(k);
                        continue;
                    }

                    // add dependents to the set of keys to reload
                    for dep in inner.dependency_graph.get(&k).into_iter().flatten() {
                        keys_to_reload.insert(dep.clone());
                    }

                    // unlock before reloading since reloading may create new entries in the cache
                    drop(inner);
                    entry.reload_dyn();
                }

                if keys_to_reload.is_empty() {
                    break;
                }
            }
        }
    }

    /// Called by providers to notify that a file has changed.
    pub fn asset_changed(&self, path: &VfsPath) {
        #[cfg(feature = "hot_reload")]
        self.dirty_paths
            .lock()
            .unwrap()
            .insert(path.path_without_fragment().to_path_buf());
    }

    /// Returns the global instance of the asset cache.
    pub fn instance() -> &'static AssetCache {
        static ASSET_CACHE: OnceLock<AssetCache> = OnceLock::new();
        let cache = ASSET_CACHE.get_or_init(|| AssetCache::new());
        cache
    }

    pub fn register_directory(path: impl AsRef<std::path::Path>) {
        let mut providers = Providers::get().write().unwrap();
        providers.register_overlay(Box::new(local_provider::LocalProvider::new(
            path.as_ref().to_path_buf(),
        )));
    }
}
