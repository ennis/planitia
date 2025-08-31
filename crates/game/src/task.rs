use crate::context::get_context;

pub use futures::future::AbortHandle;

/// Spawns a task with a name.
///
/// The name is only used for debugging purposes and is not required to be unique.
///
/// # Arguments
/// * `name` - The name of the task.
/// * `future` - The future to run as a task.
///
pub fn spawn_named<F>(name: &str, future: F)
where
    F: Future<Output = ()> + 'static,
{
    get_context().executor.spawn(name.to_string(), future);
}

/// Spawns a task on the global executor.
pub fn spawn<F>(future: F)
where
    F: Future<Output = ()> + 'static,
{
    // For now, we don't do anything with the name, but it can be used for debugging.
    spawn_named("<anonymous>", future);
}

/// Spawns a task on the global executor and returns an abort handle.
pub fn spawn_abortable<F>(future: F) -> AbortHandle
where
    F: Future<Output = ()> + 'static,
{
    let (future, handle) = futures::future::abortable(future);
    spawn(async move {
        // ignore abort result
        let _ = future.await;
    });
    handle
}
