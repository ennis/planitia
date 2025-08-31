use futures::StreamExt;
use futures::future::LocalFutureObj;
use futures::stream::FuturesUnordered;
use std::cell::RefCell;
use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

struct NamedTaskFuture {
    name: String,
    future: LocalFutureObj<'static, ()>,
}

impl NamedTaskFuture {
    fn new(name: String, future: LocalFutureObj<'static, ()>) -> Self {
        Self { name, future }
    }
}

impl Future for NamedTaskFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        unsafe { self.map_unchecked_mut(|s| &mut s.future).poll(cx) }
    }
}

pub struct LocalExecutor {
    /// Tasks.
    tasks: RefCell<FuturesUnordered<NamedTaskFuture>>,
    /// Newly created tasks.
    incoming: RefCell<Vec<NamedTaskFuture>>,
}

impl fmt::Debug for LocalExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalExecutor").finish_non_exhaustive()
    }
}

impl LocalExecutor {
    /// Creates a new local executor.
    pub(crate) fn new() -> Self {
        Self {
            tasks: RefCell::new(FuturesUnordered::new()),
            incoming: RefCell::new(Vec::new()),
        }
    }

    /// Runs tasks until all are waiting or completed.
    pub(crate) fn run_tasks_until_stalled(&self, waker: &Waker) {
        let mut tasks = self.tasks.borrow_mut();
        let mut ctx = Context::from_waker(waker);

        loop {
            // insert incoming tasks first
            tasks.extend(self.incoming.borrow_mut().drain(..));
            match tasks.poll_next_unpin(&mut ctx) {
                Poll::Ready(Some(_)) => continue,
                Poll::Pending => break,
                Poll::Ready(None) => break,
            }
        }
    }

    pub(crate) fn spawn<F>(&self, name: String, future: F)
    where
        F: Future<Output = ()> + 'static,
    {
        let task = NamedTaskFuture::new(name.to_string(), LocalFutureObj::new(Box::new(future)));

        if let Ok(tasks) = self.tasks.try_borrow_mut() {
            tasks.push(task);
        } else {
            // If the borrow fails, we push the task to the incoming queue
            // to be processed later.
            self.incoming.borrow_mut().push(task);
        }
    }
}
