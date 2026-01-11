use std::cell::{Ref, RefCell};
use std::sync::OnceLock;
use threadbound::ThreadBound;

pub struct MainThreadComponent<T: 'static> {
    inner: OnceLock<ThreadBound<&'static RefCell<T>>>,
}

impl<T: 'static + Default> MainThreadComponent<T> {
    pub const fn new() -> Self {
        Self { inner: OnceLock::new() }
    }

    fn init(&'static self) -> &'static RefCell<T> {
        self.inner
            .get_or_init(|| ThreadBound::new(Box::leak(Box::new(RefCell::new(T::default())))))
            .get_ref()
            .expect("expected access from main thread")
    }

    pub fn borrow(&'static self) -> Ref<'static, T> {
        self.init().borrow()
    }

    pub fn borrow_mut(&'static self) -> std::cell::RefMut<'static, T> {
        self.init().borrow_mut()
    }
}
