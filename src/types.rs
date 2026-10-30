use num_traits::Zero;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

pub(crate) struct Shared<T>(Rc<RefCell<T>>);

impl<T> Shared<T> {
    pub(crate) fn new(val: T) -> Self {
        Self(Rc::new(RefCell::new(val)))
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Deref for Shared<T> {
    type Target = RefCell<T>;
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

pub(crate) struct SharedSlice<T>(Shared<Box<[T]>>);

impl<T> SharedSlice<T> {
    pub(crate) fn new(v: impl Into<Box<[T]>>) -> Self {
        SharedSlice(Shared::new(v.into()))
    }
}

impl<T: Zero + Clone> SharedSlice<T> {
    pub(crate) fn zeroes(len: usize) -> Self {
        Self::new(vec![T::zero(); len])
    }
}

impl<T> Clone for SharedSlice<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Deref for SharedSlice<T> {
    type Target = RefCell<Box<[T]>>;
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

pub(crate) const SMALL_SIZE: usize = 8;
pub(crate) type SmallSlice<T> = smallvec::SmallVec<[T; SMALL_SIZE]>;
