use crate::consts::TensorDType;
use num_traits::Zero;
use std::cell::{Ref, RefCell, RefMut, UnsafeCell};
use std::ops::{Deref, Index};
use std::rc::Rc;

// --------- Shared<T> -----------------------------------------------------------------
pub(crate) struct Shared<T>(Rc<RefCell<T>>);

impl<T> Shared<T> {
    pub(crate) fn new(val: T) -> Self {
        Self(Rc::new(RefCell::new(val)))
    }
    pub(crate) fn as_rc(&self) -> &Rc<RefCell<T>> {
        &self.0
    }
    pub(crate) fn borrow(&self) -> Ref<'_, T> {
        self.0.borrow()
    }
    pub(crate) fn borrow_mut(&self) -> RefMut<'_, T> {
        self.0.borrow_mut()
    }
}

// pub(crate) struct Shared<T>(Rc<UnsafeCell<T>>);
//
// impl<T> Shared<T> {
//     pub(crate) fn new(val: T) -> Self { Self(Rc::new(UnsafeCell::new(val)).into()) }
//
//     pub(crate) fn borrow(&self) -> &T { unsafe{ &*self.0.get() } }
//     pub(crate) fn borrow_mut(&self) -> &mut T { unsafe{ &mut *self.0.get() } }
// }

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

// --------- SharedSlice<T> ------------------------------------------------------------
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
    type Target = Shared<Box<[T]>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// --------- SmallSlice<T> ------------------------------------------------------------
pub(crate) const SMALL_SIZE: usize = 8;
pub(crate) type SmallSlice<T> = smallvec::SmallVec<[T; SMALL_SIZE]>;
