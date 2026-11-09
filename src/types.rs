use num_traits::Zero;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

// --------- SharedSlice<T> ------------------------------------------------------------
pub(crate) struct SharedSlice<T>(Rc<RefCell<Box<[T]>>>);

impl<T> SharedSlice<T> {
    pub fn new(v: impl Into<Box<[T]>>) -> Self {
        Self(Rc::new(RefCell::new(v.into())))
    }

    pub fn borrow(&self) -> Ref<'_, Box<[T]>> { self.0.borrow() }
    pub fn borrow_mut(&self) -> RefMut<'_, Box<[T]>> { self.0.borrow_mut() }
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

// --------- SmallSlice<T> ------------------------------------------------------------
pub(crate) const SMALL_SIZE: usize = 8;
pub(crate) type SmallSlice<T> = smallvec::SmallVec<[T; SMALL_SIZE]>;
