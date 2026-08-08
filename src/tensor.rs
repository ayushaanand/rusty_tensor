use crate::consts::*;

use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use smallvec::{SmallVec, smallvec};

/// Type-erased metadata describing a tensor's memory layout.
///
/// Holds shape, stride, offset, and element count — no data or dtype.
#[derive(Debug)]
pub struct TensorMeta {
    pub(crate) shape: SmallDimVec,
    pub(crate) stride: SmallSizeVec,
    pub(crate) offset: SizeType,
    pub(crate) total_elems: SizeType,
}

impl TensorMeta {
    /// Creates a `TensorMeta` from a shape slice.
    ///
    /// Computes C-contiguous (row-major) strides: `stride[i] = product(shape[i+1..])`.
    /// If any dimension is `0`, `total_elems` is set to `0` and strides remain `0`.
    /// Note: stride zeroing for broadcast dims (size-`1` dimensions) is done later
    /// by [`TensorIter`](crate::iterator::TensorIter), not here.
    ///
    /// # Example
    /// ```
    /// # use tensor_lib::tensor::TensorMeta;
    /// # use tensor_lib::iterator::TensorIter;
    /// let meta = TensorMeta::new(&[2, 3]);
    /// assert_eq!(TensorIter::new([&meta], 0).unwrap().count(), 6);
    ///
    /// let empty = TensorMeta::new(&[2, 0, 3]);
    /// assert!(TensorIter::new([&empty], 0).is_none());
    /// ```
    pub fn new(shape: &[DimType]) -> Self {
        let mut res = TensorMeta {
            stride: smallvec![SIZE_ZERO; shape.len()],
            shape: SmallVec::from(shape),
            offset: SIZE_ZERO,
            total_elems: 1,
        };

        if res.shape.contains(&DIM_ZERO) {
            res.total_elems = SIZE_ZERO;
            return res;
        }

        for (sh, st) in res.shape.iter().rev().zip(res.stride.iter_mut().rev()) {
            *st = res.total_elems;
            res.total_elems *= *sh as SizeType;
        }

        res
    }
}

/// A multi-dimensional strided tensor backed by a `Vec<T>`.
///
/// Always constructed via [`Tensor::new`], which returns a [`SharedTensor<T>`](crate::consts::SharedTensor)
/// (`Rc<RefCell<Tensor<T>>>`). Interact with it through `.borrow()` / `.borrow_mut()`.
#[derive(Debug)]
pub struct Tensor<T: TensorDType> {
    pub(crate) meta: TensorMeta,
    pub(crate) data: Vec<T>,
}

impl<T: TensorDType> Tensor<T> {
    /// Creates a tensor filled with `def_val` and wraps it in a [`SharedTensor<T>`](SharedTensor).
    ///
    /// `shape` is passed to [`TensorMeta::new`] for stride computation.
    /// `def_val` is equivalent to `torch.full`.
    pub fn new(shape: &[DimType], def_val: T) -> SharedTensor<T> {
        let new_meta = TensorMeta::new(&shape);

        let new_tensor = Tensor {
            data: vec![def_val; new_meta.total_elems as usize],
            meta: new_meta,
        };

        Rc::new(RefCell::new(new_tensor))
    }

    /// Returns the number of dimensions (rank) of this tensor.
    pub fn rank(&self) -> DimType {
        self.meta.shape.len() as DimType
    }

    /// Returns the element at the given flat index, or `None` if out of bounds.
    pub fn get(&self, idx: SizeType) -> Option<T> {
        if idx < self.meta.total_elems {
            Some(self.get_unsafe(idx))
        } else {
            None
        }
    }

    /// Returns the element at the given flat index without bounds checking.
    /// `idx` must be `< total_elems`.
    pub(crate) fn get_unsafe(&self, idx: SizeType) -> T {
        self.data[idx as usize]
    }

    /// Returns an immutable reference to this tensor's [`TensorMeta`].
    pub fn get_meta(&self) -> &TensorMeta {
        &self.meta
    }
}

#[cfg(feature = "python-testing")]
#[pyo3::pymodule(name = "tensor")]
pub(crate) mod py_tests {
    use super::*;
    use pyo3::prelude::*;

    #[pyfunction(name = "compute_strides")]
    fn py_test_compute_strides(shape: Vec<DimType>) -> Vec<SizeType> {
        let res = TensorMeta::new(&SmallDimVec::from(shape));
        res.stride.to_vec()
    }
}
