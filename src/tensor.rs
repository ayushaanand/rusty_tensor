use crate::consts::*;

use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use smallvec::{smallvec, SmallVec};

/// Type-erased metadata describing a tensor's memory layout.
///
/// Contains only the shape, stride, offset, etc. — no data or type
/// information. This allows specific TensorDType agnostic methods like broadcasting.
#[derive(Debug)]
pub struct TensorMeta {
    pub(crate) shape: SmallDimVec,
    pub(crate) stride: SmallSizeVec,
    pub(crate) offset: SizeType,
    pub(crate) total_elems: SizeType,
}

impl TensorMeta {
    pub fn new(shape: SmallDimVec) -> Self {
        let mut res = TensorMeta {
            stride: smallvec![SIZE_ZERO; shape.len()],
            shape: SmallVec::from(shape),
            offset: SIZE_ZERO,
            total_elems: 1
        };

        if res.shape.contains(&DIM_ZERO) {
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
/// Composed of a [`TensorMeta`] (shape, stride, offset, total_elems) and a
/// `data: Vec<T>` holding the actual elements. All fields are `pub(crate)`.
///
/// # Constructor
/// [`new`](Tensor::new) return [`SharedTensor<T>`](crate::consts::SharedTensor).
/// The tensor is always heap-allocated and reference-counted
/// for backward grad purposes.
#[derive(Debug)]
pub struct Tensor<T: TensorDType> {
    pub(crate) meta: TensorMeta,
    pub(crate) data: Vec<T>,
}

impl<T: TensorDType> Tensor<T> {
    /// Creates a new tensor filled with `def_val` and wraps it in a [`SharedTensor<T>`](SharedTensor).
    ///
    /// Strides are computed in row-major (C-contiguous) order. Dimensions with
    /// `shape[i] == 1` get `stride[i] == 0` to enable transparent broadcasting.
    ///
    /// # Arguments
    /// * `shape` - Dimensions of the tensor (e.g., `vec![3, 2, 1]`). Ownership is
    ///   moved into the struct.
    /// * `def_val` - The value to fill every element with (equivalent to `torch.full()`).
    ///
    ///  # Returns
    /// [`SharedTensor<T>`](SharedTensor<T>) — an `Rc<RefCell<Tensor<T>>>` wrapping the newly created tensor.
    pub fn new(shape: SmallVec<[DimType; 8]>, def_val: T) -> SharedTensor<T> {
        let new_meta = TensorMeta::new(shape);

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
    ///
    /// # Arguments
    /// * `idx` - Flat index into the tensor's data array.
    pub fn get(&self, idx: SizeType) -> Option<T> {
        if idx < self.meta.total_elems {
            Some(self.get_unsafe(idx))
        } else {
            None
        }
    }

    /// Returns the element at the given flat index without bounds checking.
    ///
    /// # Arguments
    /// * `idx` - Flat index into the tensor's data array. Must be `< total_elems`.
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

    #[pyfunction(name= "compute_strides")]
    fn py_test_compute_strides(shape: Vec<DimType>) -> Vec<SizeType> {
        let res = TensorMeta::new(SmallDimVec::from(shape));
        res.stride.to_vec()
    }
}