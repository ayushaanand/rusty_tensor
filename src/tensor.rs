use crate::consts::*;

use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

/// Type-erased metadata describing a tensor's memory layout.
///
/// Contains only the shape, stride, offset, etc. — no data or type
/// information. This allows specific TensorDType agnostic methods like broadcasting.
#[derive(Debug)]
pub struct TensorMeta {
    pub(crate) shape: Vec<DimType>,
    /// `stride` is 0, whenever `shape` is 1
    pub(crate) stride: Vec<SizeType>,
    pub(crate) offset: SizeType,
    pub(crate) total_elems: SizeType,
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
    /// # Returns
    /// [`SharedTensor<T>`](SharedTensor<T>) — an `Rc<RefCell<Tensor<T>>>` wrapping the newly created tensor.
    pub fn new(shape: &[DimType], def_val: T) -> SharedTensor<T> {
        let (stride, total_elems) = Self::compute_strides(shape);

        let new_tensor = Tensor {
            meta: TensorMeta {
                shape: Vec::from(shape),
                stride,
                total_elems,
                offset: SIZE_ZERO,
            },
            data: vec![def_val; total_elems as usize],
        };

        Rc::new(RefCell::new(new_tensor))
    }

    fn compute_strides(shape: &[DimType]) -> (Vec<SizeType>, SizeType) {
        let mut stride: Vec<SizeType> = vec![SIZE_ZERO; shape.len()];
        
        if shape.contains(&DIM_ZERO) {
            return (stride, SIZE_ZERO);
        }

        let mut elems: SizeType = 1;
        for (sh, st) in shape.iter().rev().zip(stride.iter_mut().rev()) {
            *st = if *sh == 1 { SIZE_ZERO } else { elems };
            elems *= *sh as SizeType;
        }

        (stride, elems)
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
pub mod tensor_tests {
    use super::*;
    use pyo3::prelude::*;

    #[pyfunction]
    fn test_compute_strides(shape: Vec<DimType>) -> Vec<SizeType> {
        let (stride, _) = Tensor::<i32>::compute_strides(&shape);
        stride
    }
}