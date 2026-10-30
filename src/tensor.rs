use crate::consts::*;

use crate::error::ShapeError;
use crate::iterator::TensorIter;
use crate::types::SmallSlice;
use smallvec::smallvec;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

/// Type-erased metadata describing a tensor's memory layout.
/// Holds shape, stride, offset, and element count — no data or dtype.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorMeta {
    pub shape: SmallSlice<DimType>,
    pub stride: SmallSlice<SizeType>,
    pub offset: SizeType,
    pub total_elems: SizeType,
}

impl TensorMeta {
    /// Creates a `TensorMeta` from a shape slice.
    ///
    /// Computes C-contiguous (row-major) strides: `stride[i] = product(shape[i+1..]).`, with `0` offset
    /// # Example
    /// ```
    /// # use smallvec::SmallVec;
    /// # use tensor_lib::tensor::TensorMeta;
    /// let meta = TensorMeta::new(&[4, 3, 2, 1]);
    /// assert_eq!(meta.shape, SmallVec::from_buf([4, 3, 2, 1]));
    /// assert_eq!(meta.stride, SmallVec::from_buf([6, 2, 1, 1]));
    /// assert_eq!(meta.offset, 0);
    /// assert_eq!(meta.total_elems, 24)
    /// ```
    ///
    /// If any shape is `0`, strides become ALL 0, and total_elems is 0
    /// ```
    /// # use smallvec::SmallVec;
    /// # use tensor_lib::tensor::TensorMeta;
    /// let meta = TensorMeta::new(&[2, 0, 1]);
    /// assert_eq!(meta.shape, SmallVec::from_buf([2, 0, 1]));
    /// assert_eq!(meta.stride, SmallVec::from_buf([0, 0, 0]));
    /// assert_eq!(meta.offset, 0);
    /// assert_eq!(meta.total_elems, 0)
    /// ```
    pub fn new(shape: &[DimType]) -> Self {
        let (stride, total_elems) = Self::strides_and_total_elems(shape);

        TensorMeta {
            shape: SmallSlice::from(shape),
            offset: SIZE_ZERO,
            stride,
            total_elems,
        }
    }

    pub(crate) fn strides_and_total_elems(shape: &[DimType]) -> (SmallSlice<SizeType>, SizeType) {
        let mut strides = smallvec![SIZE_ZERO, shape.len()];
        let mut total_elems = 1;

        if shape.contains(&DIM_ZERO) {
            total_elems = SIZE_ZERO;
        } else {
            for (sh, st) in shape.iter().rev().zip(strides.iter_mut().rev()) {
                *st = total_elems;
                total_elems *= *sh as SizeType;
            }
        }

        (strides, total_elems)
    }
}

/// A multidimensional strided tensor, which supports all numerical types, backed by a `Box<[T]>`,
/// so number of elems cant be changed later.
/// Supports dtype which implements the [`TensorDType`](TensorDType) trait (eg, i32, f64, etc.)
/// Always constructed via [`Tensor::new`](Tensor::new)
/// (`Rc<RefCell<Tensor<T>>>`). Interact with it through `.borrow()` / `.borrow_mut()`.
#[derive(Debug, PartialEq)]
pub struct Tensor<T> {
    pub(crate) meta: TensorMeta,
    pub(crate) data: Box<[T]>,
}

impl<T: TensorDType> Tensor<T> {
    /// Creates a tensor filled with `def_val`
    pub fn new(shape: &[DimType], def_val: T) -> Self {
        let new_meta = TensorMeta::new(shape);

        Tensor {
            data: vec![def_val; new_meta.total_elems].into_boxed_slice(),
            meta: new_meta,
        }
    }

    /// Returns the number of dimensions (rank) of this tensor.
    ///
    /// # Example
    /// ```
    /// # use tensor_lib::tensor::Tensor;
    /// let t = Tensor::new(&[3, 2, 1], 0);
    /// assert_eq!(t.rank(), 3)
    pub fn rank(&self) -> DimType {
        self.meta.shape.len() as DimType
    }

    /// Returns the element at the given flat index, or `None` if out of bounds.
    pub fn get(&mut self, idx: SizeType) -> Option<T> {
        if idx < self.meta.total_elems {
            Some(self[idx])
        } else {
            None
        }
    }

    pub fn from_buf<I>(shape: &[DimType], buf: I) -> TensorResult<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let mut t = Self::new(shape, T::zero());
        let mut it = TensorIter::from_tensors([&t], 0)?;

        for i in buf {
            if it.done {
                break;
            } else {
                t[it.pos[0]] = i;
                it.advance()
            }
        }

        if it.done { Ok(t) } else { Err(ShapeError) }
    }

    pub fn unsqueeze(&mut self, dim: usize) {
        self.meta.shape.insert(dim, 1);
        self.meta.stride.insert(dim, SIZE_ZERO);
    }
}

impl<T: TensorDType> Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl<T: TensorDType> IndexMut<usize> for Tensor<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}

#[cfg(feature = "python-testing")]
#[pyo3::pymodule(name = "tensor")]
pub(crate) mod py_tests {
    use super::*;
    use pyo3::prelude::*;

    #[pyfunction(name = "compute_strides")]
    fn py_test_compute_strides(shape: Vec<DimType>) -> Vec<SizeType> {
        let res = TensorMeta::new(&SmallSlice::from(shape));
        res.stride.to_vec()
    }
}
