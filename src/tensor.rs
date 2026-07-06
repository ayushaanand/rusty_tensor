use crate::consts::*;

use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

/// Type-erased metadata describing a tensor's memory layout.
///
/// Contains only the shape, stride, offset, and element count — no data or type information.
/// This allows the [`Odometer`](crate::odometer::Odometer) to iterate over tensors of
/// different element types without requiring a generic `T` parameter.
#[derive(Debug)]
pub(crate) struct TensorMeta {
    pub(crate) shape: Vec<DimType>,
    pub(crate) stride: Vec<SizeType>,
    pub(crate) offset: SizeType,
    pub(crate) total_elems: SizeType,
}

/// A multi-dimensional tensor backed by a flat contiguous `Vec<T>`.
///
/// All fields are `pub(crate)` — external users cannot access them directly.
/// Use public getter methods for read access.
///
/// # Stride Convention
/// - Strides are computed in row-major (C-contiguous) order.
/// - Broadcast dimensions (`shape[i] == 1`) have `stride[i] == 0`,
///   causing the Odometer to repeatedly read the same position along that axis.
///
/// # Ownership
/// All constructors return `SharedTensor<T>` (`Rc<RefCell<Tensor<T>>>`).
/// The tensor is always heap-allocated and reference-counted.
#[derive(Debug)]
pub(crate) struct Tensor<T: TensorDType> {
    // Basic Variables
    pub(crate) meta: TensorMeta,
    pub(crate) data: Vec<T>,
}

impl<T: TensorDType> Tensor<T> {
    /// Creates a new tensor filled with `def_val` and wraps it in a `SharedTensor<T>`.
    ///
    /// # Arguments
    /// * `shape` - Dimensions of the tensor (e.g., `vec![3, 2, 1]`). Ownership is moved
    ///   into the struct.
    /// * `def_val` - The value to fill every element with (equivalent to `torch.full()`).
    ///
    /// # Returns
    /// `SharedTensor<T>` — an `Rc<RefCell<Tensor<T>>>` wrapping the newly created tensor.
    pub fn new(shape: Vec<DimType>, def_val: T) -> SharedTensor<T> {
        let mut elems: SizeType = 1;

        let mut stride: Vec<SizeType> = shape
            .iter()
            .rev()
            .map(|&s| {
                let curr_stride = if s == 1 { SIZE_ZERO } else { elems };
                elems *= s as SizeType;
                curr_stride
            })
            .collect();

        stride.reverse();

        let new_tensor = Tensor {
            meta: TensorMeta {
                shape,
                stride,
                total_elems: elems,
                offset: SIZE_ZERO,
            },
            data: vec![def_val; elems as usize],
        };

        Rc::new(RefCell::new(new_tensor))
    }

    /// Returns the number of dimensions (rank) of this tensor.
    pub fn dim(&self) -> DimType {
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

    /// Returns an immutable reference to this tensor's metadata.
    ///
    /// # Borrow Guidelines
    /// When passing the result to an [`Odometer`](crate::odometer::Odometer),
    /// the `RefCell` borrow guard must be bound to a `let` variable so that
    /// the `TensorMeta` reference outlives the Odometer:
    ///
    /// ```rust
    /// let t1_borrow = t1.borrow();           // guard lives until end of scope
    /// let meta = t1_borrow.get_meta();       // reference borrows from guard
    /// let odo = Odometer::new([meta], 0);    // meta outlives odo ✓
    /// ```
    ///
    /// This will **not** compile:
    /// ```rust,compile_fail
    /// let odo = Odometer::new([t1.borrow().get_meta()], 0);  // guard drops immediately ✗
    /// ```
    pub fn get_meta(&self) -> &TensorMeta {
        &self.meta
    }

    /// Returns a mutable reference to this tensor's metadata.
    ///
    /// # Safety Note
    /// Modifying `shape` without recalculating `stride` and `total_elems`
    /// will corrupt the tensor's memory layout.
    pub fn get_meta_mut(&mut self) -> &mut TensorMeta {
        &mut self.meta
    }
}
