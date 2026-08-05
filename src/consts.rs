/// Represents a single dimension value (max 255 per axis).
/// If testing, make sure to update PY_DIM_TYPE below.
pub type DimType = u8;
/// Zero-value sentinel for `DimType`.
pub const DIM_ZERO: DimType = 0 as DimType;

pub const SMALL_SIZE: usize = 8;
pub type SmallDimVec = smallvec::SmallVec<[DimType; SMALL_SIZE]>;

/// Represents flat memory positions, strides, and element counts.
/// If testing, make sure to update PY_SIZE_TYPE below.
pub type SizeType = u64;
/// Zero-value sentinel for `SizeType`.
pub const SIZE_ZERO: SizeType = 0 as SizeType;
pub type SmallSizeVec = smallvec::SmallVec<[SizeType; SMALL_SIZE]>;

use std::fmt::Display;

/// Unified super-trait for all valid tensor element types.
///
/// Blanket-implemented for any `T` satisfying `Num + Clone + Copy + Display`.
/// Use `T: TensorDType` on `impl` blocks instead of repeating the full bound list.
///
/// Valid concrete types include: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`,
/// `i64`, `f32`, `f64`.
pub trait TensorDType: num_traits::Num + Clone + Copy + Display {}
impl<T> TensorDType for T where T: num_traits::Num + Clone + Copy + Display {}

use crate::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

/// Shared ownership wrapper for all tensors in the engine.
///
/// Wraps `Tensor<T>` in `Rc` (shared ownership) and `RefCell` (interior mutability),
/// enabling safe shared mutable access required by the computation graph.
///
/// Use `.borrow()` for read access and `.borrow_mut()` for write access.
pub type SharedTensor<T> = Rc<RefCell<Tensor<T>>>;

#[cfg(feature = "python-testing")]
#[pyo3::pymodule(name = "consts")]
#[allow(dead_code)]
pub mod py_tests {
    #[pymodule_export]
    const PY_DIM_TYPE: &str = "uint8";
    #[pymodule_export]
    const PY_SIZE_TYPE: &str = "uint64";
}
