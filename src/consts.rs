/// Represents a single dimension value (max 255 per axis).
/// If testing, make sure to update PY_DIM_TYPE below.
pub(crate) type DimType = usize;
/// Zero-value sentinel for `DimType`.
pub(crate) const DIM_ZERO: DimType = 0 as DimType;

/// Represents flat memory positions, strides, and element counts.
/// If testing, make sure to update PY_SIZE_TYPE below.
pub(crate) type SizeType = usize;
/// Zero-value sentinel for `SizeType`.
pub(crate) const SIZE_ZERO: SizeType = 0 as SizeType;

use std::fmt::Display;
use std::iter::{Product, Sum};
use std::ops::{AddAssign, SubAssign};
trait PvtSuperTraits: Clone + Copy + Display + Sum + Product + AddAssign + SubAssign {}
impl<T> PvtSuperTraits for T where T: Clone + Copy + Display + Sum + Product + AddAssign + SubAssign {}

/// Unified super-trait for all valid tensor element types.
///
/// Blanket-implemented for any `T` satisfying [`Num`](num_traits::Num) + [`Clone`](Clone) +
/// [`Copy`](Copy) + [`Display`](Display) + [`Sum`](Sum) +
/// [`Product`](Product) + [`AddAssign`](AddAssign).
/// Use `T: TensorDType` on `impl` blocks instead of repeating the full bound list.
///
/// Valid concrete types include: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`,
/// `i64`, `f32`, `f64`.
#[allow(private_bounds)]
pub trait TensorDType: num_traits::Num + PvtSuperTraits {}
impl<T> TensorDType for T where T: num_traits::Num + PvtSuperTraits {}

#[allow(private_bounds)]
pub trait TensorFType: num_traits::Float + PvtSuperTraits {}
impl<T> TensorFType for T where T: num_traits::Float + PvtSuperTraits {}
use crate::error::TensorError;
pub type TensorResult<T> = Result<T, TensorError>;

#[cfg(feature = "python-testing")]
#[pyo3::pymodule(name = "consts")]
#[allow(dead_code)]
pub mod py_tests {
    #[pymodule_export]
    const PY_DIM_TYPE: &str = "uint8";
    #[pymodule_export]
    const PY_SIZE_TYPE: &str = "uint64";
}
