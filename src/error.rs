#[derive(Debug, PartialEq)]
pub enum TensorError {
    BroadcastError,
    UnexEmptyInputError,
    UnexZeroElemError,
    SkipDimError,
    AliasError,
    MatmulShapeError,
    ShapeError,
    RangeError,
}

pub use TensorError::*;
