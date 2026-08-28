#[derive(Debug)]
pub enum TensorError {
    BroadcastError,
    UnexEmptyInputError,
    UnexZeroElemError,
    SkipDimError,
    AliasError,
    MatmulShapeError,
    ShapeError,
}

pub use TensorError::*;