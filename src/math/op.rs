mod ident;
mod binary;
mod unary;
mod custom_unary;

use smallvec::SmallVec;
use crate::consts::*;
use crate::iterator::TensorIter;

pub(crate) enum Op<T: TensorDType> {
    Ident(IdentOp<T>),

    Add(BinaryOp<T>),
    Sub(BinaryOp<T>),
    Mul(BinaryOp<T>),

    ReLu(UnaryOp<T>),
    Sigmoid(UnaryOp<T>),

    Apply(CustomUnaryOp<T>),
    ApplyNoRetain(CustomUnaryOp<T>),

    MatMul(MatMulOp<T>),

    Transpose(ViewOp<T>),
    Slice(ViewOp<T>),
}


pub struct IdentOp<T: TensorDType> {
    input: SharedTensor<T>
}

struct BinaryOp<T: TensorDType> {
    input: [SharedExpr<T>; 2],
    it: TensorIter,
}

struct UnaryOp<T: TensorDType> {
    input: SharedExpr<T>,
    it: TensorIter,
}

struct CustomUnaryOp<T: TensorDType> {
    u_op: UnaryOp<T>,
    f_fwd: fn(T) -> T,
    f_bwd: fn(T) -> T,
}

struct MatMulOp<T: TensorDType> {
    a: SharedExpr<T>, b: SharedExpr<T>,
    it: TensorIter,
}

struct ViewOp<T: TensorDType> {
    input: SharedExpr<T>,
    it: TensorIter,
}