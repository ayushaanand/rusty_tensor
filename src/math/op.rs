mod binary;
mod custom_unary;
mod ident;
mod matmul;
mod unary;

use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::{Expr, Var};
use crate::shared_types::Shared;

pub(crate) enum Op<T: TensorFType> {
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

pub struct IdentOp<T: TensorFType> {
    input: Shared<Var<T>>,
    it: TensorIter,
}

pub struct BinaryOp<T: TensorFType> {
    input: [Shared<Expr<T>>; 2],
    it: TensorIter,
}

pub struct UnaryOp<T: TensorFType> {
    input: Shared<Expr<T>>,
    it: TensorIter,
}

pub struct CustomUnaryOp<T: TensorFType> {
    u_op: UnaryOp<T>,
    f_fwd: fn(T) -> T,
    f_bwd: fn(T) -> T,
}

pub struct MatMulOp<T: TensorFType> {
    a: Shared<Expr<T>>,
    b: Shared<Expr<T>>,
    it: TensorIter,
}

pub struct ViewOp<T: TensorFType> {
    input: Shared<Expr<T>>,
    it: TensorIter,
}
