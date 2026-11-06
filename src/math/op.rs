mod binary;
mod custom_unary;
mod ident;
mod matmul;
mod unary;
mod view;

use crate::consts::TensorFType;
use crate::iterator::TensorIter;
use crate::math::{Expr, Var};
use crate::types::Shared;
use binary::BinaryOp;

#[derive(Clone)]
pub enum Op<T: TensorFType> {
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

impl<T: TensorFType> Op<T> {
    pub fn fwd(&mut self, out: &mut [T]) {
        match self {
            Op::Ident(_) => {}

            Op::Add(op) => op.fwd(out, |a, b| a + b),
            Op::Sub(op) => op.fwd(out, |a, b| a - b),
            Op::Mul(op) => op.fwd(out, |a, b| a * b),

            _ => {
                unreachable!()
            }
        }
    }

    pub fn bwd(&mut self, grad: &[T], out_val: &[T]) {
        match self {
            Op::Ident(_) => {}

            Op::Add(op) => op.add_bwd(grad),
            Op::Sub(op) => op.sub_bwd(grad),
            Op::Mul(op) => op.mul_bwd(grad),

            _ => {
                unreachable!()
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct IdentOp<T: TensorFType> {
    input: Shared<Var<T>>,
}

#[derive(Clone)]
pub(crate) struct CustomUnaryOp<T: TensorFType> {
    u_op: UnaryOp<T>,
    f_fwd: fn(T) -> T,
    f_bwd: fn(T) -> T,
}

#[derive(Clone)]
pub(crate) struct MatMulOp<T: TensorFType> {
    a: Shared<Expr<T>>,
    b: Shared<Expr<T>>,
    it: TensorIter,

    // m*p @ p*n
    m: usize,
    p: usize,
    n: usize,
}

#[derive(Clone)]
pub(crate) struct UnaryOp<T: TensorFType> {
    input: Shared<Expr<T>>,
    it: TensorIter,
}

#[derive(Clone)]
pub(crate) struct ViewOp<T: TensorFType> {
    input: Shared<Expr<T>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Var;
    use crate::tensor::Tensor;
    use crate::types::Shared;

    #[test]
    fn test_ident() {}
}
