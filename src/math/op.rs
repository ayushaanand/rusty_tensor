mod binary;
mod custom_unary;
mod ident;
mod matmul;
mod unary;
mod view;

use crate::consts::TensorFType;
use crate::math::{Expr, Var};
use crate::types::Shared;
use binary::BinaryOp;
use custom_unary::CustomUnaryOp;
use matmul::MatMulOp;
use unary::UnaryOp;

#[derive(Clone)]
pub enum Op<T: TensorFType> {
    Ident(IdentOp<T>),

    Add(BinaryOp<T>),
    Sub(BinaryOp<T>),
    Mul(BinaryOp<T>),

    ReLu(UnaryOp<T>),
    SiLu(UnaryOp<T>),
    Sigmoid(UnaryOp<T>),

    Apply(CustomUnaryOp<T>),

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

            Op::ReLu(op) => op.fwd(out, |x| x.max(T::zero())),
            Op::SiLu(op) => op.fwd(out, |x| x / (T::one() + (-x).exp())),
            Op::Sigmoid(op) => op.fwd(out, |x| T::one() / (T::one() + (-x).exp())),

            Op::Apply(op) => op.fwd(out),

            Op::MatMul(op) => op.fwd(out),

            _ => {
                unreachable!()
            }
        }
    }

    pub fn bwd(&mut self, grad: &[T], out_val: &[T]) {
        let zero = T::zero();
        let one = T::one();
        match self {
            Op::Ident(_) => {}

            Op::Add(op) => op.add_bwd(grad),
            Op::Sub(op) => op.sub_bwd(grad),
            Op::Mul(op) => op.mul_bwd(grad),

            // Z = f(A)
            Op::ReLu(op) => op.bwd(grad, out_val, |z, _| if z > zero { one } else { zero }),
            Op::SiLu(op) => op.bwd(grad, out_val, |z, a|
                // f'f-1(a) = sigmoid + a(sigmoid - sigmoid^2) = z(1 + (1-z)/a) or 1/2 if a == 0
                if a == zero { one / (one + one) } else { z * (one + (one - z) / a) }
            ),
            Op::Sigmoid(op) => op.bwd(grad, out_val, |z, _| z * (one - z)),

            Op::Apply(op) => op.bwd(grad, out_val),

            Op::MatMul(op) => op.bwd(grad, out_val),

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
pub(crate) struct ViewOp<T: TensorFType> {
    input: Shared<Expr<T>>,
}
