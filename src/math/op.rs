mod binary;
mod custom_unary;
mod ident;
mod matmul;
mod unary;
mod view;

use crate::consts::TensorFType;
use binary::BinaryOp;
use custom_unary::CustomUnaryOp;
use ident::IdentOp;
use matmul::MatMulOp;
use unary::UnaryOp;
use view::ViewOp;

#[derive(Clone)]
pub enum Op<T: TensorFType> {
    Ident(IdentOp<T>),

    Add(BinaryOp<T>),
    Sub(BinaryOp<T>),
    Mul(BinaryOp<T>),

    ReLu(UnaryOp<T>),
    SiLu(UnaryOp<T>),
    Sigmoid(UnaryOp<T>),
    Softplus(UnaryOp<T>),

    Apply(CustomUnaryOp<T>),

    MatMul(MatMulOp<T>),

    Transpose(ViewOp<T>),
    Slice(ViewOp<T>),
}

impl<T: TensorFType> Op<T> {
    pub fn fwd(&self, out: &mut [T]) {
        let zero = T::zero();
        let one = T::one();
        match self {
            Op::Ident(_) => {}

            Op::Add(op) => op.fwd(out, |a, b| a + b),
            Op::Sub(op) => op.fwd(out, |a, b| a - b),
            Op::Mul(op) => op.fwd(out, |a, b| a * b),

            Op::ReLu(op) => op.fwd(out, |x| T::max(x, zero)),
            Op::SiLu(op) => op.fwd(out, |x| x / (one + T::exp(-x))),
            Op::Sigmoid(op) => op.fwd(out, |x| one / (one + T::exp(-x))),
            Op::Softplus(op) => op.fwd(out, |x| T::ln(one + T::exp(x))),

            Op::Apply(op) => op.fwd(out),

            Op::MatMul(op) => op.fwd(out),

            Op::Transpose(_) | Op::Slice(_) => {}
        }
    }

    pub fn bwd(&self, grad: &[T], out_val: &[T]) {
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
            Op::Softplus(op) => op.bwd(grad, out_val, |z, _| one - T::exp(-z)),

            Op::Apply(op) => op.bwd(grad, out_val),

            Op::MatMul(op) => op.bwd(grad),

            Op::Transpose(_) | Op::Slice(_) => {}
        }
    }
}
