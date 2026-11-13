pub mod apply;
pub mod binary;
pub mod ident;
pub mod matmul;
pub mod unary;
pub mod view;

use super::Expr;
use crate::consts::TensorFType;
use std::rc::Rc;

use apply::ApplyOp;
use binary::BinaryOp;
use ident::IdentOp;
use matmul::MatMulOp;
use unary::UnaryOp;
use view::ViewOp;

pub(crate) enum Op<T: TensorFType> {
    Ident(IdentOp),

    Add(BinaryOp<T>),
    Sub(BinaryOp<T>),
    Mul(BinaryOp<T>),

    ReLu(UnaryOp<T>),
    SiLu(UnaryOp<T>),
    Sigmoid(UnaryOp<T>),
    Softplus(UnaryOp<T>),

    Apply(ApplyOp<T>),

    MatMul(MatMulOp<T>),

    Transpose(ViewOp<T>),
    Slice(ViewOp<T>),
}

impl<T: TensorFType> Op<T> {
    pub fn fwd(&self, out: &mut [T]) {
        match self {
            Op::Ident(_) => {}

            Op::Add(op) => op.fwd(out, |a, b| a + b),
            Op::Sub(op) => op.fwd(out, |a, b| a - b),
            Op::Mul(op) => op.fwd(out, |a, b| a * b),

            Op::ReLu(op) => op.fwd(out, unary::fwd_functions::relu),
            Op::SiLu(op) => op.fwd(out, unary::fwd_functions::silu),
            Op::Sigmoid(op) => op.fwd(out, unary::fwd_functions::sigmoid),
            Op::Softplus(op) => op.fwd(out, unary::fwd_functions::softplus),

            Op::Apply(op) => op.fwd(out),

            Op::MatMul(op) => op.fwd(out),

            Op::Transpose(_) | Op::Slice(_) => {}
        }
    }

    pub fn bwd(&self, grad: &[T], out_val: &[T]) {
        match self {
            Op::Ident(_) => {}

            Op::Add(op) => op.add_bwd(grad),
            Op::Sub(op) => op.sub_bwd(grad),
            Op::Mul(op) => op.mul_bwd(grad),

            // Z = f(A)
            Op::ReLu(op) => op.bwd(grad, out_val, unary::bwd_functions::relu),
            Op::SiLu(op) => op.bwd(grad, out_val, unary::bwd_functions::silu),
            Op::Sigmoid(op) => op.bwd(grad, out_val, unary::bwd_functions::sigmoid),
            Op::Softplus(op) => op.bwd(grad, out_val, unary::bwd_functions::softplus),

            Op::Apply(op) => op.bwd(grad),

            Op::MatMul(op) => op.bwd(grad),

            Op::Transpose(_) | Op::Slice(_) => {}
        }
    }

    pub fn inputs(&self) -> &[Rc<Expr<T>>] {
        match self {
            Op::Ident(_) => &[],

            Op::Add(op) | Op::Sub(op) | Op::Mul(op) => op.inputs(),

            // Z = f(A)
            Op::ReLu(op) | Op::SiLu(op) | Op::Sigmoid(op) | Op::Softplus(op) => op.inputs(),

            Op::Apply(op) => op.inputs(),

            Op::MatMul(op) => op.inputs(),

            Op::Transpose(op) | Op::Slice(op) => op.inputs(),
        }
    }

    pub fn needs_zeroed_data(&self) -> bool {
        !matches!(self, Op::Ident(_) | Op::Transpose(_) | Op::Slice(_))
    }

    pub fn is_view(&self) -> bool {
        matches!(self, Op::Transpose(_) | Op::Slice(_))
    }
}
