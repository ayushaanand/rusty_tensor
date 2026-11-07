use super::unary::{UnaryOp, build_unary_op};
use crate::consts::*;
use crate::math::Expr;
use crate::math::op::Op;
use crate::types::Shared;

fn build_custom_unary_op<T: TensorFType>(
    input: &Shared<Expr<T>>,
    f: fn(T) -> T,
    f_dash: fn(T) -> T,
    op_var: fn(CustomUnaryOp<T>) -> Op<T>,
) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, |base| {
        op_var(CustomUnaryOp {
            u_op: base,
            f,
            f_dash,
        })
    })
}

pub fn apply<T: TensorFType>(
    input: &Shared<Expr<T>>,
    f: fn(T) -> T,
    f_dash: fn(T) -> T,
) -> TensorResult<Shared<Expr<T>>> {
    build_custom_unary_op(input, f, f_dash, Op::Apply)
}

// --------- CustomUnary --------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct CustomUnaryOp<T: TensorFType> {
    u_op: UnaryOp<T>,
    f: fn(T) -> T,
    f_dash: fn(T) -> T,
}

impl<T: TensorFType> CustomUnaryOp<T> {
    pub(super) fn fwd(&mut self, out: &mut [T]) {
        self.u_op.fwd(out, self.f)
    }

    pub(super) fn bwd(&mut self, grad: &[T], out_val: &[T]) {
        self.u_op.bwd(grad, out_val, |_, a| (self.f_dash)(a))
    }
}
