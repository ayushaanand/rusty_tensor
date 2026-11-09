use std::rc::Rc;
use super::unary::{UnaryOp, build_unary_op};
use crate::consts::*;
use crate::math::Expr;
use crate::math::op::Op;

#[derive(Clone)]
pub(crate) struct CustomUnaryOp<T: TensorFType> {
    u_op: UnaryOp<T>,
    f: fn(T) -> T,
    f_dash: fn(T) -> T,
}

fn build_custom_unary_op<T: TensorFType>(
    input: &Rc<Expr<T>>,
    f: fn(T) -> T,
    f_dash: fn(T) -> T,
    op_var: fn(CustomUnaryOp<T>) -> Op<T>,
) -> TensorResult<Rc<Expr<T>>> {
    build_unary_op(input, |base| {
        op_var(CustomUnaryOp {
            u_op: base,
            f,
            f_dash,
        })
    })
}

pub fn apply<T: TensorFType>(
    input: &Rc<Expr<T>>,
    f: fn(T) -> T,
    f_dash: fn(T) -> T,
) -> TensorResult<Rc<Expr<T>>> {
    build_custom_unary_op(input, f, f_dash, Op::Apply)
}

impl<T: TensorFType> CustomUnaryOp<T> {
    pub(super) fn fwd(&self, out: &mut [T]) {
        self.u_op.fwd(out, self.f)
    }

    pub(super) fn bwd(&self, grad: &[T], out_val: &[T]) {
        self.u_op.bwd(grad, out_val, |_, a| (self.f_dash)(a))
    }
}
