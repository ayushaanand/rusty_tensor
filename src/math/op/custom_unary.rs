use crate::consts::*;
use crate::math::Expr;
use crate::math::op::unary::build_unary_op;
use crate::math::op::{CustomUnaryOp, Op};
use crate::shared_types::Shared;

fn build_custom_unary_op<T: TensorFType>(
    input: &Shared<Expr<T>>,
    retain_data: bool,
    f_fwd: fn(T) -> T,
    f_bwd: fn(T) -> T,
    op_var: fn(CustomUnaryOp<T>) -> Op<T>,
) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, retain_data, |base| {
        op_var(CustomUnaryOp {
            u_op: base,
            f_fwd,
            f_bwd,
        })
    })
}

pub fn apply<T: TensorFType>(
    input: &Shared<Expr<T>>,
    f_fwd: fn(T) -> T,
    f_bwd: fn(T) -> T,
) -> TensorResult<Shared<Expr<T>>> {
    build_custom_unary_op(input, true, f_fwd, f_bwd, Op::Apply)
}

pub fn apply_no_retain<T: TensorFType>(
    input: &Shared<Expr<T>>,
    f_fwd: fn(T) -> T,
    f_bwd: fn(T) -> T,
) -> TensorResult<Shared<Expr<T>>> {
    build_custom_unary_op(input, false, f_fwd, f_bwd, Op::ApplyNoRetain)
}
