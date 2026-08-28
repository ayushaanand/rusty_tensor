use crate::consts::*;
use crate::math::op::{CustomUnaryOp, Op};
use crate::math::op::Op::Apply;
use crate::math::op::unary::build_unary_op;

pub fn build_custom_unary_op<T: TensorDType> (
    input: &SharedExpr<T>,
    retain_data: bool,
    f_fwd: fn(T) -> T, f_bwd: fn(T) -> T,
    op_var: fn(CustomUnaryOp<T>) -> Op<T>
) -> TensorResult<SharedExpr<T>> {
     build_unary_op(
        input, retain_data,
        |base|  op_var(CustomUnaryOp {
            u_op: base,
            f_fwd,
            f_bwd
        })
    )
}

pub fn apply<T: TensorDType>(
    input: &SharedExpr<T>,
    f_fwd: fn(T) -> T, f_bwd: fn(T) -> T
) -> TensorResult<SharedExpr<T>> {
    build_custom_unary_op(
        input, false, 
        f_fwd, f_bwd,
        Apply
    )    
}

pub fn apply_no_retain<T: TensorDType>(
    input: &SharedExpr<T>,
    f_fwd: fn(T) -> T, f_bwd: fn(T) -> T
) -> TensorResult<SharedExpr<T>> {
    build_custom_unary_op(
        input, false,
        f_fwd, f_bwd,
        Apply
    )
}