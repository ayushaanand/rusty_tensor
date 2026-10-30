use crate::consts::TensorFType;
use crate::math::op::{Op, ViewOp};
use crate::math::{Expr, Var};
use crate::tensor::TensorMeta;
use crate::types::Shared;

fn build_view_op<T: TensorFType>(
    input: &Shared<Expr<T>>,
    new_meta: TensorMeta,
    op_var: impl FnOnce(ViewOp<T>) -> Op<T>,
) -> Shared<Expr<T>> {
    let op = ViewOp {
        input: input.clone(),
    };

    let borrow = input.borrow();
    Shared::new(Expr {
        var: Var {
            meta: new_meta,
            ..borrow.var.clone()
        },
        ..borrow.clone()
    })
}

pub fn transpose<T: TensorFType>(input: &Shared<Expr<T>>, i: usize, j: usize) -> Shared<Expr<T>> {
    let borrow = input.borrow();

    let mut new_meta = borrow.var.meta.clone();
    new_meta.shape.swap(i, j);
    (new_meta.stride, new_meta.total_elems) =
        TensorMeta::strides_and_total_elems(new_meta.shape.as_slice());

    build_view_op(&input, new_meta, Op::Transpose)
}
