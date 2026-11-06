use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::op::*;
use crate::math::{Expr, Var};
use crate::tensor::TensorMeta;
use crate::types::{Shared, SharedSlice};

pub fn build_unary_op<T: TensorFType>(
    input: &Shared<Expr<T>>,
    retain_data: bool,
    op_var: impl FnOnce(UnaryOp<T>) -> Op<T>,
) -> TensorResult<Shared<Expr<T>>> {
    let borrow_input = input.borrow();

    let out_meta = TensorMeta::new(borrow_input.var.meta.shape.as_slice());

    let it = TensorIter::new([&out_meta, &borrow_input.var.meta], 0)?;

    let op = op_var(UnaryOp {
        input: input.clone(),
        it,
    });

    let total_elems = borrow_input.var.meta.total_elems;
    let grad = borrow_input
        .requires_grad()
        .then(|| SharedSlice::zeroes(total_elems));

    Ok(Shared::new(Expr {
        var: Var {
            data: SharedSlice::zeroes(total_elems),
            ..borrow_input.var.clone()
        },
        grad,
        op,
        retain_data,
    }))
}

fn relu<T: TensorFType>(input: &Shared<Expr<T>>) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, false, Op::ReLu)
}

fn sigmoid<T: TensorFType>(input: &Shared<Expr<T>>) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, false, Op::Sigmoid)
}
