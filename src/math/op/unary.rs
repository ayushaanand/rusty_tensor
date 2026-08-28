use std::cell::RefCell;
use std::rc::Rc;
use std::iter::once;

use crate::consts::*;
use crate::error::*;
use crate::iterator::TensorIter;
use crate::math::expr::Expr;
use crate::math::op::*;
use crate::tensor::TensorMeta;

pub fn build_unary_op<T: TensorDType>(
    input: &SharedExpr<T>,
    retain_data: bool,
    op_var: impl FnOnce(UnaryOp<T>) -> Op<T>
) -> TensorResult<SharedExpr<T>> {
    let borrow_input = input.borrow();

    let out_meta = TensorMeta::new(borrow_input.meta.shape.as_slice());

    let it = TensorIter::new(
        [&out_meta, &borrow_input.meta],
        0
    )?;

    let u_op = UnaryOp {
        input: input.clone(),
        it
    };

    let op = op_var(u_op);

    let total_elems = borrow_input.meta.total_elems;
    let grad = if borrow_input.requires_grad { Some(vec![T::zero(); total_elems]) } else { None };

    let expr = Expr {
        meta: borrow_input.meta.clone(),
        data: Some(Rc::new(vec![T::zero(); total_elems])),
        grad,
        op,
        requires_grad: borrow_input.requires_grad,
        retain_data: true
    };

    Ok(Rc::new(RefCell::new(expr)))
}

fn relu<T: TensorDType>(input: &SharedExpr<T>) -> TensorResult<SharedExpr<T>> {
    build_unary_op(input, false, Op::ReLu)
}

fn sigmoid<T: TensorDType>(input: &SharedExpr<T>) -> TensorResult<SharedExpr<T>> {
    build_unary_op(input, true, Op::Sigmoid)
}