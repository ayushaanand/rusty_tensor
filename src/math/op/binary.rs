use std::rc::Rc;
use std::cell::RefCell;

use crate::broadcast::compute_broadcast_shape_iter;
use crate::iterator::TensorIter;
use crate::tensor::TensorMeta;
use crate::consts::*;
use crate::error::*;
use crate::math::{expr::Expr, op::*, op::Op::*};

pub fn build_binary_op<T: TensorDType>(
    a: &SharedExpr<T>, b: &SharedExpr<T>,
    op_var: fn(BinaryOp<T>) -> Op<T>
) -> TensorResult<SharedExpr<T>> {
    if a.as_ptr() == b.as_ptr() { return Err(AliasError); }

    let inputs = [a.borrow(), b.borrow()];

    let out_shape = compute_broadcast_shape_iter(
        inputs.iter().map(|ex| ex.meta.shape.as_slice()),
        0,
    )?;

    let out_meta = TensorMeta::new(out_shape.as_slice());

    let it = TensorIter::new(
        std::iter::once(&out_meta).chain(inputs.iter().map(|i| &i.meta)),
        0
    )?;

    let op = op_var(BinaryOp {
        input: [a.clone(), b.clone()],
        it
    });

    let total_elems = out_meta.total_elems;

    let expr = Expr {
        meta: out_meta,
        data: Some(Rc::new(vec![T::zero(); total_elems])),
        grad: Some(vec![T::zero(); total_elems]),
        op,
        requires_grad: inputs.into_iter().map(|ex| ex.requires_grad).any(|b| b),
        retain_data: false
    };

    Ok(Rc::new(RefCell::new(expr)))
}

pub fn add<T: TensorDType>(a: &SharedExpr<T>, b: &SharedExpr<T>) -> TensorResult<SharedExpr<T>> {
    build_binary_op(a, b, Add)
}

pub fn sub<T: TensorDType>(a: &SharedExpr<T>, b: &SharedExpr<T>) -> TensorResult<SharedExpr<T>> {
    build_binary_op(a, b, Sub)
}

pub fn mul<T: TensorDType>(a: &SharedExpr<T>, b: &SharedExpr<T>) -> TensorResult<SharedExpr<T>> {
    build_binary_op(a, b, Mul)
}