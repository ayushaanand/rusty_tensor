use crate::consts::*;
use crate::math::{Expr, Storage, op::Op};
use crate::tensor::Tensor;

use std::rc::Rc;

pub(crate) struct IdentOp;

pub fn ident<T: TensorFType>(input: Tensor<T>, requires_grad: bool) -> Rc<Expr<T>> {
    Rc::new(Expr {
        meta: input.meta,
        storage: Rc::new(Storage::new_from(input.data, requires_grad)),
        op: Op::Ident(IdentOp),
    })
}

pub fn ident_expr<T: TensorFType>(input: &Rc<Expr<T>>) -> Rc<Expr<T>> {
    Rc::new(Expr {
        meta: input.meta.clone(),
        storage: input.storage.clone(),
        op: Op::Ident(IdentOp),
    })
}
