use crate::consts::*;
use crate::math::{Expr, Storage, op::Op};
use crate::tensor::Tensor;

use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct IdentOp;

pub fn ident<T: TensorFType>(input: Tensor<T>, requires_grad: bool) -> Rc<Expr<T>> {
    Rc::new(Expr {
        meta: input.meta,
        storage: Rc::new(Storage::new_from(input.data, requires_grad)),
        op: Op::Ident(IdentOp),
    })
}
