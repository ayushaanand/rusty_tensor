use crate::consts::*;
use crate::math::{Expr, Var, op::Op};
use crate::types::SharedSlice;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct IdentOp<T: TensorFType> {
    input: Rc<Var<T>>,
}

pub fn ident<T: TensorFType>(input: &Rc<Var<T>>, requires_grad: bool) -> Rc<Expr<T>> {
    let op = Op::Ident(IdentOp {
        input: Rc::clone(input),
    });

    let grad = requires_grad.then(|| SharedSlice::zeroes(input.meta.total_elems));

    Rc::new(Expr {
        var: Var::clone(input),
        grad,
        op,
    })
}
