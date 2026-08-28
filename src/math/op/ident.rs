use std::cell::RefCell;
use std::rc::Rc;
use crate::consts::*;
use crate::math::{expr::Expr, op::*};

pub fn ident<T: TensorDType>(input: &SharedTensor<T>, requires_grad: bool) -> SharedExpr<T> {
    let op = Op::Ident(IdentOp { input: input.clone() });

    let expr = Expr {
        meta: input.borrow().meta.clone(),
        data: None,
        grad: None,
        op,
        requires_grad,
        retain_data: false
    };

    Rc::new(RefCell::new(expr))
}