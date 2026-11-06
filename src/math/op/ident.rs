use crate::consts::*;
use crate::math::op::IdentOp;
use crate::math::{Expr, Var, op::Op};
use crate::types::{Shared, SharedSlice};

pub fn ident<T: TensorFType>(
    input: &Shared<Var<T>>,
    requires_grad: bool,
) -> TensorResult<Shared<Expr<T>>> {
    let borrow = input.borrow();

    let op = Op::Ident(IdentOp {
        input: input.clone(),
    });

    let grad = requires_grad.then(|| SharedSlice::zeroes(borrow.meta.total_elems));

    Ok(Shared::new(Expr {
        var: borrow.clone(),
        grad,
        op,
    }))
}
