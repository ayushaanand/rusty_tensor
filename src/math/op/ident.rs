use crate::consts::*;
use crate::math::{
    Expr, Var,
    op::{IdentOp, Op},
};
use crate::types::{Shared, SharedSlice};

pub fn ident<T: TensorFType>(
    input: &Shared<Var<T>>,
    requires_grad: bool,
) -> TensorResult<Shared<Expr<T>>> {
    let borrow = input.borrow();

    let op = Op::Ident(IdentOp {
        input: input.clone(),
    });

    let grad = if requires_grad {
        Some(SharedSlice::zeroes(borrow.meta.total_elems))
    } else {
        None
    };

    Ok(Shared::new(Expr {
        var: Var {
            requires_grad,
            ..borrow.clone()
        },
        grad,
        op,
        retain_data: false,
    }))
}
