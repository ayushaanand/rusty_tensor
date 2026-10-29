use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::{
    Expr, Var,
    op::{IdentOp, Op::Ident},
};
use crate::shared_types::{Shared, SharedSlice};

pub fn ident<T: TensorFType>(
    input: &Shared<Var<T>>,
    requires_grad: bool,
) -> TensorResult<Shared<Expr<T>>> {
    let borrow = input.borrow();

    let it = TensorIter::new([&borrow.meta], 0)?;
    let op = Ident(IdentOp {
        input: input.clone(),
        it,
    });

    let grad = if requires_grad {
        Some(SharedSlice::zeroes(borrow.meta.total_elems))
    } else {
        None
    };
    let expr = Expr {
        var: Var {
            meta: borrow.meta.clone(),
            data: borrow.data.clone(),
            requires_grad,
        },
        grad,
        op,
        retain_data: false,
    };

    Ok(Shared::new(expr))
}
