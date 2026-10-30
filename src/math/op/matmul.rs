use crate::broadcast::compute_broadcast_shape_const;
use crate::consts::{TensorFType, TensorResult};
use crate::iterator::TensorIter;
use crate::math::{Expr, Var, op};
use crate::tensor::TensorMeta;
use crate::types::{Shared, SharedSlice};
use std::array;

pub fn mm<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
) -> TensorResult<Shared<Expr<T>>> {
    const SKIP: usize = 2;

    let inputs: [_; 2] = [a.borrow(), b.borrow()];
    let out_shape = compute_broadcast_shape_const::<2>(
        array::from_fn(|i| inputs[i].var.meta.shape.as_slice()),
        SKIP,
    )?;
    let out_meta = TensorMeta::new(out_shape.as_slice());
    let it = TensorIter::new(
        std::iter::once(&out_meta).chain(inputs.iter().map(|i| &i.var.meta)),
        SKIP,
    )?;

    let op = op::Op::MatMul(op::MatMulOp {
        a: a.clone(),
        b: b.clone(),
        it,
    });

    let requires_grad = inputs.iter().any(|i| i.var.requires_grad);
    let grad = if requires_grad {
        Some(SharedSlice::zeroes(out_meta.total_elems))
    } else {
        None
    };

    Ok(Shared::new(Expr {
        var: Var {
            data: SharedSlice::zeroes(out_meta.total_elems),
            meta: out_meta,
            requires_grad,
        },
        grad,
        op,
        retain_data: false,
    }))
}
