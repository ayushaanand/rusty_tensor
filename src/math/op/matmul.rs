use crate::broadcast::compute_broadcast_shape_const;
use crate::consts::{TensorFType, TensorResult};
use crate::error::MatmulShapeError;
use crate::iterator::TensorIter;
use crate::math::op::MatMulOp;
use crate::math::{Expr, Var, op};
use crate::tensor::TensorMeta;
use crate::types::{Shared, SharedSlice};
use std::array;

pub fn mm<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
) -> TensorResult<Shared<Expr<T>>> {
    const SKIP: usize = 2;

    let inputs = [a.borrow(), b.borrow()];
    let shapes: [_; 2] = array::from_fn(|i| inputs[i].var.meta.shape.as_slice());
    let lens: [_; 2] = array::from_fn(|i| inputs[i].var.meta.shape.len());

    let mut out_shape = compute_broadcast_shape_const(shapes, SKIP)?;
    let out_len = out_shape.len();

    // shapes.len() >= 2 is guaranteed due to skip_dims
    let m = shapes[0][lens[0] - 2];

    let p = shapes[0][lens[0] - 1];
    if p != shapes[1][lens[1] - 2] {
        return Err(MatmulShapeError);
    }

    let n = shapes[1][lens[1] - 1];

    out_shape[out_len - 2] = m;
    out_shape[out_len - 1] = n;
    let out_meta = TensorMeta::new(out_shape.as_slice());
    let it = TensorIter::new(
        std::iter::once(&out_meta).chain(inputs.iter().map(|expr| &expr.var.meta)),
        SKIP,
    )?;

    let (m, p, n) = (m as usize, p as usize, n as usize);
    let op = op::Op::MatMul(MatMulOp {
        a: a.clone(),
        b: b.clone(),
        m,
        p,
        n,
        it,
    });

    let grad = inputs
        .iter()
        .any(|expr| expr.requires_grad())
        .then(|| SharedSlice::zeroes(out_meta.total_elems));

    Ok(Shared::new(Expr {
        var: Var {
            data: SharedSlice::zeroes(out_meta.total_elems),
            meta: out_meta,
        },
        grad,
        op,
    }))
}
