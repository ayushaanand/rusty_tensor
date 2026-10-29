use crate::broadcast::compute_broadcast_shape_iter;
use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::op::BinaryOp;
use crate::math::{Expr, Var, op::Op};
use crate::shared_types;
use crate::shared_types::Shared;
use crate::tensor::TensorMeta;
use Op::{Add, Mul, Sub};

pub fn build_binary_op<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
    op_var: fn(BinaryOp<T>) -> Op<T>,
) -> TensorResult<Shared<Expr<T>>> {
    let inputs = [a.borrow(), b.borrow()];

    let out_shape =
        compute_broadcast_shape_iter(inputs.iter().map(|ex| ex.var.meta.shape.as_slice()), 0)?;

    let out_meta = TensorMeta::new(out_shape.as_slice());

    let it = TensorIter::new(
        std::iter::once(&out_meta).chain(inputs.iter().map(|i| &i.var.meta)),
        0,
    )?;

    let op = op_var(BinaryOp {
        input: [a.clone(), b.clone()],
        it,
    });

    let total_elems = out_meta.total_elems;
    let requires_grad = inputs.iter().any(|expr| expr.var.requires_grad);
    let grad = if requires_grad {
        Some(shared_types::SharedSlice::zeroes(total_elems))
    } else {
        None
    };

    let expr = Expr {
        var: Var {
            meta: out_meta,
            data: shared_types::SharedSlice::zeroes(total_elems),
            requires_grad,
        },
        grad,
        op,
        retain_data: false,
    };

    Ok(Shared::new(expr))
}

pub fn add<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
) -> TensorResult<Shared<Expr<T>>> {
    build_binary_op(a, b, Add)
}

pub fn sub<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
) -> TensorResult<Shared<Expr<T>>> {
    build_binary_op(a, b, Sub)
}

pub fn mul<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
) -> TensorResult<Shared<Expr<T>>> {
    build_binary_op(a, b, Mul)
}
