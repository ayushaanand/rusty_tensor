use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::Expr;
use crate::math::op::*;
use crate::tensor::TensorMeta;
use crate::types::SharedSlice;

pub fn build_unary_op<T: TensorFType>(
    input: &Shared<Expr<T>>,
    retain_data: bool,
    op_var: impl FnOnce(UnaryOp<T>) -> Op<T>,
) -> TensorResult<Shared<Expr<T>>> {
    let borrow_input = input.borrow();

    let out_meta = TensorMeta::new(borrow_input.var.meta.shape.as_slice());

    let it = TensorIter::new([&out_meta, &borrow_input.var.meta], 0)?;

    let u_op = UnaryOp {
        input: input.clone(),
        it,
    };

    let op = op_var(u_op);

    let total_elems = borrow_input.var.meta.total_elems;
    let grad = if borrow_input.var.requires_grad {
        Some(SharedSlice::zeroes(total_elems))
    } else {
        None
    };

    Ok(Shared::new(Expr {
        var: Var {
            data: SharedSlice::zeroes(total_elems),
            ..borrow_input.var.clone()
        },
        grad,
        op,
        retain_data,
    }))
}

fn relu<T: TensorFType>(input: &Shared<Expr<T>>) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, false, Op::ReLu)
}

fn sigmoid<T: TensorFType>(input: &Shared<Expr<T>>) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, false, Op::Sigmoid)
}
