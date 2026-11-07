use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::op::*;
use crate::math::{Expr, Var};
use crate::tensor::TensorMeta;
use crate::types::{Shared, SharedSlice};

pub fn build_unary_op<T: TensorFType>(
    input: &Shared<Expr<T>>,
    op_var: impl FnOnce(UnaryOp<T>) -> Op<T>,
) -> TensorResult<Shared<Expr<T>>> {
    let borrow_input = input.borrow();

    let out_meta = TensorMeta::new(borrow_input.var.meta.shape.as_slice());

    let it = TensorIter::new([&out_meta, &borrow_input.var.meta], 0)?;

    let op = op_var(UnaryOp {
        input: input.clone(),
        it,
    });

    let total_elems = borrow_input.var.meta.total_elems;
    let grad = borrow_input
        .requires_grad()
        .then(|| SharedSlice::zeroes(total_elems));

    Ok(Shared::new(Expr {
        var: Var {
            data: SharedSlice::zeroes(total_elems),
            ..borrow_input.var.clone()
        },
        grad,
        op,
    }))
}

fn relu<T: TensorFType>(input: &Shared<Expr<T>>) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, Op::ReLu)
}

fn silu<T: TensorFType>(input: &Shared<Expr<T>>) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, Op::SiLu)
}

fn sigmoid<T: TensorFType>(input: &Shared<Expr<T>>) -> TensorResult<Shared<Expr<T>>> {
    build_unary_op(input, Op::Sigmoid)
}

// -------- Unary -------------------------------------------------------------------------
#[derive(Clone)]
pub(crate) struct UnaryOp<T: TensorFType> {
    input: Shared<Expr<T>>,
    it: TensorIter,
}

impl<T: TensorFType> UnaryOp<T> {
    pub(super) fn fwd<F: Fn(T) -> T>(&mut self, out: &mut [T], f: F) {
        let borrow = self.input.borrow();
        let parent_data = borrow.var.data.borrow();

        self.it.for_each(|pos| {
            out[pos[0]] += f(parent_data[pos[1]]);
        })
    }

    pub(super) fn bwd<F: Fn(T, T) -> T>(&mut self, grad: &[T], out_val: &[T], f: F) {
        let borrow = self.input.borrow();
        if borrow.requires_grad() {
            let parent_data = borrow.var.data.borrow();
            let mut parent_grad = borrow.grad.as_ref().unwrap().borrow_mut();

            self.it.for_each(|pos| {
                parent_grad[pos[1]] += grad[pos[0]] * f(out_val[pos[0]], parent_data[pos[1]]);
            });
        }
    }
}
