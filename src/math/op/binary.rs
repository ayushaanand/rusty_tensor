use crate::broadcast::compute_broadcast_shape_iter;
use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::{Expr, Var, op::Op};
use crate::tensor::TensorMeta;
use crate::types::{Shared, SharedSlice};
use Op::{Add, Mul, Sub};
use std::rc::Rc;

fn build_binary_op<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
    op_var: fn(BinaryOp<T>) -> Op<T>,
    retain_data: bool,
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
    let grad = inputs
        .iter()
        .any(|expr| expr.requires_grad())
        .then(|| SharedSlice::zeroes(total_elems));

    let expr = Expr {
        var: Var {
            meta: out_meta,
            data: SharedSlice::zeroes(total_elems),
        },
        grad,
        op,
        retain_data,
    };

    Ok(Shared::new(expr))
}

pub fn add<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
) -> TensorResult<Shared<Expr<T>>> {
    build_binary_op(a, b, Add, false)
}

pub fn sub<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
) -> TensorResult<Shared<Expr<T>>> {
    build_binary_op(a, b, Sub, false)
}

pub fn mul<T: TensorFType>(
    a: &Shared<Expr<T>>,
    b: &Shared<Expr<T>>,
) -> TensorResult<Shared<Expr<T>>> {
    build_binary_op(a, b, Mul, true)
}

// ----------- Binary -----------------------------------------------------------------------------
#[derive(Clone)]
pub(crate) struct BinaryOp<T: TensorFType> {
    input: [Shared<Expr<T>>; 2],
    it: TensorIter,
}

impl<T: TensorFType> BinaryOp<T> {
    pub(super) fn fwd<F: Fn(T, T) -> T>(&mut self, out: &mut [T], f: F) {
        let borrow_inputs = self.input.each_ref().map(|expr| expr.borrow());
        let slices = borrow_inputs.each_ref().map(|expr| expr.var.data.borrow());

        self.it.for_each(|pos| {
            out[pos[0]] += f(slices[0][pos[1]], slices[1][pos[2]]);
        })
    }

    pub(super) fn add_bwd(&mut self, grad: &[T]) {
        // parents may be same and may or may not require grad
        if !Rc::ptr_eq(self.input[0].as_rc(), self.input[1].as_rc()) {
            // parents are not same but may or may not require grad individually
            let borrow_inputs = self.input.each_ref().map(|expr| expr.borrow());

            // only parent[i] requires grad
            let mut op_on_one = |i: usize| {
                let mut parent_grad = borrow_inputs[i].get_grad().borrow_mut();
                self.it.for_each(|pos| {
                    parent_grad[pos[1 + i]] += grad[pos[0]];
                })
            };

            match borrow_inputs.each_ref().map(|expr| expr.requires_grad()) {
                [true, true] => {
                    // both parents require grad
                    // fused version of op_on_one(0, 1)
                    let mut parent_grads = borrow_inputs
                        .each_ref()
                        .map(|expr| expr.get_grad().borrow_mut());

                    self.it.for_each(|pos| {
                        parent_grads[0][pos[1]] += grad[pos[0]];
                        parent_grads[1][pos[2]] += grad[pos[0]];
                    })
                }
                [true, false] => {
                    op_on_one(0);
                }
                [false, true] => {
                    op_on_one(1);
                }
                [false, false] => {}
            }
        } else {
            // parents are same, but may not require grad
            let borrow = self.input[0].borrow();
            if borrow.requires_grad() {
                // parent require grad
                let mut parent_grad = borrow.get_grad().borrow_mut();
                self.it.for_each(|pos| {
                    // double for being both operands of addition, but no T::two() available
                    parent_grad[pos[1]] += grad[pos[0]] + grad[pos[0]];
                })
            }
        }
    }

    pub(super) fn sub_bwd(&mut self, grad: &[T]) {
        if !Rc::ptr_eq(self.input[0].as_rc(), self.input[1].as_rc()) {
            let borrow_inputs = self.input.each_ref().map(|expr| expr.borrow());

            match borrow_inputs.each_ref().map(|expr| expr.requires_grad()) {
                [true, true] => {
                    let mut parent_grads = borrow_inputs
                        .each_ref()
                        .map(|expr| expr.get_grad().borrow_mut());

                    self.it.for_each(|pos| {
                        parent_grads[0][pos[1]] += grad[pos[0]];
                        parent_grads[1][pos[2]] -= grad[pos[0]];
                    })
                }
                [true, false] => {
                    let mut parent_grad = borrow_inputs[0].get_grad().borrow_mut();
                    self.it.for_each(|pos| {
                        parent_grad[pos[1]] += grad[pos[0]];
                    })
                }
                [false, true] => {
                    let mut parent_grad = borrow_inputs[1].get_grad().borrow_mut();
                    self.it.for_each(|pos| {
                        parent_grad[pos[2]] -= grad[pos[0]];
                    })
                }
                [false, false] => {}
            }
        }
        // else both parents are same and even if they require grad, += grad and -= grad is no-op
    }

    pub(super) fn mul_bwd(&mut self, grad: &[T]) {
        if !Rc::ptr_eq(self.input[0].as_rc(), self.input[1].as_rc()) {
            let borrow_inputs = self.input.each_ref().map(|expr| expr.borrow());

            let mut op_on_one = |i: usize| {
                let mut parent_grad = borrow_inputs[i].get_grad().borrow_mut();
                let other_parent_data = borrow_inputs[1 - i].var.data.borrow();

                self.it.for_each(|pos| {
                    parent_grad[pos[1 + (i)]] += grad[pos[0]] * other_parent_data[1 + (1 - i)];
                })
            };

            match borrow_inputs.each_ref().map(|expr| expr.requires_grad()) {
                [true, true] => {
                    // fused version of op_on_one(0, 1)
                    let mut parent_grads = borrow_inputs
                        .each_ref()
                        .map(|expr| expr.get_grad().borrow_mut());
                    let parent_datas = borrow_inputs.each_ref().map(|expr| expr.var.data.borrow());

                    self.it.for_each(|pos| {
                        for i in 0..2 {
                            parent_grads[i][pos[1 + (i)]] =
                                grad[pos[0]] * parent_datas[1 - i][pos[1 + (1 - i)]];
                        }
                    })
                }
                [true, false] => {
                    op_on_one(0);
                }
                [false, true] => {
                    op_on_one(1);
                }
                [false, false] => {}
            }
        } else {
            let borrow = self.input[0].borrow();

            if borrow.requires_grad() {
                let mut parent_grad = borrow.get_grad().borrow_mut();
                let parent_data = borrow.var.data.borrow();

                self.it.for_each(|pos| {
                    parent_grad[pos[1]] +=
                        grad[pos[0]] * parent_data[pos[1]] + grad[pos[0]] * parent_data[pos[1]];
                });
            }
        }
    }
}
