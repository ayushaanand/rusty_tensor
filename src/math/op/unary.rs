use std::cell::RefCell;
use std::rc::Rc;

use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::op::Op;
use crate::math::{Expr, Storage};
use crate::tensor::TensorMeta;

#[derive(Clone)]
pub(crate) struct UnaryOp<T: TensorFType> {
    input: Rc<Expr<T>>,
    it: RefCell<TensorIter>,
}

pub(super) fn build_unary_op<T: TensorFType>(
    input: &Rc<Expr<T>>,
    op_var: impl FnOnce(UnaryOp<T>) -> Op<T>,
) -> TensorResult<Rc<Expr<T>>> {
    let out_meta = TensorMeta::new(input.meta.shape.as_slice());

    let it = RefCell::new(TensorIter::new([&out_meta, &input.meta], 0)?);

    let op = op_var(UnaryOp {
        input: Rc::clone(input),
        it,
    });

    Ok(Rc::new(Expr {
        storage: Rc::new(Storage::zeroes(out_meta.total_elems, input.requires_grad())),
        meta: out_meta,
        op,
    }))
}

pub fn relu<T: TensorFType>(input: &Rc<Expr<T>>) -> TensorResult<Rc<Expr<T>>> {
    build_unary_op(input, Op::ReLu)
}

pub fn silu<T: TensorFType>(input: &Rc<Expr<T>>) -> TensorResult<Rc<Expr<T>>> {
    build_unary_op(input, Op::SiLu)
}

pub fn sigmoid<T: TensorFType>(input: &Rc<Expr<T>>) -> TensorResult<Rc<Expr<T>>> {
    build_unary_op(input, Op::Sigmoid)
}

pub fn softplus<T: TensorFType>(input: &Rc<Expr<T>>) -> TensorResult<Rc<Expr<T>>> {
    build_unary_op(input, Op::Softplus)
}

impl<T: TensorFType> UnaryOp<T> {
    pub(super) fn fwd<F: Fn(T) -> T>(&self, out: &mut [T], f: F) {
        let parent_data = self.input.storage.data.borrow();

        self.it.borrow_mut().for_each(|pos| {
            out[pos[0]] += f(parent_data[pos[1]]);
        })
    }

    pub(super) fn bwd<F: Fn(T, T) -> T>(&self, grad: &[T], out_val: &[T], f: F) {
        let parent_data = self.input.storage.data.borrow();
        if let Some(parent_grad) = &self.input.storage.grad {
            let parent_grad = &mut *parent_grad.borrow_mut();

            self.it.borrow_mut().for_each(|pos| {
                parent_grad[pos[1]] += grad[pos[0]] * f(out_val[pos[0]], parent_data[pos[1]]);
            });
        } else { /* unreachable!() */
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_allclose;
    use crate::math::op::ident::ident;
    use crate::tensor::Tensor;

    #[test]
    fn test_relu() {
        let e1 = ident(
            Tensor::<f32>::from_buf(&[2, 3], [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).unwrap(),
            true,
        );
        assert!(e1.requires_grad());

        let e2 = relu(&e1).unwrap();
        assert!(e2.requires_grad());

        e2.eval_fwd();
        e2.get_grad()
            .borrow_mut()
            .copy_from_slice(&[-3.0, 1.0, -4.0, 0.0, 0.5, 0.01]);
        e2.eval_bwd();

        assert_allclose!(e2.storage.data.borrow(), &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        assert_allclose!(e1.get_grad().borrow(), &[0.0, 0.0, 0.0, 0.0, 0.5, 0.01]);
    }
}
