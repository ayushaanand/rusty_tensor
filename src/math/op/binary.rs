use std::cell::RefCell;
use std::rc::Rc;

use super::Op;
use crate::broadcast::compute_broadcast_shape_const;
use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::{Expr, Storage};
use crate::tensor::TensorMeta;

use Op::{Add, Mul, Sub};

pub(crate) struct BinaryOp<T: TensorFType> {
    input: [Rc<Expr<T>>; 2],
    it: RefCell<TensorIter>,
}

fn build_binary_op<T: TensorFType>(
    a: &Rc<Expr<T>>,
    b: &Rc<Expr<T>>,
    op_var: fn(BinaryOp<T>) -> Op<T>,
) -> TensorResult<Rc<Expr<T>>> {
    let inputs = [a, b];
    let out_shape = compute_broadcast_shape_const(inputs.map(|ex| ex.meta.shape.as_slice()), 0)?;

    let out_meta = TensorMeta::new(out_shape.as_slice());

    let it = RefCell::new(TensorIter::new(
        [&out_meta, &inputs[0].meta, &inputs[1].meta],
        0,
    )?);

    let op = op_var(BinaryOp {
        input: inputs.map(Rc::clone),
        it,
    });

    let requires_grad = inputs.into_iter().any(|expr| expr.requires_grad());
    let expr = Expr {
        storage: Rc::new(Storage::zeroes(out_meta.total_elems, requires_grad)),
        meta: out_meta,
        op,
    };

    Ok(Rc::new(expr))
}

pub fn add<T: TensorFType>(a: &Rc<Expr<T>>, b: &Rc<Expr<T>>) -> TensorResult<Rc<Expr<T>>> {
    build_binary_op(a, b, Add)
}

pub fn sub<T: TensorFType>(a: &Rc<Expr<T>>, b: &Rc<Expr<T>>) -> TensorResult<Rc<Expr<T>>> {
    build_binary_op(a, b, Sub)
}

pub fn mul<T: TensorFType>(a: &Rc<Expr<T>>, b: &Rc<Expr<T>>) -> TensorResult<Rc<Expr<T>>> {
    build_binary_op(a, b, Mul)
}

// Rc::ptr_eq on parents is not enough, for e.g. in A + A^T
// 1. if Rc::ptr_eq on expr -
//    a. if [0].grad.is_some() - borrow_mut and borrow [0]'s data and use 2*grad
//    b. else done
// 2. else -
//    a. if .grad.is_some on both -
//       i.  if Rc::ptr_eq on grads, borrow_mut [0].grad but index with both pos[0, 1]
//       ii. else - borrow_mut both and proceed
//    b. if .grad.is_some on only one - borrow_mut that's grad and proceed
//    c. else - done
impl<T: TensorFType> BinaryOp<T> {
    pub(super) fn fwd<F: Fn(T, T) -> T>(&self, out: &mut [T], f: F) {
        let parents = self.input.each_ref().map(|expr| expr.storage.data.borrow());

        self.it.borrow_mut().for_each(|pos| {
            out[pos[0]] += f(parents[0][pos[1]], parents[1][pos[2]]);
        })
    }

    fn bwd_unified<F0, F1>(&self, grad: &[T], f_0: F0, f_1: F1)
    where
        // Fn(&mut this_mut_grad, bwd_grad, other_parent_data)
        F0: Fn(T, T) -> T,
        F1: Fn(T, T) -> T,
    {
        let it = &mut *self.it.borrow_mut();

        if Rc::ptr_eq(&self.input[0], &self.input[1]) {
            // same parent, borrow only one and index uniformly, ie A + A
            if let Some(parent_grad) = &self.input[0].storage.grad {
                let parent_grad = &mut *parent_grad.borrow_mut();
                let parent_data = &*self.input[0].storage.data.borrow();
                it.for_each(|pos| {
                    let arg = (grad[pos[0]], parent_data[pos[1]]);
                    let tot_grad = f_0(arg.0, arg.1) + f_1(arg.0, arg.1);
                    parent_grad[pos[1]] += tot_grad;
                })
            } else { /* unreachable!() */
            }
        } else if Rc::ptr_eq(&self.input[0].storage, &self.input[1].storage) {
            // diff parent, but same grad, borrow once, but index separately
            if let Some(parent_grad) = &self.input[0].storage.grad {
                let parent_grad = &mut *parent_grad.borrow_mut();
                let parent_data = &*self.input[0].storage.data.borrow();
                it.for_each(|pos| {
                    parent_grad[pos[1]] += f_0(grad[pos[0]], parent_data[pos[2]]);
                    parent_grad[pos[2]] += f_1(grad[pos[0]], parent_data[pos[1]]);
                })
            } else { /* unreachable!() */
            }
        } else {
            let parent_datas = self.input.each_ref().map(|expr| expr.storage.data.borrow());

            // different parents, different grads, one may not require grad
            match self.input.each_ref().map(|expr| &expr.storage.grad) {
                [Some(p_grad1), Some(p_grad2)] => {
                    // separate grads, borrow and index separately
                    let mut parent_grads = [p_grad1, p_grad2].map(|grad| grad.borrow_mut());
                    it.for_each(|pos| {
                        parent_grads[0][pos[1]] += f_0(grad[pos[0]], parent_datas[1][pos[2]]);
                        parent_grads[1][pos[2]] += f_1(grad[pos[0]], parent_datas[0][pos[1]]);
                    })
                }
                // now only one requires_grad, borrow and index accordingly
                [Some(p_grad1), None] => {
                    let parent_grad = &mut *p_grad1.borrow_mut();
                    it.for_each(|pos| {
                        parent_grad[pos[1]] += f_0(grad[pos[0]], parent_datas[1][pos[2]]);
                    })
                }
                [None, Some(p_grad2)] => {
                    let parent_grad = &mut *p_grad2.borrow_mut();
                    it.for_each(|pos| {
                        parent_grad[pos[2]] += f_1(grad[pos[0]], parent_datas[0][pos[1]]);
                    })
                }
                [None, None] => { /* unreachable!() */ }
            }
        }
    }

    pub fn add_bwd(&self, grad: &[T]) {
        self.bwd_unified(grad, |g, _| g, |g, _| g)
    }

    pub fn sub_bwd(&self, grad: &[T]) {
        // Rc::ptr_eq on expr => grad += x - x situation, no-op in normal fp
        if !Rc::ptr_eq(&self.input[0], &self.input[1]) {
            self.bwd_unified(grad, |g, _| g, |g, _| -g)
        }
    }

    pub fn mul_bwd(&self, grad: &[T]) {
        self.bwd_unified(grad, |g, d2| g * d2, |g, d1| g * d1)
    }

    pub fn inputs(&self) -> &[Rc<Expr<T>>] {
        &self.input
    }
}

#[cfg(test)]
mod binary_op_tests {
    use super::*;
    use crate::math::op::ident::ident;
    use crate::tensor::Tensor;

    #[test]
    fn test_add_diff_parent_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), true);
        assert!(e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), true);
        assert!(e2.requires_grad());

        let e3 = add(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e3.eval_bwd();

        let Add(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.storage.data.borrow()[pos[0]], 3.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0);
            assert_eq!(e2.get_grad().borrow()[pos[2]], -2.0 * 3.0);
        })
    }

    #[test]
    fn test_add_diff_parent_one_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), true);
        assert!(e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), false);
        assert!(!e2.requires_grad());

        let e3 = add(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e3.eval_bwd();

        let Add(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.storage.data.borrow()[pos[0]], 3.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0);
        })
    }

    #[test]
    fn test_add_diff_parent_none_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), false);
        assert!(!e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), false);
        assert!(!e2.requires_grad());

        let e3 = add(&e1, &e2).unwrap();
        assert!(!e3.requires_grad());
        e3.eval_fwd();
        e3.eval_bwd();

        let Add(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.storage.data.borrow()[pos[0]], 3.0);
        })
    }

    #[test]
    fn test_add_same_parent_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), true);
        assert!(e1.requires_grad());

        let e2 = add(&e1, &e1).unwrap();
        assert!(e2.requires_grad());
        e2.eval_fwd();
        e2.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e2.eval_bwd();

        let Add(op) = &e2.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e2.storage.data.borrow()[pos[0]], 2.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0 + -2.0);
        })
    }

    #[test]
    fn test_add_same_parent_no_require_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), false);
        assert!(!e1.requires_grad());

        let e2 = add(&e1, &e1).unwrap();
        assert!(!e2.requires_grad());
        e2.eval_fwd();

        let Add(op) = &e2.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.storage.data.borrow()[pos[0]], 2.0);
        })
    }

    #[test]
    fn test_sub_diff_parent_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), true);
        assert!(e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), true);
        assert!(e2.requires_grad());

        let e3 = sub(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e3.eval_bwd();

        let Sub(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.storage.data.borrow()[pos[0]], -1.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0);
            assert_eq!(e2.get_grad().borrow()[pos[2]], 2.0 * 3.0);
        })
    }

    #[test]
    fn test_sub_diff_parent_one_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), true);
        assert!(e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), false);
        assert!(!e2.requires_grad());

        let e3 = sub(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e3.eval_bwd();

        let Sub(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.storage.data.borrow()[pos[0]], -1.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0);
        })
    }

    #[test]
    fn test_sub_diff_parent_none_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), false);
        assert!(!e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), false);
        assert!(!e2.requires_grad());

        let e3 = sub(&e1, &e2).unwrap();
        assert!(!e3.requires_grad());
        e3.eval_fwd();
        e3.eval_bwd();

        let Sub(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.storage.data.borrow()[pos[0]], -1.0);
        })
    }

    #[test]
    fn test_sub_same_parent_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), true);
        assert!(e1.requires_grad());

        let e2 = sub(&e1, &e1).unwrap();
        assert!(e2.requires_grad());
        e2.eval_fwd();
        e2.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e2.eval_bwd();

        let Sub(op) = &e2.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e2.storage.data.borrow()[pos[0]], 0.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], 0.0);
        })
    }

    #[test]
    fn test_sub_same_parent_no_require_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 1.0), false);
        assert!(!e1.requires_grad());

        let e2 = sub(&e1, &e1).unwrap();
        assert!(!e2.requires_grad());
        e2.eval_fwd();

        let Sub(op) = &e2.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.storage.data.borrow()[pos[0]], 0.0);
        })
    }

    #[test]
    fn test_mul_diff_parent_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 4.0), true);
        assert!(e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), true);
        assert!(e2.requires_grad());

        let e3 = mul(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e3.eval_bwd();

        let Mul(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.storage.data.borrow()[pos[0]], 8.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0 * 2.0);
            assert_eq!(e2.get_grad().borrow()[pos[2]], -2.0 * 4.0 * 3.0);
        })
    }

    #[test]
    fn test_mul_diff_parent_one_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 4.0), true);
        assert!(e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), false);
        assert!(!e2.requires_grad());

        let e3 = mul(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e3.eval_bwd();

        let Mul(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.storage.data.borrow()[pos[0]], 8.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0 * 2.0);
        })
    }

    #[test]
    fn test_mul_diff_parent_none_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 4.0), false);
        assert!(!e1.requires_grad());

        let e2 = ident(Tensor::new(&[2, 1], 2.0), false);
        assert!(!e2.requires_grad());

        let e3 = mul(&e1, &e2).unwrap();
        assert!(!e3.requires_grad());
        e3.eval_fwd();
        e3.eval_bwd();

        let Mul(op) = &e3.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.storage.data.borrow()[pos[0]], 8.0);
        })
    }

    #[test]
    fn test_mul_same_parent_requires_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 4.0), true);
        assert!(e1.requires_grad());

        let e2 = mul(&e1, &e1).unwrap();
        assert!(e2.requires_grad());
        e2.eval_fwd();
        e2.get_grad().borrow_mut().iter_mut().for_each(|grad| {
            *grad = -2.0;
        });
        e2.eval_bwd();

        let Mul(op) = &e2.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e2.storage.data.borrow()[pos[0]], 16.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], 2.0 * -2.0 * 4.0);
        })
    }

    #[test]
    fn test_mul_same_parent_no_require_grad() {
        let e1 = ident(Tensor::new(&[2, 3], 4.0), false);
        assert!(!e1.requires_grad());

        let e2 = mul(&e1, &e1).unwrap();
        assert!(!e2.requires_grad());
        e2.eval_fwd();

        let Mul(op) = &e2.op else { unreachable!() };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.storage.data.borrow()[pos[0]], 16.0);
        })
    }
}
