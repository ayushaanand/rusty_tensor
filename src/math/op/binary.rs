use crate::broadcast::compute_broadcast_shape_const;
use crate::consts::*;
use crate::iterator::TensorIter;
use crate::math::{Expr, Var, op::Op};
use crate::tensor::TensorMeta;
use crate::types::SharedSlice;
use Op::{Add, Mul, Sub};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
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
    let out_shape =
        compute_broadcast_shape_const(inputs.map(|ex| ex.var.meta.shape.as_slice()), 0)?;

    let out_meta = TensorMeta::new(out_shape.as_slice());

    let it = RefCell::new(TensorIter::new(
        std::iter::once(&out_meta).chain(inputs.map(|i| &i.var.meta)),
        0,
    )?);

    let op = op_var(BinaryOp {
        input: inputs.map(|expr| Rc::clone(expr)),
        it,
    });

    let total_elems = out_meta.total_elems;
    let grad = inputs.iter()
        .any(|expr| expr.requires_grad())
        .then(|| SharedSlice::zeroes(total_elems));

    let expr = Expr {
        var: Var {
            meta: out_meta,
            data: SharedSlice::zeroes(total_elems),
        },
        grad,
        op,
    };

    Ok(Rc::new(expr))
}

pub fn add<T: TensorFType>(
    a: &Rc<Expr<T>>,
    b: &Rc<Expr<T>>,
) -> TensorResult<Rc<Expr<T>>> {
    build_binary_op(a, b, Add)
}

pub fn sub<T: TensorFType>(
    a: &Rc<Expr<T>>,
    b: &Rc<Expr<T>>,
) -> TensorResult<Rc<Expr<T>>> {
    build_binary_op(a, b, Sub)
}

pub fn mul<T: TensorFType>(
    a: &Rc<Expr<T>>,
    b: &Rc<Expr<T>>,
) -> TensorResult<Rc<Expr<T>>> {
    build_binary_op(a, b, Mul)
}

// 1. check Rc::ptr_eq since parents may be same
// 2. if different parents -
//    a. if both require grad, make a fused loop
//    b. if only one requires, make it into a separate lambda and index using i and 1-i
//    c. if none require grad, done
// 3. else both are same parents, so one parent
//    a. if parent requires grad, apply both grads at once
//    b. else done
//
// pos[0] is for output
// pos[i -> 1..=2] are inputs and pos[1 + i] is for parent[i]
// contract logic but indexing i and 1-i in - for i in 0..2

impl<T: TensorFType> BinaryOp<T> {
    pub(super) fn fwd<F: Fn(T, T) -> T>(&self, out: &mut [T], f: F) {
        let slices = self.input.each_ref().map(|expr| expr.var.data.borrow());

        self.it.borrow_mut().for_each(|pos| {
            out[pos[0]] += f(slices[0][pos[1]], slices[1][pos[2]]);
        })
    }

    pub(super) fn add_bwd(&self, grad: &[T]) {
        // parents may be same and may or may not require grad
        let it = &mut *self.it.borrow_mut();
        
        if !Rc::ptr_eq(&self.input[0], &self.input[1]) {
            // only parent[i] requires grad
            let mut op_on_one = |i: usize| {
                let mut parent_grad = self.input[i].get_grad().borrow_mut();
                it.for_each(|pos| {
                    parent_grad[pos[1 + i]] += grad[pos[0]];
                })
            };

            match self.input.each_ref().map(|expr| expr.requires_grad()) {
                [true, true] => {
                    // both parents require grad
                    // fused version of op_on_one(0, 1)
                    let mut parent_grads = self.input
                        .each_ref()
                        .map(|expr| expr.get_grad().borrow_mut());

                    it.for_each(|pos| {
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
            let parent = &self.input[0];
            if parent.requires_grad() {
                // parent require grad
                let mut parent_grad = parent.get_grad().borrow_mut();
                it.for_each(|pos| {
                    // double for being both operands of addition, but no T::two() available
                    parent_grad[pos[1]] += grad[pos[0]] + grad[pos[0]];
                })
            }
        }
    }

    pub(super) fn sub_bwd(&self, grad: &[T]) {
        if !Rc::ptr_eq(&self.input[0], &self.input[1]) {
            let it = &mut *self.it.borrow_mut();

            match self.input.each_ref().map(|expr| expr.requires_grad()) {
                [true, true] => {
                    let mut parent_grads = self.input
                        .each_ref()
                        .map(|expr| expr.get_grad().borrow_mut());

                    it.for_each(|pos| {
                        parent_grads[0][pos[1]] += grad[pos[0]];
                        parent_grads[1][pos[2]] -= grad[pos[0]];
                    })
                }
                [true, false] => {
                    let mut parent_grad = self.input[0].get_grad().borrow_mut();
                    it.for_each(|pos| {
                        parent_grad[pos[1]] += grad[pos[0]];
                    })
                }
                [false, true] => {
                    let mut parent_grad = self.input[1].get_grad().borrow_mut();
                    it.for_each(|pos| {
                        parent_grad[pos[2]] -= grad[pos[0]];
                    })
                }
                [false, false] => {}
            }
        }
        // else both parents are same and even if they require grad, += grad and -= grad is no-op
    }

    pub(super) fn mul_bwd(&self, grad: &[T]) {
        let it = &mut *self.it.borrow_mut();
        
        if !Rc::ptr_eq(&self.input[0], &self.input[1]) {
            let mut op_on_one = |i: usize| {
                let mut parent_grad = self.input[i].get_grad().borrow_mut();
                let other_parent_data = self.input[1 - i].var.data.borrow();

                it.for_each(|pos| {
                    parent_grad[pos[1 + (i)]] += grad[pos[0]] * other_parent_data[pos[1 + (1 - i)]];
                })
            };

            match self.input.each_ref().map(|expr| expr.requires_grad()) {
                [true, true] => {
                    // fused version of op_on_one(0, 1)
                    let mut parent_grads = self.input
                        .each_ref()
                        .map(|expr| expr.get_grad().borrow_mut());
                    let parent_datas = self.input.each_ref().map(|expr| expr.var.data.borrow());

                    it.for_each(|pos| {
                        for i in 0..2 {
                            parent_grads[i][pos[1 + (i)]] +=
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
            let parent = &self.input[0];

            if parent.requires_grad() {
                let mut parent_grad = parent.get_grad().borrow_mut();
                let parent_data = parent.var.data.borrow();

                it.for_each(|pos| {
                    parent_grad[pos[1]] +=
                        grad[pos[0]] * parent_data[pos[1]] + grad[pos[0]] * parent_data[pos[1]];
                });
            }
        }
    }
}

#[cfg(test)]
mod binary_op_tests {
    use super::*;
    use crate::math::op::ident::ident;
    use crate::tensor::Tensor;

    #[test]
    fn test_add_diff_parent_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, true);
        assert!(e1.requires_grad());
        let e2 = ident(&v2, true);
        assert!(e2.requires_grad());

        let e3 = add(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e3.eval_bwd();
        
        let Add(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.var.data.borrow()[pos[0]], 3.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0);
            assert_eq!(
                e2.get_grad().borrow()[pos[2]],
                -2.0 * 3.0
            );
        })
    }

    #[test]
    fn test_add_diff_parent_one_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, true);
        assert!(e1.requires_grad());
        let e2 = ident(&v2, false);
        assert!(!e2.requires_grad());

        let e3 = add(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e3.eval_bwd();

        let Add(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.var.data.borrow()[pos[0]], 3.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0);
        })
    }

    #[test]
    fn test_add_diff_parent_none_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, false);
        assert!(!e1.requires_grad());
        let e2 = ident(&v2, false);
        assert!(!e1.requires_grad());

        let e3 = add(&e1, &e2).unwrap();
        assert!(!e3.requires_grad());
        e3.eval_fwd();
        e3.eval_bwd();

        let Add(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.var.data.borrow()[pos[0]], 3.0);
        })
    }

    #[test]
    fn test_add_same_parent_requires_grad() {
        let v = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let e1 = ident(&v, true);
        assert!(e1.requires_grad());

        let e2 = add(&e1, &e1).unwrap();
        assert!(e1.requires_grad());
        e2.eval_fwd();
        e2.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e2.eval_bwd();
        
        let Add(op) = &e2.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e2.var.data.borrow()[pos[0]], 2.0);

            assert_eq!(
                e1.get_grad().borrow()[pos[1]],
                -2.0 + -2.0
            );
        })
    }

    #[test]
    fn test_add_same_parent_no_require_grad() {
        let v = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let e1 = ident(&v, false);
        assert!(!e1.requires_grad());

        let e2 = add(&e1, &e1).unwrap();
        assert!(!e1.requires_grad());
        e2.eval_fwd();
        
        let Add(op) = &e2.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.var.data.borrow()[pos[0]], 2.0);
        })
    }

    #[test]
    fn test_sub_diff_parent_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, true);
        assert!(e1.requires_grad());
        let e2 = ident(&v2, true);
        assert!(e2.requires_grad());

        let e3 = sub(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e3.eval_bwd();

        let Sub(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.var.data.borrow()[pos[0]], -1.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0);
            assert_eq!(
                e2.get_grad().borrow()[pos[2]],
                2.0 * 3.0
            );
        })
    }

    #[test]
    fn test_sub_diff_parent_one_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, true);
        assert!(e1.requires_grad());
        let e2 = ident(&v2, false);
        assert!(!e2.requires_grad());

        let e3 = sub(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e3.eval_bwd();

        let Sub(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.var.data.borrow()[pos[0]], -1.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], -2.0);
        })
    }

    #[test]
    fn test_sub_diff_parent_none_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, false);
        assert!(!e1.requires_grad());
        let e2 = ident(&v2, false);
        assert!(!e1.requires_grad());

        let e3 = sub(&e1, &e2).unwrap();
        assert!(!e3.requires_grad());
        e3.eval_fwd();
        e3.eval_bwd();

        let Sub(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.var.data.borrow()[pos[0]], -1.0);
        })
    }

    #[test]
    fn test_sub_same_parent_requires_grad() {
        let v = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let e1 = ident(&v, true);
        assert!(e1.requires_grad());

        let e2 = sub(&e1, &e1).unwrap();
        assert!(e1.requires_grad());
        e2.eval_fwd();
        e2.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e2.eval_bwd();
        
        let Sub(op) = &e2.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e2.var.data.borrow()[pos[0]], 0.0);

            assert_eq!(e1.get_grad().borrow()[pos[1]], 0.0);
        })
    }

    #[test]
    fn test_sub_same_parent_no_require_grad() {
        let v = Rc::new(Var::from(Tensor::new(&[2, 3], 1.0)));
        let e1 = ident(&v, false);
        assert!(!e1.requires_grad());

        let e2 = sub(&e1, &e1).unwrap();
        assert!(!e1.requires_grad());
        e2.eval_fwd();
        
        let Sub(op) = &e2.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.var.data.borrow()[pos[0]], 0.0);
        })
    }

    #[test]
    fn test_mul_diff_parent_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 4.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, true);
        assert!(e1.requires_grad());
        let e2 = ident(&v2, true);
        assert!(e2.requires_grad());

        let e3 = mul(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e3.eval_bwd();

        let Mul(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.var.data.borrow()[pos[0]], 8.0);

            assert_eq!(
                e1.get_grad().borrow()[pos[1]],
                -2.0 * 2.0
            );
            assert_eq!(
                e2.get_grad().borrow()[pos[2]],
                -2.0 * 4.0 * 3.0
            );
        })
    }

    #[test]
    fn test_mul_diff_parent_one_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 4.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, true);
        assert!(e1.requires_grad());
        let e2 = ident(&v2, false);
        assert!(!e2.requires_grad());

        let e3 = mul(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e3.eval_bwd();

        let Mul(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e3.var.data.borrow()[pos[0]], 8.0);

            assert_eq!(
                e1.get_grad().borrow()[pos[1]],
                -2.0 * 2.0
            );
        })
    }

    #[test]
    fn test_mul_diff_parent_none_requires_grad() {
        let v1 = Rc::new(Var::from(Tensor::new(&[2, 3], 4.0)));
        let v2 = Rc::new(Var::from(Tensor::new(&[2, 1], 2.0)));

        let e1 = ident(&v1, false);
        assert!(!e1.requires_grad());
        let e2 = ident(&v2, false);
        assert!(!e1.requires_grad());

        let e3 = mul(&e1, &e2).unwrap();
        assert!(!e3.requires_grad());
        e3.eval_fwd();
        e3.eval_bwd();

        let Mul(op) = &e3.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e3.var.data.borrow()[pos[0]], 8.0);
        })
    }

    #[test]
    fn test_mul_same_parent_requires_grad() {
        let v = Rc::new(Var::from(Tensor::new(&[2, 3], 4.0)));
        let e1 = ident(&v, true);
        assert!(e1.requires_grad());

        let e2 = mul(&e1, &e1).unwrap();
        assert!(e1.requires_grad());
        e2.eval_fwd();
        e2.get_grad()
            .borrow_mut()
            .iter_mut()
            .for_each(|grad| {
                *grad = -2.0;
            });
        e2.eval_bwd();
        
        let Mul(op) = &e2.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.get_grad().borrow()[pos[0]], -2.0);
            assert_eq!(e2.var.data.borrow()[pos[0]], 16.0);

            assert_eq!(
                e1.get_grad().borrow()[pos[1]],
                2.0 * -2.0 * 4.0
            );
        })
    }

    #[test]
    fn test_mul_same_parent_no_require_grad() {
        let v = Rc::new(Var::from(Tensor::new(&[2, 3], 4.0)));
        let e1 = ident(&v, false);
        assert!(!e1.requires_grad());

        let e2 = mul(&e1, &e1).unwrap();
        assert!(!e1.requires_grad());
        e2.eval_fwd();
        
        let Mul(op) = &e2.op else {
            unreachable!()
        };

        op.it.borrow_mut().for_each(|pos| {
            assert_eq!(e2.var.data.borrow()[pos[0]], 16.0);
        })
    }
}
