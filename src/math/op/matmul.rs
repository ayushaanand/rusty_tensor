use std::cell::RefCell;
use crate::broadcast::compute_broadcast_shape_const;
use crate::consts::{TensorFType, TensorResult};
use crate::error::MatmulShapeError;
use crate::iterator::TensorIter;
use crate::math::{Expr, Var, op};
use crate::tensor::TensorMeta;
use crate::types::SharedSlice;
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct MatMulOp<T: TensorFType> {
    a: Rc<Expr<T>>,
    b: Rc<Expr<T>>,
    it: RefCell<TensorIter>,

    // m*p @ p*n -> (m, p, n)
    mat_dims: (usize, usize, usize),
}

pub fn mm<T: TensorFType>(
    a: &Rc<Expr<T>>,
    b: &Rc<Expr<T>>,
) -> TensorResult<Rc<Expr<T>>> {
    const SKIP: usize = 2;

    let inputs = [a, b];
    let shapes: [_; 2] = inputs.map(|expr| expr.var.meta.shape.as_slice());
    let lens: [_; 2] = shapes.map(|sh| sh.len());

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

    let it = RefCell::new(TensorIter::new(
        std::iter::once(&out_meta).chain(inputs.map(|expr| &expr.var.meta)),
        SKIP,
    )?);

    let mat_dims = (m as usize, p as usize, n as usize);
    let op = op::Op::MatMul(MatMulOp {
        a: Rc::clone(a),
        b: Rc::clone(b),
        it,
        mat_dims,
    });

    let grad = inputs
        .iter()
        .any(|expr| expr.requires_grad())
        .then(|| SharedSlice::zeroes(out_meta.total_elems));

    Ok(Rc::new(Expr {
        var: Var {
            data: SharedSlice::zeroes(out_meta.total_elems),
            meta: out_meta,
        },
        grad,
        op,
    }))
}

impl<T: TensorFType> MatMulOp<T> {
    fn mm_loop<F: FnMut([usize; 6])>(
        it: &mut TensorIter,
        mat_dims: (usize, usize, usize),
        mut f: F,
    ) {
        let (m, p, n) = mat_dims;
        it.for_each(|pos| {
            let r_o = pos[0];
            let a_o = pos[1];
            let b_o = pos[2];
            for i in 0..m {
                for k in 0..p {
                    for j in 0..n {
                        f([r_o, a_o, b_o, i, j, k]);
                    }
                }
            }
        });
    }

    pub(crate) fn fwd(&self, out: &mut [T]) {
        let a_data = self.a.var.data.borrow();
        let b_data = self.b.var.data.borrow();

        let (_, p, n) = self.mat_dims;
        Self::mm_loop(&mut *self.it.borrow_mut(), self.mat_dims, |[r_o, a_o, b_o, i, j, k]| {
            out[r_o + i * n + j] += a_data[a_o + i * p + k] * b_data[b_o + k * n + j];
        });
    }

    pub(crate) fn bwd(&self, grad: &[T]) {
        let (_, p, n) = self.mat_dims;
        if !Rc::ptr_eq(&self.a, &self.b) {
            let a_data = self.a.var.data.borrow();
            let b_data = self.b.var.data.borrow();

            match (self.a.requires_grad(), self.b.requires_grad()) {
                (true, true) => {
                    let mut a_grad = self.a.get_grad().borrow_mut();
                    let mut b_grad = self.b.get_grad().borrow_mut();

                    Self::mm_loop(&mut *self.it.borrow_mut(), self.mat_dims, |[r_o, a_o, b_o, i, j, k]| {
                        a_grad[a_o + i * p + k] += grad[r_o + i * n + j] * b_data[b_o + k * n + j]; // grad_A = grad @ B^T
                        b_grad[b_o + k * n + j] += a_data[a_o + i * p + k] * grad[r_o + i * n + j]; // grad_B = A^T @ grad
                    });
                }
                (true, false) => {
                    let mut a_grad = self.a.get_grad().borrow_mut();

                    Self::mm_loop(&mut *self.it.borrow_mut(), self.mat_dims, |[r_o, a_o, b_o, i, j, k]| {
                        a_grad[a_o + i * p + k] += grad[r_o + i * n + j] * b_data[b_o + k * n + j];
                    });
                }
                (false, true) => {
                    let mut b_grad = self.b.get_grad().borrow_mut();

                    Self::mm_loop(&mut *self.it.borrow_mut(), self.mat_dims, |[r_o, a_o, b_o, i, j, k]| {
                        b_grad[b_o + k * n + j] += a_data[a_o + i * p + k] * grad[r_o + i * n + j];
                    });
                }
                (false, false) => {}
            }
        } else {
            let parent_data = self.a.var.data.borrow();
            if self.a.requires_grad() {
                let mut parent_grad = self.a.get_grad().borrow_mut();
                Self::mm_loop(&mut *self.it.borrow_mut(), self.mat_dims, |[r_o, a_o, b_o, i, j, k]| {
                    parent_grad[a_o + i * p + k] +=
                        grad[r_o + i * n + j] * parent_data[b_o + k * n + j];
                    parent_grad[b_o + k * n + j] +=
                        parent_data[a_o + i * p + k] * grad[r_o + i * n + j];
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::op::ident::ident;
    use crate::tensor::Tensor;
    use std::iter::repeat_n;

    #[macro_export]
    macro_rules! assert_allclose {
        ($left:expr, $right:expr) => {
            assert_allclose!($left, $right, 1e-6, 1e-6);
        };
        ($left:expr, $right:expr, $atol:expr, $rtol:expr) => {{
            assert_eq!(
                $left.len(),
                $right.len(),
                "Length mismatch: left has len {}, right has len {}",
                $left.len(),
                $right.len()
            );
            for (i, (&a, &b)) in $left.iter().zip($right.iter()).enumerate() {
                let diff = (a - b).abs();
                let tol = $atol + $rtol * b.abs();
                assert!(
                    diff <= tol,
                    "Float mismatch at index {}:\n  left:  {}\n right:  {}\n  diff:  {} > tol {}",
                    i,
                    a,
                    b,
                    diff,
                    tol
                );
            }
        }};
    }

    fn repeat_slice<T: Copy>(slice: &[T], n: usize) -> Vec<T> {
        repeat_n(slice, n).flatten().copied().collect::<Vec<_>>()
    }

    #[test]
    fn test_matmul() {
        let e1 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[2, 3], (1..=6).map(|i| i as f32)).unwrap(),
            )),
            true,
        );
        let e2 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[3, 2], (1..=6).rev().map(|i| i as f32)).unwrap(),
            )),
            true,
        );

        let e3 = mm(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        assert_eq!(e3.var.meta.shape.as_slice(), &[2, 2]);
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&[-0.5, -0.2, 1.0, 0.1]);
        e3.eval_bwd();

        assert_allclose!(e3.var.data.borrow(), &[20.0, 14.0, 56.0, 41.0]);
        assert_allclose!(
            e1.get_grad().borrow(),
            &[-4.0, -2.6, -1.2, 6.5, 4.3, 2.1]
        );
        assert_allclose!(
            e2.get_grad().borrow(),
            &[3.5, 0.2, 4.0, 0.1, 4.5, 0.0]
        );
    }

    #[test]
    fn test_matmul_broadcast() {
        let e1 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[3, 2, 3], repeat_n((1..=6).map(|i| i as f32), 3).flatten())
                    .unwrap(),
            )),
            true,
        );
        let e2 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[3, 2], (1..=6).rev().map(|i| i as f32)).unwrap(),
            )),
            true,
        );

        let e3 = mm(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        assert_eq!(e3.var.meta.shape.as_slice(), &[3, 2, 2]);
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&repeat_slice(&[-0.5_f32, -0.2, 1.0, 0.1], 3));
        e3.eval_bwd();

        assert_allclose!(
            e3.var.data.borrow(),
            &repeat_slice(&[20.0_f32, 14.0, 56.0, 41.0], 3)
        );
        assert_allclose!(
            e1.get_grad().borrow(),
            &repeat_slice(&[-4.0_f32, -2.6, -1.2, 6.5, 4.3, 2.1], 3)
        );
        assert_allclose!(
            e2.get_grad().borrow(),
            &[3.5_f32, 0.2, 4.0, 0.1, 4.5, 0.0].map(|f| f * 3.0)
        );
    }

    #[test]
    fn test_matmul_only_one_require_grad() {
        let e1 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[2, 3], (1..=6).map(|i| i as f32)).unwrap(),
            )),
            true,
        );
        assert!(e1.requires_grad());
        let e2 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[3, 2], (1..=6).rev().map(|i| i as f32)).unwrap(),
            )),
            false,
        );
        assert!(!e2.requires_grad());

        let e3 = mm(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        assert_eq!(e3.var.meta.shape.as_slice(), &[2, 2]);
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&[-0.5, -0.2, 1.0, 0.1]);
        e3.eval_bwd();

        assert_allclose!(e3.var.data.borrow(), &[20.0, 14.0, 56.0, 41.0]);
        assert_allclose!(
            e1.get_grad().borrow(),
            &[-4.0, -2.6, -1.2, 6.5, 4.3, 2.1]
        );
        assert!(e2.grad.is_none());
    }

    #[test]
    fn test_matmul_none_require_grad() {
        let e1 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[2, 3], (1..=6).map(|i| i as f32)).unwrap(),
            )),
            false,
        );
        assert!(!e1.requires_grad());
        let e2 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[3, 2], (1..=6).rev().map(|i| i as f32)).unwrap(),
            )),
            false,
        );
        assert!(!e2.requires_grad());

        let e3 = mm(&e1, &e2).unwrap();
        assert!(!e3.requires_grad());
        assert_eq!(e3.var.meta.shape.as_slice(), &[2, 2]);
        e3.eval_fwd();
        e3.eval_bwd();

        assert_allclose!(e3.var.data.borrow(), &[20.0, 14.0, 56.0, 41.0]);
    }

    #[test]
    fn test_matmul_same_parent() {
        let e1 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(&[2, 2], (1..=4).map(|i| i as f32)).unwrap(),
            )),
            true,
        );
        assert!(e1.requires_grad());

        let e3 = mm(&e1, &e1).unwrap();
        assert!(e3.requires_grad());
        assert_eq!(e3.var.meta.shape.as_slice(), &[2, 2]);
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&[-0.5, -0.2, 1.0, 0.1]);
        e3.eval_bwd();

        assert_allclose!(e3.var.data.borrow(), &[7.0, 10.0, 15.0, 22.0]);
        assert_allclose!(e1.get_grad().borrow(), &[1.6, -2.2, 4.2, 3.4]);
    }

    #[test]
    fn test_matmul_same_parent_no_grad() {
        let e1 = ident(
            &Rc::new(Var::from(
                Tensor::from_buf(
                    &[3, 3],
                    [5.0_f32, -0.1, 3.2, 4.9, 7.0, -2.3, -4.3, 1.2, -7.8],
                )
                .unwrap(),
            )),
            false,
        );
        assert!(!e1.requires_grad());

        let e3 = mm(&e1, &e1).unwrap();
        assert!(!e3.requires_grad());
        assert_eq!(e3.var.meta.shape.as_slice(), &[3, 3]);
        e3.eval_fwd();
        e3.eval_bwd();

        assert_allclose!(
            e3.var.data.borrow(),
            &[10.75, 2.64, -8.73, 68.69, 45.75, 17.52, 17.92, -0.53, 44.32]
        );
    }
}
