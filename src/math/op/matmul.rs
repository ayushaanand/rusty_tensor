use crate::broadcast::compute_broadcast_shape_const;
use crate::consts::{SizeType, TensorFType, TensorResult};
use crate::error::MatmulShapeError;
use crate::iterator::TensorIter;
use crate::math::{Expr, Storage, op};
use crate::tensor::TensorMeta;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct MatMulOp<T: TensorFType> {
    a: Rc<Expr<T>>,
    b: Rc<Expr<T>>,
    it: RefCell<TensorIter>,

    // m*p @ p*n  (m, p, n)
    mat_dims: (usize, usize, usize),
}

pub fn mm<T: TensorFType>(a: &Rc<Expr<T>>, b: &Rc<Expr<T>>) -> TensorResult<Rc<Expr<T>>> {
    const SKIP_DIMS: usize = 2;

    let inputs = [a, b];
    let shapes = inputs.map(|expr| expr.meta.shape.as_slice());
    let lens = shapes.map(|sh| sh.len());

    let mut out_shape = compute_broadcast_shape_const(shapes, SKIP_DIMS)?;
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
        std::iter::once(&out_meta).chain(inputs.map(|expr| &expr.meta)),
        SKIP_DIMS,
    )?);

    let mat_dims = (m, p, n);
    let op = op::Op::MatMul(MatMulOp {
        a: Rc::clone(a),
        b: Rc::clone(b),
        it,
        mat_dims,
    });

    Ok(Rc::new(Expr {
        storage: Rc::new(Storage::zeroes(
            out_meta.total_elems,
            a.requires_grad() || b.requires_grad(),
        )),
        meta: out_meta,
        op,
    }))
}

impl<T: TensorFType> MatMulOp<T> {
    fn mm_loop<F: FnMut(usize, usize, usize)>(&self, mut f: F) {
        let (m, p, n) = self.mat_dims;

        fn inner_strides<T: TensorFType>(arg: &Rc<Expr<T>>) -> [SizeType; 2] {
            *arg.meta.stride.last_chunk::<2>().unwrap()
        }

        let inner_a_strides = inner_strides(&self.a);
        let inner_b_strides = inner_strides(&self.b);

        self.it.borrow_mut().for_each(|pos| {
            let (r_offset, a_offset, b_offset) = (pos[0], pos[1], pos[2]);
            for i in 0..m {
                let r_row = r_offset + i * n;
                let a_row = a_offset + i * inner_a_strides[0];
                for k in 0..p {
                    let a_idx = a_row + k * inner_a_strides[1];
                    let b_row = b_offset + k * inner_b_strides[0];
                    for j in 0..n {
                        let r_idx = r_row + j;
                        let b_idx = b_row + j * inner_b_strides[1];
                        // let r_idx = r_offset + i * n                  + j * 1;
                        // let a_idx = a_offset + i * inner_a_strides[0] + k * inner_a_strides[1];
                        // let b_idx = b_offset + k * inner_b_strides[0] + j * inner_b_strides[1];
                        f(r_idx, a_idx, b_idx);
                    }
                }
            }
        });
    }

    pub(crate) fn fwd(&self, out: &mut [T]) {
        let a_data = self.a.storage.data.borrow();
        let b_data = self.b.storage.data.borrow();

        self.mm_loop(|r_idx, a_idx, b_idx| {
            out[r_idx] += a_data[a_idx] * b_data[b_idx];
        });
    }

    pub(crate) fn bwd(&self, grad: &[T]) {
        // Rc::ptr_eq on expr branch is removed since uniform indexing optimization is not possible in matmul
        // write must be done through both indices, which is done in grad aliased case anyway

        if Rc::ptr_eq(&self.a.storage, &self.b.storage) {
            // grads are shared, but uniform indexing not possible due to view changes
            if let Some(parent_grad) = &self.a.storage.grad {
                let parent_grad = &mut *parent_grad.borrow_mut();
                let parent_data = &*self.a.storage.data.borrow();
                self.mm_loop(|r_idx, a_idx, b_idx| {
                    parent_grad[a_idx] += grad[r_idx] * parent_data[b_idx];
                    parent_grad[b_idx] += parent_data[a_idx] * grad[r_idx];
                })
            } else { /* unreachable!() */
            }
        } else {
            let a_data = &*self.a.storage.data.borrow();
            let b_data = &*self.b.storage.data.borrow();

            // different parents, one may not require grad
            match [&self.a.storage.grad, &self.b.storage.grad] {
                [Some(a_grad), Some(b_grad)] => {
                    // separate grads, borrow_mut and index separately
                    let a_grad = &mut *a_grad.borrow_mut();
                    let b_grad = &mut *b_grad.borrow_mut();

                    self.mm_loop(|r_idx, a_idx, b_idx| {
                        a_grad[a_idx] += grad[r_idx] * b_data[b_idx];
                        b_grad[b_idx] += a_data[a_idx] * grad[r_idx];
                    })
                }
                // now only one requires_grad, borrow and index accordingly
                [Some(a_grad), None] => {
                    let a_grad = &mut *a_grad.borrow_mut();
                    self.mm_loop(|r_idx, a_idx, b_idx| {
                        a_grad[a_idx] += grad[r_idx] * b_data[b_idx];
                    })
                }
                [None, Some(b_grad)] => {
                    let b_grad = &mut *b_grad.borrow_mut();
                    self.mm_loop(|r_idx, a_idx, b_idx| {
                        b_grad[b_idx] += a_data[a_idx] * grad[r_idx];
                    })
                }
                [None, None] => { /* unreachable!() */ }
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
            Tensor::from_buf(&[2, 3], (1..=6).map(|i| i as f32)).unwrap(),
            true,
        );
        let e2 = ident(
            Tensor::from_buf(&[3, 2], (1..=6).rev().map(|i| i as f32)).unwrap(),
            true,
        );

        let e3 = mm(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        assert_eq!(e3.meta.shape.as_slice(), &[2, 2]);
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&[-0.5, -0.2, 1.0, 0.1]);
        e3.eval_bwd();

        assert_allclose!(e3.storage.data.borrow(), &[20.0, 14.0, 56.0, 41.0]);
        assert_allclose!(e1.get_grad().borrow(), &[-4.0, -2.6, -1.2, 6.5, 4.3, 2.1]);
        assert_allclose!(e2.get_grad().borrow(), &[3.5, 0.2, 4.0, 0.1, 4.5, 0.0]);
    }

    #[test]
    fn test_matmul_broadcast() {
        let e1 = ident(
            Tensor::from_buf(&[3, 2, 3], repeat_n((1..=6).map(|i| i as f32), 3).flatten()).unwrap(),
            true,
        );
        let e2 = ident(
            Tensor::from_buf(&[3, 2], (1..=6).rev().map(|i| i as f32)).unwrap(),
            true,
        );

        let e3 = mm(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        assert_eq!(e3.meta.shape.as_slice(), &[3, 2, 2]);
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&repeat_slice(&[-0.5_f32, -0.2, 1.0, 0.1], 3));
        e3.eval_bwd();

        assert_allclose!(
            e3.storage.data.borrow(),
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
            Tensor::from_buf(&[2, 3], (1..=6).map(|i| i as f32)).unwrap(),
            true,
        );
        assert!(e1.requires_grad());
        let e2 = ident(
            Tensor::from_buf(&[3, 2], (1..=6).rev().map(|i| i as f32)).unwrap(),
            false,
        );
        assert!(!e2.requires_grad());

        let e3 = mm(&e1, &e2).unwrap();
        assert!(e3.requires_grad());
        assert_eq!(e3.meta.shape.as_slice(), &[2, 2]);
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&[-0.5, -0.2, 1.0, 0.1]);
        e3.eval_bwd();

        assert_allclose!(e3.storage.data.borrow(), &[20.0, 14.0, 56.0, 41.0]);
        assert_allclose!(e1.get_grad().borrow(), &[-4.0, -2.6, -1.2, 6.5, 4.3, 2.1]);
        assert!(e2.storage.grad.is_none());
    }

    #[test]
    fn test_matmul_none_require_grad() {
        let e1 = ident(
            Tensor::from_buf(&[2, 3], (1..=6).map(|i| i as f32)).unwrap(),
            false,
        );
        assert!(!e1.requires_grad());
        let e2 = ident(
            Tensor::from_buf(&[3, 2], (1..=6).rev().map(|i| i as f32)).unwrap(),
            false,
        );
        assert!(!e2.requires_grad());

        let e3 = mm(&e1, &e2).unwrap();
        assert!(!e3.requires_grad());
        assert_eq!(e3.meta.shape.as_slice(), &[2, 2]);
        e3.eval_fwd();
        e3.eval_bwd();

        assert_allclose!(e3.storage.data.borrow(), &[20.0, 14.0, 56.0, 41.0]);
    }

    #[test]
    fn test_matmul_same_parent() {
        let e1 = ident(
            Tensor::from_buf(&[2, 2], (1..=4).map(|i| i as f32)).unwrap(),
            true,
        );
        assert!(e1.requires_grad());

        let e3 = mm(&e1, &e1).unwrap();
        assert!(e3.requires_grad());
        assert_eq!(e3.meta.shape.as_slice(), &[2, 2]);
        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&[-0.5, -0.2, 1.0, 0.1]);
        e3.eval_bwd();

        assert_allclose!(e3.storage.data.borrow(), &[7.0, 10.0, 15.0, 22.0]);
        assert_allclose!(e1.get_grad().borrow(), &[1.6, -2.2, 4.2, 3.4]);
    }

    #[test]
    fn test_matmul_same_parent_no_grad() {
        let e1 = ident(
            Tensor::from_buf(
                &[3, 3],
                [5.0_f32, -0.1, 3.2, 4.9, 7.0, -2.3, -4.3, 1.2, -7.8],
            )
            .unwrap(),
            false,
        );
        assert!(!e1.requires_grad());

        let e3 = mm(&e1, &e1).unwrap();
        assert!(!e3.requires_grad());
        assert_eq!(e3.meta.shape.as_slice(), &[3, 3]);
        e3.eval_fwd();
        e3.eval_bwd();

        assert_allclose!(
            e3.storage.data.borrow(),
            &[10.75, 2.64, -8.73, 68.69, 45.75, 17.52, 17.92, -0.53, 44.32]
        );
    }
}
