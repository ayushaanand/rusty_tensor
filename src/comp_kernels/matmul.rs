use crate::consts::{DimType, TensorDType};
use crate::iterator::TensorIter;

fn matmul_op_accum<T: TensorDType>(
    t_r: &mut [T],
    t_a: &[T],
    t_b: &[T],
    it: &mut TensorIter,
    m: DimType,
    p: DimType,
    n: DimType,
) {
    let (m, p, n) = (m as usize, p as usize, n as usize);
    it.reset();
    while !it.done {
        let r_o = it.pos[0];
        let a_o = it.pos[1];
        let b_o = it.pos[2];
        for i in 0..m {
            for k in 0..p {
                for j in 0..n {
                    t_r[r_o + i * n + j] += t_a[a_o + i * p + k] * t_b[b_o + k * n + j];
                    // m*n = m*p @ p*n
                }
            }
        }
        it.advance_unchecked();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_matmul_sq() {
        let a = Tensor::from_buf(&[2, 2], 1..=4).unwrap();
        let b = Tensor::from_buf(&[2, 2], (1..=4).rev()).unwrap();
        let mut c = Tensor::new(&[2, 2], 0);

        let mut it = TensorIter::from_tensors([&c, &b, &a], 2).unwrap();
        matmul_op_accum(&mut c.data, &a.data, &b.data, &mut it, 2, 2, 2);

        assert_eq!(c, Tensor::from_buf(&[2, 2], [8, 5, 20, 13]).unwrap());
    }

    #[test]
    fn test_matmul_rect() {
        let a = Tensor::from_buf(&[2, 3], 1..=6).unwrap();
        let b = Tensor::from_buf(&[3, 2], (1..=6).rev()).unwrap();
        let mut c = Tensor::new(&[2, 2], 0);

        let mut it = TensorIter::from_tensors([&c, &a, &b], 2).unwrap();
        matmul_op_accum(&mut c.data, &a.data, &b.data, &mut it, 2, 3, 2);

        assert_eq!(c, Tensor::from_buf(&[2, 2], [20, 14, 56, 41]).unwrap());
    }

    #[test]
    fn test_matmul_rect_broadcast() {
        let repeats = 5;
        let a =
            Tensor::from_buf(&[repeats, 2, 3], (1..=6).cycle().take(repeats as usize * 6)).unwrap();
        let b = Tensor::from_buf(&[3, 2], (1..=6).rev()).unwrap();
        let mut c = Tensor::new(&[2, 2], 0);

        let mut it = TensorIter::from_tensors([&c, &a, &b], 2).unwrap();
        matmul_op_accum(&mut c.data, &a.data, &b.data, &mut it, 2, 3, 2);

        assert_eq!(
            c,
            Tensor::from_buf(&[2, 2], [20, 14, 56, 41].map(|x| repeats as i32 * x)).unwrap()
        );
    }
}
