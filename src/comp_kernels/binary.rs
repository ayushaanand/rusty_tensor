use crate::consts::TensorDType;
use crate::iterator::TensorIter;

pub fn binary_op_assign<T: TensorDType, F: Fn(T, T) -> T>(
    t_r: &mut [T],
    t_a: &[T], t_b: &[T],
    it: &mut TensorIter,
    f: F
) {
    it.reset();
    while !it.done {
        t_r[it.pos[0]] = f(t_a[it.pos[1]], t_b[it.pos[2]]);
        it.advance();
    }
}

pub fn binary_op_accum<T: TensorDType, F: Fn(T, T) -> T>(
    t_r: &mut [T],
    t_a: &[T], t_b: &[T],
    it: &mut TensorIter,
    f: F
) {
    it.reset();
    while !it.done {
        t_r[it.pos[0]] += f(t_a[it.pos[1]], t_b[it.pos[2]]);
        it.advance();
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use super::*;

    #[test]
    fn test_binary_op_accum_i32_add() {
        let t1 = Tensor::new(&[2, 3], 5);
        let t2 = Tensor::new(&[2, 3], 3);

        let mut t3 = Tensor::new(&[2, 3], 0);
        let mut it = TensorIter::from_tensors([&t3, &t2, &t1], 0).unwrap();

        binary_op_accum(
            &mut t3.data,
            &t1.data, &t2.data,
            &mut it,
            |a, b| a + b
        );

        it.reset();
        for i in it {
            assert_eq!(t3[i[0]], t1[i[1]] + t2[i[2]]);
        }
    }

    #[test]
    fn test_binary_op_accum_i32_xor_broadcast() {
        let t1 = Tensor::new(&[3, 2, 3], 5);
        let t2 = Tensor::new(&[2, 3], 3);

        let mut t3 = Tensor::new(&[2, 3], 0);
        let mut it = TensorIter::from_tensors([&t3, &t1, &t2], 0).unwrap();

        binary_op_accum(
            &mut t3.data,
            &t1.data, &t2.data,
            &mut it,
            |a, b| a ^ b
        );

        it.reset();
        for i in it {
            assert_eq!(t3[i[0]], 3 * (t1[i[1]] ^ t2[i[2]]));
        }
    }

    #[test]
    fn test_binary_op_assign_f32_exp_add() {
        let t1 = Tensor::new(&[2, 3], 5f32);
        let t2 = Tensor::new(&[2, 3], 3f32);

        let mut t3 = Tensor::new(&[2, 3], 6f32);
        let mut it = TensorIter::from_tensors([&t3, &t1, &t2], 0).unwrap();

        binary_op_assign(
            &mut t3.data,
            &t1.data, &t2.data,
            &mut it,
            |a, b| a.exp() + b
        );

        it.reset();
        for i in it {
            assert_eq!(t3[i[0]], t1[i[1]].exp() + t2[i[2]]);
        }
    }
}
