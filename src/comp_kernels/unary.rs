use crate::consts::*;
use crate::iterator::TensorIter;

fn unary_op_assign<T: TensorDType, F: Fn(T) -> T>(
    t_r: &mut [T],
    t_o: &[T],
    it: &mut TensorIter,
    f: F,
) {
    it.reset();
    while !it.done {
        t_r[it.pos[0]] = f(t_o[it.pos[1]]);
        it.advance();
    }
}

fn unary_op_accum<T: TensorDType, F: Fn(T) -> T>(
    t_r: &mut [T],
    t_o: &[T],
    it: &mut TensorIter,
    f: F,
) {
    it.reset();
    while !it.done {
        t_r[it.pos[0]] += f(t_o[it.pos[1]]);
        it.advance();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_unary_op_assign_i32_ident() {
        let t1 = Tensor::new(&[5, 2, 3], 2);
        let mut t2 = Tensor::new(&[5, 2, 3], 5);
        let mut it = TensorIter::from_tensors([&t2, &t1], 0).unwrap();
        unary_op_assign(&mut t2.data, &t1.data, &mut it, |i| i);
        it.for_each(|i| {
            assert_eq!(t2.data[i[0]], t1.data[i[1]]);
        });
    }

    #[test]
    fn test_unary_op_assign_i32_neg() {
        let t1 = Tensor::new(&[5, 2, 3], 2);
        let mut t2 = Tensor::new(&[5, 2, 3], 5);
        let mut it = TensorIter::from_tensors([&t2, &t1], 0).unwrap();
        unary_op_assign(&mut t2.data, &t1.data, &mut it, |i| -i);
        it.for_each(|i| {
            assert_eq!(t2.data[i[0]], -t1.data[i[1]]);
        });
    }

    #[test]
    fn test_unary_op_assign_f32_exp() {
        let t1 = Tensor::new(&[5, 2, 3], 2f32);
        let mut t2 = Tensor::new(&[5, 2, 3], 5f32);
        let mut it = TensorIter::from_tensors([&t2, &t1], 0).unwrap();
        unary_op_assign(&mut t2.data, &t1.data, &mut it, |f| f.exp());
        it.for_each(|i| {
            assert_eq!(t2.data[i[0]], t1.data[i[1]].exp());
        });
    }

    #[test]
    fn test_unary_op_assign_f32_pow() {
        let t1 = Tensor::new(&[5, 2, 3], 2f32);
        let mut t2 = Tensor::new(&[5, 2, 3], 5f32);
        let mut it = TensorIter::from_tensors([&t2, &t1], 0).unwrap();
        unary_op_assign(&mut t2.data, &t1.data, &mut it, |f| f.powf(3.0));
        it.for_each(|i| {
            assert_eq!(t2.data[i[0]], t1.data[i[1]].powf(3.0));
        });
    }

    #[test]
    fn test_unary_op_assign_f32_accum() {
        let t1 = Tensor::new(&[5, 2, 3], 2f32);
        let mut t2 = Tensor::new(&[5, 2, 3], 5f32);
        let mut it = TensorIter::from_tensors([&t2, &t1], 0).unwrap();
        unary_op_accum(&mut t2.data, &t1.data, &mut it, |f| f.powf(3.0));
        it.for_each(|i| {
            assert_eq!(t2.data[i[0]], 5f32 + t1.data[i[1]].powf(3.0));
        });
    }

    #[test]
    fn test_unary_op_assign_i32_accum_broadcast() {
        let t1 = Tensor::new(&[3, 5, 2, 3], 2i32);
        let mut t2 = Tensor::new(&[5, 2, 3], 5i32);
        let mut it = TensorIter::from_tensors([&t2, &t1], 0).unwrap();
        unary_op_accum(&mut t2.data, &t1.data, &mut it, |i| -i);
        it.for_each(|i| {
            assert_eq!(t2.data[i[0]], 5 - 3 * t1.data[i[1]]);
        });
    }
}
