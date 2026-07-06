pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub mod broadcast;
pub(crate) mod consts;
pub mod math;
pub mod odometer;
pub mod tensor;

use odometer::Odometer;
use tensor::Tensor;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::broadcast::compute_broadcast_shape;

    #[test]
    fn it_works() {
        let t1 = Tensor::new(vec![1, 4, 2, 5], 1u32);
        let t1_b = t1.borrow();
        let t2 = Tensor::new(vec![3, 1], 1u32);
        let t2_b = t2.borrow();
        let t3 = Tensor::new(vec![5, 1, 2], 1u32);
        let t3_b = t3.borrow();

        let mut odo = match Odometer::new([t2_b.get_meta(), t1_b.get_meta()], 0) {
            Some(odo) => odo,
            None => panic!(),
        };

        // for odo_iter in &mut odo {
        //     println!("{} {}", odo_iter[0], odo_iter[1]);
        // }

        println!("{:?}", t1_b);
        println!("{:?}", t2_b);

        let b = compute_broadcast_shape([t1_b.get_meta()], 2).unwrap();
        println!("{:?}", b);
    }
}
