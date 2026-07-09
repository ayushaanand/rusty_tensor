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
        let t1 = Tensor::<usize>::new(&*vec![], 1);
        let t1_b = t1.borrow();
        let t2 = Tensor::new(&*vec![], 1u32);
        let t2_b = t2.borrow();
        let t3 = Tensor::new(&*vec![], 1u32);
        let t3_b = t3.borrow();

        let mut odo = match Odometer::new(&t3_b.get_meta().shape, [t3_b.get_meta(), t2_b.get_meta(), t1_b.get_meta()], 0) {
            Some(odo) => odo,
            None => panic!(),
        };

        for odo_iter in &mut odo {
            println!("{} {} {}", odo_iter[0], odo_iter[1], odo_iter[2]);
        }

        println!("{:?}", t1_b);
        println!("{:?}", t2_b);

    }
}

#[cfg(feature = "python-testing")]
#[pyo3::pymodule]
fn tensor_lib(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_wrapped(pyo3::wrap_pymodule!(crate::consts::consts_tests))?;
    m.add_wrapped(pyo3::wrap_pymodule!(crate::tensor::tensor_tests))?;
    Ok(())
}