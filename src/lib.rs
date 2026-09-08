#![cfg_attr(test, allow(dead_code))]
pub mod broadcast;
pub mod consts;
pub mod error;
pub mod iterator;
pub mod tensor;
mod comp_kernels;

#[macro_export]
macro_rules! make_args {
    ($($arg:tt),*) => {
        {
            let new_args: [SmallDimVec; _] = [$(smallvec!$arg),*];
            new_args
        }
    };
}

#[cfg(test)]
mod tests {
    use smallvec::SmallVec;
    use crate::iterator::TensorIter;
    use crate::tensor::Tensor;

    #[test]
    fn test_usage() {
        let s1 = SmallVec::from_buf([1, 2, 3]);
        let s2 = SmallVec::from_buf([5, 1, 1]);

        let mut t1 = Tensor::new(&s1, 5f32);
        let t2 = Tensor::new(&s2, 3f32);

        let mut odo = TensorIter::from_tensors([&t1, &t2], 0).unwrap();

        while !odo.done {
            t1[odo.pos[0]] += t2[odo.pos[1]];
            odo.advance()
        }

        odo.reset();
        let mut numel: usize = 0;
        for pos in odo {
            numel += 1;
            // here is 20, not 8 because we are not doing elementwise accumulation here
            // due to broadcast, each elem of s1 interacts with full s2
            // ie s1[i][j] += \sum_{k = 0}^{4} s2[k]
            // ie s1[i][j] += 3 * 5 => t1.data[i] == 5 + 3 * 5 == 20
            assert_eq!(t1[pos[0]], 20f32);
        }

        assert_eq!(numel, 30);
    }
}

#[cfg(feature = "python-testing")]
#[pyo3::pymodule]
fn tensor_lib(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    use pyo3::wrap_pymodule;
    m.add_wrapped(wrap_pymodule!(crate::consts::py_tests))?;
    m.add_wrapped(wrap_pymodule!(crate::tensor::py_tests))?;
    m.add_wrapped(wrap_pymodule!(crate::broadcast::py_tests))?;
    Ok(())
}
