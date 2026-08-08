#![cfg_attr(test, allow(dead_code))]
pub mod broadcast;
pub mod consts;
// pub mod math;
pub mod tensor;
pub mod iterator;

#[macro_export]
macro_rules! make_args {
    ($($arg:tt),*) => {
        {
            use crate::consts::SmallDimVec;
            #[allow(unused_imports)]
            use smallvec::smallvec;
            
            let new_args: [SmallDimVec; _] = [$(smallvec!$arg),*];
            new_args
        }
    };
}

#[cfg(test)]
mod tests {
    use smallvec::{smallvec};
    use crate::consts::SmallDimVec;
    use crate::iterator::TensorIter;
    use crate::tensor::Tensor;

    #[test]
    fn test_usage() {
        let s1: SmallDimVec = smallvec![1, 2, 3];
        let s2: SmallDimVec = smallvec![5, 1, 1];

        let raw_tensors = [Tensor::new(&s1, 5), Tensor::new(&s2, 3)];

        let mut t1 = raw_tensors[0].borrow_mut();
        let t2 = raw_tensors[1].borrow();

        let metas = [&t1.meta, &t2.meta];
        let mut odo = TensorIter::new(metas, 0).unwrap();

        while !odo.done {
            t1.data[odo.pos[0]] += t2.data[odo.pos[1]];
            odo.advance()
        }

        odo.reset();
        let mut numel: usize = 0;
        for pos in odo {
            numel += 1;
            // here is 20, not 8 because we are not doing elementwise accumulation here
            // due to broadcast, each elem of s1 interacts with full s2
            // ie s1[i][j] += sum_{k = 0}^{4} s2[k]
            // ie s1[i][j] += 3 * 5 => t1.data[i] == 5 + 3 * 5 == 20
            assert_eq!(t1.data[pos[0]], 20);
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
