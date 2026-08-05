#![cfg_attr(test, allow(dead_code))]
pub mod broadcast;
pub(crate) mod consts;
// pub mod math;
mod new_odometer;
pub mod odometer;
pub mod tensor;

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
