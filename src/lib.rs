pub mod broadcast;
pub(crate) mod consts;
pub mod math;
pub mod odometer;
pub mod tensor;
mod new_odometer;

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