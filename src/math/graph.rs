use std::rc::Rc;

use crate::consts::*;
use crate::math::Expr;

pub struct GradGraph<T: TensorFType> {
    topo_seq: Vec<Rc<Expr<T>>>,
}

impl<T: TensorFType> GradGraph<T> {
    fn compile_from(heads: &[Rc<Expr<T>>]) -> Self {
        unimplemented!()
    }
}