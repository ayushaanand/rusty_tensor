use std::rc::Rc;
use crate::consts::*;
use crate::tensor::{Tensor, TensorMeta};

use crate::math::op::Op;

// Expr just helps in building the graph of computation
pub(crate) struct Expr<T: TensorDType> {
    pub(crate) meta: TensorMeta,

    pub(crate) data: Option<Rc<Vec<T>>>, // None only for ident nodes
    pub(crate) grad: Option<Vec<T>>, // None only for ident nodes
    pub(crate) op: Op<T>,

    pub(crate) requires_grad: bool,
    pub(crate) retain_data: bool,
}