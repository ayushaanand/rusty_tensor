use crate::consts::*;
use crate::math::op::Op;
use crate::tensor::TensorMeta;
use crate::types::SharedSlice;

mod op;

#[derive(Clone)]
struct Var<T: TensorFType> {
    meta: TensorMeta,
    data: SharedSlice<T>,
    requires_grad: bool,
}

#[derive(Clone)]
struct Expr<T: TensorFType> {
    var: Var<T>,
    grad: Option<SharedSlice<T>>, // None only for ident nodes
    op: Op<T>,
    retain_data: bool,
}
