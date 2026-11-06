use crate::consts::*;
use crate::math::op::Op;
use crate::tensor::{Tensor, TensorMeta};
use crate::types::SharedSlice;
use std::cell::Ref;

mod op;

// ------- Var -----------------------------------------------------------------
#[derive(Clone)]
struct Var<T: TensorFType> {
    meta: TensorMeta,
    data: SharedSlice<T>,
}

impl<T: TensorFType> From<Tensor<T>> for Var<T> {
    fn from(value: Tensor<T>) -> Self {
        Self {
            meta: value.meta,
            data: SharedSlice::new(value.data),
        }
    }
}

// ------- Expr ----------------------------------------------------------------
#[derive(Clone)]
struct Expr<T: TensorFType> {
    var: Var<T>,
    grad: Option<SharedSlice<T>>, // None only for ident nodes
    op: Op<T>,
    retain_data: bool, // usage is not currently in scope
}

impl<T: TensorFType> Expr<T> {
    pub fn requires_grad(&self) -> bool {
        self.grad.is_some()
    }
    pub fn get_grad(&self) -> &SharedSlice<T> {
        self.grad.as_ref().unwrap()
    }
}

impl<T: TensorFType> Expr<T> {
    fn eval_fwd(&mut self) {
        self.op.fwd(&mut *self.var.data.borrow_mut());
    }
    fn eval_bwd(&mut self) {
        let grad = self.grad.as_ref().unwrap_or(&self.var.data);
        self.op.bwd(&grad.borrow(), &*self.var.data.borrow());
    }
}
