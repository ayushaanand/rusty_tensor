use crate::consts::*;
use crate::math::op::Op;
use crate::tensor::{Tensor, TensorMeta};
use crate::types::SharedSlice;

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
    // TODO: remove var and SharedSlice, data and Option<grad> should be in one Rc
    grad: Option<SharedSlice<T>>,
    op: Op<T>,
}

impl<T: TensorFType> Expr<T> {
    pub fn requires_grad(&self) -> bool {
        self.grad.is_some()
    }

    /// Get the underlying grad buffer of expr
    ///
    /// # Panic:
    /// Panics if expr.[`requires_grad()`](requires_grad) == false
    fn get_grad(&self) -> &SharedSlice<T> {
        self.grad.as_ref().unwrap()
    }
}

impl<T: TensorFType> Expr<T> {
    fn eval_fwd(&self) {
        self.op.fwd(&mut self.var.data.borrow_mut());
    }
    fn eval_bwd(&self) {
        let Some(grad) = &self.grad else {
            return;
        };

        // TODO: expr requires_grad, ie one parent must require too (if expr not ident)
        //       may be optimizable
        self.op.bwd(&*grad.borrow(), &self.var.data.borrow());
    }
}
