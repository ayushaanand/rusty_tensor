use std::cell::RefCell;
use std::rc::Rc;

use crate::consts::*;
use crate::tensor::TensorMeta;

pub mod op;
pub mod graph;
use op::Op;

// ------- Storage -----------------------------------------------------------------
struct Storage<T> {
    data: RefCell<Box<[T]>>,
    grad: Option<RefCell<Box<[T]>>>,
}

impl<T: TensorDType> Storage<T> {
    fn zeroes(len: usize, requires_grad: bool) -> Self {
        let new_refcell_box = || RefCell::new(vec![T::zero(); len].into());
        Self {
            data: new_refcell_box(),
            grad: requires_grad.then(new_refcell_box),
        }
    }

    fn new_from(data: Box<[T]>, requires_grad: bool) -> Self {
        Self {
            grad: requires_grad.then(|| RefCell::new(vec![T::zero(); data.len()].into())),
            data: RefCell::new(data),
        }
    }
}

// ------- Expr ----------------------------------------------------------------
pub struct Expr<T: TensorFType> {
    meta: TensorMeta,
    storage: Rc<Storage<T>>,
    op: Op<T>,
}

impl<T: TensorFType> Expr<T> {
    pub fn requires_grad(&self) -> bool {
        self.storage.grad.is_some()
    }

    #[cfg(test)]
    /// Get the underlying grad buffer of expr
    ///
    /// # Panic:
    /// Panics if expr.[`requires_grad()`](requires_grad) == false
    fn get_grad(&self) -> &RefCell<Box<[T]>> {
        self.storage.grad.as_ref().unwrap()
    }

    pub(crate) fn inputs(&self) -> &[Rc<Expr<T>>] {
        self.op.inputs()
    }
}

impl<T: TensorFType> Expr<T> {
    pub fn eval_fwd(&self) {
        self.op.fwd(&mut self.storage.data.borrow_mut());
    }

    pub fn eval_bwd(&self) {
        let Some(grad) = &self.storage.grad else {
            return;
        };

        // TODO: expr requires_grad, ie one parent must require too (if expr not ident)
        //       may be optimizable
        self.op.bwd(&grad.borrow(), &self.storage.data.borrow());
    }
}
