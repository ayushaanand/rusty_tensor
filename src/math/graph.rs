use std::collections::HashSet;
use std::rc::Rc;

use crate::consts::*;
use crate::math::Expr;

pub struct GradGraph<T: TensorFType> {
    roots: Vec<Rc<Expr<T>>>,
    topo_seq: Vec<Rc<Expr<T>>>,
}

impl<T: TensorFType> GradGraph<T> {
    pub fn compile_from(roots: &[&Rc<Expr<T>>]) -> Self {
        let mut topo_seq = Vec::new();

        fn dfs<T: TensorFType>(
            node: &Rc<Expr<T>>,
            seq: &mut Vec<Rc<Expr<T>>>,
            visited: &mut HashSet<*const Expr<T>>,
        ) {
            // returns true if new entry, so NOT is done before for processing newly visited node
            if !visited.insert(Rc::as_ptr(node)) {
                return;
            }

            for child in node.inputs() {
                dfs(child, seq, visited);
            }

            seq.push(Rc::clone(node));
        }

        let mut visited_set = HashSet::new();
        for root in roots {
            dfs(*root, &mut topo_seq, &mut visited_set);
        }

        Self {
            roots: roots.iter().map(|expr| Rc::clone(*expr)).collect(),
            topo_seq,
        }
    }

    pub fn eval_fwd(&self) {
        for node in &self.topo_seq {
            node.eval_fwd();
        }
    }

    pub fn eval_bwd(&self) {
        for node in &self.topo_seq {
            node.fill_grad_with(T::zero());
        }

        for node in &self.roots {
            node.fill_grad_with(T::one());
        }

        for node in self.topo_seq.iter().rev() {
            node.eval_bwd();
        }
    }
}
