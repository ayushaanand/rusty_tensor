use super::unary::{UnaryOp, build_unary_op};
use crate::consts::*;
use crate::math::Expr;
use crate::math::op::Op;
use std::rc::Rc;

pub(crate) struct ApplyOp<T: TensorFType> {
    unary: UnaryOp<T>,
    apply: Box<dyn ApplyTrait<T>>,
}

trait ApplyTrait<T: TensorFType> {
    fn fwd(&self, unary: &UnaryOp<T>, out: &mut [T]);
    fn bwd(&self, unary: &UnaryOp<T>, grad: &[T], out_val: &[T]);
}

struct ApplyFunctions<F, FDash> {
    f: F,
    f_dash: Option<FDash>,
}

impl<T: TensorFType, F, FDash> ApplyTrait<T> for ApplyFunctions<F, FDash>
where 
    F: Fn(T) -> T,
    FDash: Fn(T) -> T,
{
    fn fwd(&self, unary: &UnaryOp<T>, out: &mut [T]) {
        unary.fwd(out, |x| (self.f)(x));
    }

    fn bwd(&self, unary: &UnaryOp<T>, grad: &[T], out_val: &[T]) {
        if let Some(f_dash) = &self.f_dash {
            unary.bwd(grad, out_val, |_, a| f_dash(a))
        }
        else { /* unreachable!() */ }
    }
}  

fn apply<T: TensorFType, F, FDash>(input: &Rc<Expr<T>>, f: F, f_dash: FDash,
) -> TensorResult<Rc<Expr<T>>> 
where
    F: Fn(T) -> T + 'static,
    FDash: Fn(T) -> T + 'static,
{
    let apply = Box::new(ApplyFunctions {
        f,
        f_dash: input.requires_grad().then(|| f_dash),
    });
    
    build_unary_op(input, |unary| {
        Op::Apply(ApplyOp {
            unary,
            apply,
        })
    })
}

impl<T: TensorFType> ApplyOp<T> {
    pub(super) fn fwd(&self, out: &mut [T]) {
        self.apply.fwd(&self.unary, out);
    }

    pub(super) fn bwd(&self, grad: &[T], out_val: &[T]) {
        self.apply.bwd(&self.unary, grad, out_val);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::op::ident::ident; 
    use crate::tensor::Tensor;
    
    #[test]
    fn test_apply_exp() {
        let input: [f32; _] = [0.1, 0.2, 0.3, 0.4];
        let input_exp = input.map(|x| x.exp());
        
        let e1 = ident(Tensor::from_buf(&[2, 2], input).unwrap(), true);

        let e2 = apply(&e1, f32::exp, f32::exp).unwrap();

        e2.eval_fwd();
        assert_eq!(&**e2.storage.data.borrow(), &input_exp);

        e2.get_grad().borrow_mut().copy_from_slice(&input);
        e2.eval_bwd();
        assert_eq!(&**e1.get_grad().borrow(), &input.map(|x| x * x.exp()));
    }
}