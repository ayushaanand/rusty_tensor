use crate::consts::{TensorFType, TensorResult};
use crate::error::RangeError;
use crate::iterator::TensorIter;
use crate::math::op::Op;
use crate::math::{Expr, Var};
use crate::tensor::TensorMeta;
use crate::types::{Shared, SharedSlice};

#[derive(Clone)]
pub(super) struct ViewOp<T: TensorFType> {
    input: Shared<Expr<T>>,
    it: TensorIter,
}

fn build_view_op<T: TensorFType>(
    input: &Shared<Expr<T>>,
    new_meta: TensorMeta,
    op_var: impl FnOnce(ViewOp<T>) -> Op<T>,
) -> TensorResult<Shared<Expr<T>>> {
    let grad_meta = TensorMeta::new(&new_meta.shape);
    let it = TensorIter::new([&grad_meta, &new_meta], 0)?;

    let op = op_var(ViewOp {
        input: input.clone(),
        it,
    });

    let borrow = input.borrow();
    let grad = borrow
        .requires_grad()
        .then(|| SharedSlice::zeroes(new_meta.total_elems));

    Ok(Shared::new(Expr {
        var: Var {
            meta: new_meta,
            ..borrow.var.clone()
        },
        grad,
        op,
    }))
}

pub fn transpose<T: TensorFType>(
    input: &Shared<Expr<T>>,
    i: usize,
    j: usize,
) -> TensorResult<Shared<Expr<T>>> {
    let borrow = input.borrow();

    let mut new_meta = borrow.var.meta.clone();
    new_meta.shape.swap(i, j);
    new_meta.stride.swap(i, j);

    build_view_op(input, new_meta, Op::Transpose)
}

// ------- Slicing ----------------------------------------------------
enum SlicingRange {
    Range(std::ops::Range<usize>),
    RangeInclusive(std::ops::RangeInclusive<usize>),
    RangeFrom(std::ops::RangeFrom<usize>),
    RangeTo(std::ops::RangeTo<usize>),
    RangeToInclusive(std::ops::RangeToInclusive<usize>),
    RangeFull(std::ops::RangeFull),
}

impl SlicingRange {
    fn to_tuple(&self, up_bound: usize) -> (usize, usize) {
        match self {
            SlicingRange::Range(r) => (r.start, r.end),
            SlicingRange::RangeInclusive(r) => {
                let inner = r.clone().into_inner();
                (inner.0, inner.1 + 1)
            }
            SlicingRange::RangeFrom(r) => (r.start, up_bound),
            SlicingRange::RangeTo(r) => (0, r.end),
            SlicingRange::RangeToInclusive(r) => (0, r.end + 1),
            SlicingRange::RangeFull(_) => (0, up_bound),
        }
    }
}

macro_rules! impl_slicing_from {
    ($name:ident<$ty:ty>) => {
        impl From<::std::ops::$name<$ty>> for SlicingRange {
            fn from(s: ::std::ops::$name<$ty>) -> Self {
                Self::$name(s)
            }
        }
    };

    ($name:ident) => {
        impl From<::std::ops::$name> for SlicingRange {
            fn from(s: ::std::ops::$name) -> Self {
                Self::$name(s)
            }
        }
    };
}

impl_slicing_from!(Range<usize>);
impl_slicing_from!(RangeInclusive<usize>);
impl_slicing_from!(RangeFrom<usize>);
impl_slicing_from!(RangeTo<usize>);
impl_slicing_from!(RangeToInclusive<usize>);
impl_slicing_from!(RangeFull);

#[macro_export]
macro_rules! ranges {
    ($($range:expr),* $(,)?) => { [$($crate::math::op::view::SlicingRange::from($range)),*] };
}

fn slice<T: TensorFType>(
    input: &Shared<Expr<T>>,
    ranges: &[SlicingRange],
) -> TensorResult<Shared<Expr<T>>> {
    let mut new_meta = input.borrow().var.meta.clone();

    if ranges.len() > new_meta.shape.len() {
        return Err(RangeError);
    }

    let offset = &mut new_meta.offset;
    let shape_it = new_meta.shape.as_mut_slice().iter_mut();
    let stride_it = new_meta.stride.as_mut_slice().iter_mut();
    for ((shape, stride), range) in shape_it.zip(stride_it).zip(ranges.into_iter()) {
        let (start, end) = range.to_tuple(*shape);

        *offset += start * *stride;
        *shape = if end > start { end - start } else { 0 };
    }

    new_meta.total_elems = new_meta.shape.iter().product();
    build_view_op(input, new_meta, Op::Slice)
}

impl<T: TensorFType> ViewOp<T> {
    pub(super) fn fwd(&mut self, _: &mut [T]) {}

    pub(super) fn bwd(&mut self, grad: &[T]) {
        let parent = self.input.borrow();

        if parent.requires_grad() {
            let mut parent_grad = parent.get_grad().borrow_mut();
            self.it.for_each(|pos| {
                parent_grad[pos[1]] += grad[pos[0]];
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_allclose;
    use crate::math::op::ident::ident;
    use crate::tensor::Tensor;
    use std::iter::repeat_n;

    #[test]
    fn test_transpose() {
        let e1 = ident(
            &Shared::new(Var::from(
                Tensor::from_buf(&[2, 3, 4], (0..24).map(|i| i as f32)).unwrap(),
            )),
            true,
        );

        let e2 = transpose(&e1, 0, 1).unwrap();
        assert_eq!(&*e2.borrow().var.meta.shape, &[3, 2, 4]);
        let e2borrow = &mut *e2.borrow_mut();

        let Op::Transpose(op) = &mut e2borrow.op else {
            unreachable!()
        };

        let mut v = Vec::new();
        let e2data = e2borrow.var.data.borrow();

        op.it.for_each(|pos| {
            v.push(e2data[pos[1]]);
        });

        assert_allclose!(
            v,
            &[0.0, 12.0, 4.0, 16.0, 8.0, 20.0]
                .into_iter()
                .map(|x| (0..4).map(move |i| x + (i as f32)))
                .flatten()
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slicing() {
        let e1 = ident(
            &Shared::new(Var::from(
                Tensor::from_buf(&[2, 3, 4], (0..24).map(|i| i as f32)).unwrap(),
            )),
            true,
        );

        let e2 = slice(&e1, &ranges![.., 2.., ..=1]).unwrap();
        assert_eq!(&*e2.borrow().var.meta.shape, &[2, 1, 2]);
        e2.borrow_mut().eval_fwd();
        e2.borrow()
            .get_grad()
            .borrow_mut()
            .copy_from_slice(&[-1.0, -2.0, -3.0, -4.0]);
        e2.borrow_mut().eval_bwd();

        #[rustfmt::skip]
        assert_allclose!(
            e1.borrow().get_grad().borrow(),
            &[0.0,  0.0, 0.0, 0.0,
              0.0,  0.0, 0.0, 0.0,
             -1.0, -2.0, 0.0, 0.0,

              0.0,  0.0, 0.0, 0.0,
              0.0,  0.0, 0.0, 0.0,
             -3.0, -4.0, 0.0, 0.0]
        );

        let e2borrow = &mut *e2.borrow_mut();

        let Op::Slice(op) = &mut e2borrow.op else {
            unreachable!()
        };

        let mut v = Vec::new();
        let e2data = e2borrow.var.data.borrow();

        op.it.for_each(|pos| {
            v.push(e2data[pos[1]]);
        });

        assert_allclose!(v, &[8.0, 9.0, 20.0, 21.0]);
    }
}
