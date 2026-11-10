use std::cell::RefCell;
use std::rc::Rc;

use crate::consts::{TensorFType, TensorResult};
use crate::error::RangeError;
use crate::iterator::TensorIter;
use crate::math::{Expr, op::Op};
use crate::tensor::TensorMeta;

#[derive(Clone)]
pub(super) struct ViewOp<T: TensorFType> {
    input: Rc<Expr<T>>,
    it: RefCell<TensorIter>,
}

fn build_view_op<T: TensorFType>(
    input: &Rc<Expr<T>>,
    new_meta: TensorMeta,
    op_var: impl FnOnce(ViewOp<T>) -> Op<T>,
) -> TensorResult<Rc<Expr<T>>> {
    let grad_meta = TensorMeta::new(&new_meta.shape);
    let it = RefCell::new(TensorIter::new([&grad_meta, &new_meta], 0)?);
    let op = op_var(ViewOp {
        input: Rc::clone(input),
        it,
    });

    Ok(Rc::new(Expr {
        storage: Rc::clone(&input.storage),
        meta: new_meta,
        op,
    }))
}

pub fn transpose<T: TensorFType>(
    input: &Rc<Expr<T>>,
    i: usize,
    j: usize,
) -> TensorResult<Rc<Expr<T>>> {
    let mut new_meta = input.meta.clone();
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
    input: &Rc<Expr<T>>,
    ranges: &[SlicingRange],
) -> TensorResult<Rc<Expr<T>>> {
    let mut new_meta = input.meta.clone();

    if ranges.len() > new_meta.shape.len() {
        return Err(RangeError);
    }

    let offset = &mut new_meta.offset;
    let shape_it = new_meta.shape.as_mut_slice().iter_mut();
    let stride_it = new_meta.stride.as_mut_slice().iter_mut();
    for ((shape, stride), range) in shape_it.zip(stride_it).zip(ranges) {
        let (start, end) = range.to_tuple(*shape);

        *offset += start * *stride;
        *shape = end.saturating_sub(start);
    }

    new_meta.total_elems = new_meta.shape.iter().product();
    build_view_op(input, new_meta, Op::Slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assert_allclose;
    use crate::math::op::{binary::add, ident::ident};
    use crate::tensor::Tensor;

    #[test]
    fn test_transpose() {
        let e1 = ident(
            Tensor::from_buf(&[2, 3, 4], (0..24).map(|i| i as f32)).unwrap(),
            true,
        );

        let e2 = transpose(&e1, 0, 1).unwrap();
        assert_eq!(&*e2.meta.shape, &[3, 2, 4]);

        let Op::Transpose(op) = &e2.op else {
            unreachable!()
        };

        let mut v = Vec::new();
        let e2data = e2.storage.data.borrow();

        op.it.borrow_mut().for_each(|pos| {
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
            Tensor::from_buf(&[2, 3, 4], (0..24).map(|i| i as f32)).unwrap(),
            true,
        );

        let e2 = slice(&e1, &ranges![.., 2.., ..=1]).unwrap();
        assert_eq!(&*e2.meta.shape, &[2, 1, 2]);
        e2.eval_fwd();
        let Op::Slice(ViewOp { it, .. }) = &e2.op else {
            unreachable!()
        };
        it.borrow_mut().for_each(|pos| {
            e2.get_grad().borrow_mut()[pos[1]] += -9.0;
        });
        e2.eval_bwd();

        #[rustfmt::skip]
        assert_allclose!(
            e1.get_grad().borrow(),
            &[0.0,  0.0, 0.0, 0.0,
              0.0,  0.0, 0.0, 0.0,
             -9.0, -9.0, 0.0, 0.0,

              0.0,  0.0, 0.0, 0.0,
              0.0,  0.0, 0.0, 0.0,
             -9.0, -9.0, 0.0, 0.0]
        );

        let mut v = Vec::new();
        let e2data = e2.storage.data.borrow();

        it.borrow_mut().for_each(|pos| {
            v.push(e2data[pos[1]]);
        });

        assert_allclose!(v, &[8.0, 9.0, 20.0, 21.0]);
    }

    #[test]
    fn test_view_add() {
        let e1 = ident(
            Tensor::from_buf(&[3, 3], (1..=9).map(|i| i as f32)).unwrap(),
            true,
        );
        let e2 = transpose(&e1, 0, 1).unwrap();

        let e3 = add(&e1, &e2).unwrap();

        e3.eval_fwd();
        e3.get_grad()
            .borrow_mut()
            .copy_from_slice(&(1..=9).map(|i| i as f32).collect::<Vec<_>>());
        e3.eval_bwd();

        assert_eq!(
            &**e1.get_grad().borrow(),
            &[2.0, 6.0, 10.0, 6.0, 10.0, 14.0, 10.0, 14.0, 18.0]
        )
    }
}
