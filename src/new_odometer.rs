use smallvec::SmallVec;
use std::cmp::Reverse;

use crate::broadcast::compute_broadcast_shape_iter;
use crate::consts::*;
use crate::tensor::TensorMeta;

#[derive(Debug)]
pub struct Odometer {
    pub pos: SmallSizeVec,             // len = args.len()
    pub(crate) idx: SmallDimVec,       // len = max_rank - skip_dims
    pub(crate) offsets: SmallSizeVec,  // len = args.len()
    pub(crate) out_shape: SmallDimVec, // len = max_rank - skip_dims
    pub(crate) arg_strides: SmallVec<[SmallSizeVec; SMALL_SIZE]>, // len = args.len(), len(args._) = max_rank - skip_dims
    pub done: bool,
}

// 1. args.is_empty() -> return None
// 2. any(arg.total_elems == SIZE_ZERO) -> return None
// 3. compute max_rank, if skip_dims > max_rank -> return None
// 4. copy every arg's (shape, stride) to vecs of max_rank: prepend shape with 1, stride with 0, on the left
// 5. compute_broadcast_shape
// 6. eliminate trivial dims: keep only indices d where broadcasted_shape[d] != 1
// 7. if arg.shape[i] == 1, arg.stride[i] = 0
// 8. compute permutation from arg[0].stride (assuming arg[0] is hotter): stride == 0s first, then descending by stride value
// 9. apply that permutation to out_shape, new_shapes[i], new_strides[i]
// 10. coalesce adjacent dims d, d+1 (after permutation): merge iff for every arg:
//      stride[d] == stride[d+1] * shape[d+1]
//      on merge: shape[d+1] *= shape[d], out_shape[d] = 0, drop d
// 11. build final out_shape and arg_strides
// 12. build rest fields
// *. out_shape.len == 0 is handled as special case by inserting a stub dim 1 to remove branches in advance

trait MetaIter<'a>: Iterator<Item=&'a TensorMeta> + Clone {}

impl<'a, T> MetaIter<'a> for T
where T: Iterator<Item=&'a TensorMeta> + Clone {}

impl Odometer {
    pub fn new<'a, I>(args: I, skip_dims: usize) -> Option<Self>
    where
        I: IntoIterator<Item=&'a TensorMeta>,
        I::IntoIter: Clone,
    {
        let it = args.into_iter();
        // 1, 2, 3
        if !OdometerBuilder::is_valid(it.clone(), skip_dims) {
            return None;
        }

        // 4, 5
        let mut builder = OdometerBuilder::new(it, skip_dims)?;

        // 6. eliminate trivial dims: keep only indices d where broadcasted_shape[d] != 1
        builder.part(1);

        // 7. if arg.shape[i] == 1, arg.stride[i] = 0
        builder.zero_broadcasted_dims();

        // 8. compute permutation from arg[0].stride (assuming arg[0] is hotter): stride == 0s first, then descending by stride value
        builder.sort_by_stride_of::<0>();

        // 10. coalesce adjacent dims d, d+1 (after permutation): merge iff for every arg:
        //      stride[d] == stride[d+1] * shape[d+1]
        //      on merge: shape[d+1] *= shape[d], out_shape[d] = 0, drop d
        builder.coalesce_dims();

        // 11. build final out_shape and arg_strides, remove extra dims
        builder.shrink_to_fit();

        // 12. build rest fields
        Some(builder.build())
    }

    pub fn reset(&mut self) {
        self.pos = self.offsets.clone();
        self.idx.iter_mut().for_each(|i| *i = 0);
        self.done = false;
    }

    #[inline(always)]
    pub fn advance(&mut self) {
        // out_shape[i] is never 0 or 1
        // except when scalar iteration (special case)
        let last_dim = self.out_shape.len() - 1;
        self.inc_dim(last_dim);

        if self.idx[last_dim] == self.out_shape[last_dim] {
            self.advance_carry();
        }
    }

    #[cold]
    fn advance_carry(&mut self) {
        let mut i = self.out_shape.len() - 1;
        self.idx[i] = 0;
        self.reduce_dim_by(i, self.out_shape[i] as SizeType);

        // clear reduces by shape - 1
        while i > 0 {
            i -= 1;
            if self.idx[i] < self.out_shape[i] - 1 {
                self.inc_dim(i);
                return;
            } else {
                self.clear_dim(i);
            };
        }
        self.done = true;
    }

    #[inline(always)]
    fn inc_dim(&mut self, dim: usize) {
        self.idx[dim] += 1;
        for (a, p) in self.arg_strides.iter().zip(self.pos.iter_mut()) {
            *p += a[dim];
        }
    }

    fn clear_dim(&mut self, dim: usize) {
        self.idx[dim] = 0;
        self.reduce_dim_by(dim, self.out_shape[dim] as SizeType - 1);
    }

    fn reduce_dim_by(&mut self, dim: usize, amount: SizeType) {
        for (a, p) in self.arg_strides.iter().zip(self.pos.iter_mut()) {
            *p -= a[dim] * amount;
        }
    }
}

impl Iterator for Odometer {
    type Item = SmallSizeVec;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let res = self.pos.clone();
        self.advance();
        Some(res)
    }
}

struct OdometerBuilder {
    out_shape: SmallDimVec,
    shapes: SmallVec<[SmallDimVec; SMALL_SIZE]>,
    strides: SmallVec<[SmallSizeVec; SMALL_SIZE]>,
    offsets: SmallSizeVec,
    actual_rank: usize,
}

impl OdometerBuilder {
    fn is_valid<'a>(args: impl MetaIter<'a>, skip_dims: usize) -> bool {
        // 1. args.is_empty() -> return None
        let args_len = args.clone().count();
        if args_len == 0 {
            return false;
        }

        // 2. any(arg.total_elems == SIZE_ZERO) -> return None
        if args.clone().any(|m| m.total_elems == 0) {
            return false;
        }

        // 3. compute max_rank, if skip_dims > max_rank -> return None
        let max_rank = args.clone().map(|m| m.shape.len()).max().unwrap(); // unwrap is fine because args_len != 0
        if skip_dims > max_rank {
            return false;
        }

        true
    }

    fn new<'a>(args: impl MetaIter<'a>, skip_dims: usize) -> Option<Self> {
        // args.len > 0 guaranteed from is_valid
        let max_rank = args.clone().map(|m| m.shape.len()).max().unwrap();

        // 4. copy every arg's (shape, stride) to vecs of max_rank: prepend shape with 1, stride with 0, on the left
        let avb_dims = max_rank - skip_dims; // no underflow because of 3

        // shape/stride[i].len = avb_dims = max_rank - skip_dims
        let shapes: SmallVec<[SmallDimVec; SMALL_SIZE]> = args
            .clone()
            .map(|m| Self::make_padded(&m.shape, 1 as DimType, avb_dims, skip_dims))
            .collect();
        let strides: SmallVec<[SmallSizeVec; SMALL_SIZE]> = args
            .clone()
            .map(|m| Self::make_padded(&m.stride, SIZE_ZERO, avb_dims, skip_dims))
            .collect();

        // 5. compute_broadcast_shape
        let out_shape = compute_broadcast_shape_iter(
            shapes.iter().map(|s| s.as_ref()),
            0, // because we eliminated skip_dims in make_padded
        )?;

        let offsets = args.map(|m| m.offset).collect();

        Some(OdometerBuilder {
            out_shape,
            shapes,
            strides,
            offsets,
            actual_rank: avb_dims,
        })
    }

    fn make_padded<S, T>(slice: &[T], pad_val: T, avb_dims: usize, skip_dims: usize) -> S
    where
        S: FromIterator<T>,
        T: Clone + Sized,
    {
        let avb_loc_dims = slice.len().saturating_sub(skip_dims); // avb_loc_dims = slice.len - skip_dims
        // pad with (avb_dims - avb_loc_dims) pad_vals
        // no underflow because avb_dims = max_rank - skip_dims >= loc_rank.saturating_sub(skip_dims)
        std::iter::repeat_n(pad_val, avb_dims - avb_loc_dims)
            // take the rest avb_loc_dims, avb_loc_dims + (avb_dims - avb_loc_dims) = avb_dims
            .chain(slice[..avb_loc_dims].iter().cloned())
            .collect()
    }

    fn swap_dims(&mut self, i: usize, j: usize) {
        self.out_shape.swap(i, j);
        self.shapes.iter_mut().for_each(|s| s.swap(i, j));
        self.strides.iter_mut().for_each(|s| s.swap(i, j));
    }

    fn part(&mut self, target: DimType) {
        let mut j = 0;
        for i in 0..self.actual_rank {
            if self.out_shape[i] != target {
                if i != j {
                    self.swap_dims(i, j);
                }
                j += 1;
            }
        }
        self.actual_rank = j;
    }

    fn zero_broadcasted_dims(&mut self) {
        self.shapes
            .iter_mut()
            .zip(self.strides.iter_mut())
            .for_each(|(sh, st)| {
                for i in 0..self.actual_rank {
                    if sh[i] == 1 {
                        st[i] = SIZE_ZERO
                    }
                }
            });
    }

    fn sort_by_stride_of<const I: usize>(&mut self) {
        let mut perm: SmallVec<[usize; SMALL_SIZE]> = (0..self.actual_rank).collect();
        perm.sort_unstable_by_key(|&i| {
            Reverse(if self.strides[I][i] == SIZE_ZERO {
                usize::MAX
            } else {
                self.strides[I][i] as usize
            })
        });

        for i in 0..self.actual_rank {
            while perm[i] != i {
                self.swap_dims(i, perm[i]);
                let new_pos = perm[i];
                perm.swap(i, new_pos);
            }
        }
    }

    fn coalesce_dims(&mut self) {
        for i in (0..self.actual_rank.saturating_sub(1)).rev() {
            let all_agree = !self
                .shapes
                .iter()
                .zip(self.strides.iter())
                .any(|(sh, st)| st[i] != st[i + 1] * (sh[i + 1] as SizeType));
            if all_agree {
                self.shapes
                    .iter_mut()
                    .zip(self.strides.iter_mut())
                    .for_each(|(sh, st)| {
                        sh[i] *= sh[i + 1];
                        st[i] = st[i + 1];
                    });
                self.out_shape[i] *= self.out_shape[i + 1];
                self.out_shape[i + 1] = DIM_ZERO;
            }
        }

        self.part(DIM_ZERO);
    }

    fn shrink_to_fit(&mut self) {
        fn drop_ext_dims<T>(vec: &mut SmallVec<[T; SMALL_SIZE]>, len: usize) {
            vec.truncate(len);
            vec.shrink_to_fit();
        }

        for i in 0..self.shapes.len() {
            drop_ext_dims(&mut self.shapes[i], self.actual_rank);
            drop_ext_dims(&mut self.strides[i], self.actual_rank);
        }
        drop_ext_dims(&mut self.out_shape, self.actual_rank);
    }

    fn build(mut self) -> Odometer {
        if self.actual_rank == 0 {
            self.out_shape.push(1);
            self.strides.iter_mut().for_each(|s| s.push(SIZE_ZERO));
            self.actual_rank = 1;
        }

        let pos: SmallSizeVec = self.offsets.clone();
        let idx: SmallDimVec = std::iter::repeat_n(DIM_ZERO, self.actual_rank).collect();

        Odometer {
            pos,
            idx,
            offsets: self.offsets,
            out_shape: self.out_shape,
            arg_strides: self.strides,
            done: false,
        }
    }
}

macro_rules! odometer {
    ([$([$($args:expr),*]),*], $skip:expr) => {
        {
            let metas = [$(TensorMeta::new(&[$($args),*])),*];
            Odometer::new(metas.iter(), $skip)
        }
    }
}

#[cfg(test)]
mod tests {
    use smallvec::{smallvec, SmallVec};
    use crate::consts::*;
    use crate::tensor::*;
    use super::Odometer;

    #[test]
    fn test_example_use() {
        let shapes: [SmallDimVec; _] = [smallvec![2, 3, 4], smallvec![2, 1, 4], smallvec![4], smallvec![]]; // get your shapes
        let shared_tensors: [SharedTensor<i32>; _] = shapes.map(|s| Tensor::new(s, 3)); // get your tensors,
        let borrowed_tensors = shared_tensors.each_ref().map(|t| t.borrow()); // borrow at runtime
    }

    #[test]
    fn test_empty() {
        assert!(odometer!([], 0).is_none());
        assert!(odometer!([], 1).is_none());
        assert!(odometer!([], 2).is_none());
    }

    #[test]
    fn test_0_numel() {
        assert!(odometer!([[0]], 0).is_none());
        assert!(odometer!([[0]], 1).is_none());
        assert!(odometer!([[3, 4, 5]], 0).is_some());
        assert!(odometer!([[3, 4, 0]], 0).is_none());
        assert!(odometer!([[3, 4, 5], [0]], 0).is_none());
        assert!(odometer!([[3, 4, 5], [3, 0, 1]], 0).is_none());
        assert!(odometer!([[3, 1, 5], [3, 0, 1]], 0).is_none());
    }

    #[test]
    fn test_skip_dims() {
        assert!(odometer!([[1]], 0).is_some());
        assert!(odometer!([[1]], 1).is_some());
        assert!(odometer!([[1]], 2).is_none());

        assert!(odometer!([[2, 2, 2], []], 2).is_some());
        assert!(odometer!([[2, 2, 2], [2]], 3).is_some());
        assert!(odometer!([[2, 2, 2], []], 4).is_none());
        assert!(odometer!([[2, 2, 2], [1, 2], [2], []], 3).is_some());
        assert!(odometer!([[2, 2, 2], [1, 2], [2], []], 4).is_none());
    }

    #[test]
    fn test() {
        {
            let odo = odometer!([[2, 3, 4, 5]], 0).unwrap();
            assert_eq!(odo.out_shape.len(), 1);
            assert_eq!(odo.out_shape[0], 120);
            assert_eq!(odo.arg_strides[0][0], 1);
        }
        {
            let odo = odometer!([[2, 2], [2], [2, 1]], 0).unwrap();
            assert_eq!(odo.out_shape.len(), 2);
            assert_eq!(odo.out_shape[0], 2); assert_eq!(odo.out_shape[1], 2);
            assert_eq!(odo.arg_strides[0][0], 1); assert_eq!(odo.arg_strides[0][1], 2);
            assert_eq!(odo.arg_strides[1][0], 1); assert_eq!(odo.arg_strides[1][1], 0);
            assert_eq!(odo.arg_strides[2][0], 0); assert_eq!(odo.arg_strides[2][1], 1);
        }
    }

    #[test]
    fn test_iter() {
        let mut odo = odometer!([[2, 2], [1, 2], [2, 1]], 0).unwrap();
        assert_eq!(odo.pos[0], 0); assert_eq!(odo.pos[1], 0); assert_eq!(odo.pos[2], 0); assert_eq!(odo.done, false); odo.advance();
        assert_eq!(odo.pos[0], 1); assert_eq!(odo.pos[1], 1); assert_eq!(odo.pos[2], 0); assert_eq!(odo.done, false); odo.advance();
        assert_eq!(odo.pos[0], 2); assert_eq!(odo.pos[1], 0); assert_eq!(odo.pos[2], 1); assert_eq!(odo.done, false); odo.advance();
        assert_eq!(odo.pos[0], 3); assert_eq!(odo.pos[1], 1); assert_eq!(odo.pos[2], 1); assert_eq!(odo.done, false); odo.advance();
        assert_eq!(odo.done, true);
        odo.reset();
        assert_eq!(odo.pos[0], 0); assert_eq!(odo.pos[1], 0); assert_eq!(odo.pos[2], 0); assert_eq!(odo.done, false);
    }
}
