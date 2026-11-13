use crate::broadcast::compute_broadcast_shape_iter;
use crate::consts::*;
use crate::error::*;
use crate::tensor::{Tensor, TensorMeta};
use crate::types::SmallSlice;
use std::cmp::Reverse;

/// A pre-optimized, multi-tensor flat-index iterator.
///
/// Simultaneously walks `N` tensors that share a compatible broadcast shape,
/// yielding a flat buffer index per tensor at each logical position.
/// Internally, dimensions are coalesced and sorted for cache locality, so the
/// effective iteration rank may be less than the original tensor rank.
///
/// # Hot-loop pattern
/// ```
/// # use tensor_lib::tensor::TensorMeta;
/// # use tensor_lib::iterator::TensorIter;
/// # let a = TensorMeta::new(&[2, 3]); let b = TensorMeta::new(&[1, 3]);
/// let mut iter = TensorIter::new([&a, &b], 0).unwrap();
/// while !iter.done {
///     let (idx_a, idx_b) = (iter.pos[0], iter.pos[1]);
///     // data_a[idx_a] op= data_b[idx_b] ...
///     iter.advance();
/// }
/// ```
///
/// Alternatively use the [`Iterator`](Iterator) impl for adapter chains; note it clones
/// `pos` on every call, so prefer the `while !done` form in hot paths.
///
/// # `skip_dims`
/// Excludes the `skip_dims` innermost dimensions from broadcasting and
/// iteration. The caller handles those dimensions separately (e.g., the `M×N`
/// contraction in a batched matmul). Use `0` for all elementwise ops.
#[derive(Debug, Clone)]
pub struct TensorIter {
    /// Current flat buffer index for each argument tensor.
    /// Valid to read between construction and when `done` becomes `true`.
    pub pos: SmallSlice<SizeType>,
    pub(crate) idx: SmallSlice<DimType>,
    pub(crate) offsets: SmallSlice<SizeType>,
    pub(crate) out_shape: SmallSlice<DimType>,
    pub(crate) arg_strides: SmallSlice<SmallSlice<SizeType>>,
    pub done: bool,
}

// 1. args.is_empty() -> return None
// 2. any(arg.total_elems == SIZE_ZERO) -> return None
// TODO: eliminate 2 to allow zero elem tensors too
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

trait MetaIter<'a>: Iterator<Item = &'a TensorMeta> + Clone {}

impl<'a, T> MetaIter<'a> for T where T: Iterator<Item = &'a TensorMeta> + Clone {}

impl TensorIter {
    /// Constructs a `TensorIter` for the given tensor arguments.
    ///
    /// Runs a one-time build pipeline: pads shapes for broadcasting, eliminates
    /// trivial dimensions, sorts and coalesces dimensions for cache efficiency.
    ///
    /// Returns `None` if:
    /// - `args` is empty, or any argument has `total_elems == 0`.
    /// - `skip_dims` exceeds the maximum rank across all arguments.
    /// - any two arguments have incompatible non-unit dimensions (see
    ///   [`compute_broadcast_shape`](crate::broadcast::compute_broadcast_shape)).
    ///
    /// # Example
    /// ```
    /// # use tensor_lib::tensor::TensorMeta;
    /// # use tensor_lib::iterator::TensorIter;
    /// let a = TensorMeta::new(&[2, 3]);
    /// let b = TensorMeta::new(&[1, 3]); // broadcasts against a
    /// assert!(TensorIter::new([&a, &b], 0).is_ok());
    /// assert!(TensorIter::new([&a, &b], 3).is_err()); // skip_dims > rank
    ///
    /// let c = TensorMeta::new(&[4, 3]); // incompatible with a on dim 0
    /// assert!(TensorIter::new([a, c].each_ref(), 0).is_err());
    /// ```
    pub fn new<'a, I>(args: I, skip_dims: usize) -> TensorResult<Self>
    where
        I: IntoIterator<Item = &'a TensorMeta>,
        I::IntoIter: Clone,
    {
        let it = args.into_iter();
        // 1, 2, 3
        IterBuilder::is_err(it.clone(), skip_dims)?;

        // 4, 5
        let builder = IterBuilder::new(it, skip_dims)?;

        Ok(builder
            // 6. eliminate trivial dims: keep only indices d where broadcasted_shape[d] != 1
            .part(1)
            // 7. if arg.shape[i] == 1, arg.stride[i] = 0
            .zero_broadcasted_dims()
            // 8. compute permutation from arg[0].stride (assuming arg[0] is hotter): stride == 0s first, then descending by stride value
            .sort_by_stride_of::<0>()
            // 10. coalesce adjacent dims d, d+1 (after permutation): merge iff for every arg:
            //      stride[d] == stride[d+1] * shape[d+1]
            //      on merge: shape[d+1] *= shape[d], out_shape[d] = 0, drop d
            .coalesce_dims()
            // 11. build final out_shape and arg_strides, remove extra dims
            .shrink_to_fit()
            // 12. build rest fields
            .build())
    }

    pub fn new_logical<'a, I>(args: I, skip_dims: usize) -> TensorResult<Self>
    where
        I: IntoIterator<Item = &'a TensorMeta>,
        I::IntoIter: Clone,
    {
        let it = args.into_iter();
        IterBuilder::is_err(it.clone(), skip_dims)?;
        let builder = IterBuilder::new(it, skip_dims)?;
        Ok(builder
            .part(1)
            .zero_broadcasted_dims()
            // sorting omitted to retain logical ordering
            // .sort_by_stride_of::<0>();
            .coalesce_dims()
            .shrink_to_fit()
            .build())
    }

    pub fn from_tensors<'a, I, T>(args: I, skip_dims: usize) -> TensorResult<Self>
    where
        I: IntoIterator<Item = &'a Tensor<T>>,
        T: TensorDType + 'a,
        I::IntoIter: Clone,
    {
        Self::new(args.into_iter().map(|a| &a.meta), skip_dims)
    }

    /// Resets the iterator to its initial state.
    ///
    /// Restores all positions to their starting offsets and clears `done`.
    /// Useful when the same operation executes repeatedly (e.g., per training epoch).
    ///
    /// # Example
    /// ```
    /// # use tensor_lib::tensor::TensorMeta;
    /// # use tensor_lib::iterator::TensorIter;
    /// let meta = TensorMeta::new(&[4]);
    /// let mut iter = TensorIter::new([&meta], 0).unwrap();
    /// while !iter.done { iter.advance(); }
    /// iter.reset();
    /// assert!(!iter.done);
    /// assert_eq!(iter.pos[0], 0);
    /// ```
    pub fn reset(&mut self) {
        self.pos = self.offsets.clone();
        self.idx.iter_mut().for_each(|i| *i = 0);
        self.done = false;
    }

    /// Advances to the next logical position, updating all entries in `pos`.
    ///
    /// Sets `done` when the last element has been passed.
    /// If sure about `done` being false, use [`advance_unchecked`](TensorIter::advance_unchecked)
    ///
    /// # Example
    /// ```
    /// # use tensor_lib::tensor::TensorMeta;
    /// # use tensor_lib::iterator::TensorIter;
    /// let meta = TensorMeta::new(&[3]);
    /// let mut iter = TensorIter::new([&meta], 0).unwrap();
    /// iter.advance(); iter.advance(); iter.advance();
    /// assert!(iter.done);
    /// ```
    #[inline(always)]
    pub fn advance(&mut self) {
        if !self.done {
            self.advance_unchecked();
        }
    }

    // TODO: suppress some dimensions if not used by caller
    /// Advances to the next logical position, but doesn't check `done` flag but sets it correctly.
    #[inline(always)]
    pub fn advance_unchecked(&mut self) {
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

impl TensorIter {
    pub fn for_each(&mut self, mut f: impl FnMut(&[SizeType])) {
        self.reset();
        while !self.done {
            f(self.pos.as_slice());
            self.advance_unchecked();
        }
    }
}

struct IterBuilder {
    out_shape: SmallSlice<DimType>,
    shapes: SmallSlice<SmallSlice<DimType>>,
    strides: SmallSlice<SmallSlice<SizeType>>,
    offsets: SmallSlice<SizeType>,
    actual_rank: usize,
}

impl IterBuilder {
    fn is_err<'a>(args: impl MetaIter<'a>, skip_dims: usize) -> Result<(), TensorError> {
        // 1. args.is_empty() -> return None
        let args_len = args.clone().count();
        if args_len == 0 {
            return Err(UnexEmptyInputError);
        }

        // 2. any(arg.total_elems == SIZE_ZERO) -> return None
        if args.clone().any(|m| m.total_elems == 0) {
            return Err(UnexZeroElemError);
        }

        // 3. compute max_rank, if skip_dims > max_rank -> return None
        let max_rank = args.clone().map(|m| m.shape.len()).max().unwrap(); // unwrap is fine because args_len != 0
        if skip_dims > max_rank {
            return Err(SkipDimError);
        }

        Ok(())
    }

    fn new<'a>(args: impl MetaIter<'a>, skip_dims: usize) -> TensorResult<Self> {
        // args.len > 0 guaranteed from is_err
        let max_rank = args.clone().map(|m| m.shape.len()).max().unwrap();

        // 4. copy every arg's (shape, stride) to vecs of max_rank: prepend shape with 1, stride with 0, on the left
        let avb_dims = max_rank - skip_dims; // no underflow because of 3

        // shape/stride[i].len = avb_dims = max_rank - skip_dims
        let shapes: SmallSlice<SmallSlice<_>> = args
            .clone()
            .map(|m| Self::make_padded(&m.shape, 1 as DimType, avb_dims, skip_dims))
            .collect();
        let strides = args
            .clone()
            .map(|m| Self::make_padded(&m.stride, SIZE_ZERO, avb_dims, skip_dims))
            .collect();

        // 5. compute_broadcast_shape
        let out_shape = compute_broadcast_shape_iter(
            shapes.iter().map(|s| s.as_ref()),
            0, // because we eliminated skip_dims in make_padded
        )?;

        let offsets = args.map(|m| m.offset).collect();

        Ok(Self {
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
        // no underflow because avb_dims = max_rank - skip_dims >= loc_rank - skip_dims
        std::iter::repeat_n(pad_val, avb_dims - avb_loc_dims)
            // take the rest avb_loc_dims, avb_loc_dims + (avb_dims - avb_loc_dims) = avb_dims
            .chain(slice[..avb_loc_dims].iter().cloned())
            .collect()
    }

    fn swap_dims(mut self, i: usize, j: usize) -> Self {
        self.out_shape.swap(i, j);
        self.shapes.iter_mut().for_each(|s| s.swap(i, j));
        self.strides.iter_mut().for_each(|s| s.swap(i, j));
        self
    }

    fn part(mut self, target: DimType) -> Self {
        let mut j = 0;
        for i in 0..self.actual_rank {
            if self.out_shape[i] != target {
                if i != j {
                    self = self.swap_dims(i, j);
                }
                j += 1;
            }
        }
        self.actual_rank = j;
        self
    }

    fn zero_broadcasted_dims(mut self) -> Self {
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

        self
    }

    fn sort_by_stride_of<const N: usize>(mut self) -> Self {
        let mut perm: SmallSlice<_> = (0..self.actual_rank).collect();
        perm.sort_unstable_by_key(|&i| {
            Reverse(if self.strides[N][i] == SIZE_ZERO {
                usize::MAX
            } else {
                self.strides[N][i]
            })
        });

        for i in 0..self.actual_rank {
            while perm[i] != i {
                self = self.swap_dims(i, perm[i]);
                let new_pos = perm[i];
                perm.swap(i, new_pos);
            }
        }

        self
    }

    fn coalesce_dims(mut self) -> Self {
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

        self.part(DIM_ZERO)
    }

    fn shrink_to_fit(mut self) -> Self {
        fn drop_ext_dims<T>(vec: &mut SmallSlice<T>, len: usize) {
            vec.truncate(len);
            vec.shrink_to_fit();
        }

        for i in 0..self.shapes.len() {
            drop_ext_dims(&mut self.shapes[i], self.actual_rank);
            drop_ext_dims(&mut self.strides[i], self.actual_rank);
        }
        drop_ext_dims(&mut self.out_shape, self.actual_rank);
        self
    }

    fn build(mut self) -> TensorIter {
        if self.actual_rank == 0 {
            self.out_shape.push(1);
            self.strides.iter_mut().for_each(|s| s.push(SIZE_ZERO));
            self.actual_rank = 1;
        }

        let pos: SmallSlice<SizeType> = self.offsets.clone();
        let idx: SmallSlice<DimType> = std::iter::repeat_n(DIM_ZERO, self.actual_rank).collect();

        TensorIter {
            pos,
            idx,
            offsets: self.offsets,
            out_shape: self.out_shape,
            arg_strides: self.strides,
            done: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TensorIter;
    use crate::consts::DimType;
    use smallvec::smallvec;
    macro_rules! tensor_iter {
        ([$([$($args:expr),*]),*], $skip:expr) => {
            {
                let metas = [$(TensorMeta::new(&[$($args),*])),*];
                TensorIter::new(metas.iter(), $skip)
            }
        }
    }
    use crate::tensor::*;
    use crate::types::SmallSlice;

    #[test]
    fn test_example_use() {
        let shapes: [SmallSlice<DimType>; _] = [
            smallvec![2, 3, 4],
            smallvec![2, 1, 4],
            smallvec![4],
            smallvec![],
        ]; // get your shapes
        let tensors: [Tensor<i32>; _] = shapes.map(|s| Tensor::new(&s, 3)); // get your tensors,
        let _ = TensorIter::new(tensors.each_ref().map(|t| &t.meta), 0).unwrap();
    }

    #[test]
    fn test_empty() {
        assert!(tensor_iter!([], 0).is_err());
        assert!(tensor_iter!([], 1).is_err());
        assert!(tensor_iter!([], 2).is_err());
    }

    #[test]
    fn test_0_numel() {
        assert!(tensor_iter!([[0]], 0).is_err());
        assert!(tensor_iter!([[0]], 1).is_err());
        assert!(tensor_iter!([[3, 4, 5]], 0).is_ok());
        assert!(tensor_iter!([[3, 4, 0]], 0).is_err());
        assert!(tensor_iter!([[3, 4, 5], [0]], 0).is_err());
        assert!(tensor_iter!([[3, 4, 5], [3, 0, 1]], 0).is_err());
        assert!(tensor_iter!([[3, 1, 5], [3, 0, 1]], 0).is_err());
    }

    #[test]
    fn test_skip_dims() {
        assert!(tensor_iter!([[1]], 0).is_ok());
        assert!(tensor_iter!([[1]], 1).is_ok());
        assert!(tensor_iter!([[1]], 2).is_err());

        assert!(tensor_iter!([[2, 2, 2], []], 2).is_ok());
        assert!(tensor_iter!([[2, 2, 2], [2]], 3).is_ok());
        assert!(tensor_iter!([[2, 2, 2], []], 4).is_err());
        assert!(tensor_iter!([[2, 2, 2], [1, 2], [2], []], 3).is_ok());
        assert!(tensor_iter!([[2, 2, 2], [1, 2], [2], []], 4).is_err());
    }

    #[test]
    fn test() {
        {
            let odo = tensor_iter!([[2, 3, 4, 5]], 0).unwrap();
            assert_eq!(odo.out_shape.len(), 1);
            assert_eq!(odo.out_shape[0], 120);
            assert_eq!(odo.arg_strides[0][0], 1);
        }
        {
            let odo = tensor_iter!([[2, 2], [2], [2, 1]], 0).unwrap();
            assert_eq!(odo.out_shape.as_slice(), &[2, 2]);

            assert_eq!(odo.arg_strides[0].as_slice(), &[2, 1]);
            assert_eq!(odo.arg_strides[1].as_slice(), &[0, 1]);
            assert_eq!(odo.arg_strides[2].as_slice(), &[1, 0]);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn test_iter() {
        let mut odo = tensor_iter!([[2, 2], [1, 2], [2, 1]], 0).unwrap();
        assert_eq!(odo.pos.as_slice(), &[0, 0, 0]); assert_eq!(odo.done, false); odo.advance();
        assert_eq!(odo.pos.as_slice(), &[1, 1, 0]); assert_eq!(odo.done, false); odo.advance();
        assert_eq!(odo.pos.as_slice(), &[2, 0, 1]); assert_eq!(odo.done, false); odo.advance();
        assert_eq!(odo.pos.as_slice(), &[3, 1, 1]); assert_eq!(odo.done, false); odo.advance();
        assert_eq!(odo.done, true);
        odo.reset();

        assert_eq!(odo.pos.as_slice(), &[0, 0, 0]); assert_eq!(odo.done, false);
    }
}
