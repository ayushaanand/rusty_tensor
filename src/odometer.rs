use crate::consts::*;
use crate::tensor::TensorMeta;
use std::array;

/// Multi-tensor iterator that walks through a multi-dimensional index space
/// and yields flat memory positions for each tensor argument at every step.
///
/// Operates on [`TensorMeta`] references rather than full tensors, making it
/// completely agnostic to the underlying element type `T`. This allows iteration
/// over tensors of different data types in the same Odometer.
///
/// # Convention: `args[0]` is always the output tensor's metadata
/// The Odometer reads `shape` from `args[0]` to determine iteration bounds.
/// All remaining `args[1..]` are input tensors. The caller must ensure `args[0]`
/// has the correct broadcasted output shape before constructing the Odometer,
/// or else, behaviour is UB, but generally modular wrapping may be observed.
/// Use [`compute_broadcast_shape`](crate::broadcast::compute_broadcast_shape)
/// to obtain the correct output shape.
///
/// # Generic Parameters
/// * `'a` - Lifetime of the borrowed [`TensorMeta`] references. All `RefCell`
///   borrow guards must outlive the Odometer
///   (see [`Tensor::get_meta`](crate::tensor::Tensor::get_meta)).
/// * `COUNT` - Number of tensor arguments (output + inputs, known at compile time).
///
/// # Iteration
/// Yields `[`[`SizeType`](SizeType)`; COUNT]` — an array of flat memory positions, one per tensor.
/// - `[0]`: flat index into `args[0]`'s data (the output tensor).
/// - `[1..=COUNT]`: flat indices into the input tensors' data.
///
/// Iteration order is row-major: the rightmost non-skipped dimension increments
/// fastest.
///
/// # Broadcasting
/// Handled automatically via stride. Dimensions where `stride[i] == 0`
/// (broadcast dimensions with `shape[i] == 1`) cause the corresponding `pos`
/// to remain stationary along that axis while the output position advances.
///
/// # `skip_ranks`
/// Controls how many innermost dimensions are excluded from iteration.
/// The Odometer only walks the outer (batch) dimensions; the skipped inner
/// dimensions are left for the caller's inner kernel to handle, for eg-
/// - `0`: iterate all dimensions (for elementwise ops like add, mul).
/// - `2`: skip the two inner matrix dimensions (for batched matmul).
pub struct Odometer<'a, const COUNT: usize> {
    /// `idx.len() = args[0].shape.len() - skip_ranks`
    pub(crate) idx: Vec<DimType>,
    pub(crate) out_shape: &'a Vec<DimType>,
    pub(crate) pos: [SizeType; COUNT],
    pub(crate) args: [&'a TensorMeta; COUNT],
    pub(crate) skip_ranks: usize,
    pub(crate) done: bool,
}

impl<'a, const COUNT: usize> Odometer<'a, COUNT> {
    /// Creates a new Odometer with validation.
    ///
    /// # Arguments
    /// * `args` - Fixed-size array of [`&TensorMeta`](TensorMeta) references.
    ///   `args[0]` must be the output tensor's metadata whose shape drives
    ///   iteration. `args[1..]` are inputs.
    ///   Obtain via [`Tensor::get_meta()`](crate::tensor::Tensor::get_meta).
    /// * `skip_ranks` - Number of innermost dimensions to exclude from iteration.
    ///
    /// # Returns
    /// * `Some(Odometer)` — if all tensors have rank ≥ `skip_ranks`.
    /// * `None` — if any tensor has fewer dimensions than `skip_ranks`.
    pub fn new(out_shape: &'a Vec<DimType>, args: [&'a TensorMeta; COUNT], skip_ranks: usize) -> Option<Self> {
        if args.iter().any(|&m| m.shape.len() < skip_ranks) {
            None
        } else {
            Some(Self::new_unsafe(out_shape, args, skip_ranks))
        }
    }

    /// Creates a new Odometer without validation.
    ///
    /// Same arguments and conventions as [`Odometer::new`], but no rank
    /// validation is performed. Intended for internal use by math functions
    /// that have already validated their arguments.
    ///
    /// # Returns
    /// A new `Odometer` with `idx` zeroed to length
    /// `args[0].shape.len() - skip_ranks`, `pos` initialized to each tensor's
    /// `offset`, and `done` set to `false`.
    pub(crate) fn new_unsafe(out_shape: &'a Vec<DimType>, args: [&'a TensorMeta; COUNT], skip_ranks: usize) -> Self {
        let out_rank = out_shape.len();
        Odometer {
            out_shape,
            idx: vec![0; out_rank - skip_ranks],
            skip_ranks,
            pos: array::from_fn(|i| args[i].offset),
            args,
            done: false,
        }
    }

    /// Resets the Odometer to its initial state for re-iteration.
    pub fn reset(&mut self) {
        self.done = false;
        self.idx
            .iter_mut()
            .for_each(|i| *i = 0);
        self.pos
            .iter_mut()
            .zip(self.args.iter())
            .for_each(|(p, m)| {*p = m.offset});
    }
}

impl<'a, const COUNT: usize> Iterator for Odometer<'a, COUNT> {
    type Item = [SizeType; COUNT];

    /// Yields the current flat memory positions and advances the Odometer.
    ///
    /// On each call, captures the current `pos` array, then increments the
    /// multi-dimensional index starting at the fastest-varying (rightmost
    /// non-skipped) dimension, propagating carry towards the outermost
    /// dimension on overflow.
    ///
    /// # Returns
    /// * `Some([SizeType; COUNT])` — flat memory positions for each tensor at
    ///   the current multi-dimensional index.
    /// * `None` — when all positions in the output shape (excluding
    ///   `skip_ranks` inner dims) have been exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let res = self.pos.clone();

        let out_rank = self.out_shape.len();
        if out_rank == 0 {self.done = true;}

        for curr_depth in self.skip_ranks..out_rank {
            let d = out_rank - curr_depth - 1;
            let mut is_overflow = true;

            if self.out_shape[d] != 0 {
                self.idx[d] += 1;
                if self.idx[d] != self.out_shape[d] { is_overflow = false }

                self.pos
                    .iter_mut()
                    .zip(self.args.iter())
                    .for_each(|(p, m)| {
                        if curr_depth < m.shape.len() {
                            let i = m.shape.len() - curr_depth - 1;
                            if is_overflow {
                                *p -= (m.shape[i] as SizeType - 1) * m.stride[i]
                            } else {
                                *p += m.stride[i]
                            }
                        }
                    });
            }

            if !is_overflow {
                break;
            } else if curr_depth == out_rank - 1 {
                self.done = true;
                break;
            } else {
                self.idx[d] = DIM_ZERO;
            };
        }

        Some(res)
    }
}
