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
/// has the correct broadcasted output shape before constructing the Odometer.
///
/// # Generic Parameters
/// * `'a` - Lifetime of the borrowed `TensorMeta` references. All borrow guards
///   must outlive the Odometer (see Borrow Guidelines below).
/// * `COUNT` - Number of tensor arguments (known at compile time).
///
/// # Iteration
/// Yields `[SizeType; COUNT]` — an array of flat memory positions, one per tensor.
/// - `pos[0]`: flat index into `args[0]`'s data (the output tensor).
/// - `pos[1..N]`: flat indices into the input tensors' data.
///
/// Iteration order is row-major: the rightmost non-skipped dimension increments fastest.
///
/// Broadcasting is handled automatically via stride. Dimensions where `stride == 0`
/// (broadcast dimensions with `shape[i] == 1`) cause the corresponding `pos` to
/// remain stationary along that axis.
///
/// # Borrow Guidelines
/// Because tensors are wrapped in `SharedTensor<T>` (`Rc<RefCell<...>>`), you must
/// bind borrow guards to `let` variables before constructing the Odometer. The guards
/// must live at least as long as the Odometer.
///
/// ```rust
/// // ✓ Correct: guards are bound to variables that outlive the Odometer
/// let t1_borrow = t1.borrow();
/// let t2_borrow = t2.borrow();
/// let odo = Odometer::new([t1_borrow.get_meta(), t2_borrow.get_meta()], 0);
///
/// // ✗ Wrong: temporary guards are dropped immediately
/// let odo = Odometer::new([t1.borrow().get_meta(), t2.borrow().get_meta()], 0);
/// ```
pub struct Odometer<'a, const COUNT: usize> {
    pub(crate) idx: Vec<DimType>,
    pub(crate) pos: [SizeType; COUNT],
    pub(crate) args: [&'a TensorMeta; COUNT],
    pub(crate) skip_ranks: usize,
    pub(crate) done: bool,
}

impl<'a, const COUNT: usize> Odometer<'a, COUNT> {
    /// Creates a new Odometer with validation.
    ///
    /// # Arguments
    /// * `args` - Fixed-size array of `&TensorMeta` references. `args[0]` must be the
    ///   output tensor's metadata whose shape drives iteration. `args[1..]` are inputs.
    ///   Obtain these via [`Tensor::get_meta()`](crate::tensor::Tensor::get_meta).
    /// * `skip_ranks` - Number of innermost dimensions to exclude from iteration.
    ///   Use `0` for elementwise ops (e.g., add, mul).
    ///   Use `2` to skip the inner matrix dimensions for batched matmul.
    ///
    /// # Returns
    /// * `Some(Odometer)` — if all tensors have rank ≥ `skip_ranks`.
    /// * `None` — if any tensor has fewer dimensions than `skip_ranks`.
    pub fn new(args: [&'a TensorMeta; COUNT], skip_ranks: usize) -> Option<Self> {
        if args.iter().any(|&m| m.shape.len() < skip_ranks) {
            None
        } else {
            Some(Self::new_unsafe(args, skip_ranks))
        }
    }

    /// Creates a new Odometer without validation.
    ///
    /// # Arguments
    /// * `args` - Same convention as [`Odometer::new`]. No rank validation is performed.
    /// * `skip_ranks` - Same as [`Odometer::new`].
    ///
    /// # Returns
    /// A new `Odometer` with `idx` zeroed, `pos` initialized to each tensor's `offset`,
    /// and `done` set to `false`.
    ///
    /// Intended for internal use by math functions that have already validated arguments.
    pub(crate) fn new_unsafe(args: [&'a TensorMeta; COUNT], skip_ranks: usize) -> Self {
        Odometer {
            idx: vec![0; args[0].shape.len() - skip_ranks],
            skip_ranks,
            pos: array::from_fn(|i| args[i].offset),
            args,
            done: false,
        }
    }

    /// Resets the Odometer to its initial state for re-iteration.
    ///
    /// Zeroes all indices, resets each `pos` to its tensor's `offset`,
    /// and clears the `done` flag.
    pub fn reset(&mut self) {
        self.done = false;
        self.idx.iter_mut().for_each(|i| *i = 0);
        self.pos
            .iter_mut()
            .enumerate()
            .for_each(|(i, p)| *p = self.args[i].offset);
    }
}

impl<'a, const COUNT: usize> Iterator for Odometer<'a, COUNT> {
    type Item = [SizeType; COUNT];

    /// Yields the current flat memory positions and advances the Odometer.
    ///
    /// # Returns
    /// * `Some([SizeType; COUNT])` — flat memory positions for each tensor at the
    ///   current multi-dimensional index.
    /// * `None` — when all positions in the output shape have been exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let res = self.pos;
        let out_m = self.args[0];
        let out_rank = out_m.shape.len();

        for curr_depth in self.skip_ranks..out_rank {
            let d = out_rank - curr_depth;
            self.idx[d] += 1;

            let is_overflow = self.idx[d] == out_m.shape[d];

            self.pos
                .iter_mut()
                .zip(self.args.iter())
                .for_each(|(p, m)| {
                    if curr_depth <= m.shape.len() {
                        let i = m.shape.len() - curr_depth;
                        if is_overflow {
                            *p -= (m.shape[i] as SizeType - 1) * m.stride[i]
                        } else {
                            *p += m.stride[i]
                        }
                    }
                });

            if !is_overflow {
                break;
            } else if curr_depth == out_rank {
                self.done = true;
                break;
            } else {
                self.idx[d] = DIM_ZERO;
            };
        }

        Some(res)
    }
}
