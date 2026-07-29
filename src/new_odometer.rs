use std::mem::swap;
use crate::consts::*;
use crate::tensor::*;
use smallvec::{SmallVec, smallvec};
use crate::broadcast::compute_broadcast_shape_iter;

const SMALL_SIZE: usize = 5;

pub struct Odometer {
    pub pos: SmallSizeVec, // len = COUNT
    pub(crate) idx: SmallDimVec, // len = max_rank - skip_rank
    pub(crate) out_shape: SmallDimVec,
    pub(crate) arg_strides: SmallVec<[SmallSizeVec; SMALL_SIZE]>, // len = COUNT, len(args._) = max_rank - skip_ranks
    pub(crate) skip_ranks: usize,
}

fn push_0_right() {

}

impl Odometer {
    pub fn new(args: &[TensorMeta], skip_ranks: usize) -> Option<Self> {
        let args_len = args.len();
        if args_len == 0 { return None; }

        let max_rank = args.iter().map(|m| m.shape.len()).max().unwrap();
        if skip_ranks > max_rank {  return None; }

        if args.iter().any(|m| m.total_elems == 0) { return None; }

        let avb_ranks = max_rank - skip_ranks;
        // Prepare custom shape and strides
        // // Copy shapes and strides into new space while padding with 0 strides and 1 shapes
        let mut new_shapes: SmallVec<[SmallDimVec; SMALL_SIZE]> = args.iter()
            .map(|m| {
                let avb_loc_ranks =  m.shape.len().saturating_sub(skip_ranks);
                std::iter::repeat(0)
                    .take(avb_ranks.saturating_sub(avb_loc_ranks))
                    .chain(m.shape[..avb_loc_ranks].iter().cloned())
                    .collect()
            })
            .collect();

        let mut new_strides: SmallVec<[SmallSizeVec; SMALL_SIZE]> = args.iter()
            .map(|m| {
                let avb_loc_ranks = m.stride.len().saturating_sub(skip_ranks);
                std::iter::repeat(0)
                    .take(avb_ranks.saturating_sub(avb_loc_ranks)) // pad with 0s
                    .chain(m.stride[..avb_loc_ranks].iter().cloned()) // fill remaining, except skipped
                    .collect()
            })
            .collect();

        // // Sort based on strides of arg[0]
        // // // push 0 strides to left using two-pointer approach
        // // // // relative order breaks due to swapping but its fine here because we are sorting anyway later
        let mut j = 0;
        for i in 0..max_rank {
            if new_strides[0][i] == SIZE_ZERO {
                new_shapes.iter_mut().for_each(|s| swap(&mut s[i], &mut s[j]));
                new_strides.iter_mut().for_each(|s| swap(&mut s[i], &mut s[j]));
                j += 1;
            }
        }
        // // // sort the rest using bubble sort
        // // // j is the index of first non-zero arg[0]'s stride, sort in descending after that
        for k in 0..(max_rank - j - 1) {
            for i in j..(max_rank - k - 1) {
                if new_shapes[0][i] < new_shapes[0][i + 1] {
                    new_shapes.iter_mut().for_each(|s| swap(&mut s[i], &mut s[i + 1]));
                    new_strides.iter_mut().for_each(|s| swap(&mut s[i], &mut s[i + 1]));
                }
            }
        }

        // // Compute broadcasted shape (out_shape), skip if broadcast fail
        let mut out_shape = compute_broadcast_shape_iter(new_shapes.iter().map(|s| s.as_ref()), skip_ranks)?;

        // // Eliminate dimension with all 1s
        // // Relative order is important here so 1s must be 'inserted' to right
        j = 0;
        for i in 0..max_rank {
            if out_shape[i] == 1 {
                new_shapes.iter_mut().for_each(|s| swap(&mut s[i], &mut s[j]));
                new_strides.iter_mut().for_each(|s| swap(&mut s[i], &mut s[j]));
                j += 1;
            }
        }
        // // resize to fit
        new_shapes.iter_mut().for_each(|s| s.resize(j, DIM_ZERO));
        new_strides.iter_mut().for_each(|s| s.resize(j, SIZE_ZERO));

        // // Make strides of dimensions 1s - 0
        for i in 0..args_len {
            for j in 0..max_rank {
                if new_shapes[i][j] == 1 { new_strides[i][j] = SIZE_ZERO }
            }
        }

        // // Try dimension collapse
        // // (if all tensors agree that stride[i] == stride[i + 1] * shape[i + 1])
        if max_rank > 1 {
            for i in (0..(max_rank - 1)).rev() {
                let all_agree = !new_shapes.iter().rev()
                    .zip(new_strides.iter().rev())
                    .any(|(sh, st)| {
                        st[i] != st[i + 1] * (sh[i + 1] as SizeType)
                    });
                if all_agree {
                    new_shapes.iter_mut().for_each(|s| s[i] *= s[i + 1]);
                    new_strides.iter_mut().for_each(|s| s[i] = s[i + 1]);
                    out_shape[i + 1] = 0;
                }
            }
        }

        // push 0s in out_shape to right for final resize
        j = ;
        for i in 0..max_rank {
            if new_shapes[0][i] == DIM_ZERO {
                new_shapes.iter_mut().for_each(|s| swap(&mut s[i], &mut s[j]));
                new_strides.iter_mut().for_each(|s| swap(&mut s[i], &mut s[j]));
                j += 1;
            }
        }

        // Prepare pos (all entries 0) and idx (entries start as offsets)
        // Pack in odometer
        ;
    }
}

impl Iterator for Odometer {
    type Item = SmallSizeVec;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: impl new iterator using refined system
        None
    }
}