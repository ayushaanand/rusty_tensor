use crate::consts::*;
use crate::tensor::*;

/// Computes the broadcasted output shape from an array of tensor metadata.
///
/// Follows NumPy-style broadcasting rules: dimensions are aligned from the
/// right (innermost), and at each position the output size is the non-`1`
/// value if all tensors agree, or `1` if every tensor has `1` (or is missing
/// that dimension entirely).
///
/// # Convention: `skip_ranks`
/// When `skip_ranks > 0`, the innermost `skip_ranks` dimensions are excluded
/// from broadcast validation and left as `0` in the output. The caller is
/// responsible for filling those positions with the correct values (e.g., the
/// matmul result dimensions `M` and `N`).
///
/// The same `skip_ranks` value must be passed to
/// [`Odometer::new`](crate::odometer::Odometer::new) so that the Odometer
/// iterates only the outer (batch) dimensions that were broadcast-checked here.
///
/// # Arguments
/// * `args` - Fixed-size array of [`&TensorMeta`](TensorMeta) references.
/// * `skip_ranks` - Number of innermost dimensions to exclude from broadcasting.
///   Use `0` for elementwise ops. Use `2` for batched matmul.
///
/// # Returns
/// * `Some(Vec<DimType>)` — the full output shape of length of maximum rank in args.
///   Positions `[0..max_rank - skip_ranks]` contain the broadcast-resolved
///   batch dimensions. Positions `[max_rank - skip_ranks..]` are left as `0`
///   for the caller to fill.
/// * `None` — if `COUNT == 0`, or if `max_rank < skip_ranks`, or if any two
///   tensors have incompatible dimensions (neither is `1` and they differ).
pub fn compute_broadcast_shape<const COUNT: usize>(args: [&[DimType]; COUNT], skip_ranks: usize) -> Option<Vec<DimType>> {
    if COUNT == 0 { return None; };

    let max_rank = args.iter().map(|&m| m.len()).max().unwrap_or(0);
    if max_rank < skip_ranks { return None; }
    let mut res: Vec<DimType> = vec![1; max_rank];

    // make rev iters on valid slices of args
    let mut iters: [_; COUNT] = args.map(|m| {
        let avb_rank = m.len().saturating_sub(skip_ranks);
        m[..avb_rank].iter().rev()
    });

    // make rev iter on valid slice of res
    let avb_res_rank = max_rank.saturating_sub(skip_ranks);
    let res_iter = res[..avb_res_rank].iter_mut().rev();

    for res_dim in res_iter {
        *res_dim = 1;
        for it in iters.iter_mut() {
            match it.next() {
                None | Some(1) => {},
                Some(&x) =>
                    if *res_dim == 1 { *res_dim = x; }
                    else if *res_dim == x {}
                    else { return None; }
            }
        }
    }

    Some(res)
}
