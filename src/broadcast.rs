use crate::consts::*;
use smallvec::smallvec;

/// Computes the broadcasted output shape from an array of shape slices.
///
/// Follows NumPy broadcasting rules: dimensions are aligned from the right,
/// and at each position the output is the non-`1` value if all inputs agree,
/// or `1` if every input has `1` (or is absent).
///
/// `skip_dims` excludes the innermost `skip_dims` dimensions from broadcast
/// validation, leaving them as `0` in the output for the caller to fill
/// (e.g., the `M` and `N` result dims in batched matmul). Use `0` for
/// elementwise ops.
///
/// Returns `None` if `COUNT == 0`, `max_rank < skip_dims`, or any two inputs
/// have incompatible non-unit dimensions.
pub fn compute_broadcast_shape_const<const COUNT: usize>(
    args: [&[DimType]; COUNT],
    skip_dims: usize,
) -> Option<SmallDimVec> {
    if COUNT == 0 {
        return None;
    };

    let max_rank = args.iter().map(|&m| m.len()).max().unwrap_or(0);
    if max_rank < skip_dims {
        return None;
    }
    let mut res: SmallDimVec = smallvec![1; max_rank];

    // make rev iters on valid slices of args
    let mut iters: [_; COUNT] = args.map(|m| {
        let avb_rank = m.len().saturating_sub(skip_dims);
        m[..avb_rank].iter().rev()
    });

    // make rev iter on valid slice of res
    let avb_res_rank = max_rank.saturating_sub(skip_dims);
    let res_iter = res[..avb_res_rank].iter_mut().rev();

    for res_dim in res_iter {
        for it in iters.iter_mut() {
            match it.next() {
                None | Some(1) => {}
                Some(&x) => {
                    if *res_dim == 1 {
                        *res_dim = x;
                    } else if *res_dim == x {
                    } else {
                        return None;
                    }
                }
            }
        }
    }

    Some(res)
}

/// Iterator-based variant of [`compute_broadcast_shape_const`].
///
/// Accepts any cloneable iterator of `&[DimType]` slices, suitable when the
/// number of inputs is not known at compile time. Used internally by
/// [`TensorIter`](crate::iterator::TensorIter). See [`compute_broadcast_shape_const`]
/// for full semantics and the `skip_dims` convention.
pub fn compute_broadcast_shape_iter<'a, I>(args_iter: I, skip_dims: usize) -> Option<SmallDimVec>
where
    I: Iterator<Item = &'a [DimType]> + Clone,
{
    if args_iter.clone().next() == None {
        return None;
    };

    let max_rank = args_iter.clone().map(|a| a.len()).max().unwrap_or(0);
    if max_rank < skip_dims {
        return None;
    }
    let mut res: SmallDimVec = smallvec![1; max_rank];

    let avb_rank = max_rank.saturating_sub(skip_dims);
    let res_it = res[..avb_rank].iter_mut().rev();
    let mut curr_depth = skip_dims;

    for curr_dim in res_it {
        for arg in args_iter.clone() {
            if arg.len() > curr_depth {
                let arg_dim = arg[arg.len() - curr_depth - 1];

                if arg_dim == 1 {
                } else if *curr_dim == 1 {
                    *curr_dim = arg_dim
                } else if *curr_dim == arg_dim {
                } else {
                    return None;
                }
            }
        }
        curr_depth += 1;
    }

    Some(res)
}

/// Slice-based convenience wrapper over [`compute_broadcast_shape_iter`].
///
/// Accepts any `AsRef<[DimType]>` slice (e.g., `&[SmallDimVec]`, `&[Vec<DimType>]`).
/// See [`compute_broadcast_shape_const`] for full semantics and the `skip_dims` convention.
pub fn compute_broadcast_shape<S: AsRef<[DimType]>>(
    args: &[S],
    skip_dims: usize,
) -> Option<SmallDimVec> {
    compute_broadcast_shape_iter(args.iter().map(|s| s.as_ref()), skip_dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::make_args;
    use smallvec::{smallvec};

    fn test_fn<const N: usize>(
        args: [SmallDimVec; N],
        skip_dims: usize,
        res: Option<SmallDimVec>
    ) {
        let const_args = args.each_ref().map(|f| f.as_slice());
        assert_eq!(compute_broadcast_shape_const(const_args, skip_dims), res);
        assert_eq!(compute_broadcast_shape(&const_args, skip_dims), res);
        assert_eq!(compute_broadcast_shape_iter(args.iter().map(|s| s.as_slice()), skip_dims), res);
    }
    
    macro_rules! test {
        ([$($args:tt),*], $skip:expr, $($res:tt)*) => {
            {
                let args = make_args!($($args),*);
                test_fn(args, $skip, $($res)*);
            }
        };
    }

    macro_rules! test_succ {
        ([$($args:tt),*], $skip:expr, $res:tt) => {
            test!([$($args),*], $skip, Some(smallvec!$res));
        }
    }

    macro_rules! test_fail {
        ([$($args:tt),*], $skip:expr) => {
            test!([$($args),*], $skip, None);
        }
    }

    #[test]
    fn test_scalar() {
        test_fail!([], 0);
        test_fail!([], 1);

        test_succ!([[]], 0, []);
        test_succ!([[], []], 0, []);
        test_fail!([[]], 1);
    }

    #[test]
    fn test_broadcast() {
        test_succ!([[0]], 0, [0]);
        test_succ!([[0]], 1, [1]);
        test_succ!([[1]], 1, [1]);
        test_fail!([[2]], 2);
        test_succ!([[1], [2]], 0, [2]);
        test_succ!([[2], [2]], 0, [2]);
        test_fail!([[3], [2]], 0);
        test_succ!([[3], [2]], 1, [1]);

        test_succ!([[], [1]], 0, [1]);
        test_succ!([[], [2]], 0, [2]);

        test_succ!([[], [2, 2]], 0, [2, 2]);
        test_succ!([[], [2, 2]], 1, [2, 1]);
        test_succ!([[], [2, 2]], 2, [1, 1]);
        test_fail!([[], [2, 2]], 3);

        test_succ!([[2, 1], [1, 2]], 0, [2, 2]);
        test_succ!([[2, 1], [1, 2]], 1, [2, 1]);
        test_succ!([[2, 1], [1, 2]], 2, [1, 1]);
        test_fail!([[2, 1], [1, 2]], 3);
    }
}

#[cfg(feature = "python-testing")]
#[pyo3::pymodule(name = "broadcast")]
pub(crate) mod py_tests {
    use super::compute_broadcast_shape;
    use crate::consts::DimType;
    use pyo3::prelude::*;
    use smallvec::smallvec;

    #[pyfunction(name = "compute_broadcast_shape")]
    fn py_test_compute_broadcast_shape(args: Vec<Vec<DimType>>) -> Vec<DimType> {
        if args.len() == 0 {
            return vec![];
        }
        let res = compute_broadcast_shape(&args, 0).unwrap_or(smallvec![]);
        res.to_vec()
    }
}
