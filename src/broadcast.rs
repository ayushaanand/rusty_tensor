use crate::consts::*;
use crate::tensor::*;

pub fn compute_broadcast_shape<const COUNT: usize>(args: [&TensorMeta; COUNT], skip_ranks: usize) -> Option<Vec<DimType>> {
    if COUNT == 0 { return None; };

    let max_rank = args.iter().map(|&m| m.shape.len()).max().unwrap_or(0);
    if max_rank < skip_ranks { return None; }
    let mut res: Vec<DimType> = vec![1; max_rank];

    // make rev iters on valid slices of arg's shapes
    let mut iters: [_; COUNT] = args.map(|m| {
        let avb_ranks = m.shape.len().saturating_sub(skip_ranks);
        m.shape[..avb_ranks].iter().rev()
    });

    // make rev iter on valid slice of res
    let avb_res_rank = max_rank.saturating_sub(skip_ranks);
    let mut res_iter = res[..avb_res_rank].iter_mut().rev();

    for res_dim in res_iter {
        for it in iters.iter_mut() {
            match it.next() {
                None => {},
                Some(1) => {},
                Some(&x) =>
                    if *res_dim == 1 { *res_dim = x; }
                    else if *res_dim == x {}
                    else { return None; }
            }
        }
    }

    Some(res)
}
