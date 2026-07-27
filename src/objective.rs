use crate::{matrix::BlockBuffer, util::{Vector9, dot, sum}};

pub fn compute_obj<const L: usize>(p_mat: &BlockBuffer<f64, L, 9>, x: &Vector9<f64>, eps: f64) -> f64 {
    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    let s = compute_obj_block(p_mat, &x0, eps);

    let n = p_mat.num_rows();

    return sum(&x0) - s / (n as f64) - 1.0;
}

fn compute_obj_scalar(p_mat: &[[f64; 9]], x: &[f64; 9], eps: f64) -> f64 {
    let n = p_mat.len();

    // x.sum() - obj / v - 1.0

    let mut s = 0.0;
    for row in p_mat.iter() {
        let prod = dot(row, x);
        let t = (prod + eps).ln();
        // TODO kahan?
        s += t;
    }

    return sum(x) - s / (n as f64) - 1.0;
}

fn compute_obj_block<const L: usize>(p_mat: &BlockBuffer<f64, L, 9>, x: &[f64; 9], eps: f64) -> f64 {
    let (blocks, remainder) = p_mat.as_blocks();

    let mut s = 0.0;
    for block in blocks.iter() {
        // TODO vectorize this bad boy
        for i in 0..L {
            let mut row = [0.0; 9];
            for j in 0..9 {
                row[j] = block[j][i];
            }
            let prod = dot(&row, x);
            let t = (prod + eps).ln();
            // TODO kahan?
            s += t;
        }
    }

    for row in remainder.iter() {
        let prod = dot(&row, x);
        let t = (prod + eps).ln();
        // TODO kahan?
        s += t;
    }
    s
}
