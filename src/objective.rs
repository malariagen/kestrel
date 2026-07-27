use crate::util::{Matrix9xN, Vector9, dot, sum};

pub fn compute_obj(p_mat_t: &Matrix9xN<f64>, x: &Vector9<f64>, eps: f64) -> f64 {
    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    // TODO change this later
    let (p, _) = p_mat_t.as_slice().as_chunks::<9>();

    return compute_obj_scalar(p, &x0, eps);
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
