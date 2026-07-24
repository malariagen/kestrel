use crate::util::{Matrix9xN, Vector9, dot};

pub fn compute_grad(
    p_mat_t: &Matrix9xN<f64>,
    x: &Vector9<f64>,
    eps: f64,
) -> Vector9<f64> {

    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    // TODO change this later
    let (p, _) = p_mat_t.as_slice().as_chunks::<9>();

    let ptd = compute_pt_d_scalar(&p, &x0, eps);

    let n = p.len();
    // g = 1.0 - (P^T d) / L
    let mut g = Vector9::<f64>::zeros();
    for i in 0..9 {
        g[i] = 1.0 - ptd[i] / (n as f64);
    }

    g
}

pub fn compute_pt_d_scalar(
    p_mat: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [f64; 9] {

    let mut g = [0.0; 9];

    for row in p_mat.iter() {
        let prod = dot(row, x);

        // d = 1 / (P x + eps)
        let d = 1.0 / (prod + eps);

        // g = P^T d
        for i in 0..9 {
            g[i] = row[i].mul_add(d, g[i]);
        }
    }

    g
}
