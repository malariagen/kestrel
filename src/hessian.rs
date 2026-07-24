use crate::util::{Matrix9, Matrix9xN, Vector9, dot};

pub fn compute_hess(
    p_mat_t: &Matrix9xN<f64>,
    x: &Vector9<f64>,
    eps: f64,
) -> Matrix9<f64> {

    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    let (p, _) = p_mat_t.as_slice().as_chunks::<9>();

    let ptd2p = compute_pt_d2_p_scalar(p, &x0, eps);

    let n = p.len();

    Matrix9::from_fn(|i, j| ptd2p[i][j] / (n as f64))
}

// P^T D^2 P
pub fn compute_pt_d2_p_scalar(
    p_mat: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {

    let mut h = [[0.0; 9]; 9];

    for row in p_mat.iter() {
        let prod = dot(row, x);

        // d = 1 / (P x + eps)
        let d = 1.0 / (prod + eps);

        let mut scaled_row = [0.0; 9];
        for i in 0..9 {
            scaled_row[i] = d * row[i];
        }

        // g = P^T d
        for i in 0..9 {
            // In theory we could iterate from i..9,
            // but that causes a stack spill.
            for j in 0..9 {
                h[i][j] = scaled_row[i].mul_add(scaled_row[j], h[i][j]);
            }
        }
    }

    // for i in 0..9 {
    //     for j in i..9 {
    //         h[j][i] = h[i][j];
    //     }
    // }

    h
}

pub fn compute_pt_d2_p_scalar2(
    p_mat: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {

    let mut h = [[0.0; 9]; 9];

    for row in p_mat.iter() {
        let prod = dot(row, x);

        // d = 1 / (P x + eps)
        let d = 1.0 / (prod + eps);

        let mut scaled_row = [0.0; 9];
        for i in 0..9 {
            scaled_row[i] = d * row[i];
        }

        // g = P^T d
        for i in 0..9 {
            for j in i..9 {
                h[i][j] = scaled_row[i].mul_add(scaled_row[j], h[i][j]);
            }
        }
    }

    for i in 0..9 {
        for j in i..9 {
            h[j][i] = h[i][j];
        }
    }

    h
}