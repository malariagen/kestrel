use crate::{matrix::Block, util::{Matrix9xN, Vector9, dot}};
use std::arch::x86_64::*;

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

    let mut ptd = [0.0; 9];
    compute_pt_d_scalar(&p, &x0, eps, &mut ptd);

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
    g: &mut [f64; 9]
) {
    for row in p_mat.iter() {
        let prod = dot(row, x);

        // d = 1 / (P x + eps)
        let d = 1.0 / (prod + eps);

        // g = P^T d
        for i in 0..9 {
            g[i] = row[i].mul_add(d, g[i]);
        }
    }
}


#[target_feature(enable = "avx512f")]
pub unsafe fn compute_pt_d_avx512_blocked(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [f64; 9] {

    let xs = [
        _mm512_set1_pd(x[0]),
        _mm512_set1_pd(x[1]),
        _mm512_set1_pd(x[2]),
        _mm512_set1_pd(x[3]),
        _mm512_set1_pd(x[4]),
        _mm512_set1_pd(x[5]),
        _mm512_set1_pd(x[6]),
        _mm512_set1_pd(x[7]),
        _mm512_set1_pd(x[8]),
    ];

    let mut columns = [_mm512_setzero_pd(); 9];
    let mut gradients = [_mm512_setzero_pd(); 9];

    let one = _mm512_set1_pd(1.0);

    for block in blocks.iter() {
        // Load columns of P from memory
        for col in 0..9 {
            columns[col] = unsafe { _mm512_load_pd(block[col].as_ptr()) };
        }

        // Calculate d
        // This computes a dot product between x and a row of p
        // TODO this could be manually unrolled a few times
        let mut d = _mm512_set1_pd(eps);
        for col in 0..9 {
            d = _mm512_fmadd_pd(xs[col], columns[col], d);
        }

        // TODO investigate reciprocal
        d = _mm512_div_pd(one, d);

        for col in 0..9 {
            gradients[col] = _mm512_fmadd_pd(columns[col], d, gradients[col]);
        }
    }

    let mut g = [0.0; 9];
    for col in 0..9 {
        g[col] = _mm512_reduce_add_pd(gradients[col]);
    }

    compute_pt_d_scalar(remainder, x, eps, &mut g);

    g
}