use std::arch::x86_64::*;

use crate::matrix::{Block, BlockArray};


#[target_feature(enable = "avx512f")]
pub unsafe fn compute_pt_d_avx512_column_major(
    rows: usize,
    p_mat: &[f64], // rows x 9, column major
    x: &[f64; 9],
    eps: f64,
) -> [f64; 9] {

    assert!(p_mat.len() == rows * 9);

    let p = p_mat.as_ptr();

    // In the matrix, P[r, c] = c * rows + r
    // This could be calculated on the fly, but LLVM likes the pointers to each column better
    // since it avoids arithmetic.
    let mut col_ptrs: [*const f64; 9] = std::array::from_fn(|col| unsafe { p.add(col.unchecked_mul(rows)) });

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

    let chunks = rows / 8;
    let remainder = rows % 8;

    for _ in 0..chunks {
        // Load columns of P from memory
        for col in 0..9 {
            // TODO align the memory
            columns[col] = unsafe { _mm512_loadu_pd(col_ptrs[col]) };
        }

        // Calculate d
        // This computes a dot product between x and a row of p
        let mut d = _mm512_set1_pd(eps);
        for col in 0..9 {
            d = _mm512_fmadd_pd(xs[col], columns[col], d);
        }

        d = _mm512_div_pd(one, d);

        for col in 0..9 {
            gradients[col] = _mm512_fmadd_pd(columns[col], d, gradients[col]);
        }

        for col in 0..9 {
            col_ptrs[col] = unsafe { col_ptrs[col].add(8) };
        }
    }

    if remainder > 0 {
        let mask = (1u8 << remainder) - 1;

        for col in 0..9 {
            // TODO align the memory
            columns[col] = unsafe { _mm512_maskz_loadu_pd(mask, col_ptrs[col]) };
        }

        // Calculate d
        // This computes a dot product between x and a row of p
        let mut d = _mm512_set1_pd(eps);
        for col in 0..9 {
            d = _mm512_fmadd_pd(xs[col], columns[col], d);
        }

        d = _mm512_div_pd(one, d);

        for col in 0..9 {
            gradients[col] = _mm512_mask3_fmadd_pd(columns[col], d, gradients[col], mask);
        }
    }

    let mut g = [0.0; 9];
    for col in 0..9 {
        g[col] = _mm512_reduce_add_pd(gradients[col]);
    }

    g
}


#[target_feature(enable = "avx512f")]
pub fn compute_pt_d_scalar_column(
    rows: usize,
    p_mat: &[f64], // rows x 9, column major
    x: &[f64; 9],
    eps: f64,
) -> [f64; 9] {

    assert!(p_mat.len() == rows * 9);

    let mut g = [0.0; 9];

    let p = p_mat.as_ptr();

    let mut col_ptrs: [*const f64; 9] = std::array::from_fn(|col| unsafe { p.add(col.unchecked_mul(rows)) });

    for _ in 0..rows {
        let mut d = 0.0;
        // d = P x
        for col in 0..9 {
            let c = unsafe { *col_ptrs[col] };
            d = c.mul_add(x[col], d);
        }

        d = 1.0 / (d + eps);

        // g = P^T d
        for col in 0..9 {
            let c = unsafe { *col_ptrs[col] };
            g[col] = c.mul_add(d, g[col]);
        }

        for col in 0..9 {
            col_ptrs[col] = unsafe { col_ptrs[col].add(1) };
        }
    }

    g
}
