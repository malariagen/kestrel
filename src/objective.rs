use crate::{log::log_avx512, matrix::BlockBuffer, util::{Vector9, dot, sum}};
use core::arch::x86_64::*;

pub fn compute_obj<const L: usize>(p_mat: &BlockBuffer<f64, L, 9>, x: &Vector9<f64>, eps: f64) -> f64 {
    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    let s = compute_obj_block(p_mat, &x0, eps);

    let n = p_mat.num_rows();

    return sum(&x0) - s / (n as f64) - 1.0;
}

pub fn compute_obj_avx(p_mat: &BlockBuffer<f64, 8, 9>, x: &Vector9<f64>, eps: f64) -> f64 {
    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    let s = unsafe { compute_obj_avx512(p_mat, &x0, eps) };

    let n = p_mat.num_rows();

    return sum(&x0) - s / (n as f64) - 1.0;
}

fn compute_obj_scalar(p_mat: &[[f64; 9]], x: &[f64; 9], eps: f64) -> f64 {
    let mut s = 0.0;
    for row in p_mat.iter() {
        let prod = dot(row, x);
        let t = (prod + eps).ln();
        // TODO kahan?
        s += t;
    }
    s
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

#[target_feature(enable = "avx512f")]
pub fn compute_obj_avx512(p_mat: &BlockBuffer<f64, 8, 9>, x: &[f64; 9], eps: f64) -> f64 {
    let (blocks, remainder) = p_mat.as_blocks();

    let zx: [__m512d; 9] = std::array::from_fn(|i| _mm512_set1_pd(x[i]));

    let ze =_mm512_set1_pd(eps);

    let mut zs = _mm512_setzero_pd();

    for block in blocks.iter() {

        let c: [__m512d; 9] =
            std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

        // Calculate d
        // This computes a dot product between x and a row of p
        // TODO this could be manually unrolled a few times
        // let mut d = _mm512_set1_pd(eps);
        // for col in 0..9 {
        //     d = _mm512_fmadd_pd(zx[col], c[col], d);
        // }

        let mut d0 = _mm512_fmadd_pd(zx[0], c[0], ze);
        let mut d1 = _mm512_mul_pd(zx[1], c[1]);
        let mut d2 = _mm512_mul_pd(zx[2], c[2]);

        d0 = _mm512_fmadd_pd(zx[3], c[3], d0);
        d1 = _mm512_fmadd_pd(zx[4], c[4], d1);
        d2 = _mm512_fmadd_pd(zx[5], c[5], d2);

        d0 = _mm512_fmadd_pd(zx[6], c[6], d0);
        d1 = _mm512_fmadd_pd(zx[7], c[7], d1);
        d2 = _mm512_fmadd_pd(zx[8], c[8], d2);

        let mut d = _mm512_add_pd(d0, d1);
        d = _mm512_add_pd(d, d2);

        let l = log_avx512(d);

        // TODO kahan
        zs = _mm512_add_pd(l, zs);
    }

    let mut s = _mm512_reduce_add_pd(zs);

    s += compute_obj_scalar(remainder, x, eps);

    s
}
