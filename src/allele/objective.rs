use std::arch::x86_64::*;
use crate::{algebra::Vector, blockbuffer::{Block, BlockBuffer}, log::Log};

fn compute_objective(likelihood_mats: &BlockBuffer<f64, 8, 9>, x: &Vector<4>, eps: f64) -> f64 {
    let (blocks, remainder) = likelihood_mats.as_blocks();

    let b = compute_obj_blocks(blocks, x, eps);

    let r = compute_obj_remainder(remainder, x, eps);

    let n = likelihoods_mats.num_rows() as f64;

    -(b + r) / (n as f64)
}

fn compute_obj_blocks(blocks: &[Block<f64, 8, 10>], x: &Vector<4>, eps: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return unsafe { compute_obj_avx512(blocks, x, eps) };
    }

    unreachable!("Not implemented yet")

    // compute_blocks_scalar(blocks, x, eps)
}

fn compute_objective_remainder(likelihoods: &[Vector<10>], x: &Vector<4>, eps: f64) -> f64 {
    let mut s = 0.0;

    for mat in likelihoods.iter() {
        let p = dot(x, &mul(mat, x));
        s += Log::log(p + eps);
    }

    let n = likelihoods.len();

    -s / (n as f64)
}

#[target_feature(enable = "avx512f")]
pub fn compute_obj_avx512(blocks: &[Block<f64, 10, 8>], x: &Vector<4>, eps: f64) -> f64 {
    // 2
    let mut zs = _mm512_setzero_pd();
    let ze = _mm512_set1_pd(eps);

    let x0 = _mm512_set1_pd(x[0]);
    let x1 = _mm512_set1_pd(x[1]);
    let x2 = _mm512_set1_pd(x[2]);
    let x3 = _mm512_set1_pd(x[3]);

    // 10
    let x00 = _mm512_mul_pd(x0, x0);
    let x11 = _mm512_mul_pd(x1, x1);
    let x22 = _mm512_mul_pd(x2, x2);
    let x33 = _mm512_mul_pd(x3, x3);

    let x01 = _mm512_mul_pd(x0, x1);
    let x02 = _mm512_mul_pd(x0, x2);
    let x03 = _mm512_mul_pd(x0, x3);
    let x12 = _mm512_mul_pd(x1, x2);
    let x13 = _mm512_mul_pd(x1, x3);
    let x23 = _mm512_mul_pd(x2, x3);

    let tx01 = _mm512_add_pd(x01, x01);
    let tx02 = _mm512_add_pd(x02, x02);
    let tx03 = _mm512_add_pd(x03, x03);
    let tx12 = _mm512_add_pd(x12, x12);
    let tx13 = _mm512_add_pd(x13, x13);
    let tx23 = _mm512_add_pd(x23, x23);

    // No stack spillage, very good
    for block in blocks.iter() {
        let mut p = ze;

        p = _mm512_fmadd_pd(x00, unsafe { _mm512_load_pd(block[0].as_ptr()) }, p);
        p = _mm512_fmadd_pd(tx01, unsafe { _mm512_load_pd(block[1].as_ptr()) }, p);
        p = _mm512_fmadd_pd(x11, unsafe { _mm512_load_pd(block[2].as_ptr()) }, p);
        p = _mm512_fmadd_pd(tx02, unsafe { _mm512_load_pd(block[3].as_ptr()) }, p);
        p = _mm512_fmadd_pd(tx12, unsafe { _mm512_load_pd(block[4].as_ptr()) }, p);
        p = _mm512_fmadd_pd(x22, unsafe { _mm512_load_pd(block[5].as_ptr()) }, p);
        p = _mm512_fmadd_pd(tx03, unsafe { _mm512_load_pd(block[6].as_ptr()) }, p);
        p = _mm512_fmadd_pd(tx13, unsafe { _mm512_load_pd(block[7].as_ptr()) }, p);
        p = _mm512_fmadd_pd(tx23, unsafe { _mm512_load_pd(block[8].as_ptr()) }, p);
        p = _mm512_fmadd_pd(x33, unsafe { _mm512_load_pd(block[9].as_ptr()) }, p);

        let l = Log::log(p);

        zs = _mm512_add_pd(l, zs);
    }

    _mm512_reduce_add_pd(zs)
}

