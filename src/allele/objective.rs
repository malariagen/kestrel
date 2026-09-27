use std::arch::x86_64::*;
use crate::{algebra::Vector, arith::{Arith, Lane, Lane8}, blockbuffer::Block, lanevector::LaneVector, log::Log};

pub fn compute_objective(likelihood_mats: &LaneVector<Lane8, 10>, x: &Vector<4>, eps: f64) -> f64 {
    let (blocks, remainder) = likelihood_mats.as_lanes();

    let b = compute_obj_blocks(blocks, x, eps);
    let r = compute_obj_lane::<f64, f64>(remainder, x, eps);

    let n = likelihood_mats.len() as f64;

    -(b + r) / (n as f64)
}

fn compute_obj_blocks(blocks: &[[Lane8; 10]], x: &Vector<4>, eps: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return compute_obj_lane::<Lane8, __m512d>(blocks, x, eps);
    }

    unreachable!("Not implemented yet")

    // compute_blocks_scalar(blocks, x, eps)
}

// fn compute_objective_remainder(likelihoods: &[Vector<10>], x: &Vector<4>, eps: f64) -> f64 {
//     let mut s = 0.0;

//     for mat in likelihoods.iter() {
//         let p = dot(x, &mul(mat, x));
//         s += Log::log(p + eps);
//     }

//     let n = likelihoods.len();

//     -s / (n as f64)
// }

#[target_feature(enable = "avx512f")]
pub fn compute_obj_avx512(blocks: &[Block<f64, 8, 10>], x: &Vector<4>, eps: f64) -> f64 {
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

fn compute_obj_lane<L: Lane, S: Arith<L> + Log + Copy>(blocks: &[[L; 10]], x: &Vector<4>, eps: f64) -> f64 {
    // 2
    let mut zs = S::zero();
    let ze = S::set(eps);

    let x0 = S::set(x[0]);
    let x1 = S::set(x[1]);
    let x2 = S::set(x[2]);
    let x3 = S::set(x[3]);

    // 10
    let x00 = x0.mul(x0);
    let x11 = x1.mul(x1);
    let x22 = x2.mul(x2);
    let x33 = x3.mul(x3);

    let x01 = x0.mul(x1);
    let x02 = x0.mul(x2);
    let x03 = x0.mul(x3);
    let x12 = x1.mul(x2);
    let x13 = x1.mul(x3);
    let x23 = x2.mul(x3);

    let tx01 = x01.add(x01);
    let tx02 = x02.add(x02);
    let tx03 = x03.add(x03);
    let tx12 = x12.add(x12);
    let tx13 = x13.add(x13);
    let tx23 = x23.add(x23);

    // No stack spillage, very good
    for block in blocks.iter() {
        let mut p = ze;

        p = x00.fma(S::load(&block[0]), p);
        p = tx01.fma(S::load(&block[1]), p);
        p = x11.fma(S::load(&block[2]), p);
        p = tx02.fma(S::load(&block[3]), p);
        p = tx12.fma(S::load(&block[4]), p);
        p = x22.fma(S::load(&block[5]), p);
        p = tx03.fma(S::load(&block[6]), p);
        p = tx13.fma(S::load(&block[7]), p);
        p = tx23.fma(S::load(&block[8]), p);
        p = x33.fma(S::load(&block[9]), p);

        let l = Log::log(p);

        zs = l.add(zs);
    }

    zs.radd()
}


