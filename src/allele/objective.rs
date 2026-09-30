use crate::{algebra::Vector, arith::{Arith, Lane, Lane8}, lanevector::LaneVector, log::Log};

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
        use std::arch::x86_64::__m512d;

        return compute_obj_lane::<Lane8, __m512d>(blocks, x, eps);
    }

    unreachable!("Not implemented yet")
}

fn compute_obj_lane<L: Lane, S: Arith<L> + Log>(blocks: &[[L; 10]], x: &Vector<4>, eps: f64) -> f64 {
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


