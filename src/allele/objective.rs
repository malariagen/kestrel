use crate::{
    algebra::Vector,
    arith::{
        Arith, Lane, Lane8,
        lane::{Lane2, Lane4},
    },
    lanevector::{GenericLaneVector, LaneVector},
    log::Log,
};

pub fn compute_objective(likelihood_mats: &GenericLaneVector<10>, x: &Vector<4>, eps: f64) -> f64 {
    match likelihood_mats {
        GenericLaneVector::L8(lv) => compute_obj_avx512(lv, x, eps),
        GenericLaneVector::L4(lv) => compute_obj_avx2(lv, x, eps),
        GenericLaneVector::L2(lv) => compute_obj_neon(lv, x, eps),
        GenericLaneVector::L1(lv) => compute_obj_scalar(lv, x, eps),
    }
}

fn compute_obj_avx512(likelihood_mats: &LaneVector<Lane8, 10>, x: &Vector<4>, eps: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__m512d;
        return compute_obj_generic::<Lane8, __m512d>(likelihood_mats, x, eps);
    }

    panic!("Architecture incompatible with Lane8!")
}

fn compute_obj_avx2(likelihood_mats: &LaneVector<Lane4, 10>, x: &Vector<4>, eps: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__m256d;
        unimplemented!("AHHHH")
        // return compute_obj::<Lane4, __m256d>(likelihood_mats, x, eps);
    }

    panic!("Architecture incompatible with Lane4!")
}

fn compute_obj_neon(likelihood_mats: &LaneVector<Lane2, 10>, x: &Vector<4>, eps: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::x86_64::__m256d;
        return compute_obj_generic::<Lane2, __m256d>(likelihood_mats, x, eps);
    }

    panic!("Architecture incompatible with Lane2!")
}

fn compute_obj_scalar(likelihood_mats: &LaneVector<f64, 10>, x: &Vector<4>, eps: f64) -> f64 {
    return compute_obj_generic::<f64, f64>(likelihood_mats, x, eps);
}

fn compute_obj_generic<L: Lane, S: Arith<L> + Log>(
    likelihood_mats: &LaneVector<L, 10>,
    x: &Vector<4>,
    eps: f64,
) -> f64 {
    let (blocks, remainder) = likelihood_mats.as_lanes();

    let b = compute_obj_lane::<L, S>(blocks, x, eps);
    let r = compute_obj_lane::<f64, f64>(remainder, x, eps);

    let n = likelihood_mats.len() as f64;

    -(b + r) / (n as f64)
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
