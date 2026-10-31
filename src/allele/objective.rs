use crate::{
    algebra::Vector,
    lanevector::LaneVector,
    log::Log,
    simd::{Arith, Lane, LoadStore},
};

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::Avx2Lane;
#[cfg(target_arch = "x86_64")]
use crate::simd::avx512::Avx512Lane;
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::NeonLane;

pub trait AlleleObjective<const A: usize, const G: usize>: Lane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, G>, x: &Vector<A>, eps: f64) -> f64;
}

// ==========================================
// AVX-512
// ==========================================

#[cfg(target_arch = "x86_64")]
impl AlleleObjective<2, 3> for Avx512Lane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 3>, x: &Vector<2>, eps: f64) -> f64 {
        use std::arch::x86_64::__m512d;
        compute_obj_generic2::<Self, __m512d>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "x86_64")]
impl AlleleObjective<3, 6> for Avx512Lane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 6>, x: &Vector<3>, eps: f64) -> f64 {
        use std::arch::x86_64::__m512d;
        compute_obj_generic3::<Self, __m512d>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "x86_64")]
impl AlleleObjective<4, 10> for Avx512Lane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 10>, x: &Vector<4>, eps: f64) -> f64 {
        use std::arch::x86_64::__m512d;
        compute_obj_generic4::<Self, __m512d>(likelihood_mats, x, eps)
    }
}

// ==========================================
// AVX2
// ==========================================

#[cfg(target_arch = "x86_64")]
impl AlleleObjective<2, 3> for Avx2Lane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 3>, x: &Vector<2>, eps: f64) -> f64 {
        use std::arch::x86_64::__m256d;
        compute_obj_generic2::<Self, __m256d>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "x86_64")]
impl AlleleObjective<3, 6> for Avx2Lane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 6>, x: &Vector<3>, eps: f64) -> f64 {
        use std::arch::x86_64::__m256d;
        compute_obj_generic3::<Self, __m256d>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "x86_64")]
impl AlleleObjective<4, 10> for Avx2Lane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 10>, x: &Vector<4>, eps: f64) -> f64 {
        use std::arch::x86_64::__m256d;
        compute_obj_generic4::<Self, __m256d>(likelihood_mats, x, eps)
    }
}

// ==========================================
// Neon
// ==========================================

#[cfg(target_arch = "aarch64")]
impl AlleleObjective<2, 3> for NeonLane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 3>, x: &Vector<2>, eps: f64) -> f64 {
        use std::arch::aarch64::float64x2_t;
        compute_obj_generic2::<Self, float64x2_t>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "aarch64")]
impl AlleleObjective<3, 6> for NeonLane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 6>, x: &Vector<3>, eps: f64) -> f64 {
        use std::arch::aarch64::float64x2_t;
        compute_obj_generic3::<Self, float64x2_t>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "aarch64")]
impl AlleleObjective<4, 10> for NeonLane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 10>, x: &Vector<4>, eps: f64) -> f64 {
        use std::arch::aarch64::float64x2_t;
        compute_obj_generic4::<Self, float64x2_t>(likelihood_mats, x, eps)
    }
}

// ==========================================
// Scalar
// ==========================================

impl AlleleObjective<2, 3> for f64 {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 3>, x: &Vector<2>, eps: f64) -> f64 {
        compute_obj_generic2::<Self, f64>(likelihood_mats, x, eps)
    }
}

impl AlleleObjective<3, 6> for f64 {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 6>, x: &Vector<3>, eps: f64) -> f64 {
        compute_obj_generic3::<Self, f64>(likelihood_mats, x, eps)
    }
}

impl AlleleObjective<4, 10> for f64 {
    fn compute_objective(likelihood_mats: &LaneVector<Self, 10>, x: &Vector<4>, eps: f64) -> f64 {
        compute_obj_generic4::<Self, f64>(likelihood_mats, x, eps)
    }
}


fn compute_obj_generic2<L: Lane, S: Arith + Log + LoadStore<L>>(
    likelihood_mats: &LaneVector<L, 3>,
    x: &Vector<2>,
    eps: f64,
) -> f64 {
    let blocks = likelihood_mats.lanes();
    let remainder = likelihood_mats.rem();

    let b = compute_obj_lane2::<L, S>(blocks, x, eps);
    let r = compute_obj_lane2::<f64, f64>(remainder, x, eps);

    let n = likelihood_mats.len() as f64;

    -(b + r) / n
}

fn compute_obj_generic3<L: Lane, S: Arith + Log + LoadStore<L>>(
    likelihood_mats: &LaneVector<L, 6>,
    x: &Vector<3>,
    eps: f64,
) -> f64 {
    let blocks = likelihood_mats.lanes();
    let remainder = likelihood_mats.rem();

    let b = compute_obj_lane3::<L, S>(blocks, x, eps);
    let r = compute_obj_lane3::<f64, f64>(remainder, x, eps);

    let n = likelihood_mats.len() as f64;

    -(b + r) / n
}

fn compute_obj_generic4<L: Lane, S: Arith + Log + LoadStore<L>>(
    likelihood_mats: &LaneVector<L, 10>,
    x: &Vector<4>,
    eps: f64,
) -> f64 {
    let blocks = likelihood_mats.lanes();
    let remainder = likelihood_mats.rem();

    let b = compute_obj_lane4::<L, S>(blocks, x, eps);
    let r = compute_obj_lane4::<f64, f64>(remainder, x, eps);

    let n = likelihood_mats.len() as f64;

    -(b + r) / n
}

fn compute_obj_lane2<L: Lane, S: Arith + Log + LoadStore<L>>(blocks: &[[L; 3]], x: &Vector<2>, eps: f64) -> f64 {
    // 4
    let mut zs = S::zero();
    let ze = S::set(eps);

    let x0 = S::set(x[0]);
    let x1 = S::set(x[1]);

    // 4
    let x00 = x0.mul(x0);
    let x11 = x1.mul(x1);
    let x01 = x0.mul(x1);
    let tx01 = x01.add(x01);

    for block in blocks.iter() {
        let mut p = ze;

        p = x00.fma(S::load(&block[0]), p);
        p = tx01.fma(S::load(&block[1]), p);
        p = x11.fma(S::load(&block[2]), p);

        let l = Log::log(p);

        zs = l.add(zs);
    }

    zs.radd()
}


fn compute_obj_lane3<L: Lane, S: Arith + Log + LoadStore<L>>(blocks: &[[L; 6]], x: &Vector<3>, eps: f64) -> f64 {
    // 2
    let mut zs = S::zero();
    let ze = S::set(eps);

    let x0 = S::set(x[0]);
    let x1 = S::set(x[1]);
    let x2 = S::set(x[2]);

    // 10
    let x00 = x0.mul(x0);
    let x11 = x1.mul(x1);
    let x22 = x2.mul(x2);

    let x01 = x0.mul(x1);
    let x02 = x0.mul(x2);
    let x12 = x1.mul(x2);

    let tx01 = x01.add(x01);
    let tx02 = x02.add(x02);
    let tx12 = x12.add(x12);

    // No stack spillage, very good
    for block in blocks.iter() {
        let mut p = ze;

        p = x00.fma(S::load(&block[0]), p);
        p = tx01.fma(S::load(&block[1]), p);
        p = x11.fma(S::load(&block[2]), p);
        p = tx02.fma(S::load(&block[3]), p);
        p = tx12.fma(S::load(&block[4]), p);
        p = x22.fma(S::load(&block[5]), p);

        let l = Log::log(p);

        zs = l.add(zs);
    }

    zs.radd()
}


fn compute_obj_lane4<L: Lane, S: Arith + Log + LoadStore<L>>(blocks: &[[L; 10]], x: &Vector<4>, eps: f64) -> f64 {
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
