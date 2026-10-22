use crate::{
    algebra::Vector,
    simd::{Arith, Lane, LoadStore},
    log::Log,
};

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::Avx2Lane;
#[cfg(target_arch = "x86_64")]
use crate::simd::avx512::Avx512Lane;
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::NeonLane;

pub trait JacquardObjective: Lane {
    fn compute_obj(lanes: &[[Self; 9]], rem: &[[f64; 9]], x: &Vector<9>, eps: f64) -> f64;
}

#[cfg(target_arch = "x86_64")]
impl JacquardObjective for Avx512Lane {
    fn compute_obj(lanes: &[[Self; 9]], rem: &[[f64; 9]], x: &Vector<9>, eps: f64) -> f64 {
        use std::arch::x86_64::__m512d;
        return compute_obj_generic::<Avx512Lane, __m512d>(lanes, rem, x, eps);
    }
}

#[cfg(target_arch = "x86_64")]
impl JacquardObjective for Avx2Lane {
    fn compute_obj(lanes: &[[Self; 9]], rem: &[[f64; 9]], x: &Vector<9>, eps: f64) -> f64 {
        use std::arch::x86_64::__m256d;
        return compute_obj_generic::<Avx2Lane, __m256d>(lanes, rem, x, eps);
    }
}

#[cfg(target_arch = "aarch64")]
impl JacquardObjective for NeonLane {
    fn compute_obj(lanes: &[[Self; 9]], rem: &[[f64; 9]], x: &Vector<9>, eps: f64) -> f64 {
        use std::arch::aarch64::float64x2_t;
        return compute_obj_generic::<NeonLane, float64x2_t>(lanes, rem, x, eps);
    }
}

impl JacquardObjective for f64 {
    fn compute_obj(lanes: &[[Self; 9]], rem: &[[f64; 9]], x: &Vector<9>, eps: f64) -> f64 {
        return compute_obj_generic::<f64, f64>(lanes, rem, x, eps);
    }
}

fn compute_obj_generic<L: Lane, S: Arith + LoadStore<L> + Log>(
    lanes: &[[L; 9]],
    rem: &[[f64; 9]],
    x: &Vector<9>,
    eps: f64,
) -> f64 {
    let b = compute_obj_lane::<L, S>(lanes, x, eps);
    let r = compute_obj_lane::<f64, f64>(rem, x, eps);

    let n = (lanes.len() + rem.len()) as f64;

    return -(b + r) / n;
}

fn compute_obj_lane<L: Lane, S: Arith + LoadStore<L> + Log>(blocks: &[[L; 9]], x: &[f64; 9], eps: f64) -> f64 {
    // 9
    let zx: [S; 9] = std::array::from_fn(|i| S::set(x[i]));

    // 12
    let mut zs = S::zero();
    let ze = S::set(eps);

    for block in blocks.iter() {
        // This computes a dot product between x and a row of p
        // In theory this could be manually unrolled a few times
        // But in practice that's slower.

        // 13
        let mut d = ze;
        for col in 0..9 {
            let c = S::load(&block[col]);
            // This memory access gets put directly in the fmadd instruction
            d = zx[col].fma(c, d);
        }

        let l = Log::log(d);

        zs = zs.add(l);
    }

    // Perhaps do something with zc too...
    zs.radd()
}

// MUST look at assembly. Essential!
// Making a function call requires spilling registers so they can be restored after.

// For big functions you need to manually re-order instructions to
// get the compiler to generate different code (e.g. avoid spillage)
// even though in theory they're the equivalent.
