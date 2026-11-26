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

pub trait JacquardObjective: Lane {
    fn compute_obj(p_mat: &LaneVector<Self, 9>, x: &Vector<9>, eps: f64) -> f64;
}

#[cfg(target_arch = "x86_64")]
impl JacquardObjective for Avx512Lane {
    fn compute_obj(p_mat: &LaneVector<Self, 9>, x: &Vector<9>, eps: f64) -> f64 {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner(p_mat: &LaneVector<Avx512Lane, 9>, x: &Vector<9>, eps: f64) -> f64 {
            compute_obj_generic(p_mat, x, eps)
        }
        unsafe { inner(p_mat, x, eps) }
    }
}

#[cfg(target_arch = "x86_64")]
impl JacquardObjective for Avx2Lane {
    fn compute_obj(p_mat: &LaneVector<Self, 9>, x: &Vector<9>, eps: f64) -> f64 {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn inner(p_mat: &LaneVector<Avx2Lane, 9>, x: &Vector<9>, eps: f64) -> f64 {
            compute_obj_generic(p_mat, x, eps)
        }
        unsafe { inner(p_mat, x, eps) }
    }
}

#[cfg(target_arch = "aarch64")]
impl JacquardObjective for NeonLane {
    fn compute_obj(p_mat: &LaneVector<Self, 9>, x: &Vector<9>, eps: f64) -> f64 {
        #[target_feature(enable = "neon")]
        unsafe fn inner(p_mat: &LaneVector<Avx2Lane, 9>, x: &Vector<9>, eps: f64) -> f64 {
            compute_obj_generic(p_mat, x, eps)
        }
        unsafe { inner(p_mat, x, eps) }
    }
}

impl JacquardObjective for f64 {
    fn compute_obj(p_mat: &LaneVector<Self, 9>, x: &Vector<9>, eps: f64) -> f64 {
        compute_obj_generic(p_mat, x, eps)
    }
}

#[inline(always)]
fn compute_obj_generic<L: Lane>(
    p_mat: &LaneVector<L, 9>,
    x: &Vector<9>,
    eps: f64,
) -> f64 where L::Reg: Arith + Log {
    let b = compute_obj_lane(p_mat.lanes(), x, eps);
    let r = compute_obj_lane(p_mat.rem(), x, eps);

    let n = p_mat.len() as f64;

    return -(b + r) / n;
}

#[inline(always)]
fn compute_obj_lane<L: Lane>(blocks: &[[L; 9]], x: &[f64; 9], eps: f64) -> f64
where L::Reg: Arith + Log  {
    // 9
    let zx: [L::Reg; 9] = std::array::from_fn(|i| L::Reg::set(x[i]));

    // 12
    let mut zs = L::Reg::zero();
    let ze = L::Reg::set(eps);

    for block in blocks.iter() {
        // This computes a dot product between x and a row of p
        // In theory this could be manually unrolled a few times
        // But in practice that's slower.

        // 13
        let mut d = ze;
        for col in 0..9 {
            let c = L::Reg::load(&block[col]);
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
