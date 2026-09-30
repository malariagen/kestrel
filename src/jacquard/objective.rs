use crate::{
    algebra::{Vector, dot, sum}, arith::{Arith, Lane, Lane8, LoadStore, lane::{Lane2, Lane4}}, lanevector::{GenericLaneVector, LaneVector}, log::Log,
};

pub fn compute_obj(likelihood_mats: &GenericLaneVector<9>, x: &Vector<9>, eps: f64) -> f64 {
    match likelihood_mats {
        GenericLaneVector::L8(lv) => compute_obj_avx512(lv, x, eps),
        GenericLaneVector::L4(lv) => compute_obj_avx2(lv, x, eps),
        GenericLaneVector::L2(lv) => compute_obj_neon(lv, x, eps),
        GenericLaneVector::L1(lv) => compute_obj_scalar(lv, x, eps),
    }
}

fn compute_obj_avx512(p_mat: &LaneVector<Lane8, 9>, x: &Vector<9>, eps: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__m512d;
        return compute_obj_generic::<Lane8, __m512d>(p_mat, x, eps);
    }

    panic!("Architecture incompatible with Lane8!")
}

fn compute_obj_avx2(p_mat: &LaneVector<Lane4, 9>, x: &Vector<9>, eps: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__m256d;
        unimplemented!("AHHHH")
        // return compute_obj::<Lane4, __m256d>(likelihood_mats, x, eps);
    }

    panic!("Architecture incompatible with Lane4!")
}

fn compute_obj_neon(p_mat: &LaneVector<Lane2, 9>, x: &Vector<9>, eps: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::x86_64::__m256d;
        return compute_obj_generic::<Lane2, __m256d>(p_mat, x, eps);
    }

    panic!("Architecture incompatible with Lane2!")
}

fn compute_obj_scalar(p_mat: &LaneVector<f64, 9>, x: &Vector<9>, eps: f64) -> f64 {
    return compute_obj_generic::<f64, f64>(p_mat, x, eps);
}

fn compute_obj_generic<L : Lane, S : Arith + LoadStore<L> + Log>(p_mat: &LaneVector<L, 9>, x: &Vector<9>, eps: f64) -> f64 {
    let (blocks, remainder) = p_mat.as_lanes();

    let b = compute_obj_lane::<L, S>(blocks, x, eps);
    let r = compute_obj_lane::<f64, f64>(remainder, x, eps);

    let n = p_mat.len() as f64;

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
