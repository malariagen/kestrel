use crate::{
    algebra::{Vector, dot, sum},
    arith::{Arith, Lane, Lane8},
    lanevector::LaneVector,
    log::Log,
};

pub fn compute_obj(p_mat: &LaneVector<Lane8, 9>, x: &Vector<9>, eps: f64) -> f64 {
    let (blocks, remainder) = p_mat.as_lanes();

    let b = compute_obj_blocks(blocks, x, eps);

    let r = compute_obj_lane::<f64, f64>(remainder, x, eps);

    let n = p_mat.len() as f64;

    return -(b + r) / n;
}

pub fn compute_obj_blocks(blocks: &[[Lane8; 9]], x: &Vector<9>, eps: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        use std::arch::x86_64::__m512d;

        return compute_obj_lane::<Lane8, __m512d>(blocks, x, eps);
    }

    unreachable!("Not implemented yet")
}

pub fn compute_obj_lane<L: Lane, S: Arith<L> + Log>(blocks: &[[L; 9]], x: &[f64; 9], eps: f64) -> f64 {
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
