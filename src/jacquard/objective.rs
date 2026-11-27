use crate::{
    algebra::Vector,
    lanevector::LaneVector,
    log::Log,
    simd::{Arith, Lane, LoadStore},
};

#[inline(always)]
pub fn compute_jacquard_obj<L: Lane>(
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
