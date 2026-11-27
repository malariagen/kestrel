use crate::{
    conditional::MIis,
    iis::accumulate_iis,
    lanevector::LaneVectorBuilder,
    simd::{Arith, Lane, LoadStore},
};

#[inline(always)]
pub fn calculate_mixture_component_matrix<L: Lane, M: Lane, const A: usize, const G: usize>(
    all_joint_genotypes: &MIis<G>,
    allele_frequencies: &[[L; A]],
    likelihoods_x: &[[L; G]],
    likelihoods_y: &[[L; G]],
    builder: &mut LaneVectorBuilder<M, 9>,
) where L::Reg: Arith {
    let len = allele_frequencies.len();
    assert!(likelihoods_x.len() == len);
    assert!(likelihoods_y.len() == len);

    // p = lx * ly * M
    for locus in 0..len {
        let like_x = unsafe { likelihoods_x.get_unchecked(locus) };
        let like_y = unsafe { likelihoods_y.get_unchecked(locus) };
        let af = unsafe { allele_frequencies.get_unchecked(locus) };

        // 9 registers
        let mut a: [L::Reg; 9] = std::array::from_fn(|_| L::Reg::zero());

        // G x G multiplications
        // G x G x 9 FMA

        for gx in 0..G {
            let lx = L::Reg::load(&like_x[gx]);
            for gy in 0..G {
                // For architectures with enough registers this will get hoisted
                // out of the loop
                let ly = L::Reg::load(&like_y[gy]);
                let g = lx.mul(ly);

                let ((i, j), (k, l), iis) = all_joint_genotypes[gx][gy];

                // This will always be safe due to the construction all_joint_genotypes
                let pi = L::Reg::load(unsafe { af.get_unchecked(i) });
                let pj = L::Reg::load(unsafe { af.get_unchecked(j) });
                let pk = L::Reg::load(unsafe { af.get_unchecked(k) });
                let pl = L::Reg::load(unsafe { af.get_unchecked(l) });

                accumulate_iis(&mut a, g, pi, pj, pk, pl, iis);
            }
        }

        let mut out = [L::default(); 9];

        for k in 0..9 {
            a[k].store(&mut out[k]);
        }

        builder.push_block(&out);
    }
}

#[inline(always)]
pub fn calculate_mixture_component_matrix2<L: Lane, M: Lane>(
    allele_frequencies: &[[L; 2]],
    likelihoods_x: &[[L; 3]],
    likelihoods_y: &[[L; 3]],
    builder: &mut LaneVectorBuilder<M, 9>,
) where L::Reg: Arith {
    let len = allele_frequencies.len();
    assert!(likelihoods_x.len() == len);
    assert!(likelihoods_y.len() == len);

    // p = lx * ly * M
    for locus in 0..len {
        let like_x = unsafe { likelihoods_x.get_unchecked(locus) };
        let like_y = unsafe { likelihoods_y.get_unchecked(locus) };
        let af = unsafe { allele_frequencies.get_unchecked(locus) };

        // 9 registers
        mixture_row2(builder, like_x, like_y, af);
    }
}

// It is possible to pre-calculate the conditional probability matrix M for each locus, and then re-use it for
// each pair of individuals (x, y). However, that requires loading 9 x 9 = 81 doubles, as opposed to just
// 2 for the allele frequencies. Furthermore, once you have loaded that matrix M, it takes
// 3 x 3 = 9 muls to get g, and 9 x 9 = 81 FMAs. The function below takes 9 muls to get g, 18 mul/adds to get
// the M entries, and then 45 FMAs. Even if you were to store only the 45 non-zero entries of M in memory, you
// are loading 45 elements vs. computing 18 multiplications. The latter is faster. Another option would be to
// do a partial contraction of M against the genotype likelihoods of x, write that to memory, and then load that
// to do a partial contraction against y. If you store the non-zero entries that might compete ... but it's very
// complicated and not a sure winner.
// We don't have analogous mixture_row3 and mixture_row4 functions since those have 36 and 100 cases respectively,
// and the compiler struggles to optimize that. Hence leave those to the general function.
#[inline(always)]
fn mixture_row2<L: Lane, M: Lane>(
    builder: &mut LaneVectorBuilder<M, 9>,
    like_x: &[L; 3],
    like_y: &[L; 3],
    af: &[L; 2],
) where L::Reg: Arith {
    // 15 registers
    let mut a: [L::Reg; 9] = std::array::from_fn(|_| L::Reg::zero());
    let lx: [L::Reg; 3] = std::array::from_fn(|i| L::Reg::load(&like_x[i]));
    let ly: [L::Reg; 3] = std::array::from_fn(|i| L::Reg::load(&like_y[i]));

    // 2
    let p = L::Reg::load(&af[0]);
    let q = L::Reg::load(&af[1]);

    // 10
    let p2 = p.mul(p);
    let p3 = p2.mul(p);
    let q2 = q.mul(q);
    let q3 = q2.mul(q);

    let pq = p.mul(q);
    let p2q = p2.mul(q);
    let pq2 = p.mul(q2);
    let p2q2 = p2.mul(q2);

    let p3q = p3.mul(q);
    let pq3 = p.mul(q3);

    // 5
    let twopq = pq.add(pq);
    let twop2q = p2q.add(p2q);
    let twop3q = p3q.add(p3q);
    let twopq2 = pq2.add(pq2);
    let twopq3 = pq3.add(pq3);

    // We have 32 registers exactly! The compiler needs a few
    // more for the intermediate multiplications. However due
    // to non-overlapping lifetimes for the constants it will
    // be able to fit everything without spilling to the stack.

    // genotypes - gl vcf index - iis mode
    // (0, 0) (0, 0) - 0 0 - M1
    {
        let g = lx[0].mul(ly[0]);

        a[0] = g.fma(p, a[0]);
        a[1] = g.fma(p2, a[1]);
        a[2] = g.fma(p2, a[2]);
        a[3] = g.fma(p3, a[3]);
        a[4] = g.fma(p2, a[4]);
        a[5] = g.fma(p3, a[5]);
        a[6] = g.fma(p2, a[6]);
        a[7] = g.fma(p3, a[7]);
        a[8] = g.fma(p2.mul(p2), a[8]);
    }
    // (0, 0) (0, 1) - 0 1 - M3
    {
        let g = lx[0].mul(ly[1]);

        a[2] = g.fma(pq, a[2]);
        a[3] = g.fma(twop2q, a[3]);
        a[7] = g.fma(p2q, a[7]);
        a[8] = g.fma(twop3q, a[8]);
    }
    // (0, 0) (1, 1) - 0 2 - M2
    {
        let g = lx[0].mul(ly[2]);

        a[1] = g.fma(pq, a[1]);
        a[3] = g.fma(pq2, a[3]);
        a[5] = g.fma(p2q, a[5]);
        a[8] = g.fma(p2q2, a[8]);
    }
    // (0, 1) (0, 0) - 1 0 - M6
    {
        let g = lx[1].mul(ly[0]);

        a[4] = g.fma(pq, a[4]);
        a[5] = g.fma(twop2q, a[5]);
        a[7] = g.fma(p2q, a[7]);
        a[8] = g.fma(twop3q, a[8]);
    }
    // (0, 1) (0, 1) - 1 1 - M9
    {
        let g = lx[1].mul(ly[1]);

        a[6] = g.fma(twopq, a[6]);
        a[7] = g.fma(pq, a[7]);
        a[8] = g.fma(twopq.mul(twopq), a[8]);
    }
    // (0, 1) (1, 1) - 1 2 - M7
    {
        let g = lx[1].mul(ly[2]);

        a[4] = g.fma(pq, a[4]);
        a[5] = g.fma(twopq2, a[5]);
        a[7] = g.fma(pq2, a[7]);
        a[8] = g.fma(twopq3, a[8]);
    }
    // (1, 1) (0, 0) - 2 0 - M2
    {
        let g = lx[2].mul(ly[0]);

        a[1] = g.fma(pq, a[1]);
        a[3] = g.fma(p2q, a[3]);
        a[5] = g.fma(pq2, a[5]);
        a[8] = g.fma(p2q2, a[8]);
    }
    // (1, 1) (0, 1) - 2 1 - M4
    {
        let g = lx[2].mul(ly[1]);

        a[2] = g.fma(pq, a[2]);
        a[3] = g.fma(twopq2, a[3]);
        a[7] = g.fma(pq2, a[7]);
        a[8] = g.fma(twopq3, a[8]);
    }
    // (1, 1) (1, 1) - 2 2 - M1
    {
        let g = lx[2].mul(ly[2]);

        a[0] = g.fma(q, a[0]);
        a[1] = g.fma(q2, a[1]);
        a[2] = g.fma(q2, a[2]);
        a[3] = g.fma(q3, a[3]);
        a[4] = g.fma(q2, a[4]);
        a[5] = g.fma(q3, a[5]);
        a[6] = g.fma(q2, a[6]);
        a[7] = g.fma(q3, a[7]);
        a[8] = g.fma(q2.mul(q2), a[8]);
    }

    let mut out = [L::default(); 9];

    for k in 0..9 {
        a[k].store(&mut out[k]);
    }

    builder.push_block(&out);
}
