use crate::{
    conditional::MIis, iis::accumulate_iis, simd::{Arith, Lane, LoadStore},
};

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::Avx2Lane;
#[cfg(target_arch = "x86_64")]
use crate::simd::avx512::Avx512Lane;
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::NeonLane;

pub trait MixtureMatrix: Lane {
    fn mixture_component_matrix2(
        allele_frequencies: &[[Self; 2]],
        likelihoods_x: &[[Self; 3]],
        likelihoods_y: &[[Self; 3]],
        p_mat: &mut [[Self; 9]],
    );
    fn mixture_component_matrix<const A: usize, const G: usize>(
        all_joint_genotypes: &MIis<G>,
        allele_frequencies: &[[Self; A]],
        likelihoods_x: &[[Self; G]],
        likelihoods_y: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    );
}

// pub fn test_2asm(
        // all_joint_genotypes: &MIis<6>,
        // allele_frequencies: &[[Avx512Lane; 3]],
        // likelihoods_x: &[[Avx512Lane; 6]],
        // likelihoods_y: &[[Avx512Lane; 6]],
        // p_mat: &mut [[Avx512Lane; 9]],
// ) {
    // use std::arch::x86_64::__m512d;
    // return calculate_mixture_component_matrix::<Avx512Lane, __m512d, 3, 6>(all_joint_genotypes, allele_frequencies, likelihoods_x, likelihoods_y, p_mat);
    // return calculate_mixture_component_matrix3::<Avx512Lane, __m512d>(allele_frequencies, likelihoods_x, likelihoods_y, p_mat);
// }


// pub fn test_asm_mix(
// out: &mut [Avx512Lane; 9], like_x: &[Avx512Lane; 3], like_y: &[Avx512Lane; 3], af: &[Avx512Lane; 2]
// ) {
//     use std::arch::x86_64::__m512d;
//     mixture_row2::<Avx512Lane, __m512d>(out, like_x, like_y, af);
// }

// pub fn test_asm_bi(
//         allele_frequencies: &[[Avx512Lane; 2]],
//         likelihoods_x: &[[Avx512Lane; 3]],
//         likelihoods_y: &[[Avx512Lane; 3]],
//         p_mat: &mut [[Avx512Lane; 9]],
// ) {
//     use std::arch::x86_64::__m512d;
//     return calculate_mixture_component_matrix2::<Avx512Lane, __m512d>(allele_frequencies, likelihoods_x, likelihoods_y, p_mat);
// }

#[cfg(target_arch = "x86_64")]
impl MixtureMatrix for Avx512Lane {
    fn mixture_component_matrix<const A: usize, const G: usize>(
        all_joint_genotypes: &MIis<G>,
        allele_frequencies: &[[Self; A]],
        likelihoods_x: &[[Self; G]],
        likelihoods_y: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::x86_64::__m512d;
        calculate_mixture_component_matrix::<Self, __m512d, A, G>(all_joint_genotypes, allele_frequencies, likelihoods_x, likelihoods_y, p_mat)
    }

    fn mixture_component_matrix2(
        allele_frequencies: &[[Self; 2]],
        likelihoods_x: &[[Self; 3]],
        likelihoods_y: &[[Self; 3]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::x86_64::__m512d;
        calculate_mixture_component_matrix2::<Self, __m512d>(allele_frequencies, likelihoods_x, likelihoods_y, p_mat)
    }
}

#[cfg(target_arch = "x86_64")]
impl MixtureMatrix for Avx2Lane {
    fn mixture_component_matrix<const A: usize, const G: usize>(
        all_joint_genotypes: &MIis<G>,
        allele_frequencies: &[[Self; A]],
        likelihoods_x: &[[Self; G]],
        likelihoods_y: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::x86_64::__m256d;
        calculate_mixture_component_matrix::<Self, __m256d, A, G>(all_joint_genotypes, allele_frequencies, likelihoods_x, likelihoods_y, p_mat)
    }

    fn mixture_component_matrix2(
        allele_frequencies: &[[Self; 2]],
        likelihoods_x: &[[Self; 3]],
        likelihoods_y: &[[Self; 3]],
        p_mat: &mut [[Self; 9]],
    )
    {
        use std::arch::x86_64::__m256d;
        calculate_mixture_component_matrix2::<Self, __m256d>(allele_frequencies, likelihoods_x, likelihoods_y, p_mat)
    }
}

#[cfg(target_arch = "aarch64")]
impl MixtureMatrix for NeonLane {
    fn mixture_component_matrix<const A: usize, const G: usize>(
        all_joint_genotypes: &MIis<G>,
        allele_frequencies: &[[Self; A]],
        likelihoods_x: &[[Self; G]],
        likelihoods_y: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::aarch64::float64x2_t;
        calculate_mixture_component_matrix::<float64x2_t, __m256d, A, G>(all_joint_genotypes, allele_frequencies, likelihoods_x, likelihoods_y, p_mat)
    }

    fn mixture_component_matrix2(
        allele_frequencies: &[[Self; 2]],
        likelihoods_x: &[[Self; 3]],
        likelihoods_y: &[[Self; 3]],
        p_mat: &mut [[Self; 9]],
    )
    {
        use std::arch::aarch64::float64x2_t;
        calculate_mixture_component_matrix2::<Self, float64x2_t>(allele_frequencies, likelihoods_x, likelihoods_y, p_mat)
    }
}

impl MixtureMatrix for f64 {
    fn mixture_component_matrix<const A: usize, const G: usize>(
        all_joint_genotypes: &MIis<G>,
        allele_frequencies: &[[Self; A]],
        likelihoods_x: &[[Self; G]],
        likelihoods_y: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    ) {
        calculate_mixture_component_matrix::<Self, f64, A, G>(all_joint_genotypes, allele_frequencies, likelihoods_x, likelihoods_y, p_mat)
    }

    fn mixture_component_matrix2(
        allele_frequencies: &[[Self; 2]],
        likelihoods_x: &[[Self; 3]],
        likelihoods_y: &[[Self; 3]],
        p_mat: &mut [[Self; 9]],
    ) {
        calculate_mixture_component_matrix2::<Self, f64>(allele_frequencies, likelihoods_x, likelihoods_y, p_mat)
    }
}

#[inline]
fn calculate_mixture_component_matrix<L: Lane, S: Arith + LoadStore<L>, const A: usize, const G: usize>(
    all_joint_genotypes: &MIis<G>,
    allele_frequencies: &[[L; A]],
    likelihoods_x: &[[L; G]],
    likelihoods_y: &[[L; G]],
    p_mat: &mut [[L; 9]],
) {
    let len = allele_frequencies.len();
    assert!(likelihoods_x.len() == len);
    assert!(likelihoods_y.len() == len);
    assert!(p_mat.len() == len);

    // p = lx * ly * M
    for locus in 0..len {
        let out = unsafe { p_mat.get_unchecked_mut(locus) };
        let like_x = unsafe { likelihoods_x.get_unchecked(locus) };
        let like_y = unsafe { likelihoods_y.get_unchecked(locus) };
        let af = unsafe { allele_frequencies.get_unchecked(locus) };

        // 9 registers
        let mut a: [S; 9] = std::array::from_fn(|_| S::zero());

        // G x G multiplications
        // G x G x 9 FMA

        for gx in 0..G {
            let lx = S::load(&like_x[gx]);
            for gy in 0..G {
                // For architectures with enough registers this will get hoisted
                // out of the loop
                let ly = S::load(&like_y[gy]);
                let g = lx.mul(ly);

                let ((i, j), (k, l), iis) = all_joint_genotypes[gx][gy];

                // This will always be safe due to the construction all_joint_genotypes
                let pi = S::load( unsafe { af.get_unchecked(i) });
                let pj = S::load( unsafe { af.get_unchecked(j) });
                let pk = S::load( unsafe { af.get_unchecked(k) });
                let pl = S::load( unsafe { af.get_unchecked(l) });

                accumulate_iis(&mut a, g, pi, pj, pk, pl, iis);
            }
        }

        for k in 0..9 {
            a[k].store(&mut out[k]);
        }
    }
}

#[inline]
fn calculate_mixture_component_matrix2<L: Lane, S: Arith + LoadStore<L>>(
    allele_frequencies: &[[L; 2]],
    likelihoods_x: &[[L; 3]],
    likelihoods_y: &[[L; 3]],
    p_mat: &mut [[L; 9]],
) {
    let len = allele_frequencies.len();
    assert!(likelihoods_x.len() == len);
    assert!(likelihoods_y.len() == len);
    assert!(p_mat.len() == len);

    // p = lx * ly * M
    for locus in 0..len {
        let out = unsafe { p_mat.get_unchecked_mut(locus) };
        let like_x = unsafe { likelihoods_x.get_unchecked(locus) };
        let like_y = unsafe { likelihoods_y.get_unchecked(locus) };
        let af = unsafe { allele_frequencies.get_unchecked(locus) };

        // 9 registers
        mixture_row2::<L, S>(out, like_x, like_y, af);
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
fn mixture_row2<L: Lane, S: Arith + LoadStore<L>>(out: &mut [L; 9], like_x: &[L; 3], like_y: &[L; 3], af: &[L; 2]) {
    // 15 registers
    let mut a: [S; 9] = std::array::from_fn(|_| S::zero());
    let lx: [S; 3] = std::array::from_fn(|i| S::load(&like_x[i]));
    let ly: [S; 3] = std::array::from_fn(|i| S::load(&like_y[i]));

    // 2
    let p = S::load(&af[0]);
    let q = S::load(&af[1]);

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

    for k in 0..9 {
        a[k].store(&mut out[k]);
    }
}

pub struct LanePartialVector<L: Lane, const G: usize> {
    pub lanes: Vec<[[L; 9]; G]>,
    pub rem: Vec<[[f64; 9]; G]>,
}

impl<L: Lane, const G: usize> LanePartialVector<L, G> {
    pub fn new(len: usize) -> Self {
        let lanes = len / L::N;
        let rem = len % L::N;

        let lane_buf = vec![[[L::default(); 9]; G]; lanes];
        let rem_buf = vec![[[0.0; 9]; G]; rem];

        LanePartialVector {
            lanes: lane_buf,
            rem: rem_buf,
        }
    }
}

pub struct LaneCompleteVector<L: Lane> {
    lanes: Vec<[L; 9]>,
    rem: Vec<[f64; 9]>,
    num_lanes2: usize,
    num_lanes3: usize,
    num_rem2: usize,
    num_rem3: usize,
}

impl<L: Lane> LaneCompleteVector<L> {
    pub fn new(n2: usize, n3: usize, n4: usize) -> Self {
        let num_lanes2 = n2 / L::N;
        let num_lanes3 = n3 / L::N;
        let num_lanes4 = n4 / L::N;
        // In theory one could combine this into some larger blocks...
        // But not worth it
        let num_rem2 = n2 % L::N;
        let num_rem3 = n3 % L::N;
        let num_rem4 = n4 % L::N;

        let num_lanes = num_lanes2 + num_lanes3 + num_lanes4;
        let num_rem = num_rem2 + num_rem3 + num_rem4;

        let lanes = vec![[L::default(); 9]; num_lanes];
        let rem = vec![[0.0; 9]; num_rem];

        LaneCompleteVector {
            lanes,
            rem,
            num_lanes2,
            num_lanes3,
            num_rem2,
            num_rem3,
        }
    }

    pub fn lanes(&self) -> &[[L; 9]] {
        &self.lanes
    }

    pub fn rem(&self) -> &[[f64; 9]] {
        &self.rem
    }

    pub fn lanes2(&mut self) -> &mut [[L; 9]] {
        &mut self.lanes[0..self.num_lanes2]
    }

    pub fn rem2(&mut self) -> &mut [[f64; 9]] {
        &mut self.rem[0..self.num_rem2]
    }

    pub fn lanes3(&mut self) -> &mut [[L; 9]] {
        &mut self.lanes[self.num_lanes2..(self.num_lanes2 + self.num_lanes3)]
    }

    pub fn rem3(&mut self) -> &mut [[f64; 9]] {
        &mut self.rem[self.num_rem2..(self.num_rem2 + self.num_rem3)]
    }

    pub fn lanes4(&mut self) -> &mut [[L; 9]] {
        &mut self.lanes[(self.num_lanes2 + self.num_lanes3)..]
    }

    pub fn rem4(&mut self) -> &mut [[f64; 9]] {
        &mut self.rem[(self.num_rem2 + self.num_rem3)..]
    }
}
