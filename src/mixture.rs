use crate::{
    conditional::{IIS_2, MIis}, iis, lanematrix::M, simd::{Arith, Lane, LoadStore},
};

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::Avx2Lane;
#[cfg(target_arch = "x86_64")]
use crate::simd::avx512::Avx512Lane;
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::NeonLane;

pub trait MixtureMatrix: Lane {
    fn partial_mixture_component_matrix<const G: usize>(
        m_arrays: &[M<Self, G>],
        likelihoods: &[[Self; G]],
        partial_mat: &mut [[[Self; 9]; G]],
    );
    fn complete_mixture_component_matrix<const G: usize>(
        partial_mat: &[[[Self; 9]; G]],
        likelihoods: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    );

    fn mixture_component_matrix<const A: usize, const G: usize>(
        all_joint_genotypes: &MIis<G>,
        allele_frequencies: &[[Self; A]],
        likelihoods_x: &[[Self; G]],
        likelihoods_y: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    );

    fn mixture_component_matrix2(
        allele_frequencies: &[[Self; 2]],
        likelihoods_x: &[[Self; 3]],
        likelihoods_y: &[[Self; 3]],
        p_mat: &mut [[Self; 9]],
    );
}

pub fn test_asm(
        all_joint_genotypes: &MIis<3>,
        allele_frequencies: &[[Avx512Lane; 2]],
        likelihoods_x: &[[Avx512Lane; 3]],
        likelihoods_y: &[[Avx512Lane; 3]],
        p_mat: &mut [[Avx512Lane; 9]],
) {
    use std::arch::x86_64::__m512d;
    return calculate_mixture_component_matrix::<Avx512Lane, __m512d, 2, 3>(all_joint_genotypes, allele_frequencies, likelihoods_x, likelihoods_y, p_mat);
}

pub fn test_asm_bi(
        allele_frequencies: &[[Avx512Lane; 2]],
        likelihoods_x: &[[Avx512Lane; 3]],
        likelihoods_y: &[[Avx512Lane; 3]],
        p_mat: &mut [[Avx512Lane; 9]],
) {
    use std::arch::x86_64::__m512d;
    return calculate_mixture_component_matrix2::<Avx512Lane, __m512d>(allele_frequencies, likelihoods_x, likelihoods_y, p_mat);
}

#[cfg(target_arch = "x86_64")]
impl MixtureMatrix for Avx512Lane {
    fn partial_mixture_component_matrix<const G: usize>(
        m_arrays: &[M<Self, G>],
        likelihoods: &[[Self; G]],
        partial_mat: &mut [[[Self; 9]; G]],
    ) {
        use std::arch::x86_64::__m512d;
        return calculate_partial_mixture_component_matrix::<Avx512Lane, __m512d, G>(
            m_arrays,
            likelihoods,
            partial_mat,
        );
    }

    fn complete_mixture_component_matrix<const G: usize>(
        partial_mat: &[[[Self; 9]; G]],
        likelihoods: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::x86_64::__m512d;
        return calculate_complete_mixture_component_matrix::<Avx512Lane, __m512d, G>(partial_mat, likelihoods, p_mat);
    }

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
    fn partial_mixture_component_matrix<const G: usize>(
        m_arrays: &[M<Self, G>],
        likelihoods: &[[Self; G]],
        partial_mat: &mut [[[Self; 9]; G]],
    ) {
        use std::arch::x86_64::__m256d;
        return calculate_partial_mixture_component_matrix::<Avx2Lane, __m256d, G>(m_arrays, likelihoods, partial_mat);
    }

    fn complete_mixture_component_matrix<const G: usize>(
        partial_mat: &[[[Self; 9]; G]],
        likelihoods: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::x86_64::__m256d;
        return calculate_complete_mixture_component_matrix::<Avx2Lane, __m256d, G>(partial_mat, likelihoods, p_mat);
    }

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
        unimplemented!()
    }
}

#[cfg(target_arch = "aarch64")]
impl MixtureMatrix for NeonLane {
    fn partial_mixture_component_matrix<const G: usize>(
        m_arrays: &[M<Self, G>],
        likelihoods: &[[Self; G]],
        partial_mat: &mut [[[Self; 9]; G]],
    ) {
        use std::arch::aarch64::float64x2_t;
        return calculate_partial_mixture_component_matrix::<NeonLane, float64x2_t, G>(
            m_arrays,
            likelihoods,
            partial_mat,
        );
    }

    fn complete_mixture_component_matrix<const G: usize>(
        partial_mat: &[[[Self; 9]; G]],
        likelihoods: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::aarch64::float64x2_t;
        return calculate_complete_mixture_component_matrix::<NeonLane, float64x2_t, G>(
            partial_mat,
            likelihoods,
            p_mat,
        );
    }

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
}

impl MixtureMatrix for f64 {
    fn partial_mixture_component_matrix<const G: usize>(
        m_arrays: &[M<Self, G>],
        likelihoods: &[[Self; G]],
        partial_mat: &mut [[[Self; 9]; G]],
    ) {
        return calculate_partial_mixture_component_matrix::<f64, f64, G>(m_arrays, likelihoods, partial_mat);
    }

    fn complete_mixture_component_matrix<const G: usize>(
        partial_mat: &[[[Self; 9]; G]],
        likelihoods: &[[Self; G]],
        p_mat: &mut [[Self; 9]],
    ) {
        return calculate_complete_mixture_component_matrix::<f64, f64, G>(partial_mat, likelihoods, p_mat);
    }

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

#[inline(always)]
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
        let mut p: [S; 9] = std::array::from_fn(|_| S::zero());

        for gx in 0..G {
            let lx = S::load(&like_x[gx]);
            for gy in 0..G {
                // For architectures with enough registers this will get hoisted
                // out of the loop
                let ly = S::load(&like_y[gy]);
                let g = lx.mul(ly);

                let ((i, j), (k, l), iis_mode) = all_joint_genotypes[gx][gy];

                // This will always be safe due to the construction all_joint_genotypes
                let pi = S::load( unsafe { af.get_unchecked(i) });
                let pj = S::load( unsafe { af.get_unchecked(j) });
                let pk = S::load( unsafe { af.get_unchecked(k) });
                let pl = S::load( unsafe { af.get_unchecked(l) });

                let m_row = iis::conditional_probabilities(pi, pj, pk, pl, iis_mode);
                for a in 0..9 {
                    p[a] = g.fma(m_row[a], p[a]);
                }
            }
        }

        for k in 0..9 {
            p[k].store(&mut out[k]);
        }
    }
}

#[inline(always)]
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
        mixture_row2(out, like_x, like_y, af);
    }
}

fn mixture_row2<S: Arith + LoadStore<L>>(out: &mut [L; 9], like_x: &[L; 3], like_y: &[L; 3], af: &[L; 2]) {

    let mut p_row: [S; 9] = std::array::from_fn(|_| S::zero());
    let lx = std::array::from_fn(|i| S::load&(like_x));
    let ly = std::array::from_fn(|i| S::load&(like_y));

    let p = S::load(&af[0]);
    let q = S::load(&af[1]);

    // genotypes - gl vcf index - iis mode
    // (0, 0) (0, 0) - 0, 0 - M1

    for gx in 0..3 {
        let lx = S::load(&like_x[gx]);
        for gy in 0..3 {
            // For architectures with enough registers this will get hoisted
            // out of the loop
            let ly = S::load(&like_y[gy]);
            let g = lx.mul(ly);

            let ((i, j), (k, l), iis_mode) = IIS_2[gx][gy];

            // This will always be safe due to the construction of IIS_2
            let pi = S::load( unsafe { af.get_unchecked(i) });
            let pj = S::load( unsafe { af.get_unchecked(j) });
            let pk = S::load( unsafe { af.get_unchecked(k) });
            let pl = S::load( unsafe { af.get_unchecked(l) });

            let m_row = iis::conditional_probabilities(pi, pj, pk, pl, iis_mode);
            for a in 0..9 {
                p_row[a] = g.fma(m_row[a], p_row[a]);
            }
        }
    }

    for k in 0..9 {
        p_row[k].store(&mut out[k]);
    }
}

// We have K x G x 9 x G
// And K x G
// Need to turn it into K x G x 9
#[inline]
fn calculate_partial_mixture_component_matrix<L: Lane, S: Arith + LoadStore<L>, const G: usize>(
    m_arrays: &[M<L, G>],
    likelihoods: &[[L; G]],
    partial_mat: &mut [[[L; 9]; G]],
) {
    let len = m_arrays.len();
    assert!(likelihoods.len() == len);
    assert!(partial_mat.len() == len);

    for l in 0..len {
        let out = unsafe { partial_mat.get_unchecked_mut(l) };
        let like = unsafe { likelihoods.get_unchecked(l) };
        let m = unsafe { m_arrays.get_unchecked(l) };

        for j in 0..G {
            // 9 registers
            let mut p: [S; 9] = std::array::from_fn(|_| S::zero());

            for i in 0..G {
                // This will get hoisted outside the loop
                let li = S::load(&like[i]);

                for k in 0..9 {
                    p[k] = li.fma(S::load(&m[j][i][k]), p[k]);
                }
            }

            for k in 0..9 {
                p[k].store(&mut out[j][k]);
            }
        }
    }
}

#[inline]
fn calculate_complete_mixture_component_matrix<L: Lane, S: Arith + LoadStore<L>, const G: usize>(
    partial_mat: &[[[L; 9]; G]],
    likelihoods: &[[L; G]],
    p_mat: &mut [[L; 9]],
) {
    let len = partial_mat.len();
    assert!(likelihoods.len() == len);
    assert!(p_mat.len() == len);

    for l in 0..len {
        let out = unsafe { p_mat.get_unchecked_mut(l) };
        let like = unsafe { likelihoods.get_unchecked(l) };
        let partial = unsafe { partial_mat.get_unchecked(l) };

        let mut p: [S; 9] = std::array::from_fn(|_| S::zero());

        for j in 0..G {
            let lj = S::load(&like[j]);

            for k in 0..9 {
                p[k] = lj.fma(S::load(&partial[j][k]), p[k]);
            }
        }

        for k in 0..9 {
            p[k].store(&mut out[k]);
        }
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
