use crate::{
    arith::{Arith, Lane, LoadStore},
    lanematrix::M4,
};

#[cfg(target_arch = "x86_64")]
use crate::arith::avx2::Avx2Lane;
#[cfg(target_arch = "x86_64")]
use crate::arith::avx512::Avx512Lane;
#[cfg(target_arch = "aarch64")]
use crate::arith::neon::NeonLane;

pub trait MixtureMatrix: Lane {
    fn mixture_component_matrix(
        m_matrices: &[M4<Self>],
        likelihoods_x: &[[Self; 10]],
        likelihoods_y: &[[Self; 10]],
        p_mat: &mut [[Self; 9]],
    );
}

pub fn test_asm(
    m_matrices: &[M4<Avx512Lane>],
    likelihoods_x: &[[Avx512Lane; 10]],
    likelihoods_y: &[[Avx512Lane; 10]],
    p_mat: &mut [[Avx512Lane; 9]],
) {
    use std::arch::x86_64::__m512d;
    return calculate_mixture_component_matrix::<Avx512Lane, __m512d>(m_matrices, likelihoods_x, likelihoods_y, p_mat);
}

#[cfg(target_arch = "x86_64")]
impl MixtureMatrix for Avx512Lane {
    fn mixture_component_matrix(
        m_matrices: &[M4<Self>],
        likelihoods_x: &[[Self; 10]],
        likelihoods_y: &[[Self; 10]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::x86_64::__m512d;
        return calculate_mixture_component_matrix::<Avx512Lane, __m512d>(m_matrices, likelihoods_x, likelihoods_y, p_mat);
    }
}

#[cfg(target_arch = "x86_64")]
impl MixtureMatrix for Avx2Lane {
    #[allow(unused_variables, unreachable_code)]
    fn mixture_component_matrix(
        m_matrices: &[M4<Self>],
        likelihoods_x: &[[Self; 10]],
        likelihoods_y: &[[Self; 10]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::x86_64::__m256d;
        return calculate_mixture_component_matrix::<Avx2Lane, __m256d>(m_matrices, likelihoods_x, likelihoods_y, p_mat);
    }
}

#[cfg(target_arch = "aarch64")]
impl MixtureMatrix for NeonLane {
    #[allow(unused_variables, unreachable_code)]
    fn mixture_component_matrix(
        m_matrices: &[M4<Self>],
        likelihoods_x: &[[Self; 10]],
        likelihoods_y: &[[Self; 10]],
        p_mat: &mut [[Self; 9]],
    ) {
        use std::arch::aarch64::float64x2_t;
        return calculate_mixture_component_matrix::<NeonLane, float64x2_t>(
            m_matrices,
            likelihoods_x,
            likelihoods_y,
            p_mat,
        );
    }
}

impl MixtureMatrix for f64 {
    fn mixture_component_matrix(
        m_matrices: &[M4<Self>],
        likelihoods_x: &[[Self; 10]],
        likelihoods_y: &[[Self; 10]],
        p_mat: &mut [[Self; 9]],
    ) {
        calculate_mixture_component_matrix::<f64, f64>(m_matrices, likelihoods_x, likelihoods_y, p_mat)
    }
}

#[inline(always)]
fn calculate_mixture_component_matrix<L: Lane, S: Arith + LoadStore<L>>(
    m_matrices: &[M4<L>],
    likelihoods_x: &[[L; 10]],
    likelihoods_y: &[[L; 10]],
    p_mat: &mut [[L; 9]],
) {
    let len = m_matrices.len();
    assert!(likelihoods_x.len() == len);
    assert!(likelihoods_y.len() == len);
    assert!(p_mat.len() == len);

    // p = lx * ly * M
    for l in 0..len {
        let out = unsafe { p_mat.get_unchecked_mut(l) };
        let like_x = unsafe { likelihoods_x.get_unchecked(l) };
        let like_y = unsafe { likelihoods_y.get_unchecked(l) };
        let m = unsafe { m_matrices.get_unchecked(l) };

        // 9 registers
        let mut p: [S; 9] = std::array::from_fn(|_| S::zero());

        for i in 0..10 {
            let lx = S::load(&like_x[i]);
            for j in 0..10 {
                // For architectures with enough registers this will get hoisted
                // out of the loop
                let ly = S::load(&like_y[j]);
                // This needs to match the order of the rows of the M matrix
                // In calculate_all_joint_genotypes, we iterate over the second
                // individual completely before doing the first
                let g = lx.mul(ly);
                for k in 0..9 {
                    p[k] = g.fma(S::load(&m[i][j][k]), p[k]);
                }
            }
        }

        for k in 0..9 {
            p[k].store(&mut out[k]);
        }
    }
}
