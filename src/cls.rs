use crate::algebra::{Vector, add, scale_div, sub};
use crate::conditional::{Genotype, MIis};
use crate::iis::{self, Iis};
use ndarray::{Array4, ArrayView1, ArrayView2};

pub fn calculate_m_matrices<const A: usize, const G: usize>(
    all_joint_genotypes: &MIis<G>,
    allele_frequencies: &[[f64; A]],
) -> Vec<[[Vector<9>; G]; G]> {
    allele_frequencies.iter().map(|freq| {
        std::array::from_fn(|g1| {
            std::array::from_fn(|g2| {
                let ((i, j), (k, l), iis_mode) = all_joint_genotypes[g1][g2];
                let pi = freq[i];
                let pj = freq[j];
                let pk = freq[k];
                let pl = freq[l];

                iis::conditional_probabilities(pi, pj, pk, pl, iis_mode)
            })
        })
    }).collect()
}

pub fn calculate_quadratic_c<const G: usize>(
    m_matrices: &[[[Vector<9>; G]; G]],
    genotypes_x: ArrayView1<u8>,
    genotypes_y: ArrayView1<u8>,
) -> Vector<9> {
    let num_v = m_matrices.len();
    assert!(num_v == genotypes_x.len());
    assert!(num_v == genotypes_y.len());

    let mut c = [0.0; 9];

    for v in 0..num_v {
        let gx = unsafe { genotypes_x.uget(v) };
        let gy = unsafe { genotypes_y.uget(v) };
        let m = unsafe { m_matrices.get_unchecked(v) };

        let row = m[usize::from(*gx)][usize::from(*gy)];
        c = add(&c, &row);
    }

    c
}
