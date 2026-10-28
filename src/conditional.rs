use ndarray::ArrayView1;

use crate::{iis::{self, Iis}, lanematrix::LaneM, simd::Lane};

pub type Genotype = (usize, usize);
pub type MIis<const G: usize> = [[(Genotype, Genotype, Iis); G]; G];

pub const IIS_2: MIis<3> = calculate_joint_genotype_iis::<2, 3>();
pub const IIS_3: MIis<6> = calculate_joint_genotype_iis::<3, 6>();
pub const IIS_4: MIis<10> = calculate_joint_genotype_iis::<4, 10>();

const fn calculate_joint_genotype_iis<const A: usize, const G: usize>() -> MIis<G> {
    assert!(G == A * (A + 1) / 2);
    // This is a default value that will always be overwritten
    let mut table = [[((0, 0), (0, 0), Iis::M1); G]; G];

    // We iterate in the order of genotypes in the VCF spec (triangular order)
    // (0, 0), (0, 1), (1, 1), etc.

    // Genotype for individual x
    let mut gx = 0;

    // const for loops not stable yet, so sad
    let mut j = 0;
    while j < A {
        let mut i = 0;
        while i <= j {
            // Genotype for individual y
            let mut gy = 0;
            let mut l = 0;
            while l < A {
                let mut k = 0;
                while k <= l {
                    let iis_mode = iis::calc_iis_mode(i, j, k, l);
                    table[gx][gy] = ((i, j), (k, l), iis_mode);
                    gy += 1;
                    k += 1;
                }
                l += 1;
            }
            gx += 1;
            i += 1;
            assert!(gy == G);
        }

        j += 1;
    }

    assert!(gx == G);

    table
}

pub fn calculate_conditional_probability_arrays<L: Lane, const A: usize, const G: usize>(
    allele_frequencies: ArrayView1<[f64; A]>,
) -> LaneM<L, G> {
    let num_v = allele_frequencies.dim();

    let all_joint_genotypes = calculate_joint_genotype_iis::<A, G>();

    let iter = allele_frequencies.iter().map(|site_p| {
        std::array::from_fn(|x| {
            std::array::from_fn(|y| {
                let ((i, j), (k, l), iis_mode) = all_joint_genotypes[x][y];
                let pi = site_p[i];
                let pj = site_p[j];
                let pk = site_p[k];
                let pl = site_p[l];

                iis::conditional_probabilities(pi, pj, pk, pl, iis_mode)
            })
        })
    });

    let mut lanes = LaneM::new(num_v);
    lanes.fill_from_iter(iter);

    lanes
}
