use ndarray::{ArrayView1};

use crate::{simd::Lane, iis, lanematrix::LaneM};

type Genotype = (usize, usize);
type MIis<const G: usize> = [[(Genotype, Genotype, usize); G]; G];

pub fn calculate_joint_genotype_iis<const A: usize, const G: usize>() -> MIis<G> {
    let mut table = [[((0, 0), (0, 0), 0); G]; G];
    // This is the order of genotypes in the VCF spec (triangular order)
    // (0, 0), (0, 1), (1, 1), etc.
    // Individual x
    let mut x = 0;
    for j in 0..A {
        for i in 0..=j {
            // Individual y
            let mut y = 0;
            for l in 0..A {
                for k in 0..=l {
                    let iis_mode = iis::calc_iis_mode(i, j, k, l);
                    table[x][y] = ((i, j), (k, l), iis_mode);
                    y += 1;
                }
            }
            x += 1;
        }
    }

    table
}

pub fn calculate_conditional_probability_arrays<L: Lane, const A: usize, const G: usize>(allele_frequencies: ArrayView1<[f64; A]>) -> LaneM<L, G> {
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

                std::array::from_fn(|ibd_mode| iis::conditional_probability(pi, pj, pk, pl, iis_mode, ibd_mode + 1))
            })
        })
    });

    let mut lanes = LaneM::new(num_v);
    lanes.fill_from_iter(iter);

    lanes
}
