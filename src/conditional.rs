use ndarray::ArrayRef2;

use crate::iis;

pub type M = [[[f64; 9]; 10]; 10];
type G = (usize, usize);
type MTable = [[(G, G, usize); 10]; 10];

pub fn calculate_joint_genotype_table() -> MTable {
    let mut table = [[((0, 0), (0, 0), 0); 10]; 10];
    // This is the order of genotypes in the VCF spec (triangular order)
    // (0, 0), (0, 1), (1, 1), etc.
    // Individual x
    let mut x = 0;
    for j in 0..4 {
        for i in 0..=j {
            // Individual y
            let mut y = 0;
            for l in 0..4 {
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

pub fn calculate_m_matrices(allele_frequencies: &ArrayRef2<f64>) -> Vec<M> {
    let num_a = allele_frequencies.shape()[1];
    assert_eq!(num_a, 4);

    let all_joint_genotypes = calculate_joint_genotype_table();

    let matrices = allele_frequencies
        .outer_iter()
        .map(|site_p| {
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
        })
        .collect::<Vec<_>>();

    matrices
}
