use crate::algebra::{Vector, scale_div, sub};
use crate::iis;
use ndarray::{Array4, ArrayView2};

pub fn calculate_stacked_m(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    allele_frequencies: ArrayView2<f64>,
) -> Vec<Vector<9>> {
    let num_v = allele_frequencies.shape()[0];
    let num_g = all_joint_genotypes.len();

    let mut stacked_m = Vec::with_capacity(num_v * num_g);

    for v in 0..num_v {
        for g in 0..num_g {
            let ((i, j), (k, l), iis_mode) = all_joint_genotypes[g];
            let pi = allele_frequencies[[v, i]];
            let pj = allele_frequencies[[v, j]];
            let pk = allele_frequencies[[v, k]];
            let pl = allele_frequencies[[v, l]];

            let row =
                std::array::from_fn(|ibd_mode| iis::conditional_probability(pi, pj, pk, pl, iis_mode, ibd_mode + 1));

            stacked_m.push(row);
        }
    }

    stacked_m
}

pub fn calculate_quadratic_c(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    stacked_m: &[Vector<9>],
    genotypes_x: ArrayView2<u8>,
    genotypes_y: ArrayView2<u8>,
    lookup_table: &Array4<usize>,
) -> Vector<9> {
    let num_g = all_joint_genotypes.len();
    let num_v = genotypes_x.shape()[0];

    let mut c = [0.0; 9];

    let (iter_x, _) = genotypes_x.as_slice().unwrap().as_chunks::<2>();
    let (iter_y, _) = genotypes_y.as_slice().unwrap().as_chunks::<2>();

    for (locus, (geno_x, geno_y)) in iter_x.iter().zip(iter_y).enumerate() {
        let [i, j] = *geno_x;
        let [k, l] = *geno_y;

        let g = unsafe { lookup_table.uget((usize::from(i), usize::from(j), usize::from(k), usize::from(l))) };
        // let g = lookup_table[(i as usize, j as usize, k as usize, l as usize)];

        let row = unsafe { stacked_m.get_unchecked(locus.unchecked_mul(num_g).unchecked_add(*g)) };
        c = sub(&c, row);
    }

    scale_div(num_v as f64, &c)
}

pub fn calculate_all_joint_genotypes(num_a: usize) -> Vec<((usize, usize), (usize, usize), usize)> {
    let num_single_genotypes = (num_a * (num_a + 1)) / 2;
    let num_joint_genotypes = num_single_genotypes * num_single_genotypes;
    let mut joint_genotypes = Vec::with_capacity(num_joint_genotypes);

    // This is the order of genotypes in the VCF spec (triangular order)
    // (0, 0), (0, 1), (1, 1), etc.
    // Individual x
    for j in 0..num_a {
        for i in 0..=j {
            // Individual y
            for l in 0..num_a {
                for k in 0..=l {
                    let iis_mode = iis::calc_iis_mode(i, j, k, l);
                    joint_genotypes.push(((i, j), (k, l), iis_mode));
                }
            }
        }
    }

    joint_genotypes
}

pub fn calculate_joint_genotype_lookup_table(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    num_a: usize,
) -> Array4<usize> {
    let mut lookup = Array4::<usize>::zeros((num_a, num_a, num_a, num_a));

    for (g, ((i, j), (k, l), _)) in all_joint_genotypes.iter().enumerate() {
        lookup[(*i, *j, *k, *l)] = g;
    }

    lookup
}
