use ndarray::{Array2, Array4, ArrayRef2, ArrayView2};
use crate::iis;
use crate::util::{Matrix9, Matrix9xN, MatrixNx9, Vector9};

pub fn calculate_stacked_m(all_joint_genotypes: &[((usize, usize), (usize, usize), i8)], allele_frequencies: &ArrayRef2<f64>) -> MatrixNx9<f64> {
    let num_v = allele_frequencies.shape()[0];
    let num_g = all_joint_genotypes.len();

    let mut stacked_m = MatrixNx9::<f64>::zeros(num_v * num_g);

    // nalgebra stores its matrices in column-major format, so one might think that
    // calculating the transpose of the matrix is faster since the writes would
    // be adjacent in memory. Turns out it is not.
    for v in 0..num_v {
        for g in 0..num_g {
            let ((i, j), (k, l), iis_mode) = all_joint_genotypes[g];
            let pi = allele_frequencies[[v, i]];
            let pj = allele_frequencies[[v, j]];
            let pk = allele_frequencies[[v, k]];
            let pl = allele_frequencies[[v, l]];

            for ibd_mode in 0..9 {
                stacked_m[(v * num_g + g, ibd_mode)] = iis::conditional_probability(pi, pj, pk, pl, iis_mode, ibd_mode as i8 + 1);
            }
        }
    }

    stacked_m
}

pub fn calculate_quadratic_q_mat(stacked_m: &MatrixNx9<f64>, num_v: usize) -> Matrix9<f64> {
    let mut q = Matrix9::<f64>::zeros();

    let scale = 1.0 / num_v as f64;

    q.gemm_tr(scale, &stacked_m, &stacked_m, 0.0);

    q
}

pub fn calculate_quadratic_c(all_joint_genotypes: &[((i8, i8), (i8, i8), i8)], stacked_m: &MatrixNx9<f64>, genotypes_x: ArrayView2<i8>,
genotypes_y: ArrayView2<i8>, allele_frequencies: &ArrayRef2<f64>) -> Vector9<f64> {

    let num_g = all_joint_genotypes.len();
    let num_v = genotypes_x.shape()[0];

    let mut c = Vector9::<f64>::zeros();

    let iter_x = genotypes_x.as_slice().unwrap().chunks_exact(2);
    let iter_y = genotypes_y.as_slice().unwrap().chunks_exact(2);

    for (locus, (geno_x, geno_y)) in iter_x.zip(iter_y).enumerate() {

        let gx = (geno_x[0], geno_x[1]);
        let gy = (geno_y[0], geno_y[1]);

        let iis_mode = iis::calc_iis_mode(gx.0, gx.1, gy.0, gy.1);

        let pi = allele_frequencies[[locus, gx.0 as usize]];
        let pj = allele_frequencies[[locus, gx.1 as usize]];
        let pk = allele_frequencies[[locus, gy.0 as usize]];
        let pl = allele_frequencies[[locus, gy.1 as usize]];

        for ibd in 0..9 {
            c[ibd] -= iis::conditional_probability(pi, pj, pk, pl, iis_mode, ibd as i8 + 1)
        }

        // for (g, (ogx, ogy, _)) in all_joint_genotypes.iter().enumerate() {
        //     // TODO if some of the genotype data is missing this won't match
        //     // so we need to update stacked_m to account for that
        //     // Also, nalgebra is column-major. Perhaps look into that
        //     if gx == *ogx && gy == *ogy {
        //         c -= stacked_m.row(locus * num_g + g).transpose();
        //         break;
        //     }
        // }
    }

    c / (num_v as f64)
}

pub fn calculate_quadratic_c_t(all_joint_genotypes: &[((usize, usize), (usize, usize), i8)], stacked_m_t: &Matrix9xN<f64>, genotypes_x: ArrayView2<i8>,
genotypes_y: ArrayView2<i8>, lookup_table: &Array4<usize>) -> Vector9<f64> {

    let num_g = all_joint_genotypes.len();
    let num_v = genotypes_x.shape()[0];

    let mut c = Vector9::<f64>::zeros();

    // let (chunks_x, _) = genotypes_x.as_slice().unwrap().as_chunks::<2>();
    // let (chunks_y, _) = genotypes_y.as_slice().unwrap().as_chunks::<2>();

    let iter_x = genotypes_x.as_slice().unwrap().chunks_exact(2);
    let iter_y = genotypes_y.as_slice().unwrap().chunks_exact(2);

    // for (locus, (geno_x, geno_y)) in chunks_x.iter().copied().zip(chunks_y.iter().copied()).enumerate() {
    for (locus, (geno_x, geno_y)) in iter_x.zip(iter_y).enumerate() {
        // let [i, j] = geno_x;
        // let [k, l] = geno_y;
        let (i, j) = (geno_x[0], geno_x[1]);
        let (k, l) = (geno_y[0], geno_y[1]);

        // TODO do a check here for missing data
        // if i < 0 || j < 0 || k < 0 || l < 0 {
        //     continue;
        // }

        // let g = unsafe { lookup_table.uget((i as usize, j as usize, k as usize, l as usize)) };
        let g = lookup_table[(i as usize, j as usize, k as usize, l as usize)];

        c -= stacked_m_t.column(locus * num_g + g);

        // We need a map from gx, gy to the g index, that is faster than a linear search
        // for (g, (ogx, ogy, _)) in all_joint_genotypes.iter().enumerate() {
        //     // TODO if some of the genotype data is missing this won't match
        //     // so we need to update stacked_m to account for that
        //     if gx == *ogx && gy == *ogy {
        //         c -= stacked_m_t.column(locus * num_g + g);
        //         break;
        //     }
        // }
    }

    c / (num_v as f64)
}

pub fn calculate_all_joint_genotypes(num_a: usize) -> Vec<((usize, usize), (usize, usize), i8)> {
    // let num_single_genotypes = ((A as usize)  * (A as usize + 1)) / 2;
    // let num_joint_genotypes = num_single_genotypes * num_single_genotypes;
    // let mut joint_genotypes = Vec::with_capacity(num_joint_genotypes);
    let mut joint_genotypes = Vec::new();

    for i in 0..num_a {
        for j in i..num_a {
            for k in 0..num_a {
                for l in k..num_a {
                    let iis_mode = iis::calc_iis_mode(i as i8, j as i8, k as i8, l as i8);
                    joint_genotypes.push(((i, j), (k, l), iis_mode));
                }
            }
        }
    }

    joint_genotypes
}

pub fn calculate_joint_genotype_lookup_table(all_joint_genotypes: &[((usize, usize), (usize, usize), i8)], num_a: usize) -> Array4<usize> {

    let mut lookup = Array4::<usize>::zeros((num_a, num_a, num_a, num_a));

    for (g, ((i, j), (k, l), _)) in all_joint_genotypes.iter().enumerate() {
        lookup[(*i, *j, *k, *l)] = g;
    }

    lookup
}