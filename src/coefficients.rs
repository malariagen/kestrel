use std::num::NonZeroU64;

use itertools::Itertools;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use paralight::{
    iter::{
        ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelRefSource, ParallelIteratorExt,
        ZipableSource,
    },
    threads::ThreadPool,
};

use lockfree_progress_bar::ProgressBar;

use crate::{
    algebra::{Vector, dot, mul}, arith::{Arith, Lane, LoadStore, simd::Simd}, cls, conditional::{self, M}, jacquard::{grad_hess::{self, JacquardGradHess}, objective::{self, JacquardObjective}}, lanearray2::{LaneArray2, convert_to_lanes}, lanematrix::{LaneM4, M4}, lanevector::LaneVector, mixture::MixtureMatrix, quadratic::{self, ClsQuadratic}, sqp::{self, Tuneables},
};

pub fn calculate_allele_frequencies(genotypes: ArrayView3<u8>) -> Array2<f64> {
    let num_v = genotypes.shape()[0];
    let num_s = genotypes.shape()[1];
    let num_h = genotypes.shape()[2];

    let max_a: usize = (genotypes.iter().max().unwrap() + 1).into();

    let mut freq = Array2::<f64>::zeros((num_v, max_a));

    // TODO less bounds checks
    for v in 0..num_v {
        for s in 0..num_s {
            for h in 0..num_h {
                let allele = genotypes[(v, s, h)];
                freq[(v, usize::from(allele))] += 1.0;
            }
        }
    }

    for v in 0..num_v {
        let mut total = 0.0;
        for a in 0..max_a {
            total += freq[(v, a)];
        }

        if total > 0.0 {
            for a in 0..max_a {
                freq[(v, a)] /= total;
            }
        }
    }

    freq
}

fn reorder_genotypes(mut genotypes: ArrayView3<u8>) -> Array3<u8> {
    genotypes.swap_axes(0, 1);
    let mut genotypes = genotypes.as_standard_layout().into_owned();

    let (pairs, _) = genotypes.as_slice_mut().unwrap().as_chunks_mut::<2>();

    for pair in pairs {
        pair.sort_unstable();
    }

    genotypes
}

pub fn calculate_relatedness_coefficients_gl<L: JacquardGradHess + JacquardObjective + MixtureMatrix>(
    likelihoods: ArrayView3<f64>,
    allele_frequencies: ArrayView2<f64>,
    thread_pool: &mut ThreadPool,
    iterations: NonZeroU64,
) -> Vec<Output> {
    let num_v = allele_frequencies.shape()[0];

    // TODO calculate this across each locus to figure out how many alleles there are
    // Then we can condense the stacked matrix to make it smaller? Idk, complicated...

    // let m_matrices = conditional::calculate_m_matrices(allele_frequencies);
    let m_matrices = conditional::calculate_m_matrices2::<L>(allele_frequencies);

    let likelihoods = convert_to_lanes(likelihoods);

    // Only to analyze the data, before I make it faster...
    // let tmp = swapped.slice(s![0..16, .., ..]);
    // Use every sample
    // let tmp = swapped.view();

    let pairs: Vec<_> = likelihoods.iter_rows()
        .enumerate()
        .array_combinations_with_replacement()
        .collect();
    let mut outputs = vec![Output::default(); pairs.len()];

    log::info!(
        "Calculating Jacquard coefficients for {} pairs using {} sites",
        pairs.len(),
        num_v
    );

    let bar = ProgressBar::new(pairs.len().try_into().unwrap())
        .with_eta()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    let tune = Tuneables::new(iterations, NonZeroU64::new(10).unwrap());

    (outputs.par_iter_mut(), pairs.par_iter())
        .zip_eq()
        .with_thread_pool(thread_pool)
        .for_each_init(
            || LaneVector::<L, 9>::new(num_v),
            |p_mat, (out, [(x, likelihoods_x), (y, likelihoods_y)])| {
                let delta = if x == y {
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                } else {
                    [1.0 / 9.0; 9]
                };

                L::mixture_component_matrix(&m_matrices.lanes, likelihoods_x.0, likelihoods_y.0, &mut p_mat.lanes);
                f64::mixture_component_matrix(&m_matrices.rem, likelihoods_x.1, likelihoods_y.1, &mut p_mat.rem);

                let obj = |x: &Vector<9>, eps| L::compute_obj(p_mat, &x, eps);
                let grad_hess = |x: &Vector<9>, eps| L::compute_grad_hess(p_mat, &x, eps);
                let (f, delta, iters) = sqp::solve_sqp(obj, grad_hess, &delta, &tune);

                if iters >= tune.sqp_max_iter.get() {
                    log::warn!(
                        "No convergence for Jacquard coefficients, max iterations {} exceeded",
                        tune.sqp_max_iter
                    );
                }

                *out = Output {
                    x: *x,
                    y: *y,
                    jacquard: delta,
                    iters,
                    obj: f,
                };
                handle.inc();
            },
        );

    bar.done();

    outputs
}

#[derive(Clone, Copy)]
pub struct Output {
    pub x: usize,
    pub y: usize,
    pub jacquard: [f64; 9],
    pub obj: f64,
    pub iters: u64,
}

impl Default for Output {
    fn default() -> Self {
        Output {
            x: 0,
            y: 0,
            jacquard: [0.0; 9],
            iters: 0,
            obj: 0.0,
        }
    }
}

pub fn calculate_coefficients_gt<L : ClsQuadratic>(
    genotypes: ArrayView3<u8>,
    allele_frequencies: ArrayView2<f64>,
    thread_pool: &mut ThreadPool,
    iterations: NonZeroU64,
) -> Vec<Output> {
    let genotypes = reorder_genotypes(genotypes);

    // TODO calculate this across each locus to figure out how many alleles there are
    // Then we can condense the stacked matrix to make it smaller
    let num_a = allele_frequencies.shape()[1];

    let all_joint_genotypes = cls::calculate_all_joint_genotypes(num_a);
    let stacked_m = cls::calculate_stacked_m(&all_joint_genotypes, allele_frequencies);
    let lookup_table = cls::calculate_joint_genotype_lookup_table(&all_joint_genotypes, num_a);

    let mut lv = LaneVector::new(stacked_m.len());
    lv.fill_from_iter(stacked_m.iter().copied());

    let num_v = allele_frequencies.shape()[0];
    let quadratic_q = L::compute_quadratic(&lv, num_v);

    let pairs: Vec<[(usize, ArrayView2<u8>); 2]> = genotypes
        .outer_iter()
        .enumerate()
        .array_combinations_with_replacement()
        .collect();
    let mut output = vec![Output::default(); pairs.len()];

    log::info!(
        "Calculating Jacquard coefficients for {} pairs using {} variants",
        pairs.len(),
        num_v
    );

    let bar = ProgressBar::new(pairs.len().try_into().unwrap())
        .with_eta()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    // The SQP iterations don't matter here
    let tune = Tuneables::new(iterations, iterations);

    (output.par_iter_mut(), pairs.par_iter())
        .zip_eq()
        .with_thread_pool(thread_pool)
        .for_each(|(out, [(x, genotypes_x), (y, genotypes_y)])| {
            let delta = if x == y {
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
            } else {
                [1.0 / 9.0; 9]
            };
            let c = cls::calculate_quadratic_c(
                &all_joint_genotypes,
                &stacked_m,
                *genotypes_x,
                *genotypes_y,
                &lookup_table,
            );

            let (delta, iters) = sqp::solve_qp_active_set(&quadratic_q, &c, &delta, true, &tune);

            let obj = dot(&delta, &mul(&quadratic_q, &delta)) / 2.0 + dot(&c, &delta);

            if iters >= tune.qp_max_iter.get() {
                log::warn!(
                    "No convergence for Jacquard coefficients, max iterations {} exceeded",
                    tune.qp_max_iter
                );
            }

            *out = Output {
                x: *x,
                y: *y,
                jacquard: delta,
                iters,
                obj,
            };
            handle.inc();
        });

    bar.done();

    output
}

pub fn calculate_mixture_component_matrix_gl<L: Lane>(
    m_matrices: &[M],
    likelihoods_x: ArrayView2<f64>,
    likelihoods_y: ArrayView2<f64>,
    p_mat: &mut LaneVector<L, 9>,
) {
    let (chunks_x, rem_x) = likelihoods_x.as_slice().unwrap().as_chunks::<10>();
    let (chunks_y, rem_y) = likelihoods_y.as_slice().unwrap().as_chunks::<10>();

    assert!(rem_x.is_empty());
    assert!(rem_y.is_empty());

    assert_eq!(m_matrices.len(), chunks_x.len());
    assert_eq!(m_matrices.len(), chunks_y.len());

    // p = lx * ly * M
    // We have the number of samples

    let iter = chunks_x
        .iter()
        .zip(chunks_y.iter())
        .zip(m_matrices.iter())
        .map(|((like_x, like_y), m)| {
            // g^T M
            let mut p = [0.0f64; 9];
            for i in 0..10 {
                for j in 0..10 {
                    // This needs to match the order of the rows of the M matrix
                    // In calculate_all_joint_genotypes, we iterate over the second
                    // individual completely before doing the first
                    let g = like_x[i] * like_y[j];
                    for k in 0..9 {
                        p[k] = g.mul_add(m[i][j][k], p[k]);
                    }
                }
            }

            p
        });

    p_mat.fill_from_iter(iter);
}

#[cfg(test)]
mod test {
    use crate::coefficients::reorder_genotypes;
    use ndarray::array;

    #[test]
    fn test_reorder_genotypes() {
        let genotypes = array![[[1, 2], [1, 0], [4, 3]], [[3, 0], [1, 1], [1, 1]]];

        let expected_genotypes = array![[[1, 2], [0, 3]], [[1, 0], [1, 1]], [[3, 4], [1, 1]],];

        assert_eq!(reorder_genotypes(genotypes.view()), expected_genotypes);
    }
}
