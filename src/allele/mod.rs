mod grad_hess;
mod objective;

use std::num::NonZeroU64;

use lockfree_progress_bar::ProgressBar;
use ndarray::{Array1, ArrayView2};

use paralight::{
    iter::{
        ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelRefSource, ParallelIteratorExt,
        ZipableSource,
    },
    threads::ThreadPool,
};

use crate::{
    algebra::{Vector, sum_to_one},
    allele::{grad_hess::AlleleGradHess, objective::AlleleObjective},
    lanevector::LaneVector,
    sqp::{self, Tuneables},
};

pub fn calculate_allele_frequencies<L: AlleleGradHess + AlleleObjective>(
    likelihoods: ArrayView2<[f64; 10]>,
    thread_pool: &mut ThreadPool,
    iterations: NonZeroU64,
    maf: f64,
) -> Array1<[f64; 4]> {
    let num_variants = likelihoods.shape()[0];
    let num_samples = likelihoods.shape()[1];

    log::info!(
        "Calculating allele frequencies for {} sites using {} samples",
        num_variants,
        num_samples
    );

    let bar = ProgressBar::new(num_variants.try_into().unwrap())
        .with_eta()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    let tune = Tuneables::new(iterations, NonZeroU64::new(5).unwrap());

    // TODO maybe better way
    let var_likelihoods: Vec<_> = likelihoods.outer_iter().collect();

    let mut outputs = vec![[0.0; 4]; num_variants];
    (outputs.par_iter_mut(), var_likelihoods.par_iter())
        .zip_eq()
        .with_thread_pool(thread_pool)
        .for_each_init(
            || LaneVector::<L, 10>::new(num_samples),
            |buffer, (out, variant_likelihood)| {
                let sample_likelihoods = variant_likelihood.as_slice().unwrap();

                buffer.fill_from_slice(&sample_likelihoods);

                let x0 = [0.25; 4];

                let obj = |x: &Vector<4>, eps| L::compute_objective(buffer, &x, eps);
                let grad_hess = |x: &Vector<4>, eps| L::compute_grad_hess(buffer, &x, eps);

                let (_, mut x, iter) = sqp::solve_sqp(obj, grad_hess, &x0, &tune);

                if iter >= tune.sqp_max_iter.get() {
                    log::warn!(
                        "No convergence for allele frequencies, max iterations {} exceeded",
                        tune.sqp_max_iter
                    );
                }

                // More rigourous test is LRT
                for i in 0..4 {
                    if x[i] > 0.0 && x[i] < maf {
                        x[i] = 0.0;
                    }
                }

                let x = sum_to_one(&x);

                *out = x;

                handle.inc();
            },
        );

    bar.done();

    Array1::from_vec(outputs)
}
