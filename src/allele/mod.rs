// pub mod allele;
mod objective;
mod grad_hess;

use std::{env::var, num::NonZeroUsize};

use lockfree_progress_bar::ProgressBar;
use ndarray::{Array2, Array3, Array4};

use paralight::{
    iter::{
        ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelRefSource, ParallelIteratorExt,
        ZipableSource,
    }, threads::{ThreadPool},
};

use crate::{
    algebra::{Vector, sum_to_one}, lanevector::LaneVector, sqp::{self, Tuneables},
};

pub fn calculate_allele_frequencies(likelihoods: &Array3<f64>, thread_pool: &mut ThreadPool) -> Array2<f64> {

    let num_variants = likelihoods.shape()[0];
    let num_samples = likelihoods.shape()[1];

    println!("Calculating allele frequencies for {} sites using {} samples", num_variants, num_samples);

    let bar = ProgressBar::new(num_variants.try_into().unwrap())
        .with_eta()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    // TODO maybe better way
    let var_likelihoods: Vec<_> = likelihoods.outer_iter().collect();

    let mut outputs = vec![[0.0; 4]; num_variants];
    (outputs.par_iter_mut(), var_likelihoods.par_iter())
        .zip_eq()
        .with_thread_pool(thread_pool)
        .for_each_init(
            || LaneVector::new(num_samples),
            |buffer, (out, variant_likelihood)| {

        let sample_likelihoods = variant_likelihood.as_slice().unwrap().as_chunks::<10>();
        assert!(sample_likelihoods.1.is_empty());
        buffer.fill_from_iter(sample_likelihoods.0.iter().copied());

        let x0 = [0.25; 4];

        let obj = |x: &Vector<4>, eps| objective::compute_objective(buffer, &x, eps);
        let grad_hess = |x: &Vector<4>, eps| grad_hess::compute_grad_hess(buffer, &x, eps);

        let tune = Tuneables::new();
        let (_, mut x, iter) = sqp::solve_sqp(obj, grad_hess, &x0, &tune);

        if iter >= tune.sqp_max_iter {
            println!("WARNING: no convergence for allele frequencies, max iterations {} exceeded", tune.sqp_max_iter);
        }

        // More rigourous test is LRT
        for i in 0..4 {
            if x[i] > 0.0 && x[i] < (1.0 / (2.0 * num_samples as f64)) {
                // println!("{}", x[i]);
                x[i] = 0.0;
            }
        }

        let x = sum_to_one(&x);

        *out = x;

        handle.inc();

    });

    bar.done();

    let mut af = Array2::zeros((num_variants, 4));
    let mut multi = 0;

    for (v, x) in outputs.iter().enumerate() {

        if x.iter().filter(|&i| *i > 0.0).count() >= 3 {
            multi += 1;
        }

        for i in 0..4 {
            af[[v, i]] = x[i];
        }
    }

    println!("Multi-allelic sites {multi}");

    af
}
