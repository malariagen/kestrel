// pub mod allele;
mod objective;
mod grad_hess;

use lockfree_progress_bar::ProgressBar;
use ndarray::{Array2, Array3, Array4};

use crate::{
    algebra::{Matrix, Vector, dot, mul, outer, scale_div}, lane::Lane8, lanevector::LaneVector, sqp::{self, Tuneables},
};

pub fn calculate_allele_frequencies(likelihoods: &Array3<f64>) -> Array2<f64> {

    let num_variants = likelihoods.shape()[0];
    let num_samples = likelihoods.shape()[1];

    let mut af = Array2::zeros((num_variants, 4));

    println!("Calculating allele frequencies for {} sites using {} samples", num_variants, num_samples);

    let bar = ProgressBar::new(num_variants.try_into().unwrap())
        .with_eta()
        // .disable_color()
        // .with_cpu_usage()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    let mut multi = 0;
    let mut buffer = LaneVector::new(num_samples);
    for (variant, variant_likelihood) in likelihoods.outer_iter().enumerate() {

        let sample_likelihoods = variant_likelihood.as_slice().unwrap().as_chunks::<10>();
        assert!(sample_likelihoods.1.is_empty());
        buffer.fill_from_iter(sample_likelihoods.0.iter().copied());

        let x0 = [0.25; 4];

        let obj = |x: &Vector<4>, eps| objective::compute_objective(&buffer, &x, eps);
        let grad_hess = |x: &Vector<4>, eps| grad_hess::compute_grad_hess(&buffer, &x, eps);

        let tune = Tuneables::new();
        let (_, x, iter) = sqp::solve_sqp(obj, grad_hess, &x0, &tune);

        if iter >= tune.sqp_max_iter {
            println!("WARNING: no convergence for allele frequencies, max iterations {} exceeded", tune.sqp_max_iter);
        }

        if x.iter().filter(|&i| *i > 0.0).count() >= 3 {
            multi += 1;
        }

        for i in 0..4 {
            af[[variant, i]] = x[i];
        }

        handle.inc();
    }

    bar.done();

    println!("Multi-allelic sites {multi}");

    af
}
