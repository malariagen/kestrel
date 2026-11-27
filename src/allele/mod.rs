mod grad_hess;
mod objective;

use std::num::NonZeroU64;

use lockfree_progress_bar::{ProgressBar, ProgressHandle};

use ndarray::Array2;
use paralight::{
    iter::{
        ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelRefSource, ParallelIteratorExt,
        ZipableSource,
    },
    threads::ThreadPool,
};

use crate::{
    algebra::{Matrix, Vector, sum_to_one}, allele::{grad_hess::AlleleGradHess, objective::AlleleObjective}, lanevector::LaneVector, simd::Lane, sqp::{self, GradHess, Objective, Tuneables}, vcf::{AlleleFrequencies, GenotypeLikelihoods},
};

struct AlleleFrequencyData<'a, L: Lane, const G: usize> {
    lv: &'a LaneVector<L, G>
}

impl<L: AlleleObjective<A, G>, const A: usize, const G: usize> Objective<A> for AlleleFrequencyData<'_, L, G> {
    fn objective(&self, x: &Vector<A>, eps: f64) -> f64 {
        L::vectorize(#[inline(always)] || L::compute_objective(self.lv, &x, eps))
    }
}

impl<L: AlleleGradHess<A, G>, const A: usize, const G: usize> GradHess<A> for AlleleFrequencyData<'_, L, G> {
    fn grad_hess(&self, x: &Vector<A>, eps: f64) -> (Vector<A>, Matrix<A>) {
        L::vectorize(#[inline(always)] || L::compute_grad_hess(self.lv, &x, eps))
    }
}

pub fn calculate_allele_frequencies<L>(
    genotype_likelihoods: &GenotypeLikelihoods,
    thread_pool: &mut ThreadPool,
    iterations: NonZeroU64,
    maf: f64,
) -> AlleleFrequencies
// Fun fun fun
where
    L: AlleleObjective<2, 3>
        + AlleleGradHess<2, 3>
        + AlleleObjective<3, 6>
        + AlleleGradHess<3, 6>
        + AlleleObjective<4, 10>
        + AlleleGradHess<4, 10>,
{
    let num_variants = genotype_likelihoods.variants;
    let num_samples = genotype_likelihoods.samples;

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

    let af2 = calc_frequencies::<L, 2, 3>(&genotype_likelihoods.gl2, thread_pool, &handle, maf, iterations);
    let af3 = calc_frequencies::<L, 3, 6>(&genotype_likelihoods.gl3, thread_pool, &handle, maf, iterations);
    let af4 = calc_frequencies::<L, 4, 10>(&genotype_likelihoods.gl4, thread_pool, &handle, maf, iterations);

    bar.done();

    AlleleFrequencies { af2, af3, af4 }
}

fn calc_frequencies<L, const A: usize, const G: usize>(
    gl: &Array2<[f64; G]>,
    thread_pool: &mut ThreadPool,
    handle: &ProgressHandle,
    maf: f64,
    iterations: NonZeroU64,
) -> Vec<[f64; A]>
where
    L: AlleleObjective<A, G> + AlleleGradHess<A, G>,
{
    let num_samples = gl.dim().1;

    // TODO replace this array with a better type
    let gl: Vec<_> = gl.outer_iter().collect();

    let tune = Tuneables::new(iterations, NonZeroU64::new(A as u64 + 1).unwrap());

    let mut af = vec![[0.0; A]; gl.len()];
    (af.par_iter_mut(), gl.par_iter())
        .zip_eq()
        .with_thread_pool(thread_pool)
        .for_each_init(
            || LaneVector::<L, G>::new(num_samples),
            |buffer, (out, variant_likelihood)| {
                let sample_likelihoods = variant_likelihood.as_slice().unwrap();

                buffer.fill_from_slice(sample_likelihoods);

                let x0 = [1.0f64 / A as f64; A];

                let data = AlleleFrequencyData { lv: buffer };

                let (_, mut x, iter, _) = sqp::solve_sqp(&data, &x0, &tune);

                if iter >= tune.sqp_max_iter.get() {
                    log::warn!(
                        "No convergence for allele frequencies, max iterations {} exceeded",
                        tune.sqp_max_iter
                    );
                }

                // More rigourous test is LRT
                for i in 0..A {
                    if x[i] > 0.0 && x[i] < maf {
                        x[i] = 0.0;
                    }
                }

                let x = sum_to_one(&x);

                *out = x;

                handle.inc();
            },
        );
    af
}
