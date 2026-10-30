use std::num::NonZeroU64;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use paralight::{
    iter::{
        ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelSource, ParallelIteratorExt,
        ZipableSource,
    },
    threads::ThreadPool,
};

use lockfree_progress_bar::ProgressBar;

use crate::{
    algebra::{Vector, dot, mul}, cls::{self}, conditional::{IIS_3, IIS_4}, jacquard::{grad_hess::JacquardGradHess, objective::JacquardObjective}, lanearray2::convert_to_lanes, lanevector::{LaneVector, LaneVectorBuilder}, mixture::{MixtureMatrix}, quadratic::ClsQuadratic, sqp::{self, Tuneables}, wrapper::GlAf,
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
    glaf: &GlAf<L>,
    thread_pool: &mut ThreadPool,
    iterations: NonZeroU64,
) -> Vec<Vec<Output>> {
    let n2 = glaf.af2.len();
    let n3 = glaf.af3.len();
    let n4 = glaf.af4.len();

    let num_v = n2 + n3 + n4;

    let likelihoods2 = convert_to_lanes::<L, 3>(glaf.gl2.view());
    let likelihoods3 = convert_to_lanes::<L, 6>(glaf.gl3.view());
    let likelihoods4 = convert_to_lanes::<L, 10>(glaf.gl4.view());

    // let num_samples = likelihoods2.rows;
    let num_samples = 100;

    let num_pairs = (num_samples * (num_samples + 1)) / 2;

    log::info!(
        "Calculating Jacquard coefficients for {} pairs using {} sites",
        num_pairs,
        num_v
    );

    let bar = ProgressBar::new(num_pairs.try_into().unwrap())
        .with_eta()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    let tune = Tuneables::new(iterations, NonZeroU64::new(10).unwrap());

    let mut nested_outputs = vec![vec![]; num_samples];

    ((0..num_samples).into_par_iter(), nested_outputs.par_iter_mut())
        .zip_eq()
        .with_thread_pool(thread_pool)
        .for_each_init(
            || LaneVector::new(num_v),
            |buffer, (x, out)| {
                let like2_x = likelihoods2.sample(x);
                let like3_x = likelihoods3.sample(x);
                let like4_x = likelihoods4.sample(x);

                let iter = (x..num_samples).map(|y| {
                    let delta = if x == y {
                        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                    } else {
                        [1.0 / 9.0; 9]
                    };

                    let like2_y = likelihoods2.sample(y);
                    let like3_y = likelihoods3.sample(y);
                    let like4_y = likelihoods4.sample(y);

                    let mut builder = LaneVectorBuilder::new(buffer);

                    L::mixture_component_matrix2(glaf.af2.lanes(), like2_x.0, like2_y.0, &mut builder);
                    L::mixture_component_matrix(&IIS_3, glaf.af3.lanes(), like3_x.0, like3_y.0, &mut builder);
                    L::mixture_component_matrix(&IIS_4, glaf.af4.lanes(), like4_x.0, like4_y.0, &mut builder);

                    f64::mixture_component_matrix2(glaf.af2.rem(), like2_x.1, like2_y.1, &mut builder);
                    f64::mixture_component_matrix(&IIS_3, glaf.af3.rem(), like3_x.1, like3_y.1, &mut builder);
                    f64::mixture_component_matrix(&IIS_4, glaf.af4.rem(), like4_x.1, like4_y.1, &mut builder);

                    builder.finish();

                    let obj = |x: &Vector<9>, eps| L::compute_obj(&buffer, x, eps);
                    let grad_hess =
                        |x: &Vector<9>, eps| L::compute_grad_hess(&buffer, x, eps);
                    let (f, delta, iters) = sqp::solve_sqp(obj, grad_hess, &delta, &tune);

                    if iters >= tune.sqp_max_iter.get() {
                        log::warn!(
                            "No convergence for Jacquard coefficients of pair {x} {y}, max iterations {} exceeded",
                            tune.sqp_max_iter
                        );
                    }

                    handle.inc();

                    Output {
                        x,
                        y,
                        jacquard: delta,
                        iters,
                        obj: f,
                    }
                });

                out.extend(iter);
                // handle.add(out.len().try_into().unwrap());
            },
        );

    bar.done();

    nested_outputs
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

pub fn calculate_coefficients_gt<L: ClsQuadratic>(
    genotypes: ArrayView3<u8>,
    allele_frequencies: ArrayView2<f64>,
    thread_pool: &mut ThreadPool,
    iterations: NonZeroU64,
) -> Vec<Vec<Output>> {
    let genotypes = reorder_genotypes(genotypes);

    // TODO calculate this across each locus to figure out how many alleles there are
    // Then we can condense the stacked matrix to make it smaller
    let num_a = allele_frequencies.shape()[1];

    let all_joint_genotypes = cls::calculate_all_joint_genotypes(num_a);
    let stacked_m = cls::calculate_stacked_m(&all_joint_genotypes, allele_frequencies);
    let lookup_table = cls::calculate_joint_genotype_lookup_table(&all_joint_genotypes, num_a);

    let mut lv = LaneVector::new(stacked_m.len());
    lv.fill_from_slice(&stacked_m);

    let num_v = allele_frequencies.shape()[0];
    let quadratic_q = L::compute_quadratic(&lv, num_v);

    let num_samples = genotypes.dim().0;

    let num_pairs = (num_samples * (num_samples + 1)) / 2;

    log::info!(
        "Calculating Jacquard coefficients for {} pairs using {} variants",
        num_pairs,
        num_v
    );

    let bar = ProgressBar::new(num_pairs.try_into().unwrap())
        .with_eta()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    // The SQP iterations don't matter here
    let tune = Tuneables::new(iterations, iterations);

    let mut nested_outputs = vec![vec![]; num_samples];

    (nested_outputs.par_iter_mut(), (0..num_samples).into_par_iter())
        .zip_eq()
        .with_thread_pool(thread_pool)
        .for_each(|(out, x)| {
            let genotypes_x = genotypes.index_axis(Axis(0), x);
            let iter = (x..num_samples).map(|y| {
                let genotypes_y = genotypes.index_axis(Axis(0), y);
                let delta = if x == y {
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                } else {
                    [1.0 / 9.0; 9]
                };
                let c = cls::calculate_quadratic_c(
                    &all_joint_genotypes,
                    &stacked_m,
                    genotypes_x,
                    genotypes_y,
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

                Output {
                    x,
                    y,
                    jacquard: delta,
                    iters,
                    obj,
                }
            });

            out.extend(iter);

            handle.add(out.len().try_into().unwrap());
        });

    bar.done();

    nested_outputs
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
