use std::num::NonZeroU64;

use ndarray::{Array2, Axis};
use paralight::{
    iter::{
        ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelSource, ParallelIteratorExt,
        ZipableSource,
    },
    threads::ThreadPool,
};

use lockfree_progress_bar::ProgressBar;

use crate::{
    algebra::{Matrix, Vector, add, dot, mul, scale_div}, cls::{self}, conditional::{IIS_2, IIS_3, IIS_4}, jacquard::{grad_hess::JacquardGradHess, objective::JacquardObjective}, lanevector::{LaneVector, LaneVectorBuilder}, mixture::MixtureMatrix, quadratic::ClsQuadratic, sqp::{self, Tuneables, estimate_condition_number}, vcf::{AlleleFrequencies, Genotypes}, wrapper::GlAf,
};

// We consume the input array so it gets dropped
fn genotype_indices(genotypes: Array2<[u8; 2]>) -> Array2<u8> {
    let (num_v, num_s) = genotypes.dim();

    let mut indices = Vec::with_capacity(num_s * num_v);

    for sample_gts in genotypes.columns() {
        for [a, b] in sample_gts {
            // We only use unphased genotypes
            let i = a.min(b);
            let j = a.max(b);

            // The index of (i, j) where i <= j (see the VCF spec)
            // For A <= 4 this will not overflow a u8
            let index = j * (j + 1) / 2 + i;
            indices.push(index);
        }
    }

    Array2::from_shape_vec((num_s, num_v), indices).unwrap()
}

pub fn calculate_relatedness_coefficients_gl<L: JacquardGradHess + JacquardObjective + MixtureMatrix>(
    glaf: &GlAf<L>,
    thread_pool: &mut ThreadPool,
    iterations: NonZeroU64,
    num_samples: Option<usize>,
) -> Vec<Vec<Output>> {
    let n2 = glaf.af2.len();
    let n3 = glaf.af3.len();
    let n4 = glaf.af4.len();

    let num_v = n2 + n3 + n4;

    let num_samples = num_samples.unwrap_or(glaf.num_samples);

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
                let like2_x = glaf.gl2.sample(x);
                let like3_x = glaf.gl3.sample(x);
                let like4_x = glaf.gl4.sample(x);

                let iter = (x..num_samples).map(|y| {
                    let delta = if x == y {
                        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                    } else {
                        [1.0 / 9.0; 9]
                    };

                    let like2_y = glaf.gl2.sample(y);
                    let like3_y = glaf.gl3.sample(y);
                    let like4_y = glaf.gl4.sample(y);

                    let mut builder = LaneVectorBuilder::new(buffer);

                    L::mixture_component_matrix2(glaf.af2.lanes(), like2_x.0, like2_y.0, &mut builder);
                    L::mixture_component_matrix(&IIS_3, glaf.af3.lanes(), like3_x.0, like3_y.0, &mut builder);
                    L::mixture_component_matrix(&IIS_4, glaf.af4.lanes(), like4_x.0, like4_y.0, &mut builder);

                    f64::mixture_component_matrix2(glaf.af2.rem(), like2_x.1, like2_y.1, &mut builder);
                    f64::mixture_component_matrix(&IIS_3, glaf.af3.rem(), like3_x.1, like3_y.1, &mut builder);
                    f64::mixture_component_matrix(&IIS_4, glaf.af4.rem(), like4_x.1, like4_y.1, &mut builder);

                    builder.finish(true);

                    let obj = |x: &Vector<9>, eps| L::compute_obj(&buffer, x, eps);
                    let grad_hess = |x: &Vector<9>, eps| L::compute_grad_hess(&buffer, x, eps);
                    let (f, delta, iters, cond) = sqp::solve_sqp(obj, grad_hess, &delta, &tune);

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
                        cond,
                    }
                });

                out.extend(iter);
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
    pub cond: f64,
}

impl Default for Output {
    fn default() -> Self {
        Output {
            x: 0,
            y: 0,
            jacquard: [0.0; 9],
            iters: 0,
            obj: 0.0,
            cond: 0.0,
        }
    }
}

pub fn calculate_coefficients_gt<L: ClsQuadratic>(
    genotypes: Genotypes,
    allele_frequencies: AlleleFrequencies,
    thread_pool: &mut ThreadPool,
    iterations: NonZeroU64,
    num_samples: Option<usize>,
) -> Vec<Vec<Output>> {
    let num_samples = num_samples.unwrap_or(genotypes.samples);
    let num_variants = genotypes.variants;

    let g2 = genotype_indices(genotypes.gt2);
    let g3 = genotype_indices(genotypes.gt3);
    let g4 = genotype_indices(genotypes.gt4);

    let m2 = cls::calculate_m_matrices(&IIS_2, &allele_frequencies.af2);
    let m3 = cls::calculate_m_matrices(&IIS_3, &allele_frequencies.af3);
    let m4 = cls::calculate_m_matrices(&IIS_4, &allele_frequencies.af4);

    let quadratic_q = compute_quadratic_matrix::<L>(num_variants, &m2, &m3, &m4);

    let num_pairs = (num_samples * (num_samples + 1)) / 2;

    log::info!(
        "Calculating Jacquard coefficients for {} pairs using {} sites",
        num_pairs,
        num_variants
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

            let x2 = g2.index_axis(Axis(0), x);
            let x3 = g3.index_axis(Axis(0), x);
            let x4 = g4.index_axis(Axis(0), x);

            let iter = (x..num_samples).map(|y| {

                let y2 = g2.index_axis(Axis(0), y);
                let y3 = g3.index_axis(Axis(0), y);
                let y4 = g4.index_axis(Axis(0), y);

                let c2 = cls::calculate_quadratic_c( &m2, x2.view(), y2.view(),);
                let c3 = cls::calculate_quadratic_c( &m3, x3.view(), y3.view(),);
                let c4 = cls::calculate_quadratic_c( &m4, x4.view(), y4.view(),);

                let c = scale_div(-(num_variants as f64), &add(&c2, &add(&c3, &c4)));

                let delta = if x == y {
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                } else {
                    [1.0 / 9.0; 9]
                };
                let (delta, iters) = sqp::solve_qp_active_set(&quadratic_q, &c, &delta, true, &tune);

                if iters >= tune.qp_max_iter.get() {
                    log::warn!(
                        "No convergence for Jacquard coefficients, max iterations {} exceeded",
                        tune.qp_max_iter
                    );
                }

                let brier = dot(&delta, &mul(&quadratic_q, &delta)) + 2.0 * dot(&c, &delta) + 1.0;

                // Gradient of the QP
                let g = add(&mul(&quadratic_q, &delta), &c);
                let cond = estimate_condition_number(&delta, &g, &quadratic_q, tune.sqp_conv_tol);

                Output {
                    x,
                    y,
                    jacquard: delta,
                    iters,
                    obj: brier,
                    cond,
                }
            });

            out.extend(iter);

            handle.add(out.len().try_into().unwrap());
        });

    bar.done();

    nested_outputs
}

fn compute_quadratic_matrix<L: ClsQuadratic>(
    num_variants: usize,
    m2: &Vec<[[[f64; 9]; 3]; 3]>,
    m3: &Vec<[[[f64; 9]; 6]; 6]>,
    m4: &Vec<[[[f64; 9]; 10]; 10]>
) -> Matrix<9> {
    let mut lv = LaneVector::new(num_variants);
    let mut builder = LaneVectorBuilder::new(&mut lv);

    // Append all rows from all M matrices into the lane vector
    // to make one giant matrix A. Then we calculate A^T A and
    // scale it.

    for m in m2.iter() {
        for gx in 0..3 {
            for gy in 0..3 {
                builder.push(&m[gx][gy]);
            }
        }
    }

    for m in m3.iter() {
        for gx in 0..6 {
            for gy in 0..6 {
                builder.push(&m[gx][gy]);
            }
        }
    }

    for m in m4.iter() {
        for gx in 0..10 {
            for gy in 0..10 {
                builder.push(&m[gx][gy]);
            }
        }
    }

    builder.finish(true);

    L::compute_quadratic(&lv, num_variants)
}
