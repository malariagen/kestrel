use std::num::NonZeroUsize;

use itertools::Itertools;
use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3, s};
use paralight::{
    iter::{
        ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelRefSource, ParallelIteratorExt,
        ZipableSource,
    },
    threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPool, ThreadPoolBuilder},
};

use lockfree_progress_bar::ProgressBar;

use crate::{
    algebra::{Vector, dot},
    arith::Lane8,
    blockbuffer::BlockBuffer,
    cls,
    conditional::{self, M},
    jacquard::{grad_hess, objective},
    lanevector::LaneVector,
    sqp::{self, Tuneables},
};

pub fn calculate_relatedness_coefficients_gt(genotypes: &Array3<i8>, allele_frequencies: &Array2<f64>) -> Array2<f64> {
    let num_v = genotypes.shape()[0];
    let num_s = genotypes.shape()[1];
    let num_h = genotypes.shape()[2];

    assert_eq!(num_h, 2);

    let genotypes = reorder_genotypes(genotypes.view());

    calculate_coefficients_inner(&genotypes, allele_frequencies)
}

pub fn calculate_relatedness_coefficients_no_freq(genotypes: ArrayView3<i8>) -> Array2<f64> {
    let num_v = genotypes.shape()[0];
    let num_s = genotypes.shape()[1];
    let num_h = genotypes.shape()[2];

    let allele_frequencies = &calculate_allele_frequencies(genotypes.view());

    assert_eq!(num_h, 2);

    let genotypes = reorder_genotypes(genotypes.view());

    calculate_coefficients_inner(&genotypes, allele_frequencies)
}

fn calculate_allele_frequencies(genotypes: ArrayView3<i8>) -> Array2<f64> {
    let num_v = genotypes.shape()[0];
    let num_s = genotypes.shape()[1];
    let num_h = genotypes.shape()[2];

    let max_a = (genotypes.iter().max().unwrap() + 1) as usize;

    let mut freq = Array2::<f64>::zeros((num_v, max_a));

    for v in 0..num_v {
        for s in 0..num_s {
            for h in 0..num_h {
                let allele = genotypes[(v, s, h)];
                if allele >= 0 {
                    freq[(v, allele as usize)] += 1.0;
                }
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

fn reorder_genotypes(mut genotypes: ArrayView3<i8>) -> Array3<i8> {
    genotypes.swap_axes(0, 1);
    let mut genotypes = genotypes.as_standard_layout().into_owned();

    let (pairs, _) = genotypes.as_slice_mut().unwrap().as_chunks_mut::<2>();

    for pair in pairs {
        pair.sort_unstable();
    }

    genotypes
}

fn calculate_max_alleles(genotypes: ArrayView3<i8>) -> Vec<usize> {
    genotypes
        .outer_iter()
        .map(|variant| {
            let max_allele = *variant.iter().max().unwrap();
            usize::try_from(max_allele).unwrap() + 1
        })
        .collect()
}

// fn calculate_all_joint_genotypes(per_locus_alleles: &[i8]) -> Vec<> {

// }

struct ThreadBuffers {
    p_mat: BlockBuffer<f64, 8, 9>,
}

impl ThreadBuffers {
    fn new(num_loci: usize) -> Self {
        ThreadBuffers {
            p_mat: BlockBuffer::new(num_loci),
        }
    }
}

pub fn calculate_relatedness_coefficients_gl(
    mut likelihoods: Array3<f64>,
    allele_frequencies: &Array2<f64>,
    thread_pool: &mut ThreadPool,
) -> Vec<Output> {
    let num_v = allele_frequencies.shape()[0];

    // TODO calculate this across each locus to figure out how many alleles there are
    // Then we can condense the stacked matrix to make it smaller? Idk, complicated...
    let num_a = allele_frequencies.shape()[1];

    let m_matrices = conditional::calculate_m_matrices(allele_frequencies);

    likelihoods.swap_axes(0, 1);
    let swapped = likelihoods.as_standard_layout();

    let num_s = swapped.shape()[0];

    assert_eq!(swapped.shape()[1], num_v);

    // Only to analyze the data, before I make it faster...
    let tmp = swapped.slice(s![0..16, .., ..]);

    let pairs: Vec<[(usize, ArrayView2<f64>); 2]> = tmp
        .outer_iter()
        .enumerate()
        .array_combinations_with_replacement()
        .collect();
    let mut outputs = vec![Output::default(); pairs.len()];

    println!(
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

    (outputs.par_iter_mut(), pairs.par_iter())
        .zip_eq()
        .with_thread_pool(thread_pool)
        .for_each_init(
            || LaneVector::new(num_v),
            |p_mat, (out, [(x, likelihoods_x), (y, likelihoods_y)])| {
                let delta = if x == y {
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                } else {
                    [1.0 / 9.0; 9]
                };

                calculate_mixture_component_matrix_gl(&m_matrices, likelihoods_x, likelihoods_y, p_mat);

                let tune = Tuneables::new();

                let obj = |x: &Vector<9>, eps| objective::compute_obj(p_mat, &x, eps);
                let grad_hess = |x: &Vector<9>, eps| grad_hess::compute_grad_hess(p_mat, &x, eps);
                let (f, delta, iters) = sqp::solve_sqp(obj, grad_hess, &delta, &tune);

                if iters >= tune.sqp_max_iter {
                    println!(
                        "WARNING: no convergence for Jacquard coefficients, max iterations {} exceeded",
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

fn calculate_coefficients_inner(genotypes: &Array3<i8>, allele_frequencies: &Array2<f64>) -> Array2<f64> {
    let num_v = allele_frequencies.shape()[0];

    // TODO calculate this across each locus to figure out how many alleles there are
    // Then we can condense the stacked matrix to make it smaller
    let num_a = allele_frequencies.shape()[1];

    let all_joint_genotypes = cls::calculate_all_joint_genotypes(num_a);
    let stacked_m = cls::calculate_stacked_m(&all_joint_genotypes, allele_frequencies);
    let lookup_table = cls::calculate_joint_genotype_lookup_table(&all_joint_genotypes, num_a);
    let quadratic_q = cls::calculate_quadratic_q_mat(&stacked_m, num_v);

    // let eigen = quadratic_q.symmetric_eigenvalues();
    // let delta = 1e-8;
    // let tau = 0.0f64.max(delta - eigen.min());

    // for i in 0..9 {
    //     quadratic_q[(i, i)] += tau;
    // }

    let kinship_vec = [1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0];

    let num_s = genotypes.shape()[0];

    let mut thread_pool = ThreadPoolBuilder {
        num_threads: ThreadCount::Count(NonZeroUsize::new(10).unwrap()),
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::No,
    }
    .build();

    let pairs: Vec<[(usize, ArrayView2<i8>); 2]> = genotypes
        .outer_iter()
        .enumerate()
        .array_combinations_with_replacement()
        .collect();
    let mut output = vec![Output::default(); pairs.len()];

    println!(
        "Calculating Jacquard coefficients for {} pairs using {} variants",
        pairs.len(),
        num_v
    );

    let bar = ProgressBar::new(pairs.len().try_into().unwrap())
        .with_eta()
        // .disable_color()
        // .with_cpu_usage()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    (output.par_iter_mut(), pairs.par_iter())
        .zip_eq()
        .with_thread_pool(&mut thread_pool)
        .for_each_init(
            || ThreadBuffers::new(num_v),
            |buffers, (out, [(x, genotypes_x), (y, genotypes_y)])| {
                let delta = if x == y {
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                } else {
                    [1.0 / 9.0; 9]
                };
                // let c = cls::calculate_quadratic_c(
                //     &all_joint_genotypes,
                //     &stacked_m,
                //     *genotypes_x,
                //     *genotypes_y,
                //     &lookup_table,
                // );

                // let (delta, _) = sqp::solve_qp_active_set(&quadratic_q, &c, &delta, true, &Tuneables::new());

                calculate_mixture_component_matrix(
                    &all_joint_genotypes,
                    &stacked_m,
                    genotypes_x,
                    genotypes_y,
                    &lookup_table,
                    &mut buffers.p_mat,
                );

                let tune = Tuneables::new();

                unimplemented!("No")

                // let obj = |x: &Vector<9>, eps| objective::compute_obj(&buffers.p_mat, &x, eps);
                // let grad_hess = |x: &Vector<9>, eps| grad_hess::compute_grad_hess(&buffers.p_mat, &x, eps);
                // let (f, delta, iters) = sqp::solve_sqp(obj, grad_hess, &delta, &tune);

                // if iters >= tune.sqp_max_iter {
                //     println!("WARNING: no convergence for Jacquard coefficients, max iterations {} exceeded", tune.sqp_max_iter);
                // }

                // let kinship = dot(&delta, &kinship_vec);
                // *out = Output { x: *x, y: *y, jacquard: delta, iters, obj : f };
                // handle.inc();
            },
        );

    bar.done();

    let mut kinship_mat = Array2::<f64>::zeros((num_s, num_s));

    let mut total_iters = 0;
    for out in output.iter() {
        total_iters += out.iters;
        // kinship_mat[(*x, *y)] = *kinship;
    }

    println!("Total iters {total_iters}");

    kinship_mat
}

pub fn calculate_mixture_component_matrix_gl(
    m_matrices: &[M],
    likelihoods_x: &ArrayView2<f64>,
    likelihoods_y: &ArrayView2<f64>,
    p_mat: &mut LaneVector<Lane8, 9>,
) {
    let (chunks_x, rem_x) = likelihoods_x.as_slice().unwrap().as_chunks::<10>();
    let (chunks_y, rem_y) = likelihoods_y.as_slice().unwrap().as_chunks::<10>();

    assert!(rem_x.is_empty());
    assert!(rem_y.is_empty());

    assert_eq!(m_matrices.len(), chunks_x.len());
    assert_eq!(m_matrices.len(), chunks_y.len());

    // TODO may have to lanebuffer all of this stuff...
    // Have a Buffer<Vec<>> and Buffer<Mat>>
    // For each pair (x, y), need to calculate 900 elements x number of sites
    // Oof that's a lot. For hard-called you just look up a row of M, no calculation needed

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

fn calculate_mixture_component_matrix<const L: usize>(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    stacked_m: &[Vector<9>],
    genotypes_x: &ArrayView2<i8>,
    genotypes_y: &ArrayView2<i8>,
    lookup_table: &Array4<usize>,
    p_mat: &mut BlockBuffer<f64, L, 9>,
) {
    let num_g = all_joint_genotypes.len();

    let iter_x = genotypes_x.as_slice().unwrap().chunks_exact(2);
    let iter_y = genotypes_y.as_slice().unwrap().chunks_exact(2);

    let iter = iter_x.zip(iter_y).enumerate().map(|(locus, (geno_x, geno_y))| {
        let (i, j) = (geno_x[0], geno_x[1]);
        let (k, l) = (geno_y[0], geno_y[1]);

        // TODO do a check here for missing data
        // if i < 0 || j < 0 || k < 0 || l < 0 {
        //     continue;
        // }

        let g = unsafe { lookup_table.uget((i as usize, j as usize, k as usize, l as usize)) };
        // let g = lookup_table[(i as usize, j as usize, k as usize, l as usize)];

        let row = unsafe { stacked_m.get_unchecked(locus.unchecked_mul(num_g).unchecked_add(*g)) };
        // let row = stacked_m[locus * num_g + g];
        *row
    });

    p_mat.fill_from_rows(iter);
}

#[cfg(test)]
mod test {
    use crate::coefficients::{calculate_max_alleles, reorder_genotypes};
    use ndarray::array;

    #[test]
    fn test_reorder_genotypes() {
        let genotypes = array![[[1, 2], [-1, 0], [4, 3]], [[3, 0], [1, -1], [1, 1]]];

        let expected_genotypes = array![[[1, 2], [0, 3]], [[-1, 0], [-1, 1]], [[3, 4], [1, 1]],];

        assert_eq!(reorder_genotypes(genotypes.view()), expected_genotypes);
    }

    #[test]
    fn test_total_alleles() {
        let genotypes = array![[[1, 2], [-1, 0], [4, 3]], [[3, 0], [1, -1], [1, 1]]];

        let max_alleles = vec![5, 4];

        assert_eq!(calculate_max_alleles(genotypes.view()), max_alleles);
    }
}
