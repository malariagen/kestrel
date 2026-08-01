use std::num::NonZeroUsize;

use itertools::Itertools;
use nalgebra::vector;
use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3};
use paralight::{
    iter::{
        ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelRefSource,
        ParallelIteratorExt, ZipableSource,
    },
    threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder},
};

use lockfree_progress_bar::ProgressBar;

use crate::{
    cls,
    blockbuffer::BlockBuffer,
    sqp::{self, Tuneables},
    util::{Matrix9xN, MatrixNx9, Vector9},
};

pub fn calculate_relatedness_coefficients(
    genotypes: &Array3<i8>,
    allele_frequencies: &Array2<f64>,
) -> Array2<f64> {
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

fn calculate_coefficients_inner(
    genotypes: &Array3<i8>,
    allele_frequencies: &Array2<f64>,
) -> Array2<f64> {
    let num_v = allele_frequencies.shape()[0];

    // TODO calculate this across each locus to figure out how many alleles there are
    let num_a = allele_frequencies.shape()[1];

    let all_joint_genotypes = cls::calculate_all_joint_genotypes(num_a);
    let stacked_m = cls::calculate_stacked_m(&all_joint_genotypes, allele_frequencies);
    let lookup_table = cls::calculate_joint_genotype_lookup_table(&all_joint_genotypes, num_a);
    let stacked_m_t = stacked_m.transpose().to_owned();
    let mut quadratic_q = cls::calculate_quadratic_q_mat(&stacked_m, num_v);

    // let eigen = quadratic_q.symmetric_eigenvalues();
    // let delta = 1e-8;
    // let tau = 0.0f64.max(delta - eigen.min());

    // for i in 0..9 {
    //     quadratic_q[(i, i)] += tau;
    // }

    let kinship_vec: Vector9<f64> = vector![1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0];

    let num_s = genotypes.shape()[0];

    let mut thread_pool = ThreadPoolBuilder {
        num_threads: ThreadCount::Count(NonZeroUsize::new(12).unwrap()),
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::No,
    }
    .build();

    let pairs: Vec<[(usize, ArrayView2<i8>); 2]> = genotypes
        .outer_iter()
        .enumerate()
        .array_combinations_with_replacement()
        .collect();
    let mut kinship = vec![(0, 0, 0.0f64); pairs.len()];

    let bar = ProgressBar::new(pairs.len().try_into().unwrap())
        .with_eta()
        // .disable_color()
        // .with_cpu_usage()
        .with_bar_width(50)
        .with_update_interval(100)
        .start();

    let handle = bar.clone_handle();

    (kinship.par_iter_mut(), pairs.par_iter())
        .zip_eq()
        .with_thread_pool(&mut thread_pool)
        .for_each_init(
            || ThreadBuffers::new(num_v),
            |buffers, (out, [(x, genotypes_x), (y, genotypes_y)])| {
                let c = cls::calculate_quadratic_c_t(
                    &all_joint_genotypes,
                    &stacked_m_t,
                    *genotypes_x,
                    *genotypes_y,
                    &lookup_table,
                );
                // let c = cls::calculate_quadratic_c(&all_joint_genotypes, &stacked_m, *genotypes_x, *genotypes_y, allele_frequencies);

                let delta0 = if x == y {
                    vector![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                } else {
                    Vector9::<f64>::from_element(1.0 / 9.0)
                };

                let (delta, _) = sqp::solve_qp_active_set(
                    &quadratic_q,
                    &c,
                    &delta0,
                    true,
                    true,
                    &Tuneables::new(),
                );

                // calculate_mixture_component_matrix2(
                //     &all_joint_genotypes,
                //     &stacked_m_t,
                //     *genotypes_x,
                //     *genotypes_y,
                //     &lookup_table,
                //     &mut buffers.p_mat,
                // );

                // let (delta, _) = sqp::solve_sqp(
                //     &buffers.p_mat,
                //     &delta,
                //     &Tuneables::new(),
                // );

                // println!("{} {} {}", x, y, delta.transpose());
                let kinship = delta.dot(&kinship_vec);
                *out = (*x, *y, kinship);
                handle.inc();
            },
        );

    bar.done();

    let mut kinship_mat = Array2::<f64>::zeros((num_s, num_s));

    for (x, y, kinship) in kinship.iter() {
        kinship_mat[(*x, *y)] = *kinship;
    }

    kinship_mat
}

fn calculate_mixture_component_matrix(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    stacked_m_t: &Matrix9xN<f64>,
    genotypes_x: ArrayView2<i8>,
    genotypes_y: ArrayView2<i8>,
    lookup_table: &Array4<usize>,
    p_mat: &mut MatrixNx9<f64>,
) {
    let num_g = all_joint_genotypes.len();

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

        // TODO should we make this more cache friendly?
        p_mat.set_row(locus, &stacked_m_t.column(locus * num_g + g).transpose());
    }
}

fn calculate_mixture_component_matrix2<const L: usize>(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    stacked_m_t: &Matrix9xN<f64>,
    genotypes_x: ArrayView2<i8>,
    genotypes_y: ArrayView2<i8>,
    lookup_table: &Array4<usize>,
    p_mat: &mut BlockBuffer<f64, L, 9>,
) {
    let num_g = all_joint_genotypes.len();

    let iter_x = genotypes_x.as_slice().unwrap().chunks_exact(2);
    let iter_y = genotypes_y.as_slice().unwrap().chunks_exact(2);

    let iter = iter_x
        .zip(iter_y)
        .enumerate()
        .map(|(locus, (geno_x, geno_y))| {
            let (i, j) = (geno_x[0], geno_x[1]);
            let (k, l) = (geno_y[0], geno_y[1]);

            // TODO do a check here for missing data
            // if i < 0 || j < 0 || k < 0 || l < 0 {
            //     continue;
            // }

            // let g = unsafe { lookup_table.uget((i as usize, j as usize, k as usize, l as usize)) };
            let g = lookup_table[(i as usize, j as usize, k as usize, l as usize)];

            let column = stacked_m_t.column(locus * num_g + g);
            let mut row = [0.0; 9];
            for i in 0..9 {
                row[i] = column[i];
            }
            row
        });

    p_mat.fill_from_rows(iter);
}

#[cfg(test)]
mod test {
    use crate::jacquard::{calculate_max_alleles, reorder_genotypes};
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
