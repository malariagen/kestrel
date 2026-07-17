use std::num::NonZeroUsize;

use itertools::Itertools;
use nalgebra::vector;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use paralight::{iter::{ExactParallelSourceExt, IntoExactParallelRefMutSource, IntoExactParallelRefSource, ParallelIteratorExt, ZipableSource}, threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder}};

use lockfree_progress_bar::ProgressBar;

use crate::{cls, sqp::{self, QpTune}, util::Vector9};

pub fn calculate_relatedness_coefficients(genotypes: &Array3<i8>, allele_frequencies: &Array2<f64>) -> Array2<f64> {

    let num_v = genotypes.shape()[0];
    let num_s = genotypes.shape()[1];
    let num_h = genotypes.shape()[2];

    assert_eq!(num_h, 2);

    let genotypes = reorder_genotypes(genotypes.view());

    calculate_coefficients_inner(&genotypes, allele_frequencies)
}

fn reorder_genotypes(mut genotypes: ArrayView3<i8>) -> Array3<i8> {
    genotypes.swap_axes(0, 1);
    let mut genotypes = genotypes.as_standard_layout().into_owned();

    for pair in genotypes.as_slice_mut().unwrap().chunks_exact_mut(2) {
        pair.sort();
    }

    genotypes
}

fn calculate_coefficients_inner(
    genotypes: &Array3<i8>, allele_frequencies: &Array2<f64>
) -> Array2<f64> {
    let num_v = allele_frequencies.shape()[0];
    let num_a = allele_frequencies.shape()[1];

    let all_joint_genotypes = cls::calculate_all_joint_genotypes(num_a as i8);
    let stacked_m = cls::calculate_stacked_m(&all_joint_genotypes, allele_frequencies);
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
        num_threads: ThreadCount::Count(NonZeroUsize::new(20).unwrap()),
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::No,
    }.build();

    let pairs: Vec<[(usize, ArrayView2<i8>); 2]> = genotypes.outer_iter().enumerate().array_combinations_with_replacement().collect();
    let mut kinship = vec![(0, 0, 0.0f64); pairs.len()];

    let bar = ProgressBar::new(pairs.len().try_into().unwrap())
    .with_eta()
    // .disable_color()
    // .with_cpu_usage()
    .with_bar_width(50)
    .with_update_interval(100)
    .start();

    let handle = bar.clone_handle();

    (kinship.par_iter_mut(), pairs.par_iter()).zip_eq().with_thread_pool(&mut thread_pool).for_each (|(out, [(x, genotypes_x), (y, genotypes_y)])|
    {

        let c = cls::calculate_quadratic_c(&all_joint_genotypes, &stacked_m, *genotypes_x, *genotypes_y, allele_frequencies);

        let delta0 = if x == y { vector![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0] } else {
            Vector9::<f64>::from_element(1.0 / 9.0)
        };

        let (delta, _) = sqp::solve_qp_active_set(&quadratic_q, &c, &delta0, true, true, &QpTune::new());
        // println!("{} {} {}", x, y, delta.transpose());
        let kinship = delta.dot(&kinship_vec);
        *out = (*x, *y, kinship);
        handle.inc();
    });

    bar.done();

    let mut kinship_mat = Array2::<f64>::zeros((num_s, num_s));

    for (x, y, kinship) in kinship.iter() {
        kinship_mat[(*x, *y)] = *kinship;
    }

    kinship_mat
}

#[cfg(test)]
mod test {
    use ndarray::array;
    use crate::jacquard::reorder_genotypes;

    #[test]
    fn test_reorder_genotypes() {
        let genotypes = array![
            [[1, 2], [-1, 0], [4, 3]],
            [[3, 0], [1, -1], [1, 1]]
        ];

        let expected_genotypes = array![
            [[1, 2], [0, 3]],
            [[-1, 0], [-1, 1]],
            [[3, 4], [1, 1]],
        ];

        assert_eq!(reorder_genotypes(genotypes.view()), expected_genotypes);
    }
}
