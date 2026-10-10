use std::num::NonZeroUsize;

use ndarray::{Array3, ArrayView2, ArrayView3};
use paralight::threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPool, ThreadPoolBuilder};

use crate::{arith::simd::Simd, coefficients::Output};

mod algebra;
mod allele;
mod arith;
mod ata;
mod blockbuffer;
mod buffer;
mod cholesky;
pub mod cli;
mod cls;
mod coefficients;
mod conditional;
mod eigenval;
mod iis;
mod jacquard;
mod lanematrix;
mod lanevector;
mod log;
mod sqp;
mod vcf;

#[pyo3::pymodule]
mod kestrel {
    use crate::{build_jacquard_array, cli};
    use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
    use pyo3::{exceptions::PyRuntimeError, prelude::*};

    #[pyfunction]
    fn relatedness_coefficients_gt<'py>(
        py: Python<'py>,
        genotypes: PyReadonlyArray3<'py, u8>,
    ) -> Bound<'py, PyArray3<f64>> {
        let gt_view = genotypes.as_array();

        let af = crate::coefficients::calculate_allele_frequencies(gt_view);

        let outputs = crate::calculate_relatedness_coefficients_gt(gt_view, af.view());

        let num_samples = gt_view.shape()[1];
        let jacquard = build_jacquard_array(num_samples, &outputs);

        jacquard.into_pyarray(py)
    }

    #[pyfunction]
    fn relatedness_coefficients_gt_af<'py>(
        py: Python<'py>,
        genotypes: PyReadonlyArray3<'py, u8>,
        allele_frequencies: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray3<f64>> {
        let gt_view = genotypes.as_array();
        let af_view = allele_frequencies.as_array();

        let outputs = crate::calculate_relatedness_coefficients_gt(gt_view, af_view);

        let num_samples = gt_view.shape()[1];
        let jacquard = build_jacquard_array(num_samples, &outputs);

        jacquard.into_pyarray(py)
    }

    // https://www.maturin.rs/bindings.html#both-binary-and-library
    #[pyfunction]
    fn run_cli(py: Python) -> PyResult<()> {
        let args = py.import("sys")?.getattr("argv")?.extract::<Vec<String>>()?;

        cli::run_cli(&args).map_err(|e| PyRuntimeError::new_err(format!("{:#}", e)))
    }
}

pub fn calculate_relatedness_coefficients_gt(
    genotypes: ArrayView3<u8>,
    allele_frequencies: ArrayView2<f64>,
) -> Vec<Output> {
    let num_v = genotypes.shape()[0];
    let num_s = genotypes.shape()[1];
    let num_h = genotypes.shape()[2];

    assert!(num_v > 0, "Must have at least one variant");
    assert!(num_s > 0, "Must have at least one sample");
    assert!(num_h == 2, "Must have a ploidy of 2");

    // TODO in theory we could do some checks here that the af are non-negative, sum to one
    // Indexing will work, etc.

    assert!(allele_frequencies.shape()[0] == num_v, "Must have same dimension");
    assert!(allele_frequencies.shape()[1] > 0, "Must have at least one allele");

    let simd = Simd::detect();

    let mut thread_pool = build_thread_pool(None);

    let outputs = crate::coefficients::calculate_coefficients_gt(genotypes, allele_frequencies, &mut thread_pool, simd);

    outputs
}

fn build_thread_pool(threads: Option<NonZeroUsize>) -> ThreadPool {
    let num_threads = threads.map_or(ThreadCount::AvailableParallelism, |t| ThreadCount::Count(t));

    let pool = ThreadPoolBuilder {
        num_threads: num_threads,
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::No,
    }
    .build();

    pool
}

fn build_jacquard_array(num_samples: usize, outputs: &[Output]) -> Array3<f64> {

    let mut jacquard = Array3::<f64>::zeros((num_samples, num_samples, 9));

    for out in outputs.iter() {
        let x = out.x;
        let y = out.y;

        for i in 0..9 {
            jacquard[[x, y, i]] = out.jacquard[i];
        }

        if x != y {
            // Need to swap D3 and D5, and D4 and D6
            // The rest are the same.
            jacquard[[y, x, 0]] = jacquard[[x, y, 0]];
            jacquard[[y, x, 1]] = jacquard[[x, y, 1]];
            jacquard[[y, x, 2]] = jacquard[[x, y, 4]];
            jacquard[[y, x, 3]] = jacquard[[x, y, 5]];
            jacquard[[y, x, 4]] = jacquard[[x, y, 2]];
            jacquard[[y, x, 5]] = jacquard[[x, y, 3]];
            jacquard[[y, x, 6]] = jacquard[[x, y, 6]];
            jacquard[[y, x, 7]] = jacquard[[x, y, 7]];
            jacquard[[y, x, 8]] = jacquard[[x, y, 8]];
        }
    }

    jacquard
}