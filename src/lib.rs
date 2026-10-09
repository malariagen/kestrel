use std::num::NonZeroUsize;

use ndarray::{Array3, ArrayView2, ArrayView3};
use paralight::threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPool, ThreadPoolBuilder};

use crate::arith::simd::Simd;

mod ata;
pub mod cli;
mod algebra;
mod allele;
mod arith;
mod blockbuffer;
mod buffer;
mod cholesky;
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
    use crate::cli;
    use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
    use pyo3::{exceptions::PyRuntimeError, prelude::*};

    #[pyfunction]
    fn relatedness_coefficients_gt<'py>(
        py: Python<'py>,
        genotypes: PyReadonlyArray3<'py, u8>,
    ) -> Bound<'py, PyArray3<f64>> {
        let gt_view = genotypes.as_array();

        let af = crate::coefficients::calculate_allele_frequencies(gt_view);

        let jacquard = crate::calculate_relatedness_coefficients_gt(gt_view, af.view());

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

        let jacquard = crate::calculate_relatedness_coefficients_gt(gt_view, af_view);

        jacquard.into_pyarray(py)
    }

    // https://www.maturin.rs/bindings.html#both-binary-and-library
    #[pyfunction]
    fn run_cli(py: Python) -> PyResult<()> {
        let args = py.import("sys")?.getattr("argv")?.extract::<Vec<String>>()?;

        cli::run_cli(&args).map_err(|e| PyRuntimeError::new_err(format!("{:#}", e)))
    }
}

pub fn calculate_relatedness_coefficients_gt(genotypes: ArrayView3<u8>, allele_frequencies: ArrayView2<f64>) -> Array3<f64> {
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

    let mut jacquard_mat = Array3::<f64>::zeros((num_s, num_s, 9));

    for out in outputs.iter() {
        for i in 0..9 {
            jacquard_mat[(out.x, out.y, i)] = out.jacquard[i];
        }
    }

    jacquard_mat
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
