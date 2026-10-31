use ndarray::{Array1, Array2, Array3};

use crate::{algebra::dot, coefficients::Output};

mod algebra;
mod allele;
mod cholesky;
pub mod cli;
mod cls;
mod coefficients;
mod conditional;
mod quadratic;
mod simd;
//mod eigenval;
mod iis;
mod jacquard;
mod lanearray2;
mod lanevector;
mod log;
pub mod mixture;
mod sqp;
mod vcf;
mod wrapper;
mod util;

#[pyo3::pymodule]
mod kestrel {

    use std::num::{NonZeroU64, NonZeroUsize};

    use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
    use pyo3::{exceptions::PyRuntimeError, prelude::*};

    use crate::{build_arrays, cli};

    #[pyfunction]
    #[pyo3(signature = (genotypes, allele_frequencies = None, threads = None, iterations = NonZeroU64::new(100).unwrap()))]
    fn relatedness_coefficients_gt<'py>(
        py: Python<'py>,
        genotypes: PyReadonlyArray3<'py, u8>,
        allele_frequencies: Option<PyReadonlyArray2<'py, f64>>,
        threads: Option<NonZeroUsize>,
        iterations: NonZeroU64,
    ) -> (
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let gt_view = genotypes.as_array();
        let af_view = allele_frequencies.as_ref().map(|a| a.as_array());

        let outputs = crate::wrapper::calculate_relatedness_coefficients_gt(gt_view, af_view, threads, iterations);

        let (jacquard, kinship, inbreeding) = build_arrays(&outputs);

        (
            jacquard.into_pyarray(py),
            kinship.into_pyarray(py),
            inbreeding.into_pyarray(py),
        )
    }

    #[pyfunction]
    #[pyo3(signature = (genotype_likelihoods, allele_frequencies = None, threads = None, iterations = NonZeroU64::new(100).unwrap(), minimum_allele_frequency = None))]
    fn relatedness_coefficients_gl<'py>(
        py: Python<'py>,
        genotype_likelihoods: PyReadonlyArray3<'py, f64>,
        allele_frequencies: Option<PyReadonlyArray2<'py, f64>>,
        threads: Option<NonZeroUsize>,
        iterations: NonZeroU64,
        minimum_allele_frequency: Option<f64>,
    ) -> (
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let gl_view = genotype_likelihoods.as_array();
        let af_view = allele_frequencies.as_ref().map(|a| a.as_array());

        let outputs = crate::wrapper::calculate_relatedness_coefficients_gl(
            gl_view,
            af_view,
            threads,
            iterations,
            minimum_allele_frequency,
        );

        let (jacquard, kinship, inbreeding) = build_arrays(&outputs);

        (
            jacquard.into_pyarray(py),
            kinship.into_pyarray(py),
            inbreeding.into_pyarray(py),
        )
    }

    // https://www.maturin.rs/bindings.html#both-binary-and-library
    #[pyfunction]
    fn run_cli(py: Python) -> PyResult<()> {
        let args = py.import("sys")?.getattr("argv")?.extract::<Vec<String>>()?;

        cli::run_cli(&args).map_err(|e| PyRuntimeError::new_err(format!("{:#}", e)))
    }
}

fn build_arrays(outputs: &[Vec<Output>]) -> (Array3<f64>, Array2<f64>, Array1<f64>) {
    let num_samples = outputs.len();

    let mut jacquard = Array3::<f64>::zeros((num_samples, num_samples, 9));
    let mut kinship = Array2::<f64>::zeros((num_samples, num_samples));
    let mut inbreeding = Array1::<f64>::zeros(num_samples);

    let kinship_vec = [1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0];

    for out in outputs.iter().flatten() {
        let x = out.x;
        let y = out.y;

        for i in 0..9 {
            jacquard[[x, y, i]] = out.jacquard[i];
        }

        if x != y {
            // Need to swap D3 and D5, and D4 and D6
            // The rest are the same.
            jacquard[[y, x, 0]] = out.jacquard[0];
            jacquard[[y, x, 1]] = out.jacquard[1];
            jacquard[[y, x, 2]] = out.jacquard[4];
            jacquard[[y, x, 3]] = out.jacquard[5];
            jacquard[[y, x, 4]] = out.jacquard[2];
            jacquard[[y, x, 5]] = out.jacquard[3];
            jacquard[[y, x, 6]] = out.jacquard[6];
            jacquard[[y, x, 7]] = out.jacquard[7];
            jacquard[[y, x, 8]] = out.jacquard[8];
        }

        kinship[[x, y]] = dot(&out.jacquard, &kinship_vec);

        if x != y {
            kinship[[y, x]] = kinship[[x, y]];
        } else {
            inbreeding[x] = 2.0 * kinship[[x, x]] - 1.0;
        }
    }

    (jacquard, kinship, inbreeding)
}
