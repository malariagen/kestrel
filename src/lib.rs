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
    use crate::{cli, coefficients};
    use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
    use pyo3::{exceptions::PyRuntimeError, prelude::*};

    #[pyfunction]
    fn relatedness_coefficients_gt<'py>(
        py: Python<'py>,
        genotypes: PyReadonlyArray3<'py, u8>,
    ) -> Bound<'py, PyArray3<f64>> {
        let genotypes_view = genotypes.as_array();

        let kinship = coefficients::calculate_relatedness_coefficients_gt(genotypes_view);

        kinship.into_pyarray(py)
    }

    // https://www.maturin.rs/bindings.html#both-binary-and-library
    #[pyfunction]
    fn run_cli(py: Python) -> PyResult<()> {
        let args = py.import("sys")?.getattr("argv")?.extract::<Vec<String>>()?;

        cli::run_cli(&args).map_err(|e| PyRuntimeError::new_err(format!("{:#}", e)))
    }
}
