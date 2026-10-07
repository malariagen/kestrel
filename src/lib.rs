use pyo3::prelude::*;

#[pymodule]
mod kestrel {
    use crate::coefficients;
    use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
    use pyo3::prelude::*;

    #[pyfunction]
    fn relatedness_coefficients_gt<'py>(
        py: Python<'py>,
        genotypes: PyReadonlyArray3<'py, u8>,
    ) -> Bound<'py, PyArray3<f64>> {
        let genotypes_view = genotypes.as_array();

        let kinship = coefficients::calculate_relatedness_coefficients_gt(genotypes_view);

        kinship.into_pyarray(py)
    }
}

extern crate openblas_src;

mod ata;
pub mod algebra;
pub mod allele;
pub mod arith;
pub mod blockbuffer;
pub mod buffer;
pub mod cholesky;
pub mod cls;
pub mod coefficients;
pub mod conditional;
pub mod eigenval;
pub mod iis;
pub mod jacquard;
pub mod lanematrix;
pub mod lanevector;
pub mod log;
mod sqp;
pub mod vcf;
