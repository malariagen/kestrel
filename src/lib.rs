use pyo3::prelude::*;

#[pymodule]
mod kestrel {
    use crate::jacquard;
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3};
    use pyo3::prelude::*;

    // TODO remove any variants with unknown data

    #[pyfunction]
    fn calculate_relatedness_coefficients<'py>(
        py: Python<'py>,
        genotypes: PyReadonlyArray3<'py, i8>,
    ) -> Bound<'py, PyArray2<f64>> {
        let genotypes_view = genotypes.as_array();

        let kinship = jacquard::calculate_relatedness_coefficients_no_freq(genotypes_view);

        kinship.into_pyarray(py)
    }
}

extern crate openblas_src;

pub mod algebra;
pub mod allele;
pub mod blockbuffer;
pub mod buffer;
pub mod cholesky;
pub mod cls;
pub mod eigenval;
pub mod fused;
pub mod iis;
pub mod jacquard;
pub mod lane;
pub mod log;
pub mod objective;
pub mod sqp;
pub mod vcf;
