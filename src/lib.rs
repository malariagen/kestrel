// use pyo3::prelude::*;

// /// A Python module implemented in Rust.
// #[pymodule]
// mod kestrel {
//     use pyo3::prelude::*;

//     /// Formats the sum of two numbers as string.
//     #[pyfunction]
//     fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//         Ok((a + b).to_string())
//     }
// }

extern crate openblas_src;

pub mod cholesky;
pub mod cls;
pub mod iis;
pub mod jacquard;
pub mod simd;
//pub mod genotypes;
pub mod gradient;
pub mod hessian;
pub mod objective;
pub mod sqp;
pub mod util;
pub mod vcf;
pub mod fused;
pub mod buffer;
pub mod matrix;
