use nalgebra::{ArrayStorage, Dyn, Matrix, U1, U9, VecStorage};

pub type Vector9<T> = Matrix<T, U9, U1, ArrayStorage<T, 9, 1>>;
pub type Matrix9<T> = Matrix<T, U9, U9, ArrayStorage<T, 9, 9>>;
pub type MatrixNx9<T> = Matrix<T, Dyn, U9, VecStorage<T, Dyn, U9>>;
pub type Matrix9xN<T> = Matrix<T, U9, Dyn, VecStorage<T, U9, Dyn>>;
