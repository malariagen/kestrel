use crate::{algebra::{Matrix, Vector}, jacquard::{grad_hess::compute_jacquard_grad_hess, objective::compute_jacquard_obj}, lanevector::LaneVector, log::Log, simd::{Arith, Lane}, sqp::{GradHess, Objective}};

mod grad_hess;
mod objective;

pub struct JacquardCoefficientData<'a, L: Lane> {
    pub lv: &'a LaneVector<L, 9>
}

impl<L: Lane> Objective<9> for JacquardCoefficientData<'_, L> where L::Reg: Arith + Log {
    fn objective(&self, x: &Vector<9>, eps: f64) -> f64 {
        L::vectorize(#[inline(always)] || compute_jacquard_obj(&self.lv, x, eps))

    }
}

impl<L: Lane> GradHess<9> for JacquardCoefficientData<'_, L> where L::Reg: Arith {
    fn grad_hess(&self, x: &Vector<9>, eps: f64) -> (Vector<9>, Matrix<9>) {
        L::vectorize(#[inline(always)] || compute_jacquard_grad_hess(&self.lv, x, eps))
    }
}