#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod scalar;
pub mod simd;
pub mod lane;

pub trait Arith<L : Lane> : Copy {
    fn zero() -> Self;
    fn load(val: &L) -> Self;
    fn store(self, val: &mut L);
    fn set(val: f64) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn fma(self, a: Self, b: Self) -> Self;
    fn fms(self, a: Self, b: Self) -> Self;
    fn fnma(self, a: Self, b: Self) -> Self;
    fn fnms(self, a: Self, b: Self) -> Self;
    fn radd(self) -> f64;
}

pub trait Lane : Copy {
    const N: usize;

    // TODO use Index + Default?
    fn get(&self, i: usize) -> f64;
    fn set(&mut self, i: usize, val: f64);
    fn zero() -> Self;
}

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Lane8([f64; 8]);