mod avx512;
mod scalar;

pub trait Arith<L : Lane> {
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

pub trait Lane {
    const N: usize;

    // This could be indexing tbh
    fn get(&self, i: usize) -> f64;
    fn set(&mut self, i: usize, val: f64);

    fn zero() -> Self;
}

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Lane8(pub [f64; 8]);

#[repr(align(16))]
#[derive(Clone, Copy)]
pub struct Lane2(pub [f64; 2]);

#[repr(align(32))]
#[derive(Clone, Copy)]
pub struct Lane4(pub [f64; 4]);
