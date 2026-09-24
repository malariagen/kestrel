mod avx512;
mod scalar;

pub trait Arith<Lane> {
    fn zero() -> Self;
    fn load(val: &Lane) -> Self;
    fn store(self, val: &mut Lane);
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

#[repr(align(16))]
#[derive(Clone, Copy)]
pub struct Lane2(pub [f64; 2]);

#[repr(align(32))]
#[derive(Clone, Copy)]
pub struct Lane4(pub [f64; 4]);

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct Lane8(pub [f64; 8]);