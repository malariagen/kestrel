#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod scalar;
pub mod simd;

pub trait LoadStore<L: Lane> {
    fn load(val: &L) -> Self;
    fn store(self, val: &mut L);
}

#[allow(unused)]
pub trait Arith: Copy {
    fn zero() -> Self;
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

#[allow(unused)]
pub trait Lane: Copy + Default + Sync {
    const N: usize;

    // TODO use Index
    fn get(&self, i: usize) -> f64;
    fn set(&mut self, i: usize, val: f64);
}
