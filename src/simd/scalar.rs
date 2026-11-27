use crate::simd::{Arith, Lane, LoadStore};

impl Lane for f64 {
    const N: usize = 1;

    type Reg = f64;

    #[inline(always)]
    fn get(&self, _i: usize) -> f64 {
        *self
    }

    #[inline(always)]
    fn set(&mut self, _i: usize, val: f64) {
        *self = val;
    }

    #[inline(always)]
    fn vectorize<F: FnOnce() -> R, R>(f: F) -> R {
        f()
    }
}

impl LoadStore<f64> for f64 {
    #[inline(always)]
    fn load(val: &f64) -> Self {
        *val
    }

    #[inline(always)]
    fn store(self, val: &mut f64) {
        *val = self;
    }
}

impl Arith for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn set(val: f64) -> Self {
        val
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self - other
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        self * other
    }

    #[inline(always)]
    fn div(self, other: Self) -> Self {
        self / other
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        self.mul_add(a, -b)
    }

    #[inline(always)]
    fn fnma(self, a: Self, b: Self) -> Self {
        (-self).mul_add(a, b)
    }

    #[inline(always)]
    fn fnms(self, a: Self, b: Self) -> Self {
        (-self).mul_add(a, -b)
    }

    #[inline(always)]
    fn radd(self) -> f64 {
        self
    }
}
