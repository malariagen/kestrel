use crate::arith::{Arith, Lane, LoadStore};

impl Lane for f64 {
    const N: usize = 1;

    #[inline]
    fn get(&self, _i: usize) -> f64 {
        *self
    }

    #[inline]
    fn set(&mut self, _i: usize, val: f64) {
        *self = val;
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }
}

impl LoadStore<f64> for f64 {
    #[inline]
    fn load(val: &f64) -> Self {
        *val
    }

    #[inline]
    fn store(self, val: &mut f64) {
        *val = self;
    }
}

impl Arith for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn set(val: f64) -> Self {
        val
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        self - other
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        self * other
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        self / other
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        self.mul_add(a, -b)
    }

    #[inline]
    fn fnma(self, a: Self, b: Self) -> Self {
        (-self).mul_add(a, b)
    }

    #[inline]
    fn fnms(self, a: Self, b: Self) -> Self {
        (-self).mul_add(a, -b)
    }

    #[inline]
    fn radd(self) -> f64 {
        self
    }
}
