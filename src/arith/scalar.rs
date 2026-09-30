use crate::arith::{Arith, Lane, lane::Lane1};

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

impl Arith<f64> for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn set(val: f64) -> Self {
        val
    }

    #[inline]
    fn load(val: &f64) -> Self {
        *val
    }

    #[inline]
    fn store(self, val: &mut f64) {
        *val = self;
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

impl Arith<Lane1> for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn set(val: f64) -> Self {
        val
    }

    #[inline]
    fn load(val: &Lane1) -> Self {
        val.0[0]
    }

    #[inline]
    fn store(self, val: &mut Lane1) {
        val.0[0] = self;
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
