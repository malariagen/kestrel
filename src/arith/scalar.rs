use crate::arith::Arith;

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

    fn radd(self) -> f64 {
        self
    }
}