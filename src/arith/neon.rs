use core::arch::aarch64::*;

use crate::arith::lane::Lane2;
use crate::arith::{Arith, LoadStore};

impl LoadStore<Lane2> for float64x2_t {
    #[inline]
    fn load(val: &Lane2) -> Self {
        unsafe { vld1q_f64(val.0.as_ptr()) }
    }

    #[inline]
    fn store(self, val: &mut Lane2) {
        unsafe { vst1q_f64(val.0.as_mut_ptr(), self) }
    }
}

impl Arith for float64x2_t {
    #[inline]
    fn zero() -> Self {
        unsafe { vdupq_n_f64(0.0) }
    }

    #[inline]
    fn set(val: f64) -> Self {
        unsafe { vdupq_n_f64(val) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { vmulq_f64(self, other) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { vdivq_f64(self, other) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { vaddq_f64(self, other) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { vsubq_f64(self, other) }
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        unsafe { vfmaq_f64(b, self, a) }
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        // In theory we could propagate this negative up to remove it
        // Not worth it, I think...
        unsafe { vnegq_f64(vfmsq_f64(b, self, a)) }
    }

    #[inline]
    fn fnma(self, a: Self, b: Self) -> Self {
        unsafe { vfmsq_f64(b, self, a) }
    }

    #[inline]
    fn fnms(self, a: Self, b: Self) -> Self {
        unsafe { vnegq_f64(vfmaq_f64(b, self, a)) }
    }

    #[inline]
    fn radd(self) -> f64 {
        unsafe { vaddvq_f64(self) }
    }
}
