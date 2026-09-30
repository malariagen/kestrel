use core::arch::x86_64::*;

use crate::arith::Arith;
use crate::arith::lane::Lane4;

impl Arith<Lane4> for __m256d {
    #[inline]
    fn zero() -> Self {
        unsafe { _mm256_setzero_pd() }
    }

    #[inline]
    fn set(val: f64) -> Self {
        unsafe { _mm256_set1_pd(val) }
    }

    #[inline]
    fn load(val: &Lane4) -> Self {
        unsafe { _mm256_load_pd(val.0.as_ptr()) }
    }

    #[inline]
    fn store(self, val: &mut Lane4) {
        unsafe { _mm256_store_pd(val.0.as_mut_ptr(), self) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { _mm256_mul_pd(self, other) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { _mm256_div_pd(self, other) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { _mm256_add_pd(self, other) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { _mm256_sub_pd(self, other) }
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        unsafe { _mm256_fmadd_pd(self, a, b) }
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        unsafe { _mm256_fmsub_pd(self, a, b) }
    }

    #[inline]
    fn fnma(self, a: Self, b: Self) -> Self {
        unsafe { _mm256_fnmadd_pd(self, a, b) }
    }

    #[inline]
    fn fnms(self, a: Self, b: Self) -> Self {
        unsafe { _mm256_fnmsub_pd(self, a, b) }
    }

    #[inline]
    fn radd(self) -> f64 {
        // https://stackoverflow.com/a/49943540
        let low128 = unsafe { _mm256_castpd256_pd128(self) };
        let high128 = unsafe { _mm256_extractf128_pd(self, 1) };
        let sum128 = unsafe { _mm_add_pd(low128, high128) };

        // This copies the high64 bits into both halves of 128 bit register
        let high64 = unsafe { _mm_unpackhi_pd(sum128, sum128) };
        // The lower half of sum will be low64 + high64
        let sum64 = unsafe { _mm_add_sd(sum128, high64) };
        // Now convert lower half to f64
        return unsafe { _mm_cvtsd_f64(sum64) };
    }
}
