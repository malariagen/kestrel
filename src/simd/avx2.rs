use core::arch::x86_64::*;

use crate::simd::{Arith, Lane, LoadStore};

#[repr(align(32))]
#[derive(Clone, Copy, Debug)]
pub struct Avx2Lane([f64; 4]);

impl Lane for Avx2Lane {
    const N: usize = 4;

    type Reg = __m256d;

    #[inline(always)]
    fn get(&self, i: usize) -> f64 {
        self.0[i]
    }

    #[inline(always)]
    fn set(&mut self, i: usize, val: f64) {
        self.0[i] = val;
    }

    fn vectorize<F: FnOnce() -> R, R>(f: F) -> R {
        #[target_feature(enable = "avx2,fma,bmi1,bmi2,lzcnt")]
        fn inner<F: FnOnce() ->R, R>(f: F) -> R {
            f()
        }
        unsafe { inner(f)}
    }
}

impl Default for Avx2Lane {
    #[inline(always)]
    fn default() -> Self {
        Self([0.0; Self::N])
    }
}

impl LoadStore<Avx2Lane> for __m256d {
    #[inline(always)]
    fn load(val: &Avx2Lane) -> Self {
        unsafe { _mm256_load_pd(val.0.as_ptr()) }
    }

    #[inline(always)]
    fn store(self, val: &mut Avx2Lane) {
        unsafe { _mm256_store_pd(val.0.as_mut_ptr(), self) }
    }
}

impl Arith for __m256d {
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm256_setzero_pd() }
    }

    #[inline(always)]
    fn set(val: f64) -> Self {
        unsafe { _mm256_set1_pd(val) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        unsafe { _mm256_mul_pd(self, other) }
    }

    #[inline(always)]
    fn div(self, other: Self) -> Self {
        unsafe { _mm256_div_pd(self, other) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        unsafe { _mm256_add_pd(self, other) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        unsafe { _mm256_sub_pd(self, other) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        unsafe { _mm256_fmadd_pd(self, a, b) }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        unsafe { _mm256_fmsub_pd(self, a, b) }
    }

    #[inline(always)]
    fn fnma(self, a: Self, b: Self) -> Self {
        unsafe { _mm256_fnmadd_pd(self, a, b) }
    }

    #[inline(always)]
    fn fnms(self, a: Self, b: Self) -> Self {
        unsafe { _mm256_fnmsub_pd(self, a, b) }
    }

    #[inline(always)]
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
