use core::arch::x86_64::*;

use crate::simd::{Arith, Lane, LoadStore};

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Avx512Lane([f64; 8]);

impl Lane for Avx512Lane {
    const N: usize = 8;

    type Reg = __m512d;

    #[inline(always)]
    fn get(&self, i: usize) -> f64 {
        self.0[i]
    }

    #[inline(always)]
    fn set(&mut self, i: usize, val: f64) {
        self.0[i] = val;
    }
}

impl Default for Avx512Lane {
    #[inline(always)]
    fn default() -> Self {
        Self([0.0; Self::N])
    }
}

impl LoadStore<Avx512Lane> for __m512d {
    #[inline(always)]
    fn load(val: &Avx512Lane) -> Self {
        unsafe { _mm512_load_pd(val.0.as_ptr()) }
    }

    #[inline(always)]
    fn store(self, val: &mut Avx512Lane) {
        unsafe { _mm512_store_pd(val.0.as_mut_ptr(), self) }
    }
}

impl Arith for __m512d {
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm512_setzero_pd() }
    }

    #[inline(always)]
    fn set(val: f64) -> Self {
        unsafe { _mm512_set1_pd(val) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        unsafe { _mm512_mul_pd(self, other) }
    }

    #[inline(always)]
    fn div(self, other: Self) -> Self {
        unsafe { _mm512_div_pd(self, other) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        unsafe { _mm512_add_pd(self, other) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        unsafe { _mm512_sub_pd(self, other) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        unsafe { _mm512_fmadd_pd(self, a, b) }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        unsafe { _mm512_fmsub_pd(self, a, b) }
    }

    #[inline(always)]
    fn fnma(self, a: Self, b: Self) -> Self {
        unsafe { _mm512_fnmadd_pd(self, a, b) }
    }

    #[inline(always)]
    fn fnms(self, a: Self, b: Self) -> Self {
        unsafe { _mm512_fnmsub_pd(self, a, b) }
    }

    #[inline(always)]
    fn radd(self) -> f64 {
        unsafe { _mm512_reduce_add_pd(self) }
    }
}
