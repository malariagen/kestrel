use core::arch::x86_64::*;

use crate::arith::{Arith, Lane, Lane8, LoadStore};

impl Lane for Lane8 {
    const N: usize = 8;

    #[inline]
    fn get(&self, i: usize) -> f64 {
        self.0[i]
    }

    #[inline]
    fn set(&mut self, i: usize, val: f64) {
        self.0[i] = val;
    }

    #[inline]
    fn zero() -> Self {
        Self([0.0; Self::N])
    }
}

impl LoadStore<Lane8> for __m512d {
    #[inline]
    fn load(val: &Lane8) -> Self {
        unsafe { _mm512_load_pd(val.0.as_ptr()) }
    }

    #[inline]
    fn store(self, val: &mut Lane8) {
        unsafe { _mm512_store_pd(val.0.as_mut_ptr(), self) }
    }
}

impl Arith for __m512d {
    #[inline]
    fn zero() -> Self {
        unsafe { _mm512_setzero_pd() }
    }

    #[inline]
    fn set(val: f64) -> Self {
        unsafe { _mm512_set1_pd(val) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { _mm512_mul_pd(self, other) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { _mm512_div_pd(self, other) }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { _mm512_add_pd(self, other) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { _mm512_sub_pd(self, other) }
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        unsafe { _mm512_fmadd_pd(self, a, b) }
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        unsafe { _mm512_fmsub_pd(self, a, b) }
    }

    #[inline]
    fn fnma(self, a: Self, b: Self) -> Self {
        unsafe { _mm512_fnmadd_pd(self, a, b) }
    }

    #[inline]
    fn fnms(self, a: Self, b: Self) -> Self {
        unsafe { _mm512_fnmsub_pd(self, a, b) }
    }

    #[inline]
    fn radd(self) -> f64 {
        unsafe { _mm512_reduce_add_pd(self) }
    }
}
