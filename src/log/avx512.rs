use core::arch::x86_64::*;

use crate::log::{C_6, C_5, C_4, C_3, C_2, C_1, C_0, LOG_2_HI, LOG_2_LO};

// Adapted from https://github.com/burrbull/sleef-rs/blob/master/src/f64x/u10.rs
// and https://github.com/shibatch/sleef/blob/master/src/libm/sleefsimddp.c

#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
// There are 14 constants in this assembly.
// In a loop, all of them may be broadcasted to registers...
pub fn log_avx512(d: __m512d) -> __m512d {
    let p = _mm512_mul_pd(d, _mm512_set1_pd(1.0 / 0.75));
    let not_inf = !_mm512_fpclass_pd_mask(p, 0x08);
    // The above multiplication could overflow, so we overwrite the exponent to 1024
    let e = _mm512_mask_getexp_pd(_mm512_set1_pd(1024.0), not_inf, p);

    let m = _mm512_getmant_pd(d, _MM_MANT_NORM_P75_1P5, _MM_MANT_SIGN_NAN);

    let one = _mm512_set1_pd(1.0);
    // m is strictly within [0.75, 1.5)
    // For x = m and y = 1.0, via Sterbenz lemma u = m - 1 is computed exactly
    // let u = sub_as_doubled(m, one);
    let u = _mm512_sub_pd(m, one);
    let l = add_as_doubled(m, one);

    let x = div(u, l);
    let x2 = _mm512_mul_pd(x.0, x.0);
    let x4 = _mm512_mul_pd(x2, x2);
    let x8 = _mm512_mul_pd(x4, x4);

    let t = poly7( x2, x4, x8, C_6, C_5, C_4, C_3, C_2, C_1, C_0);

    let log2_hi = _mm512_set1_pd(LOG_2_HI);
    let log2_lo = _mm512_set1_pd(LOG_2_LO);
    let mut s = mul((log2_hi, log2_lo), e);

    // For IEEE floats, 2*x = x + x
    s = add_checked_double(s, (_mm512_add_pd(x.0, x.0), _mm512_add_pd(x.1, x.1)));
    s = add_checked(s, _mm512_mul_pd(_mm512_mul_pd(x2, x.0), t));

    let r = _mm512_add_pd(s.0, s.1);

    const CONTROL: i64 = (4 << (2 * 4)) | (3 << (4 * 4)) | (5 << (5 * 4)) | (2 << (6 * 4));
    _mm512_fixupimm_pd(r, d, _mm512_set1_epi64(CONTROL), 0)
}

// TODO this is just Knuth 2sum, rename to clarify

#[inline]
#[target_feature(enable = "avx512f")]
fn add_as_doubled(s: __m512d, o: __m512d) -> (__m512d, __m512d) {
    let r0 = _mm512_add_pd(s, o);
    let v = _mm512_sub_pd(r0, s);

    let a = _mm512_sub_pd(r0, v);
    let b = _mm512_sub_pd(o, v);
    let c = _mm512_sub_pd(s, a);
    let d = _mm512_add_pd(c, b);
    (r0, d)
}

// https://en.wikipedia.org/wiki/2Sum
#[inline]
#[target_feature(enable = "avx512f")]
fn two_sum(a: __m512d, b: __m512d) -> (__m512d, __m512d) {
    let s = _mm512_add_pd(a, b);
    let ap = _mm512_sub_pd(s, b);
    let bp = _mm512_sub_pd(s, ap);
    let d_a = _mm512_sub_pd(a, ap);
    let d_b = _mm512_sub_pd(b, bp);
    let t = _mm512_add_pd(d_a, d_b);
    (s, t)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn add_checked_double(s: (__m512d, __m512d), o: (__m512d, __m512d)) -> (__m512d, __m512d) {
    let r0 = _mm512_add_pd(s.0, o.0);

    let a = _mm512_sub_pd(s.0, r0);
    let b = _mm512_add_pd(a, o.0);
    let c = _mm512_add_pd(b, s.1);
    let d = _mm512_add_pd(c, o.1);
    (r0, d)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn add_checked(s: (__m512d, __m512d), o: __m512d) -> (__m512d, __m512d) {
    let r0 = _mm512_add_pd(s.0, o);

    let a = _mm512_sub_pd(s.0, r0);
    let b = _mm512_add_pd(a, o);
    let c = _mm512_add_pd(b, s.1);
    (r0, c)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn mul(s: (__m512d, __m512d), o: __m512d) -> (__m512d, __m512d) {
    let r0 = _mm512_mul_pd(s.0, o);
    let a = _mm512_fmadd_pd(s.1, o, _mm512_fmsub_pd(s.0, o, r0));
    (r0, a)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn div(s: __m512d, o: (__m512d, __m512d)) -> (__m512d, __m512d) {
    let one = _mm512_set1_pd(1.0);

    let t = _mm512_div_pd(one, o.0);

    let q0 = _mm512_mul_pd(s, t);
    let u = _mm512_fmsub_pd(t, s, q0);

    let mut q1 = _mm512_fnmadd_pd(o.1, t, _mm512_fnmadd_pd(o.0, t, one));
    q1 = _mm512_fmadd_pd(q0, q1, u);

    (q0, q1)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn poly7(x: __m512d, x2: __m512d, x4: __m512d, c6: f64, c5: f64, c4: f64, c3: f64, c2: f64, c1: f64, c0: f64) -> __m512d {
    _mm512_fmadd_pd(x4, poly3(x, x2, c6, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn poly3(x: __m512d, x2: __m512d, c2: f64, c1: f64, c0: f64) -> __m512d {
    _mm512_fmadd_pd(x2, _mm512_set1_pd(c2), _mm512_fmadd_pd(x, _mm512_set1_pd(c1), _mm512_set1_pd(c0)))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn poly4(x: __m512d, x2: __m512d, c3: f64, c2: f64, c1: f64, c0: f64) -> __m512d {
    _mm512_fmadd_pd(x2,
        _mm512_fmadd_pd(x, _mm512_set1_pd(c3), _mm512_set1_pd(c2)),
        _mm512_fmadd_pd(x, _mm512_set1_pd(c1), _mm512_set1_pd(c0))
    )
}