use core::arch::x86_64::*;

use crate::log::{C_0, C_1, C_2, C_3, C_4, C_5, C_6, LOG_2_HI, LOG_2_LO, Log};

// Adapted from https://github.com/burrbull/sleef-rs/blob/master/src/f64x/u10.rs
// and https://github.com/shibatch/sleef/blob/master/src/libm/sleefsimddp.c

impl Log for __m512d {
    #[inline]
    fn log(self) -> Self {
        unsafe { log_avx512(self) }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
// There are 12 constants in this assembly.
// In a loop, all of them may be broadcasted to registers...
fn log_avx512(d: __m512d) -> __m512d {
    let one = _mm512_set1_pd(1.0);

    let e = _mm512_getexp_pd(d);
    // Sign zero always makes m positive. If d is negative, the fixup
    // at the end will return NaN instead.
    let m = _mm512_getmant_pd(d, _MM_MANT_NORM_P75_1P5, _MM_MANT_SIGN_ZERO);

    // A (normalized) float is of the form m * 2^e, where m is in [1, 2)
    // getexp and getmant will always return e and m in the above form
    // (normalizing subnormals as appropriate).
    // We want the mantissa to be scaled to be within [0.75, 1.5) as follows:
    //   If 1 <= m < 1.5, then return m
    //   If 1.5 <= m < 2, then return m' = m / 2, where 0.75 <= m / 2 < 1
    // In the second scenario, we rewrite as (m/2) * 2^(e+1), so we need
    // to add one to the exponent when m' < 1

    // https://stackoverflow.com/questions/16988199/how-to-choose-avx-compare-predicate-variants
    let mask = _mm512_cmp_pd_mask(m, one, _CMP_LT_OQ);
    let e = _mm512_mask_add_pd(e, mask, e, one);

    // m is strictly within [0.75, 1.5)
    // For x = m and y = 1.0, via Sterbenz lemma u = m - 1 is computed exactly
    let u = _mm512_sub_pd(m, one);
    // Furthermore, e_one = 0, and
    //   If 1 <= m < 1.5, then e_m = 0
    //   If 0.75 <= m < 1, then e_m = -1
    // So we can use Fast2Sum
    let l = fast_two_sum_ss(one, m);

    let x = div_sd(u, l);
    let x2 = _mm512_mul_pd(x.0, x.0);
    let x4 = _mm512_mul_pd(x2, x2);
    let x8 = _mm512_mul_pd(x4, x4);

    let t = poly7(
        x2,
        x4,
        x8,
        _mm512_set1_pd(C_6),
        _mm512_set1_pd(C_5),
        _mm512_set1_pd(C_4),
        _mm512_set1_pd(C_3),
        _mm512_set1_pd(C_2),
        _mm512_set1_pd(C_1),
        _mm512_set1_pd(C_0),
    );

    let log2_hi = _mm512_set1_pd(LOG_2_HI);
    let log2_lo = _mm512_set1_pd(LOG_2_LO);
    let mut s = mul_ds((log2_hi, log2_lo), e);

    // For IEEE floats, 2*x = x + x
    s = fast_two_sum_dd(s, (_mm512_add_pd(x.0, x.0), _mm512_add_pd(x.1, x.1)));
    s = fast_two_sum_ds(s, _mm512_mul_pd(_mm512_mul_pd(x2, x.0), t));

    let r = _mm512_add_pd(s.0, s.1);

    const CONTROL: i64 = (4 << (2 * 4)) | (3 << (4 * 4)) | (5 << (5 * 4)) | (2 << (6 * 4));
    _mm512_fixupimm_pd(r, d, _mm512_set1_epi64(CONTROL), 0)
}

// https://en.wikipedia.org/wiki/2Sum
#[inline]
#[target_feature(enable = "avx512f")]
fn two_sum_ss(a: __m512d, b: __m512d) -> (__m512d, __m512d) {
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
fn fast_two_sum_dd(s: (__m512d, __m512d), o: (__m512d, __m512d)) -> (__m512d, __m512d) {
    let r0 = _mm512_add_pd(s.0, o.0);

    let a = _mm512_sub_pd(s.0, r0);
    let b = _mm512_add_pd(a, o.0);
    let c = _mm512_add_pd(b, s.1);
    let d = _mm512_add_pd(c, o.1);
    (r0, d)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn fast_two_sum_ds(s: (__m512d, __m512d), o: __m512d) -> (__m512d, __m512d) {
    let r0 = _mm512_add_pd(s.0, o);

    let a = _mm512_sub_pd(s.0, r0);
    let b = _mm512_add_pd(a, o);
    let c = _mm512_add_pd(b, s.1);
    (r0, c)
}

// Assumption: e_a >= e_b
// https://en.wikipedia.org/wiki/2Sum
#[inline]
#[target_feature(enable = "avx512f")]
fn fast_two_sum_ss(a: __m512d, b: __m512d) -> (__m512d, __m512d) {
    let s = _mm512_add_pd(a, b);
    let z = _mm512_sub_pd(s, a);
    let t = _mm512_sub_pd(b, z);
    (s, t)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn mul_ds(s: (__m512d, __m512d), o: __m512d) -> (__m512d, __m512d) {
    let r0 = _mm512_mul_pd(s.0, o);
    let a = _mm512_fmadd_pd(s.1, o, _mm512_fmsub_pd(s.0, o, r0));
    (r0, a)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn div_sd(s: __m512d, o: (__m512d, __m512d)) -> (__m512d, __m512d) {
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
fn poly7(
    x: __m512d,
    x2: __m512d,
    x4: __m512d,
    c6: __m512d,
    c5: __m512d,
    c4: __m512d,
    c3: __m512d,
    c2: __m512d,
    c1: __m512d,
    c0: __m512d,
) -> __m512d {
    _mm512_fmadd_pd(x4, poly3(x, x2, c6, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn poly3(x: __m512d, x2: __m512d, c2: __m512d, c1: __m512d, c0: __m512d) -> __m512d {
    _mm512_fmadd_pd(x2, c2, _mm512_fmadd_pd(x, c1, c0))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn poly4(x: __m512d, x2: __m512d, c3: __m512d, c2: __m512d, c1: __m512d, c0: __m512d) -> __m512d {
    _mm512_fmadd_pd(x2, _mm512_fmadd_pd(x, c3, c2), _mm512_fmadd_pd(x, c1, c0))
}
