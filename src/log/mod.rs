use crate::arith::Arith;

const D1_32: f64 = (1u64 << 32) as f64;

const LOG_2_HI: f64 = 0.693_147_180_559_945_286_226_764;
const LOG_2_LO: f64 = 2.319_046_813_846_299_558_417_771_e-17;

const C_6: f64 = 0.153_207_698_850_270_135_3;
const C_5: f64 = 0.152_562_905_100_342_871_6;
const C_4: f64 = 0.181_860_593_293_778_599_6;
const C_3: f64 = 0.222_221_451_983_938_000_9;
const C_2: f64 = 0.285_714_293_279_429_931_7;
const C_1: f64 = 0.399_999_999_963_525_199;
const C_0: f64 = 0.666_666_666_666_733_354_1;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
pub mod scalar;

pub trait Log {
    fn log(self) -> Self;
}

// Assumption: e_a >= e_b
// https://en.wikipedia.org/wiki/2Sum
// (a + b, (b - ((a + b) - a)))
#[inline]
fn fast_two_sum_ss<S: Arith>(a: S, b: S) -> (S, S) {
    let s = a.add(b);
    let z = s.sub(a);
    let t = b.sub(z);
    (s, t)
}

#[inline]
fn fast_two_sum_ds<S: Arith>(a: (S, S), b: S) -> (S, S) {
    let (s, t) = fast_two_sum_ss(a.0, b);
    let e = t.add(a.1);
    (s, e)
}

#[inline]
fn fast_two_sum_dd<S: Arith>(a: (S, S), b: (S, S)) -> (S, S) {
    let (s, t) = fast_two_sum_ss(a.0, b.0);
    // The sleef code calculates (t + a.1) + b.1
    // We calculate t + (a.1 + b.1), which should be just as accurate
    // but also faster, since the addition can be done independent of
    // calculating t.
    let e = a.1.add(b.1);
    let e = t.add(e);
    (s, e)
}

#[inline]
fn fast_two_mult_ss<S: Arith>(a: S, b: S) -> (S, S) {
    let p = a.mul(b);
    let r = a.fms(b, p);
    (p, r)
}

#[inline]
fn mul_ds<S: Arith>(a: (S, S), b: S) -> (S, S) {
    let (p, r) = fast_two_mult_ss(a.0, b);
    let e = a.1.fma(b, r);
    (p, e)
}

#[inline]
fn div_sd<S: Arith>(a: S, b: (S, S)) -> (S, S) {
    let one = S::set(1.0);

    let r = one.div(b.0);

    let (q0, u) = fast_two_mult_ss(a, r);

    let e = b.0.fnma(r, one);
    let q1 = b.1.fnma(r, e);
    let q1 = q0.fma(q1, u);

    (q0, q1)
}

#[inline]
fn poly7<S: Arith>(x: S, x2: S, x4: S, c6: S, c5: S, c4: S, c3: S, c2: S, c1: S, c0: S) -> S {
    x4.fma(poly3(x, x2, c6, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

#[inline]
fn poly3<S: Arith>(x: S, x2: S, c2: S, c1: S, c0: S) -> S {
    x2.fma(c2, x.fma(c1, c0))
}

#[inline]
fn poly4<S: Arith>(x: S, x2: S, c3: S, c2: S, c1: S, c0: S) -> S {
    x2.fma(x.fma(c3, c2), x.fma(c1, c0))
}
