use core::arch::aarch64::*;

use super::{C_0, C_1, C_2, C_3, C_4, C_5, C_6, D1_32, LOG_2_HI, LOG_2_LO, Log};

impl Log for float64x2_t {
    #[inline]
    fn log(self) -> Self {
        unsafe { log_neon(self) }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn log_neon(mut d: float64x2_t) -> float64x2_t {
    // Normalize d if subnormal
    let mask_subnormal = vcltq_f64(d, vdupq_n_f64(f64::MIN_POSITIVE));
    let nd = vmulq_f64(d, vdupq_n_f64(D1_32 * D1_32));
    d = vbslq_f64(mask_subnormal, nd, d);

    let sd = vmulq_f64(d, vdupq_n_f64(1.0 / 0.75));
    let mut e = ilogb2k(sd);
    let m = ldexp3k(d, e);

    let zero = vdupq_n_f64(0.0);
    let one = vdupq_n_f64(1.0);

    // Subtract by 64 to account for normalizing the subnormals
    let offset = vandq_s64(vreinterpretq_s64_u64(mask_subnormal), vdupq_n_s64(64));
    e = vsubq_s64(e, offset);

    let u = vsubq_f64(m, one);
    let l = super::fast_two_sum_ss(one, m);

    let x = super::div_sd(u, l);
    let x2 = vmulq_f64(x.0, x.0);
    let x4 = vmulq_f64(x2, x2);
    let x8 = vmulq_f64(x4, x4);

    let t = super::poly7(
        x2,
        x4,
        x8,
        vdupq_n_f64(C_6),
        vdupq_n_f64(C_5),
        vdupq_n_f64(C_4),
        vdupq_n_f64(C_3),
        vdupq_n_f64(C_2),
        vdupq_n_f64(C_1),
        vdupq_n_f64(C_0),
    );

    let ed = vcvtq_f64_s64(e);

    let mut s = super::mul_ds((vdupq_n_f64(LOG_2_HI), vdupq_n_f64(LOG_2_LO)), ed);
    s = super::fast_two_sum_dd(s, (vaddq_f64(x.0, x.0), vaddq_f64(x.1, x.1)));
    s = super::fast_two_sum_ds(s, vmulq_f64(vmulq_f64(x2, x.0), t));

    let mask_zero = vceqq_f64(d, zero);
    // If not (d >= 0.0), then either negative or NaN
    let mask_neg_nan = vmvnq_u64(vcgteq_f64(d, zero));
    let mask_inf = vceqq_f64(d, vdupq_n_f64(f64::INFINITY));

    let mut result = vaddq_f64(s.0, s.1);
    result = vbslq_f64(mask_zero, vdupq_n_f64(f64::NEG_INFINITY), result);
    result = vbslq_f64(mask_neg_nan, vdupq_n_f64(f64::NAN), result);
    result = vbslq_f64(mask_inf, vdupq_n_f64(f64::INFINITY), result);

    result
}

// Both of these assume the input is not subnormal
#[inline]
#[target_feature(enable = "neon")]
fn ilogb2k(d: float64x2_t) -> int64x2_t {
    let bits = vreinterpretq_u64_f64(d);
    let shifted = vshrq_n_u64::<52>(bits);
    let masked = vandq_u64(shifted, vdupq_n_u64(0x7ff));
    let e = vsubq_s64(vreinterpretq_s64_u64(masked), vdupq_n_s64(0x3ff));
    e
}

#[inline]
#[target_feature(enable = "neon")]
fn ldexp3k(d: float64x2_t, e: int64x2_t) -> float64x2_t {
    let bits = vreinterpretq_s64_f64(d);
    let shifted = vshlq_n_s64::<52>(e);
    let diff = vsubq_s64(bits, shifted);
    let m = vreinterpretq_f64_s64(diff);
    m
}
