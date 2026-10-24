use super::{C_0, C_1, C_2, C_3, C_4, C_5, C_6, D1_32, LOG_2_HI, LOG_2_LO, Log};

// Why not just use the ln() function from libm? In the objective calculation, using
// that requires a function call, which requires storing registers on the stack.
// LLVM generates sub-optimal assembly for the ZMM registers, so it's better to
// make our own that can be inlined.

// Adapted from https://github.com/burrbull/sleef-rs/blob/master/src/f64/u10.rs

impl Log for f64 {
    #[inline]
    fn log(self) -> Self {
        log_scalar(self)
    }
}

#[inline]
fn log_scalar(mut d: f64) -> f64 {
    // Normalize d if subnormal
    let o = d < f64::MIN_POSITIVE;
    if o {
        d *= D1_32 * D1_32;
    }

    // In IEEE, d = m * 2^e , where 1 <= m < 2.
    // However, we want to write d = m' * 2^e', where 0.75 <= m' < 1.5.
    let mut e = ilogb2k(d * (1.0 / 0.75));
    // This mantissa will always be within [0.75, 1.5)
    let m = ldexp3k(d, e);

    if o {
        e -= 64;
    }

    let u = m - 1.0;
    let l = super::fast_two_sum_ss(1.0, m);

    let x = super::div_sd(u, l);
    let x2 = x.0 * x.0;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let t = super::poly7(x2, x4, x8, C_6, C_5, C_4, C_3, C_2, C_1, C_0);

    let mut s = super::mul_ds((LOG_2_HI, LOG_2_LO), e as f64);
    s = super::fast_two_sum_dd(s, (x.0 + x.0, x.1 + x.1));
    s = super::fast_two_sum_ds(s, (x2 * x.0) * t);

    if d == 0.0 {
        f64::NEG_INFINITY
    } else if (d < 0.0) || d.is_nan() {
        f64::NAN
    } else if d.is_infinite() {
        f64::INFINITY
    } else {
        s.0 + s.1
    }
}

// Both of these assume the input is not subnormal
#[inline]
fn ilogb2k(d: f64) -> i32 {
    (((d.to_bits() >> 52) & 0x7ff) as i32) - 0x3ff
}

#[inline]
fn ldexp3k(d: f64, e: i32) -> f64 {
    // For two's complement arithmetic, -e << 52  = (-e) * 2^52 = -(e * 2^52) = -(e << 52)
    f64::from_bits(((d.to_bits() as i64) - ((e as i64) << 52)) as u64)
}
