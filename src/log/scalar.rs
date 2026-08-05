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

fn log_scalar(mut d: f64) -> f64 {
    // Normalize d if subnormal
    let o = d < f64::MIN_POSITIVE;
    if o {
        d *= D1_32 * D1_32;
    }

    let mut e = ilogb2k(d * (1.0 / 0.75));
    // This mantissa will always be within [0.75, 1.5)
    let m = ldexp3k(d, -e);

    if o {
        e -= 64;
    }

    let u = m - 1.0;
    let l = two_sum_ss(m, 1.0);

    let x = div_sd(u, l);
    let x2 = x.0 * x.0;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let t = poly7(x2, x4, x8, C_6, C_5, C_4, C_3, C_2, C_1, C_0);

    let mut s = mul_ds((LOG_2_HI, LOG_2_LO), e as f64);
    s = fast_two_sum_dd(s, (x.0 + x.0, x.1 + x.1));
    s = fast_two_sum_ds(s, (x2 * x.0) * t);

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
    // The addition will never overflow for floats
    f64::from_bits(((d.to_bits() as i64).wrapping_add((e as i64) << 52)) as u64)
}

#[inline]
fn two_sum_ss(s: f64, o: f64) -> (f64, f64) {
    let r0 = s + o;
    let v = r0 - s;

    let a = r0 - v;
    let b = o - v;
    let c = s - a;
    let d = c + b;
    (r0, d)
}

#[inline]
fn fast_two_sum_dd(s: (f64, f64), o: (f64, f64)) -> (f64, f64) {
    let r0 = s.0 + o.0;

    let a = s.0 - r0;
    let b = a + o.0;
    let c = b + s.1;
    let d = c + o.1;
    (r0, d)
}

#[inline]
fn fast_two_sum_ds(s: (f64, f64), o: f64) -> (f64, f64) {
    let r0 = s.0 + o;

    let a = s.0 - r0;
    let b = a + o;
    let c = b + s.1;
    (r0, c)
}

#[inline]
fn mul_ds(s: (f64, f64), o: f64) -> (f64, f64) {
    let r0 = s.0 * o;
    let a = s.1.mul_add(o, s.0.mul_add(o, -r0));
    (r0, a)
}

#[inline]
fn div_sd(s: f64, o: (f64, f64)) -> (f64, f64) {
    let t = 1.0 / o.0;

    let q0 = s * t;
    let u = t.mul_add(s, -q0);

    let mut q1 = (-o.1).mul_add(t, (-o.0).mul_add(t, 1.0));
    q1 = q0.mul_add(q1, u);

    (q0, q1)
}

#[inline]
fn poly7(
    x: f64,
    x2: f64,
    x4: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> f64 {
    x4.mul_add(poly3(x, x2, c6, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

#[inline]
fn poly3(x: f64, x2: f64, c2: f64, c1: f64, c0: f64) -> f64 {
    x2.mul_add(c2, x.mul_add(c1, c0))
}

#[inline]
fn poly4(x: f64, x2: f64, c3: f64, c2: f64, c1: f64, c0: f64) -> f64 {
    x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0))
}
