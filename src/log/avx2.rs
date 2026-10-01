use core::arch::x86_64::*;

use super::{C_0, C_1, C_2, C_3, C_4, C_5, C_6, D1_32, LOG_2_HI, LOG_2_LO, Log};

impl Log for __m256d {
    #[inline]
    fn log(self) -> Self {
        unsafe { log_avx2(self) }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn log_avx2(mut d: __m256d) -> __m256d {
    // Normalize d if subnormal
    let maskd = _mm256_cmp_pd(d, _mm256_set1_pd(f64::MIN_POSITIVE), _CMP_LT_OQ);
    let nd = _mm256_mul_pd(d, _mm256_set1_pd(D1_32 * D1_32));
    d = _mm256_blendv_pd(d, nd, maskd);

    let sd = _mm256_mul_pd(d, _mm256_set1_pd(1.0 / 0.75));
    let mut e = ilogb2k(sd);
    let m = ldexp3k(d, e);

    let zero = _mm256_setzero_pd();
    let one = _mm256_set1_pd(1.0);

    // Subtract by 64 to account for normalizing the subnormals
    let maski = _mm256_castpd_si256(maskd);
    let offset = _mm256_and_si256(maski, _mm256_set1_epi64x(64));
    e = _mm256_sub_epi64(e, offset);

    let u = _mm256_sub_pd(m, one);
    let l = super::fast_two_sum_ss(one, m);

    let x = super::div_sd(u, l);
    let x2 = _mm256_mul_pd(x.0, x.0);
    let x4 = _mm256_mul_pd(x2, x2);
    let x8 = _mm256_mul_pd(x4, x4);

    let t = super::poly7(
        x2,
        x4,
        x8,
        _mm256_set1_pd(C_6),
        _mm256_set1_pd(C_5),
        _mm256_set1_pd(C_4),
        _mm256_set1_pd(C_3),
        _mm256_set1_pd(C_2),
        _mm256_set1_pd(C_1),
        _mm256_set1_pd(C_0),
    );

    // This is needed to cast i64 e to f64, incredibly cursed
    let shuf = _mm256_shuffle_epi32(e, 0x08);
    let perm = _mm256_permute4x64_epi64(shuf, 0x08);
    let e128 = _mm256_castsi256_si128(perm);
    let ed = _mm256_cvtepi32_pd(e128);

    let mut s = super::mul_ds((_mm256_set1_pd(LOG_2_HI), _mm256_set1_pd(LOG_2_LO)), ed);
    s = super::fast_two_sum_dd(s, (_mm256_add_pd(x.0, x.0), _mm256_add_pd(x.1, x.1)));
    s = super::fast_two_sum_ds(s, _mm256_mul_pd(_mm256_mul_pd(x2, x.0), t));

    let mask_zero = _mm256_cmp_pd(d, zero, _CMP_EQ_OQ);
    // d NGE_UQ 0.0 returns true if d is negative or a nan
    let mask_neg_nan = _mm256_cmp_pd(d, zero, _CMP_NGE_UQ);
    let mask_inf = _mm256_cmp_pd(d, _mm256_set1_pd(f64::INFINITY), _CMP_EQ_OQ);

    let mut result = _mm256_add_pd(s.0, s.1);
    result = _mm256_blendv_pd(result, _mm256_set1_pd(f64::NEG_INFINITY), mask_zero);
    result = _mm256_blendv_pd(result, _mm256_set1_pd(f64::NAN), mask_neg_nan);
    result = _mm256_blendv_pd(result, _mm256_set1_pd(f64::INFINITY), mask_inf);

    result
}

// Both of these assume the input is not subnormal
#[inline]
#[target_feature(enable = "avx2")]
fn ilogb2k(d: __m256d) -> __m256i {
    let bits = _mm256_castpd_si256(d);
    let shifted = _mm256_srli_epi64(bits, 52);
    let masked = _mm256_and_si256(shifted, _mm256_set1_epi64x(0x7ff));
    let e = _mm256_sub_epi64(masked, _mm256_set1_epi64x(0x3ff));
    e
}

#[inline]
#[target_feature(enable = "avx2")]
fn ldexp3k(d: __m256d, e: __m256i) -> __m256d {
    let bits = _mm256_castpd_si256(d);
    let shifted = _mm256_slli_epi64(e, 52);
    let diff = _mm256_sub_epi64(bits, shifted);
    let m = _mm256_castsi256_pd(diff);
    m
}
