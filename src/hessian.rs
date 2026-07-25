use crate::{matrix::Block, util::{Matrix9, Matrix9xN, Vector9, dot}};

use core::arch::x86_64::*;

pub fn compute_hess(
    p_mat_t: &Matrix9xN<f64>,
    x: &Vector9<f64>,
    eps: f64,
) -> Matrix9<f64> {

    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    let (p, _) = p_mat_t.as_slice().as_chunks::<9>();

    let ptd2p = compute_pt_d2_p_scalar(p, &x0, eps);

    let n = p.len();

    Matrix9::from_fn(|i, j| ptd2p[i][j] / (n as f64))
}

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {

    let one = _mm512_set1_pd(1.0);

    let mut h = [[0.0; 9]; 9];

    // ---------------------------
    // Pass 1
    // ---------------------------

    // First row
    let mut h00 = _mm512_setzero_pd();
    let mut h01 = _mm512_setzero_pd();
    let mut h02 = _mm512_setzero_pd();
    let mut h03 = _mm512_setzero_pd();
    let mut h04 = _mm512_setzero_pd();
    let mut h05 = _mm512_setzero_pd();
    let mut h06 = _mm512_setzero_pd();
    let mut h07 = _mm512_setzero_pd();
    let mut h08 = _mm512_setzero_pd();
    // Second row
    let mut h11 = _mm512_setzero_pd();
    let mut h12 = _mm512_setzero_pd();
    let mut h13 = _mm512_setzero_pd();
    let mut h14 = _mm512_setzero_pd();
    let mut h15 = _mm512_setzero_pd();
    let mut h16 = _mm512_setzero_pd();
    let mut h17 = _mm512_setzero_pd();
    let mut h18 = _mm512_setzero_pd();
    // Third row
    let mut h22 = _mm512_setzero_pd();
    let mut h23 = _mm512_setzero_pd();
    let mut h24 = _mm512_setzero_pd();
    let mut h25 = _mm512_setzero_pd();
    let mut h26 = _mm512_setzero_pd();
    let mut h27 = _mm512_setzero_pd();
    let mut h28 = _mm512_setzero_pd();

    // let mut hrow0 = [_mm512_setzero_pd(); 9];
    // let mut hrow1 = [_mm512_setzero_pd(); 8];
    // let mut hrow2 = [_mm512_setzero_pd(); 7];

    for block in blocks.iter() {
        // Calculate d
        // This computes a dot product between x and a row of p
        // TODO this could be manually unrolled a few times
        let mut d = _mm512_set1_pd(eps);
        for col in 0..9 {
            let column = unsafe { _mm512_load_pd(block[col].as_ptr()) };
            d = _mm512_fmadd_pd(_mm512_set1_pd(x[col]), column, d);
        }

        // TODO investigate reciprocal
        d = _mm512_div_pd(one, d);

        // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

        let s0 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[0].as_ptr()) }, d);
        let s1 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[1].as_ptr()) }, d);
        let s2 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[2].as_ptr()) }, d);

        // Accumulate the triangle
        // hrow0[0] = _mm512_fmadd_pd(s0, s0, hrow0[0]);
        // hrow0[1] = _mm512_fmadd_pd(s0, s1, hrow0[1]);
        // hrow0[2] = _mm512_fmadd_pd(s0, s2, hrow0[2]);

        // hrow1[0] = _mm512_fmadd_pd(s1, s1, hrow1[0]);
        // hrow1[1] = _mm512_fmadd_pd(s1, s2, hrow1[1]);

        // hrow2[0] = _mm512_fmadd_pd(s2, s2, hrow2[0]);

        h00 = _mm512_fmadd_pd(s0, s0, h00);
        h01 = _mm512_fmadd_pd(s0, s1, h01);
        h02 = _mm512_fmadd_pd(s0, s2, h02);

        h11 = _mm512_fmadd_pd(s1, s1, h11);
        h12 = _mm512_fmadd_pd(s1, s2, h12);

        h22 = _mm512_fmadd_pd(s2, s2, h22);

        // for j in 3..9 {
        //     let scaled = _mm512_mul_pd(unsafe { _mm512_load_pd(block[j].as_ptr()) }, d);
        //     hrow0[j] = _mm512_fmadd_pd(s0, scaled, hrow0[j]);
        //     hrow1[j-1] = _mm512_fmadd_pd(s1, scaled, hrow1[j-1]);
        //     hrow2[j-2] = _mm512_fmadd_pd(s2, scaled, hrow2[j-2]);
        // }

        // Column 3
        let s3 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[3].as_ptr()) }, d);
        h03 = _mm512_fmadd_pd(s0, s3, h03);
        h13 = _mm512_fmadd_pd(s1, s3, h13);
        h23 = _mm512_fmadd_pd(s2, s3, h23);

        // Column 4
        let s4 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[4].as_ptr()) }, d);
        h04 = _mm512_fmadd_pd(s0, s4, h04);
        h14 = _mm512_fmadd_pd(s1, s4, h14);
        h24 = _mm512_fmadd_pd(s2, s4, h24);

        // Column 5
        let s5 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[5].as_ptr()) }, d);
        h05 = _mm512_fmadd_pd(s0, s5, h05);
        h15 = _mm512_fmadd_pd(s1, s5, h15);
        h25 = _mm512_fmadd_pd(s2, s5, h25);

        // Column 6
        let s6 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[6].as_ptr()) }, d);
        h06 = _mm512_fmadd_pd(s0, s6, h06);
        h16 = _mm512_fmadd_pd(s1, s6, h16);
        h26 = _mm512_fmadd_pd(s2, s6, h26);

        // Column 7
        let s7 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[7].as_ptr()) }, d);
        h07 = _mm512_fmadd_pd(s0, s7, h07);
        h17 = _mm512_fmadd_pd(s1, s7, h17);
        h27 = _mm512_fmadd_pd(s2, s7, h27);

        // Column 8
        let s8 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[8].as_ptr()) }, d);
        h08 = _mm512_fmadd_pd(s0, s8, h08);
        h18 = _mm512_fmadd_pd(s1, s8, h18);
        h28 = _mm512_fmadd_pd(s2, s8, h28);
    }

    // let regs = [
    //     hrow0[0],
    //     hrow0[1],
    //     hrow0[1],
    //     hrow0[1],

    // ]

    // for j in 0..9 {
    //     h[0][j] = _mm512_reduce_add_pd(hrow0[j]);
    // }

    // for j in 0..8 {
    //     h[1][j+1] = _mm512_reduce_add_pd(hrow1[j]);
    // }

    // for j in 0..7 {
    //     h[2][j+2] = _mm512_reduce_add_pd(hrow2[j]);
    // }

    h[0][0] =_mm512_reduce_add_pd(h00);
    h[0][1] =_mm512_reduce_add_pd(h01);
    h[0][2] =_mm512_reduce_add_pd(h02);
    h[0][3] =_mm512_reduce_add_pd(h03);
    h[0][4] =_mm512_reduce_add_pd(h04);
    h[0][5] =_mm512_reduce_add_pd(h05);
    h[0][6] =_mm512_reduce_add_pd(h06);
    h[0][7] =_mm512_reduce_add_pd(h07);
    h[0][8] =_mm512_reduce_add_pd(h08);

    h[1][1] =_mm512_reduce_add_pd(h11);
    h[1][2] =_mm512_reduce_add_pd(h12);
    h[1][3] =_mm512_reduce_add_pd(h13);
    h[1][4] =_mm512_reduce_add_pd(h14);
    h[1][5] =_mm512_reduce_add_pd(h15);
    h[1][6] =_mm512_reduce_add_pd(h16);
    h[1][7] =_mm512_reduce_add_pd(h17);
    h[1][8] =_mm512_reduce_add_pd(h18);

    h[2][2] =_mm512_reduce_add_pd(h22);
    h[2][3] =_mm512_reduce_add_pd(h23);
    h[2][4] =_mm512_reduce_add_pd(h24);
    h[2][5] =_mm512_reduce_add_pd(h25);
    h[2][6] =_mm512_reduce_add_pd(h26);
    h[2][7] =_mm512_reduce_add_pd(h27);
    h[2][8] =_mm512_reduce_add_pd(h28);

    // ---------------------------
    // Pass 2
    // ---------------------------

    // let mut hrow3 = [_mm512_setzero_pd(); 6];
    // let mut hrow4 = [_mm512_setzero_pd(); 5];
    // let mut hrow5 = [_mm512_setzero_pd(); 4];
    // let mut hrow6 = [_mm512_setzero_pd(); 3];
    // let mut hrow7 = [_mm512_setzero_pd(); 2];
    // let mut hrow8 = [_mm512_setzero_pd(); 1];
    h



}


// P^T D^2 P
pub fn compute_pt_d2_p_scalar(
    p_mat: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {

    let mut h = [[0.0; 9]; 9];

    for row in p_mat.iter() {
        let prod = dot(row, x);

        // d = 1 / (P x + eps)
        let d = 1.0 / (prod + eps);

        let mut scaled_row = [0.0; 9];
        for i in 0..9 {
            scaled_row[i] = d * row[i];
        }

        // g = P^T d
        for i in 0..9 {
            // In theory we could iterate from i..9,
            // but that causes a stack spill.
            for j in 0..9 {
                h[i][j] = scaled_row[i].mul_add(scaled_row[j], h[i][j]);
            }
        }
    }

    // for i in 0..9 {
    //     for j in i..9 {
    //         h[j][i] = h[i][j];
    //     }
    // }

    h
}

pub fn compute_pt_d2_p_scalar2(
    p_mat: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {

    let mut h = [[0.0; 9]; 9];

    for row in p_mat.iter() {
        let prod = dot(row, x);

        // d = 1 / (P x + eps)
        let d = 1.0 / (prod + eps);

        let mut scaled_row = [0.0; 9];
        for i in 0..9 {
            scaled_row[i] = d * row[i];
        }

        // g = P^T d
        for i in 0..9 {
            for j in i..9 {
                h[i][j] = scaled_row[i].mul_add(scaled_row[j], h[i][j]);
            }
        }
    }

    for i in 0..9 {
        for j in i..9 {
            h[j][i] = h[i][j];
        }
    }

    h
}

#[target_feature(enable = "avx512f")]
pub unsafe fn reduce_512_to_256(v: __m512d) -> __m256d {
    let low = _mm512_castpd512_pd256(v);
    let high = _mm512_extractf64x4_pd::<1>(v);
    _mm256_add_pd(low, high)
}

#[target_feature(enable = "avx512f")]
pub unsafe fn reduce_256_to_128(v: __m256d) -> __m128d {
    let low = _mm256_castpd256_pd128(v);
    let high = _mm256_extractf128_pd::<1>(v);
    _mm_add_pd(low, high)
}

#[target_feature(enable = "avx512f")]
pub unsafe fn reduce_128_to_64(v: __m128d) -> f64 {
    let high = _mm_unpackhi_pd(v, v);
    let sum = _mm_add_sd(v, high);
    _mm_cvtsd_f64(sum)
}

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_three_passes(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {

    let one = _mm512_set1_pd(1.0);

    let mut h = [[0.0; 9]; 9];

    // ---------------------------
    // Pass 1
    // ---------------------------

    // First row
    let mut h00 = _mm512_setzero_pd();
    let mut h01 = _mm512_setzero_pd();
    let mut h02 = _mm512_setzero_pd();
    let mut h03 = _mm512_setzero_pd();
    let mut h04 = _mm512_setzero_pd();
    let mut h05 = _mm512_setzero_pd();
    let mut h06 = _mm512_setzero_pd();
    let mut h07 = _mm512_setzero_pd();
    let mut h08 = _mm512_setzero_pd();
    // Second row
    let mut h11 = _mm512_setzero_pd();
    let mut h12 = _mm512_setzero_pd();
    let mut h13 = _mm512_setzero_pd();
    let mut h14 = _mm512_setzero_pd();
    let mut h15 = _mm512_setzero_pd();
    let mut h16 = _mm512_setzero_pd();

    for block in blocks.iter() {
        let mut c: [__m512d; 9] = std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

        // Calculate d
        // This computes a dot product between x and a row of p
        // TODO this could be manually unrolled a few times
        let mut d = _mm512_set1_pd(eps);
        for col in 0..9 {
            d = _mm512_fmadd_pd(_mm512_set1_pd(x[col]), c[col], d);
        }

        // TODO investigate reciprocal
        d = _mm512_div_pd(one, d);

        // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

        for col in 0.. 9 {
            c[col] = _mm512_mul_pd(c[col], d);
        }

        h00 = _mm512_fmadd_pd(c[0], c[0], h00);
        h01 = _mm512_fmadd_pd(c[0], c[1], h01);
        h02 = _mm512_fmadd_pd(c[0], c[2], h02);
        h03 = _mm512_fmadd_pd(c[0], c[3], h03);
        h04 = _mm512_fmadd_pd(c[0], c[4], h04);
        h05 = _mm512_fmadd_pd(c[0], c[5], h05);
        h06 = _mm512_fmadd_pd(c[0], c[6], h06);
        h07 = _mm512_fmadd_pd(c[0], c[7], h07);
        h08 = _mm512_fmadd_pd(c[0], c[8], h08);

        h11 = _mm512_fmadd_pd(c[1], c[1], h11);
        h12 = _mm512_fmadd_pd(c[1], c[2], h12);
        h13 = _mm512_fmadd_pd(c[1], c[3], h13);
        h14 = _mm512_fmadd_pd(c[1], c[4], h14);
        h15 = _mm512_fmadd_pd(c[1], c[5], h15);
        h16 = _mm512_fmadd_pd(c[1], c[6], h16);
    }

    // First row
    h[0][0] =_mm512_reduce_add_pd(h00);
    h[0][1] =_mm512_reduce_add_pd(h01);
    h[0][2] =_mm512_reduce_add_pd(h02);
    h[0][3] =_mm512_reduce_add_pd(h03);
    h[0][4] =_mm512_reduce_add_pd(h04);
    h[0][5] =_mm512_reduce_add_pd(h05);
    h[0][6] =_mm512_reduce_add_pd(h06);
    h[0][7] =_mm512_reduce_add_pd(h07);
    h[0][8] =_mm512_reduce_add_pd(h08);

    // Second row
    h[1][1] =_mm512_reduce_add_pd(h11);
    h[1][2] =_mm512_reduce_add_pd(h12);
    h[1][3] =_mm512_reduce_add_pd(h13);
    h[1][4] =_mm512_reduce_add_pd(h14);
    h[1][5] =_mm512_reduce_add_pd(h15);
    h[1][6] =_mm512_reduce_add_pd(h16);

    h
}