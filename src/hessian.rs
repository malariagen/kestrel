use crate::{
    buffer::Lane8,
    matrix::Block,
    util::{Matrix9, Matrix9xN, Vector9, dot},
};

use core::arch::x86_64::*;

pub fn compute_hess(p_mat_t: &Matrix9xN<f64>, x: &Vector9<f64>, eps: f64) -> Matrix9<f64> {
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

    h[0][0] = _mm512_reduce_add_pd(h00);
    h[0][1] = _mm512_reduce_add_pd(h01);
    h[0][2] = _mm512_reduce_add_pd(h02);
    h[0][3] = _mm512_reduce_add_pd(h03);
    h[0][4] = _mm512_reduce_add_pd(h04);
    h[0][5] = _mm512_reduce_add_pd(h05);
    h[0][6] = _mm512_reduce_add_pd(h06);
    h[0][7] = _mm512_reduce_add_pd(h07);
    h[0][8] = _mm512_reduce_add_pd(h08);

    h[1][1] = _mm512_reduce_add_pd(h11);
    h[1][2] = _mm512_reduce_add_pd(h12);
    h[1][3] = _mm512_reduce_add_pd(h13);
    h[1][4] = _mm512_reduce_add_pd(h14);
    h[1][5] = _mm512_reduce_add_pd(h15);
    h[1][6] = _mm512_reduce_add_pd(h16);
    h[1][7] = _mm512_reduce_add_pd(h17);
    h[1][8] = _mm512_reduce_add_pd(h18);

    h[2][2] = _mm512_reduce_add_pd(h22);
    h[2][3] = _mm512_reduce_add_pd(h23);
    h[2][4] = _mm512_reduce_add_pd(h24);
    h[2][5] = _mm512_reduce_add_pd(h25);
    h[2][6] = _mm512_reduce_add_pd(h26);
    h[2][7] = _mm512_reduce_add_pd(h27);
    h[2][8] = _mm512_reduce_add_pd(h28);

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
pub fn compute_pt_d2_p_scalar(p_mat: &[[f64; 9]], x: &[f64; 9], eps: f64) -> [[f64; 9]; 9] {
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

pub fn compute_pt_d2_p_scalar_fused(p_mat: &[[f64; 9]], x: &[f64; 9], eps: f64) -> [[f64; 9]; 9] {
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

pub fn compute_pt_d2_p_scalar2(p_mat: &[[f64; 9]], x: &[f64; 9], eps: f64) -> [[f64; 9]; 9] {
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
        let mut c: [__m512d; 9] =
            std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

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

        for col in 0..9 {
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
    h[0][0] = _mm512_reduce_add_pd(h00);
    h[0][1] = _mm512_reduce_add_pd(h01);
    h[0][2] = _mm512_reduce_add_pd(h02);
    h[0][3] = _mm512_reduce_add_pd(h03);
    h[0][4] = _mm512_reduce_add_pd(h04);
    h[0][5] = _mm512_reduce_add_pd(h05);
    h[0][6] = _mm512_reduce_add_pd(h06);
    h[0][7] = _mm512_reduce_add_pd(h07);
    h[0][8] = _mm512_reduce_add_pd(h08);

    // Second row
    h[1][1] = _mm512_reduce_add_pd(h11);
    h[1][2] = _mm512_reduce_add_pd(h12);
    h[1][3] = _mm512_reduce_add_pd(h13);
    h[1][4] = _mm512_reduce_add_pd(h14);
    h[1][5] = _mm512_reduce_add_pd(h15);
    h[1][6] = _mm512_reduce_add_pd(h16);

    h
}

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_one_pass(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {
    const BLOCKS: usize = 32;

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
    // Fourth row
    let mut h33 = _mm512_setzero_pd();
    let mut h34 = _mm512_setzero_pd();
    let mut h35 = _mm512_setzero_pd();
    let mut h36 = _mm512_setzero_pd();
    let mut h37 = _mm512_setzero_pd();
    let mut h38 = _mm512_setzero_pd();
    // Fifth row
    let mut h44 = _mm512_setzero_pd();
    let mut h45 = _mm512_setzero_pd();
    let mut h46 = _mm512_setzero_pd();
    let mut h47 = _mm512_setzero_pd();
    let mut h48 = _mm512_setzero_pd();
    // Sixth row
    let mut h55 = _mm512_setzero_pd();
    let mut h56 = _mm512_setzero_pd();
    let mut h57 = _mm512_setzero_pd();
    let mut h58 = _mm512_setzero_pd();
    // Seventh row
    let mut h66 = _mm512_setzero_pd();
    let mut h67 = _mm512_setzero_pd();
    let mut h68 = _mm512_setzero_pd();
    // Eigth row
    let mut h77 = _mm512_setzero_pd();
    let mut h78 = _mm512_setzero_pd();
    // Ninth row
    let mut h88 = _mm512_setzero_pd();

    // A block = [Lane8; 9]
    // This is one tile
    let mut scaled_column_buf = [[Lane8::zero(); 9]; BLOCKS];

    let (tiles, rem) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        fun_name(x, eps, &mut scaled_column_buf, tile);

        for block in scaled_column_buf.iter() {
            let c: [__m512d; 9] =
                std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

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
            h17 = _mm512_fmadd_pd(c[1], c[7], h17);
            h18 = _mm512_fmadd_pd(c[1], c[8], h18);

            h22 = _mm512_fmadd_pd(c[2], c[2], h22);
            h23 = _mm512_fmadd_pd(c[2], c[3], h23);
            h24 = _mm512_fmadd_pd(c[2], c[4], h24);
            h25 = _mm512_fmadd_pd(c[2], c[5], h25);
            h26 = _mm512_fmadd_pd(c[2], c[6], h26);
            h27 = _mm512_fmadd_pd(c[2], c[7], h27);
            h28 = _mm512_fmadd_pd(c[2], c[8], h28);

            h33 = _mm512_fmadd_pd(c[3], c[3], h33);
            h34 = _mm512_fmadd_pd(c[3], c[4], h34);
            h35 = _mm512_fmadd_pd(c[3], c[5], h35);
            h36 = _mm512_fmadd_pd(c[3], c[6], h36);
            h37 = _mm512_fmadd_pd(c[3], c[7], h37);
            h38 = _mm512_fmadd_pd(c[3], c[8], h38);

            h44 = _mm512_fmadd_pd(c[4], c[4], h44);
            h45 = _mm512_fmadd_pd(c[4], c[5], h45);
            h46 = _mm512_fmadd_pd(c[4], c[6], h46);
            h47 = _mm512_fmadd_pd(c[4], c[7], h47);
            h48 = _mm512_fmadd_pd(c[4], c[8], h48);

            h55 = _mm512_fmadd_pd(c[5], c[5], h55);
            h56 = _mm512_fmadd_pd(c[5], c[6], h56);
            h57 = _mm512_fmadd_pd(c[5], c[7], h57);
            h58 = _mm512_fmadd_pd(c[5], c[8], h58);

            h66 = _mm512_fmadd_pd(c[6], c[6], h66);
            h67 = _mm512_fmadd_pd(c[6], c[7], h67);
            h68 = _mm512_fmadd_pd(c[6], c[8], h68);

            h77 = _mm512_fmadd_pd(c[7], c[7], h77);
            h78 = _mm512_fmadd_pd(c[7], c[8], h78);

            h88 = _mm512_fmadd_pd(c[8], c[8], h88);
        }

        // First row
    }

    let mut h = [[0.0; 9]; 9];

    h[0][0] = _mm512_reduce_add_pd(h00);
    h[0][1] = _mm512_reduce_add_pd(h01);
    h[0][2] = _mm512_reduce_add_pd(h02);
    h[0][3] = _mm512_reduce_add_pd(h03);
    h[0][4] = _mm512_reduce_add_pd(h04);
    h[0][5] = _mm512_reduce_add_pd(h05);
    h[0][6] = _mm512_reduce_add_pd(h06);
    h[0][7] = _mm512_reduce_add_pd(h07);
    h[0][8] = _mm512_reduce_add_pd(h08);

    // Second row
    h[1][1] = _mm512_reduce_add_pd(h11);
    h[1][2] = _mm512_reduce_add_pd(h12);
    h[1][3] = _mm512_reduce_add_pd(h13);
    h[1][4] = _mm512_reduce_add_pd(h14);
    h[1][5] = _mm512_reduce_add_pd(h15);
    h[1][6] = _mm512_reduce_add_pd(h16);
    h[1][7] = _mm512_reduce_add_pd(h17);
    h[1][8] = _mm512_reduce_add_pd(h18);

    h[2][2] = _mm512_reduce_add_pd(h22);
    h[2][3] = _mm512_reduce_add_pd(h23);
    h[2][4] = _mm512_reduce_add_pd(h24);
    h[2][5] = _mm512_reduce_add_pd(h25);
    h[2][6] = _mm512_reduce_add_pd(h26);
    h[2][7] = _mm512_reduce_add_pd(h27);
    h[2][8] = _mm512_reduce_add_pd(h28);

    h[3][3] = _mm512_reduce_add_pd(h33);
    h[3][4] = _mm512_reduce_add_pd(h34);
    h[3][5] = _mm512_reduce_add_pd(h35);
    h[3][6] = _mm512_reduce_add_pd(h36);
    h[3][7] = _mm512_reduce_add_pd(h37);
    h[3][8] = _mm512_reduce_add_pd(h38);

    h[4][4] = _mm512_reduce_add_pd(h44);
    h[4][5] = _mm512_reduce_add_pd(h45);
    h[4][6] = _mm512_reduce_add_pd(h46);
    h[4][7] = _mm512_reduce_add_pd(h47);
    h[4][8] = _mm512_reduce_add_pd(h48);

    h[5][5] = _mm512_reduce_add_pd(h55);
    h[5][6] = _mm512_reduce_add_pd(h56);
    h[5][7] = _mm512_reduce_add_pd(h57);
    h[5][8] = _mm512_reduce_add_pd(h58);

    h[6][6] = _mm512_reduce_add_pd(h66);
    h[6][7] = _mm512_reduce_add_pd(h67);
    h[6][8] = _mm512_reduce_add_pd(h68);

    h[7][7] = _mm512_reduce_add_pd(h77);
    h[7][8] = _mm512_reduce_add_pd(h78);

    h[8][8] = _mm512_reduce_add_pd(h88);

    for i in 0..9 {
        for j in i..9 {
            h[j][i] = h[i][j];
        }
    }

    h
}

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_two_passes(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {
    const BLOCKS: usize = 32;

    // First row
    let mut h00 = Lane8::zero();
    let mut h01 = Lane8::zero();
    let mut h02 = Lane8::zero();
    let mut h03 = Lane8::zero();
    let mut h04 = Lane8::zero();
    let mut h05 = Lane8::zero();
    let mut h06 = Lane8::zero();
    let mut h07 = Lane8::zero();
    let mut h08 = Lane8::zero();
    // Second row
    let mut h11 = Lane8::zero();
    let mut h12 = Lane8::zero();
    let mut h13 = Lane8::zero();
    let mut h14 = Lane8::zero();
    let mut h15 = Lane8::zero();
    let mut h16 = Lane8::zero();
    let mut h17 = Lane8::zero();
    let mut h18 = Lane8::zero();
    // Third row
    let mut h22 = Lane8::zero();
    let mut h23 = Lane8::zero();
    let mut h24 = Lane8::zero();
    let mut h25 = Lane8::zero();
    let mut h26 = Lane8::zero();
    let mut h27 = Lane8::zero();
    let mut h28 = Lane8::zero();
    // Fourth row
    let mut h33 = Lane8::zero();
    let mut h34 = Lane8::zero();
    let mut h35 = Lane8::zero();
    let mut h36 = Lane8::zero();
    let mut h37 = Lane8::zero();
    let mut h38 = Lane8::zero();
    // Fifth row
    let mut h44 = Lane8::zero();
    let mut h45 = Lane8::zero();
    let mut h46 = Lane8::zero();
    let mut h47 = Lane8::zero();
    let mut h48 = Lane8::zero();
    // Sixth row
    let mut h55 = Lane8::zero();
    let mut h56 = Lane8::zero();
    let mut h57 = Lane8::zero();
    let mut h58 = Lane8::zero();
    // Seventh row
    let mut h66 = Lane8::zero();
    let mut h67 = Lane8::zero();
    let mut h68 = Lane8::zero();
    // Eigth row
    let mut h77 = Lane8::zero();
    let mut h78 = Lane8::zero();
    // Ninth row
    let mut h88 = Lane8::zero();

    // A block = [Lane8; 9]
    // This is one tile
    let mut scaled_column_buf = [[Lane8::zero(); 9]; BLOCKS];

    let (tiles, rem) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        fun_name(x, eps, &mut scaled_column_buf, tile);

        {
            // Row 0
            let mut z00 = h00.load();
            let mut z01 = h01.load();
            let mut z02 = h02.load();
            let mut z03 = h03.load();
            let mut z04 = h04.load();
            let mut z05 = h05.load();
            let mut z06 = h06.load();
            let mut z07 = h07.load();
            let mut z08 = h08.load();

            // Row 1
            let mut z11 = h11.load();
            let mut z12 = h12.load();
            let mut z13 = h13.load();
            let mut z14 = h14.load();
            let mut z15 = h15.load();
            let mut z16 = h16.load();
            let mut z17 = h17.load();
            let mut z18 = h18.load();

            // Row 2
            let mut z22 = h22.load();
            let mut z23 = h23.load();
            let mut z24 = h24.load();
            let mut z25 = h25.load();
            let mut z26 = h26.load();
            let mut z27 = h27.load();
            // let mut z28 = h28.load();

            // // Row 3
            // let mut z33 = h33.load();
            // let mut z34 = h34.load();
            // let mut z35 = h35.load();
            // let mut z36 = h36.load();
            // let mut z37 = h37.load();
            // let mut z38 = h38.load();

            // // Row 4
            // let mut z44 = h44.load();
            // let mut z45 = h45.load();
            // let mut z46 = h46.load();
            // let mut z47 = h47.load();
            // let mut z48 = h48.load();

            // // Row 5
            // let mut z55 = h55.load();
            // let mut z56 = h56.load();
            // let mut z57 = h57.load();
            // let mut z58 = h58.load();

            // // Row 6
            // let mut z66 = h66.load();
            // let mut z67 = h67.load();
            // let mut z68 = h68.load();

            // // Row 7
            // let mut z77 = h77.load();
            // let mut z78 = h78.load();

            // // Row 8
            // let mut z88 = h88.load();

            for block in scaled_column_buf.iter() {
                let c: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                z00 = _mm512_fmadd_pd(c[0], c[0], z00);
                z01 = _mm512_fmadd_pd(c[0], c[1], z01);
                z02 = _mm512_fmadd_pd(c[0], c[2], z02);
                z03 = _mm512_fmadd_pd(c[0], c[3], z03);
                z04 = _mm512_fmadd_pd(c[0], c[4], z04);
                z05 = _mm512_fmadd_pd(c[0], c[5], z05);
                z06 = _mm512_fmadd_pd(c[0], c[6], z06);
                z07 = _mm512_fmadd_pd(c[0], c[7], z07);
                z08 = _mm512_fmadd_pd(c[0], c[8], z08);

                z11 = _mm512_fmadd_pd(c[1], c[1], z11);
                z12 = _mm512_fmadd_pd(c[1], c[2], z12);
                z13 = _mm512_fmadd_pd(c[1], c[3], z13);
                z14 = _mm512_fmadd_pd(c[1], c[4], z14);
                z15 = _mm512_fmadd_pd(c[1], c[5], z15);
                z16 = _mm512_fmadd_pd(c[1], c[6], z16);
                z17 = _mm512_fmadd_pd(c[1], c[7], z17);
                z18 = _mm512_fmadd_pd(c[1], c[8], z18);

                z22 = _mm512_fmadd_pd(c[2], c[2], z22);
                z23 = _mm512_fmadd_pd(c[2], c[3], z23);
                z24 = _mm512_fmadd_pd(c[2], c[4], z24);
                z25 = _mm512_fmadd_pd(c[2], c[5], z25);
                z26 = _mm512_fmadd_pd(c[2], c[6], z26);
                z27 = _mm512_fmadd_pd(c[2], c[7], z27);
                // z28 = _mm512_fmadd_pd(c[2], c[8], z28);

                // z33 = _mm512_fmadd_pd(c[3], c[3], z33);
                // z34 = _mm512_fmadd_pd(c[3], c[4], z34);
                // z35 = _mm512_fmadd_pd(c[3], c[5], z35);
                // z36 = _mm512_fmadd_pd(c[3], c[6], z36);
                // z37 = _mm512_fmadd_pd(c[3], c[7], z37);
                // z38 = _mm512_fmadd_pd(c[3], c[8], z38);

                // z44 = _mm512_fmadd_pd(c[4], c[4], z44);
                // z45 = _mm512_fmadd_pd(c[4], c[5], z45);
                // z46 = _mm512_fmadd_pd(c[4], c[6], z46);
                // z47 = _mm512_fmadd_pd(c[4], c[7], z47);
                // z48 = _mm512_fmadd_pd(c[4], c[8], z48);

                // z55 = _mm512_fmadd_pd(c[5], c[5], z55);
                // z56 = _mm512_fmadd_pd(c[5], c[6], z56);
                // z57 = _mm512_fmadd_pd(c[5], c[7], z57);
                // z58 = _mm512_fmadd_pd(c[5], c[8], z58);

                // z66 = _mm512_fmadd_pd(c[6], c[6], z66);
                // z67 = _mm512_fmadd_pd(c[6], c[7], z67);
                // z68 = _mm512_fmadd_pd(c[6], c[8], z68);

                // z77 = _mm512_fmadd_pd(c[7], c[7], z77);
                // z78 = _mm512_fmadd_pd(c[7], c[8], z78);

                // z88 = _mm512_fmadd_pd(c[8], c[8], z88);
            }

            // Row 0
            h00.store(z00);
            h01.store(z01);
            h02.store(z02);
            h03.store(z03);
            h04.store(z04);
            h05.store(z05);
            h06.store(z06);
            h07.store(z07);
            h08.store(z08);

            // Row 1
            h11.store(z11);
            h12.store(z12);
            h13.store(z13);
            h14.store(z14);
            h15.store(z15);
            h16.store(z16);
            h17.store(z17);
            h18.store(z18);

            // Row 2
            h22.store(z22);
            h23.store(z23);
            h24.store(z24);
            h25.store(z25);
            h26.store(z26);
            h27.store(z27);
            // h28.store(z28);
        }

        // // Row 3
        // h33.store(z33);
        // h34.store(z34);
        // h35.store(z35);
        // h36.store(z36);
        // h37.store(z37);
        // h38.store(z38);

        // // Row 4
        // h44.store(z44);
        // h45.store(z45);
        // h46.store(z46);
        // h47.store(z47);
        // h48.store(z48);

        // // Row 5
        // h55.store(z55);
        // h56.store(z56);
        // h57.store(z57);
        // h58.store(z58);

        // // Row 6
        // h66.store(z66);
        // h67.store(z67);
        // h68.store(z68);

        // // Row 7
        // h77.store(z77);
        // h78.store(z78);

        // // Row 8
        // h88.store(z88);

        // // Row 0
        // let mut z00 = h00.load();
        // let mut z01 = h01.load();
        // let mut z02 = h02.load();
        // let mut z03 = h03.load();
        // let mut z04 = h04.load();
        // let mut z05 = h05.load();
        // let mut z06 = h06.load();
        // let mut z07 = h07.load();
        // let mut z08 = h08.load();

        // // Row 1
        // let mut z11 = h11.load();
        // let mut z12 = h12.load();
        // let mut z13 = h13.load();
        // let mut z14 = h14.load();
        // let mut z15 = h15.load();
        // let mut z16 = h16.load();
        // let mut z17 = h17.load();
        // let mut z18 = h18.load();

        // // Row 2
        // let mut z22 = h22.load();
        // let mut z23 = h23.load();
        // let mut z24 = h24.load();
        // let mut z25 = h25.load();
        // let mut z26 = h26.load();
        // let mut z27 = h27.load();
        {
            let mut z28 = h28.load();

            // Row 3
            let mut z33 = h33.load();
            let mut z34 = h34.load();
            let mut z35 = h35.load();
            let mut z36 = h36.load();
            let mut z37 = h37.load();
            let mut z38 = h38.load();

            // Row 4
            let mut z44 = h44.load();
            let mut z45 = h45.load();
            let mut z46 = h46.load();
            let mut z47 = h47.load();
            let mut z48 = h48.load();

            // Row 5
            let mut z55 = h55.load();
            let mut z56 = h56.load();
            let mut z57 = h57.load();
            let mut z58 = h58.load();

            // Row 6
            let mut z66 = h66.load();
            let mut z67 = h67.load();
            let mut z68 = h68.load();

            // Row 7
            let mut z77 = h77.load();
            let mut z78 = h78.load();

            // Row 8
            let mut z88 = h88.load();

            for block in scaled_column_buf.iter() {
                let c: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                // z00 = _mm512_fmadd_pd(c[0], c[0], z00);
                // z01 = _mm512_fmadd_pd(c[0], c[1], z01);
                // z02 = _mm512_fmadd_pd(c[0], c[2], z02);
                // z03 = _mm512_fmadd_pd(c[0], c[3], z03);
                // z04 = _mm512_fmadd_pd(c[0], c[4], z04);
                // z05 = _mm512_fmadd_pd(c[0], c[5], z05);
                // z06 = _mm512_fmadd_pd(c[0], c[6], z06);
                // z07 = _mm512_fmadd_pd(c[0], c[7], z07);
                // z08 = _mm512_fmadd_pd(c[0], c[8], z08);

                // z11 = _mm512_fmadd_pd(c[1], c[1], z11);
                // z12 = _mm512_fmadd_pd(c[1], c[2], z12);
                // z13 = _mm512_fmadd_pd(c[1], c[3], z13);
                // z14 = _mm512_fmadd_pd(c[1], c[4], z14);
                // z15 = _mm512_fmadd_pd(c[1], c[5], z15);
                // z16 = _mm512_fmadd_pd(c[1], c[6], z16);
                // z17 = _mm512_fmadd_pd(c[1], c[7], z17);
                // z18 = _mm512_fmadd_pd(c[1], c[8], z18);

                // z22 = _mm512_fmadd_pd(c[2], c[2], z22);
                // z23 = _mm512_fmadd_pd(c[2], c[3], z23);
                // z24 = _mm512_fmadd_pd(c[2], c[4], z24);
                // z25 = _mm512_fmadd_pd(c[2], c[5], z25);
                // z26 = _mm512_fmadd_pd(c[2], c[6], z26);
                // z27 = _mm512_fmadd_pd(c[2], c[7], z27);
                z28 = _mm512_fmadd_pd(c[2], c[8], z28);

                z33 = _mm512_fmadd_pd(c[3], c[3], z33);
                z34 = _mm512_fmadd_pd(c[3], c[4], z34);
                z35 = _mm512_fmadd_pd(c[3], c[5], z35);
                z36 = _mm512_fmadd_pd(c[3], c[6], z36);
                z37 = _mm512_fmadd_pd(c[3], c[7], z37);
                z38 = _mm512_fmadd_pd(c[3], c[8], z38);

                z44 = _mm512_fmadd_pd(c[4], c[4], z44);
                z45 = _mm512_fmadd_pd(c[4], c[5], z45);
                z46 = _mm512_fmadd_pd(c[4], c[6], z46);
                z47 = _mm512_fmadd_pd(c[4], c[7], z47);
                z48 = _mm512_fmadd_pd(c[4], c[8], z48);

                z55 = _mm512_fmadd_pd(c[5], c[5], z55);
                z56 = _mm512_fmadd_pd(c[5], c[6], z56);
                z57 = _mm512_fmadd_pd(c[5], c[7], z57);
                z58 = _mm512_fmadd_pd(c[5], c[8], z58);

                z66 = _mm512_fmadd_pd(c[6], c[6], z66);
                z67 = _mm512_fmadd_pd(c[6], c[7], z67);
                z68 = _mm512_fmadd_pd(c[6], c[8], z68);

                z77 = _mm512_fmadd_pd(c[7], c[7], z77);
                z78 = _mm512_fmadd_pd(c[7], c[8], z78);

                z88 = _mm512_fmadd_pd(c[8], c[8], z88);
            }

            // Row 0
            // h00.store(z00);
            // h01.store(z01);
            // h02.store(z02);
            // h03.store(z03);
            // h04.store(z04);
            // h05.store(z05);
            // h06.store(z06);
            // h07.store(z07);
            // h08.store(z08);

            // // Row 1
            // h11.store(z11);
            // h12.store(z12);
            // h13.store(z13);
            // h14.store(z14);
            // h15.store(z15);
            // h16.store(z16);
            // h17.store(z17);
            // h18.store(z18);

            // // Row 2
            // h22.store(z22);
            // h23.store(z23);
            // h24.store(z24);
            // h25.store(z25);
            // h26.store(z26);
            // h27.store(z27);
            h28.store(z28);

            // Row 3
            h33.store(z33);
            h34.store(z34);
            h35.store(z35);
            h36.store(z36);
            h37.store(z37);
            h38.store(z38);

            // Row 4
            h44.store(z44);
            h45.store(z45);
            h46.store(z46);
            h47.store(z47);
            h48.store(z48);

            // Row 5
            h55.store(z55);
            h56.store(z56);
            h57.store(z57);
            h58.store(z58);

            // Row 6
            h66.store(z66);
            h67.store(z67);
            h68.store(z68);

            // Row 7
            h77.store(z77);
            h78.store(z78);

            // Row 8
            h88.store(z88);
        }
    }

    let mut h = [[0.0; 9]; 9];

    h[0][0] = _mm512_reduce_add_pd(h00.load());
    h[0][1] = _mm512_reduce_add_pd(h01.load());
    h[0][2] = _mm512_reduce_add_pd(h02.load());
    h[0][3] = _mm512_reduce_add_pd(h03.load());
    h[0][4] = _mm512_reduce_add_pd(h04.load());
    h[0][5] = _mm512_reduce_add_pd(h05.load());
    h[0][6] = _mm512_reduce_add_pd(h06.load());
    h[0][7] = _mm512_reduce_add_pd(h07.load());
    h[0][8] = _mm512_reduce_add_pd(h08.load());

    // Second row
    h[1][1] = _mm512_reduce_add_pd(h11.load());
    h[1][2] = _mm512_reduce_add_pd(h12.load());
    h[1][3] = _mm512_reduce_add_pd(h13.load());
    h[1][4] = _mm512_reduce_add_pd(h14.load());
    h[1][5] = _mm512_reduce_add_pd(h15.load());
    h[1][6] = _mm512_reduce_add_pd(h16.load());
    h[1][7] = _mm512_reduce_add_pd(h17.load());
    h[1][8] = _mm512_reduce_add_pd(h18.load());

    h[2][2] = _mm512_reduce_add_pd(h22.load());
    h[2][3] = _mm512_reduce_add_pd(h23.load());
    h[2][4] = _mm512_reduce_add_pd(h24.load());
    h[2][5] = _mm512_reduce_add_pd(h25.load());
    h[2][6] = _mm512_reduce_add_pd(h26.load());
    h[2][7] = _mm512_reduce_add_pd(h27.load());
    h[2][8] = _mm512_reduce_add_pd(h28.load());

    h[3][3] = _mm512_reduce_add_pd(h33.load());
    h[3][4] = _mm512_reduce_add_pd(h34.load());
    h[3][5] = _mm512_reduce_add_pd(h35.load());
    h[3][6] = _mm512_reduce_add_pd(h36.load());
    h[3][7] = _mm512_reduce_add_pd(h37.load());
    h[3][8] = _mm512_reduce_add_pd(h38.load());

    h[4][4] = _mm512_reduce_add_pd(h44.load());
    h[4][5] = _mm512_reduce_add_pd(h45.load());
    h[4][6] = _mm512_reduce_add_pd(h46.load());
    h[4][7] = _mm512_reduce_add_pd(h47.load());
    h[4][8] = _mm512_reduce_add_pd(h48.load());

    h[5][5] = _mm512_reduce_add_pd(h55.load());
    h[5][6] = _mm512_reduce_add_pd(h56.load());
    h[5][7] = _mm512_reduce_add_pd(h57.load());
    h[5][8] = _mm512_reduce_add_pd(h58.load());

    h[6][6] = _mm512_reduce_add_pd(h66.load());
    h[6][7] = _mm512_reduce_add_pd(h67.load());
    h[6][8] = _mm512_reduce_add_pd(h68.load());

    h[7][7] = _mm512_reduce_add_pd(h77.load());
    h[7][8] = _mm512_reduce_add_pd(h78.load());

    h[8][8] = _mm512_reduce_add_pd(h88.load());

    for i in 0..9 {
        for j in i..9 {
            h[j][i] = h[i][j];
        }
    }

    h
}

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_three_passes(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {
    const BLOCKS: usize = 32;

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
    // Fourth row
    let mut h33 = _mm512_setzero_pd();
    let mut h34 = _mm512_setzero_pd();
    let mut h35 = _mm512_setzero_pd();
    let mut h36 = _mm512_setzero_pd();
    let mut h37 = _mm512_setzero_pd();
    let mut h38 = _mm512_setzero_pd();
    // Fifth row
    let mut h44 = _mm512_setzero_pd();
    let mut h45 = _mm512_setzero_pd();
    let mut h46 = _mm512_setzero_pd();
    let mut h47 = _mm512_setzero_pd();
    let mut h48 = _mm512_setzero_pd();
    // Sixth row
    let mut h55 = _mm512_setzero_pd();
    let mut h56 = _mm512_setzero_pd();
    let mut h57 = _mm512_setzero_pd();
    let mut h58 = _mm512_setzero_pd();
    // Seventh row
    let mut h66 = _mm512_setzero_pd();
    let mut h67 = _mm512_setzero_pd();
    let mut h68 = _mm512_setzero_pd();
    // Eigth row
    let mut h77 = _mm512_setzero_pd();
    let mut h78 = _mm512_setzero_pd();
    // Ninth row
    let mut h88 = _mm512_setzero_pd();

    // A block = [Lane8; 9]
    // This is one tile
    let mut scaled_column_buf = [[Lane8::zero(); 9]; BLOCKS];

    let (tiles, rem) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        fun_name(x, eps, &mut scaled_column_buf, tile);

        for block in scaled_column_buf.iter() {
            let c: [__m512d; 9] =
                std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

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
            // h17 = _mm512_fmadd_pd(c[1], c[7], h17);
            // h18 = _mm512_fmadd_pd(c[1], c[8], h18);

            // h22 = _mm512_fmadd_pd(c[2], c[2], h22);
            // h23 = _mm512_fmadd_pd(c[2], c[3], h23);
            // h24 = _mm512_fmadd_pd(c[2], c[4], h24);
            // h25 = _mm512_fmadd_pd(c[2], c[5], h25);
            // h26 = _mm512_fmadd_pd(c[2], c[6], h26);
            // h27 = _mm512_fmadd_pd(c[2], c[7], h27);
            // h28 = _mm512_fmadd_pd(c[2], c[8], h28);

            // h33 = _mm512_fmadd_pd(c[3], c[3], h33);
            // h34 = _mm512_fmadd_pd(c[3], c[4], h34);
            // h35 = _mm512_fmadd_pd(c[3], c[5], h35);
            // h36 = _mm512_fmadd_pd(c[3], c[6], h36);
            // h37 = _mm512_fmadd_pd(c[3], c[7], h37);
            // h38 = _mm512_fmadd_pd(c[3], c[8], h38);

            // h44 = _mm512_fmadd_pd(c[4], c[4], h44);
            // h45 = _mm512_fmadd_pd(c[4], c[5], h45);
            // h46 = _mm512_fmadd_pd(c[4], c[6], h46);
            // h47 = _mm512_fmadd_pd(c[4], c[7], h47);
            // h48 = _mm512_fmadd_pd(c[4], c[8], h48);

            // h55 = _mm512_fmadd_pd(c[5], c[5], h55);
            // h56 = _mm512_fmadd_pd(c[5], c[6], h56);
            // h57 = _mm512_fmadd_pd(c[5], c[7], h57);
            // h58 = _mm512_fmadd_pd(c[5], c[8], h58);

            // h66 = _mm512_fmadd_pd(c[6], c[6], h66);
            // h67 = _mm512_fmadd_pd(c[6], c[7], h67);
            // h68 = _mm512_fmadd_pd(c[6], c[8], h68);

            // h77 = _mm512_fmadd_pd(c[7], c[7], h77);
            // h78 = _mm512_fmadd_pd(c[7], c[8], h78);

            // h88 = _mm512_fmadd_pd(c[8], c[8], h88);
        }

        for block in scaled_column_buf.iter() {
            let c: [__m512d; 9] =
                std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

            // h00 = _mm512_fmadd_pd(c[0], c[0], h00);
            // h01 = _mm512_fmadd_pd(c[0], c[1], h01);
            // h02 = _mm512_fmadd_pd(c[0], c[2], h02);
            // h03 = _mm512_fmadd_pd(c[0], c[3], h03);
            // h04 = _mm512_fmadd_pd(c[0], c[4], h04);
            // h05 = _mm512_fmadd_pd(c[0], c[5], h05);
            // h06 = _mm512_fmadd_pd(c[0], c[6], h06);
            // h07 = _mm512_fmadd_pd(c[0], c[7], h07);
            // h08 = _mm512_fmadd_pd(c[0], c[8], h08);

            // h11 = _mm512_fmadd_pd(c[1], c[1], h11);
            // h12 = _mm512_fmadd_pd(c[1], c[2], h12);
            // h13 = _mm512_fmadd_pd(c[1], c[3], h13);
            // h14 = _mm512_fmadd_pd(c[1], c[4], h14);
            // h15 = _mm512_fmadd_pd(c[1], c[5], h15);
            // h16 = _mm512_fmadd_pd(c[1], c[6], h16);
            h17 = _mm512_fmadd_pd(c[1], c[7], h17);
            h18 = _mm512_fmadd_pd(c[1], c[8], h18);

            h22 = _mm512_fmadd_pd(c[2], c[2], h22);
            h23 = _mm512_fmadd_pd(c[2], c[3], h23);
            h24 = _mm512_fmadd_pd(c[2], c[4], h24);
            h25 = _mm512_fmadd_pd(c[2], c[5], h25);
            h26 = _mm512_fmadd_pd(c[2], c[6], h26);
            h27 = _mm512_fmadd_pd(c[2], c[7], h27);
            h28 = _mm512_fmadd_pd(c[2], c[8], h28);

            h33 = _mm512_fmadd_pd(c[3], c[3], h33);
            h34 = _mm512_fmadd_pd(c[3], c[4], h34);
            h35 = _mm512_fmadd_pd(c[3], c[5], h35);
            h36 = _mm512_fmadd_pd(c[3], c[6], h36);
            h37 = _mm512_fmadd_pd(c[3], c[7], h37);
            h38 = _mm512_fmadd_pd(c[3], c[8], h38);

            // h44 = _mm512_fmadd_pd(c[4], c[4], h44);
            // h45 = _mm512_fmadd_pd(c[4], c[5], h45);
            // h46 = _mm512_fmadd_pd(c[4], c[6], h46);
            // h47 = _mm512_fmadd_pd(c[4], c[7], h47);
            // h48 = _mm512_fmadd_pd(c[4], c[8], h48);

            // h55 = _mm512_fmadd_pd(c[5], c[5], h55);
            // h56 = _mm512_fmadd_pd(c[5], c[6], h56);
            // h57 = _mm512_fmadd_pd(c[5], c[7], h57);
            // h58 = _mm512_fmadd_pd(c[5], c[8], h58);

            // h66 = _mm512_fmadd_pd(c[6], c[6], h66);
            // h67 = _mm512_fmadd_pd(c[6], c[7], h67);
            // h68 = _mm512_fmadd_pd(c[6], c[8], h68);

            // h77 = _mm512_fmadd_pd(c[7], c[7], h77);
            // h78 = _mm512_fmadd_pd(c[7], c[8], h78);

            // h88 = _mm512_fmadd_pd(c[8], c[8], h88);
        }

        for block in scaled_column_buf.iter() {
            let c: [__m512d; 9] =
                std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

            // h00 = _mm512_fmadd_pd(c[0], c[0], h00);
            // h01 = _mm512_fmadd_pd(c[0], c[1], h01);
            // h02 = _mm512_fmadd_pd(c[0], c[2], h02);
            // h03 = _mm512_fmadd_pd(c[0], c[3], h03);
            // h04 = _mm512_fmadd_pd(c[0], c[4], h04);
            // h05 = _mm512_fmadd_pd(c[0], c[5], h05);
            // h06 = _mm512_fmadd_pd(c[0], c[6], h06);
            // h07 = _mm512_fmadd_pd(c[0], c[7], h07);
            // h08 = _mm512_fmadd_pd(c[0], c[8], h08);

            // h11 = _mm512_fmadd_pd(c[1], c[1], h11);
            // h12 = _mm512_fmadd_pd(c[1], c[2], h12);
            // h13 = _mm512_fmadd_pd(c[1], c[3], h13);
            // h14 = _mm512_fmadd_pd(c[1], c[4], h14);
            // h15 = _mm512_fmadd_pd(c[1], c[5], h15);
            // h16 = _mm512_fmadd_pd(c[1], c[6], h16);
            // h17 = _mm512_fmadd_pd(c[1], c[7], h17);
            // h18 = _mm512_fmadd_pd(c[1], c[8], h18);

            // h22 = _mm512_fmadd_pd(c[2], c[2], h22);
            // h23 = _mm512_fmadd_pd(c[2], c[3], h23);
            // h24 = _mm512_fmadd_pd(c[2], c[4], h24);
            // h25 = _mm512_fmadd_pd(c[2], c[5], h25);
            // h26 = _mm512_fmadd_pd(c[2], c[6], h26);
            // h27 = _mm512_fmadd_pd(c[2], c[7], h27);
            // h28 = _mm512_fmadd_pd(c[2], c[8], h28);

            // h33 = _mm512_fmadd_pd(c[3], c[3], h33);
            // h34 = _mm512_fmadd_pd(c[3], c[4], h34);
            // h35 = _mm512_fmadd_pd(c[3], c[5], h35);
            // h36 = _mm512_fmadd_pd(c[3], c[6], h36);
            // h37 = _mm512_fmadd_pd(c[3], c[7], h37);
            // h38 = _mm512_fmadd_pd(c[3], c[8], h38);

            h44 = _mm512_fmadd_pd(c[4], c[4], h44);
            h45 = _mm512_fmadd_pd(c[4], c[5], h45);
            h46 = _mm512_fmadd_pd(c[4], c[6], h46);
            h47 = _mm512_fmadd_pd(c[4], c[7], h47);
            h48 = _mm512_fmadd_pd(c[4], c[8], h48);

            h55 = _mm512_fmadd_pd(c[5], c[5], h55);
            h56 = _mm512_fmadd_pd(c[5], c[6], h56);
            h57 = _mm512_fmadd_pd(c[5], c[7], h57);
            h58 = _mm512_fmadd_pd(c[5], c[8], h58);

            h66 = _mm512_fmadd_pd(c[6], c[6], h66);
            h67 = _mm512_fmadd_pd(c[6], c[7], h67);
            h68 = _mm512_fmadd_pd(c[6], c[8], h68);

            h77 = _mm512_fmadd_pd(c[7], c[7], h77);
            h78 = _mm512_fmadd_pd(c[7], c[8], h78);

            h88 = _mm512_fmadd_pd(c[8], c[8], h88);
        }
    }

    let mut h = [[0.0; 9]; 9];

    h[0][0] = _mm512_reduce_add_pd(h00);
    h[0][1] = _mm512_reduce_add_pd(h01);
    h[0][2] = _mm512_reduce_add_pd(h02);
    h[0][3] = _mm512_reduce_add_pd(h03);
    h[0][4] = _mm512_reduce_add_pd(h04);
    h[0][5] = _mm512_reduce_add_pd(h05);
    h[0][6] = _mm512_reduce_add_pd(h06);
    h[0][7] = _mm512_reduce_add_pd(h07);
    h[0][8] = _mm512_reduce_add_pd(h08);

    // Second row
    h[1][1] = _mm512_reduce_add_pd(h11);
    h[1][2] = _mm512_reduce_add_pd(h12);
    h[1][3] = _mm512_reduce_add_pd(h13);
    h[1][4] = _mm512_reduce_add_pd(h14);
    h[1][5] = _mm512_reduce_add_pd(h15);
    h[1][6] = _mm512_reduce_add_pd(h16);
    h[1][7] = _mm512_reduce_add_pd(h17);
    h[1][8] = _mm512_reduce_add_pd(h18);

    h[2][2] = _mm512_reduce_add_pd(h22);
    h[2][3] = _mm512_reduce_add_pd(h23);
    h[2][4] = _mm512_reduce_add_pd(h24);
    h[2][5] = _mm512_reduce_add_pd(h25);
    h[2][6] = _mm512_reduce_add_pd(h26);
    h[2][7] = _mm512_reduce_add_pd(h27);
    h[2][8] = _mm512_reduce_add_pd(h28);

    h[3][3] = _mm512_reduce_add_pd(h33);
    h[3][4] = _mm512_reduce_add_pd(h34);
    h[3][5] = _mm512_reduce_add_pd(h35);
    h[3][6] = _mm512_reduce_add_pd(h36);
    h[3][7] = _mm512_reduce_add_pd(h37);
    h[3][8] = _mm512_reduce_add_pd(h38);

    h[4][4] = _mm512_reduce_add_pd(h44);
    h[4][5] = _mm512_reduce_add_pd(h45);
    h[4][6] = _mm512_reduce_add_pd(h46);
    h[4][7] = _mm512_reduce_add_pd(h47);
    h[4][8] = _mm512_reduce_add_pd(h48);

    h[5][5] = _mm512_reduce_add_pd(h55);
    h[5][6] = _mm512_reduce_add_pd(h56);
    h[5][7] = _mm512_reduce_add_pd(h57);
    h[5][8] = _mm512_reduce_add_pd(h58);

    h[6][6] = _mm512_reduce_add_pd(h66);
    h[6][7] = _mm512_reduce_add_pd(h67);
    h[6][8] = _mm512_reduce_add_pd(h68);

    h[7][7] = _mm512_reduce_add_pd(h77);
    h[7][8] = _mm512_reduce_add_pd(h78);

    h[8][8] = _mm512_reduce_add_pd(h88);

    for i in 0..9 {
        for j in i..9 {
            h[j][i] = h[i][j];
        }
    }

    h
}

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_three_passes2(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {
    const BLOCKS: usize = 32;

    // First row
    let mut h00 = Lane8::zero();
    let mut h01 = Lane8::zero();
    let mut h02 = Lane8::zero();
    let mut h03 = Lane8::zero();
    let mut h04 = Lane8::zero();
    let mut h05 = Lane8::zero();
    let mut h06 = Lane8::zero();
    let mut h07 = Lane8::zero();
    let mut h08 = Lane8::zero();
    // Second row
    let mut h11 = Lane8::zero();
    let mut h12 = Lane8::zero();
    let mut h13 = Lane8::zero();
    let mut h14 = Lane8::zero();
    let mut h15 = Lane8::zero();
    let mut h16 = Lane8::zero();
    let mut h17 = Lane8::zero();
    let mut h18 = Lane8::zero();
    // Third row
    let mut h22 = Lane8::zero();
    let mut h23 = Lane8::zero();
    let mut h24 = Lane8::zero();
    let mut h25 = Lane8::zero();
    let mut h26 = Lane8::zero();
    let mut h27 = Lane8::zero();
    let mut h28 = Lane8::zero();
    // Fourth row
    let mut h33 = Lane8::zero();
    let mut h34 = Lane8::zero();
    let mut h35 = Lane8::zero();
    let mut h36 = Lane8::zero();
    let mut h37 = Lane8::zero();
    let mut h38 = Lane8::zero();
    // Fifth row
    let mut h44 = Lane8::zero();
    let mut h45 = Lane8::zero();
    let mut h46 = Lane8::zero();
    let mut h47 = Lane8::zero();
    let mut h48 = Lane8::zero();
    // Sixth row
    let mut h55 = Lane8::zero();
    let mut h56 = Lane8::zero();
    let mut h57 = Lane8::zero();
    let mut h58 = Lane8::zero();
    // Seventh row
    let mut h66 = Lane8::zero();
    let mut h67 = Lane8::zero();
    let mut h68 = Lane8::zero();
    // Eigth row
    let mut h77 = Lane8::zero();
    let mut h78 = Lane8::zero();
    // Ninth row
    let mut h88 = Lane8::zero();

    // A block = [Lane8; 9]
    // This is one tile
    let mut scaled_column_buf = [[Lane8::zero(); 9]; BLOCKS];

    let (tiles, rem) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        fun_name(x, eps, &mut scaled_column_buf, tile);

        {
            // Row 0
            let mut z00 = h00.load();
            let mut z01 = h01.load();
            let mut z02 = h02.load();
            let mut z03 = h03.load();
            let mut z04 = h04.load();
            let mut z05 = h05.load();
            let mut z06 = h06.load();
            let mut z07 = h07.load();
            let mut z08 = h08.load();

            // Row 1
            let mut z11 = h11.load();
            let mut z12 = h12.load();
            let mut z13 = h13.load();
            let mut z14 = h14.load();
            let mut z15 = h15.load();
            let mut z16 = h16.load();

            for block in scaled_column_buf.iter() {
                let c: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                z00 = _mm512_fmadd_pd(c[0], c[0], z00);
                z01 = _mm512_fmadd_pd(c[0], c[1], z01);
                z02 = _mm512_fmadd_pd(c[0], c[2], z02);
                z03 = _mm512_fmadd_pd(c[0], c[3], z03);
                z04 = _mm512_fmadd_pd(c[0], c[4], z04);
                z05 = _mm512_fmadd_pd(c[0], c[5], z05);
                z06 = _mm512_fmadd_pd(c[0], c[6], z06);
                z07 = _mm512_fmadd_pd(c[0], c[7], z07);
                z08 = _mm512_fmadd_pd(c[0], c[8], z08);

                z11 = _mm512_fmadd_pd(c[1], c[1], z11);
                z12 = _mm512_fmadd_pd(c[1], c[2], z12);
                z13 = _mm512_fmadd_pd(c[1], c[3], z13);
                z14 = _mm512_fmadd_pd(c[1], c[4], z14);
                z15 = _mm512_fmadd_pd(c[1], c[5], z15);
                z16 = _mm512_fmadd_pd(c[1], c[6], z16);
            }

            // Row 0
            h00.store(z00);
            h01.store(z01);
            h02.store(z02);
            h03.store(z03);
            h04.store(z04);
            h05.store(z05);
            h06.store(z06);
            h07.store(z07);
            h08.store(z08);

            // Row 1
            h11.store(z11);
            h12.store(z12);
            h13.store(z13);
            h14.store(z14);
            h15.store(z15);
            h16.store(z16);
        }

        {
            let mut z17 = h17.load();
            let mut z18 = h18.load();

            // Row 2
            let mut z22 = h22.load();
            let mut z23 = h23.load();
            let mut z24 = h24.load();
            let mut z25 = h25.load();
            let mut z26 = h26.load();
            let mut z27 = h27.load();
            let mut z28 = h28.load();

            // Row 3
            let mut z33 = h33.load();
            let mut z34 = h34.load();
            let mut z35 = h35.load();
            let mut z36 = h36.load();
            let mut z37 = h37.load();
            let mut z38 = h38.load();

            for block in scaled_column_buf.iter() {
                let c: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                z17 = _mm512_fmadd_pd(c[1], c[7], z17);
                z18 = _mm512_fmadd_pd(c[1], c[8], z18);

                z22 = _mm512_fmadd_pd(c[2], c[2], z22);
                z23 = _mm512_fmadd_pd(c[2], c[3], z23);
                z24 = _mm512_fmadd_pd(c[2], c[4], z24);
                z25 = _mm512_fmadd_pd(c[2], c[5], z25);
                z26 = _mm512_fmadd_pd(c[2], c[6], z26);
                z27 = _mm512_fmadd_pd(c[2], c[7], z27);
                z28 = _mm512_fmadd_pd(c[2], c[8], z28);

                z33 = _mm512_fmadd_pd(c[3], c[3], z33);
                z34 = _mm512_fmadd_pd(c[3], c[4], z34);
                z35 = _mm512_fmadd_pd(c[3], c[5], z35);
                z36 = _mm512_fmadd_pd(c[3], c[6], z36);
                z37 = _mm512_fmadd_pd(c[3], c[7], z37);
                z38 = _mm512_fmadd_pd(c[3], c[8], z38);
            }

            h17.store(z17);
            h18.store(z18);

            // Row 2
            h22.store(z22);
            h23.store(z23);
            h24.store(z24);
            h25.store(z25);
            h26.store(z26);
            h27.store(z27);
            h28.store(z28);

            // Row 3
            h33.store(z33);
            h34.store(z34);
            h35.store(z35);
            h36.store(z36);
            h37.store(z37);
            h38.store(z38);
        }

        {
            // Row 4
            let mut z44 = h44.load();
            let mut z45 = h45.load();
            let mut z46 = h46.load();
            let mut z47 = h47.load();
            let mut z48 = h48.load();

            // Row 5
            let mut z55 = h55.load();
            let mut z56 = h56.load();
            let mut z57 = h57.load();
            let mut z58 = h58.load();

            // Row 6
            let mut z66 = h66.load();
            let mut z67 = h67.load();
            let mut z68 = h68.load();

            // Row 7
            let mut z77 = h77.load();
            let mut z78 = h78.load();

            // Row 8
            let mut z88 = h88.load();

            for block in scaled_column_buf.iter() {
                let c4 = block[4].load();
                let c5 = block[5].load();
                let c6 = block[6].load();
                let c7 = block[7].load();
                let c8 = block[8].load();

                z44 = _mm512_fmadd_pd(c4, c4, z44);
                z45 = _mm512_fmadd_pd(c4, c5, z45);
                z46 = _mm512_fmadd_pd(c4, c6, z46);
                z47 = _mm512_fmadd_pd(c4, c7, z47);
                z48 = _mm512_fmadd_pd(c4, c8, z48);

                z55 = _mm512_fmadd_pd(c5, c5, z55);
                z56 = _mm512_fmadd_pd(c5, c6, z56);
                z57 = _mm512_fmadd_pd(c5, c7, z57);
                z58 = _mm512_fmadd_pd(c5, c8, z58);

                z66 = _mm512_fmadd_pd(c6, c6, z66);
                z67 = _mm512_fmadd_pd(c6, c7, z67);
                z68 = _mm512_fmadd_pd(c6, c8, z68);

                z77 = _mm512_fmadd_pd(c7, c7, z77);
                z78 = _mm512_fmadd_pd(c7, c8, z78);

                z88 = _mm512_fmadd_pd(c8, c8, z88);
            }

            // Row 4
            h44.store(z44);
            h45.store(z45);
            h46.store(z46);
            h47.store(z47);
            h48.store(z48);

            // Row 5
            h55.store(z55);
            h56.store(z56);
            h57.store(z57);
            h58.store(z58);

            // Row 6
            h66.store(z66);
            h67.store(z67);
            h68.store(z68);

            // Row 7
            h77.store(z77);
            h78.store(z78);

            // Row 8
            h88.store(z88);
        }
    }

    let mut h = [[0.0; 9]; 9];

    // First row
    h[0][0] = _mm512_reduce_add_pd(h00.load());
    h[0][1] = _mm512_reduce_add_pd(h01.load());
    h[0][2] = _mm512_reduce_add_pd(h02.load());
    h[0][3] = _mm512_reduce_add_pd(h03.load());
    h[0][4] = _mm512_reduce_add_pd(h04.load());
    h[0][5] = _mm512_reduce_add_pd(h05.load());
    h[0][6] = _mm512_reduce_add_pd(h06.load());
    h[0][7] = _mm512_reduce_add_pd(h07.load());
    h[0][8] = _mm512_reduce_add_pd(h08.load());

    // Second row
    h[1][1] = _mm512_reduce_add_pd(h11.load());
    h[1][2] = _mm512_reduce_add_pd(h12.load());
    h[1][3] = _mm512_reduce_add_pd(h13.load());
    h[1][4] = _mm512_reduce_add_pd(h14.load());
    h[1][5] = _mm512_reduce_add_pd(h15.load());
    h[1][6] = _mm512_reduce_add_pd(h16.load());
    h[1][7] = _mm512_reduce_add_pd(h17.load());
    h[1][8] = _mm512_reduce_add_pd(h18.load());

    h[2][2] = _mm512_reduce_add_pd(h22.load());
    h[2][3] = _mm512_reduce_add_pd(h23.load());
    h[2][4] = _mm512_reduce_add_pd(h24.load());
    h[2][5] = _mm512_reduce_add_pd(h25.load());
    h[2][6] = _mm512_reduce_add_pd(h26.load());
    h[2][7] = _mm512_reduce_add_pd(h27.load());
    h[2][8] = _mm512_reduce_add_pd(h28.load());

    h[3][3] = _mm512_reduce_add_pd(h33.load());
    h[3][4] = _mm512_reduce_add_pd(h34.load());
    h[3][5] = _mm512_reduce_add_pd(h35.load());
    h[3][6] = _mm512_reduce_add_pd(h36.load());
    h[3][7] = _mm512_reduce_add_pd(h37.load());
    h[3][8] = _mm512_reduce_add_pd(h38.load());

    h[4][4] = _mm512_reduce_add_pd(h44.load());
    h[4][5] = _mm512_reduce_add_pd(h45.load());
    h[4][6] = _mm512_reduce_add_pd(h46.load());
    h[4][7] = _mm512_reduce_add_pd(h47.load());
    h[4][8] = _mm512_reduce_add_pd(h48.load());

    h[5][5] = _mm512_reduce_add_pd(h55.load());
    h[5][6] = _mm512_reduce_add_pd(h56.load());
    h[5][7] = _mm512_reduce_add_pd(h57.load());
    h[5][8] = _mm512_reduce_add_pd(h58.load());

    h[6][6] = _mm512_reduce_add_pd(h66.load());
    h[6][7] = _mm512_reduce_add_pd(h67.load());
    h[6][8] = _mm512_reduce_add_pd(h68.load());

    h[7][7] = _mm512_reduce_add_pd(h77.load());
    h[7][8] = _mm512_reduce_add_pd(h78.load());

    h[8][8] = _mm512_reduce_add_pd(h88.load());

    for i in 0..9 {
        for j in i..9 {
            h[j][i] = h[i][j];
        }
    }

    h
}

// Current winner!
#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_three_passes3(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [[f64; 9]; 9]) {
    const BLOCKS: usize = 32;

    let mut g = [Lane8::zero(); 9];

    // First row
    let mut h00 = Lane8::zero();
    let mut h01 = Lane8::zero();
    let mut h02 = Lane8::zero();
    let mut h03 = Lane8::zero();
    let mut h04 = Lane8::zero();
    let mut h05 = Lane8::zero();
    let mut h06 = Lane8::zero();
    let mut h07 = Lane8::zero();
    let mut h08 = Lane8::zero();
    // Second row
    let mut h11 = Lane8::zero();
    let mut h12 = Lane8::zero();
    let mut h13 = Lane8::zero();
    let mut h14 = Lane8::zero();
    let mut h15 = Lane8::zero();
    let mut h16 = Lane8::zero();
    let mut h17 = Lane8::zero();
    let mut h18 = Lane8::zero();
    // Third row
    let mut h22 = Lane8::zero();
    let mut h23 = Lane8::zero();
    let mut h24 = Lane8::zero();
    let mut h25 = Lane8::zero();
    let mut h26 = Lane8::zero();
    let mut h27 = Lane8::zero();
    let mut h28 = Lane8::zero();
    // Fourth row
    let mut h33 = Lane8::zero();
    let mut h34 = Lane8::zero();
    let mut h35 = Lane8::zero();
    let mut h36 = Lane8::zero();
    let mut h37 = Lane8::zero();
    let mut h38 = Lane8::zero();
    // Fifth row
    let mut h44 = Lane8::zero();
    let mut h45 = Lane8::zero();
    let mut h46 = Lane8::zero();
    let mut h47 = Lane8::zero();
    let mut h48 = Lane8::zero();
    // Sixth row
    let mut h55 = Lane8::zero();
    let mut h56 = Lane8::zero();
    let mut h57 = Lane8::zero();
    let mut h58 = Lane8::zero();
    // Seventh row
    let mut h66 = Lane8::zero();
    let mut h67 = Lane8::zero();
    let mut h68 = Lane8::zero();
    // Eigth row
    let mut h77 = Lane8::zero();
    let mut h78 = Lane8::zero();
    // Ninth row
    let mut h88 = Lane8::zero();

    // Register allocation:
    // - 9 for x
    // - 1 for one
    // For calculation of d: 1 for d + 9 columns + 9 gradient = 19
    // For each Hessian pass: 15 accumulators + at most 5 columns = 20

    let one = _mm512_set1_pd(1.0);

    let xs = [
        _mm512_set1_pd(x[0]),
        _mm512_set1_pd(x[1]),
        _mm512_set1_pd(x[2]),
        _mm512_set1_pd(x[3]),
        _mm512_set1_pd(x[4]),
        _mm512_set1_pd(x[5]),
        _mm512_set1_pd(x[6]),
        _mm512_set1_pd(x[7]),
        _mm512_set1_pd(x[8]),
    ];

    // A block = [Lane8; 9]
    // This is one tile
    let mut scaled_column_buf = [[Lane8::zero(); 9]; BLOCKS];

    let (tiles, rem) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        {
            let mut z = [_mm512_setzero_pd(); 9];

            for (buffer, block) in scaled_column_buf.iter_mut().zip(tile.iter()) {
                let mut c: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                // Calculate d
                // This computes a dot product between x and a row of p
                // TODO this could be manually unrolled a few times
                let mut d = _mm512_set1_pd(eps);
                for col in 0..9 {
                    d = _mm512_fmadd_pd(xs[col], c[col], d);
                }

                // TODO investigate reciprocal
                d = _mm512_div_pd(one, d);

                // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

                for col in 0..9 {
                    c[col] = _mm512_mul_pd(c[col], d);
                }

                for col in 0..9 {
                    buffer[col].store(c[col]);
                }

                for col in 0..9 {
                    z[col] = _mm512_fmadd_pd(c[col], d, z[col]);
                }
            }

            for col in 0..9 {
                z[col] = _mm512_add_pd(g[col].load(), z[col]);
            }

            for col in 0..9 {
                g[col].store(z[col]);
            }
        }


        {
            // Row 0
            let mut z00 = h00.load();
            let mut z01 = h01.load();
            let mut z02 = h02.load();
            let mut z03 = h03.load();
            let mut z04 = h04.load();
            let mut z05 = h05.load();
            let mut z06 = h06.load();
            let mut z07 = h07.load();
            let mut z08 = h08.load();

            // Row 1
            let mut z11 = h11.load();
            let mut z12 = h12.load();
            let mut z13 = h13.load();
            let mut z14 = h14.load();
            let mut z15 = h15.load();
            let mut z16 = h16.load();

            for block in scaled_column_buf.iter() {
                // 2 permanent + 1 temporary = 3 registers
                let c0 = block[0].load();
                let c1 = block[1].load();

                z00 = _mm512_fmadd_pd(c0, c0, z00);

                z01 = _mm512_fmadd_pd(c0, c1, z01);
                z11 = _mm512_fmadd_pd(c1, c1, z11);

                let c2 = block[2].load();
                z02 = _mm512_fmadd_pd(c0, c2, z02);
                z12 = _mm512_fmadd_pd(c1, c2, z12);

                let c3 = block[3].load();
                z03 = _mm512_fmadd_pd(c0, c3, z03);
                z13 = _mm512_fmadd_pd(c1, c3, z13);

                let c4 = block[4].load();
                z04 = _mm512_fmadd_pd(c0, c4, z04);
                z14 = _mm512_fmadd_pd(c1, c4, z14);

                let c5 = block[4].load();
                z05 = _mm512_fmadd_pd(c0, c5, z05);
                z15 = _mm512_fmadd_pd(c1, c5, z15);

                let c6 = block[4].load();
                z06 = _mm512_fmadd_pd(c0, c6, z06);
                z16 = _mm512_fmadd_pd(c1, c6, z16);

                let c7 = block[7].load();
                z07 = _mm512_fmadd_pd(c0, c7, z07);

                let c8 = block[8].load();
                z08 = _mm512_fmadd_pd(c0, c8, z08);

                // z00 = _mm512_fmadd_pd(c[0], c[0], z00);
                // z01 = _mm512_fmadd_pd(c[0], c[1], z01);
                // z02 = _mm512_fmadd_pd(c[0], c[2], z02);
                // z03 = _mm512_fmadd_pd(c[0], c[3], z03);
                // z04 = _mm512_fmadd_pd(c[0], c[4], z04);
                // z05 = _mm512_fmadd_pd(c[0], c[5], z05);
                // z06 = _mm512_fmadd_pd(c[0], c[6], z06);
                // z07 = _mm512_fmadd_pd(c[0], c[7], z07);
                // z08 = _mm512_fmadd_pd(c[0], c[8], z08);

                // z11 = _mm512_fmadd_pd(c[1], c[1], z11);
                // z12 = _mm512_fmadd_pd(c[1], c[2], z12);
                // z13 = _mm512_fmadd_pd(c[1], c[3], z13);
                // z14 = _mm512_fmadd_pd(c[1], c[4], z14);
                // z15 = _mm512_fmadd_pd(c[1], c[5], z15);
                // z16 = _mm512_fmadd_pd(c[1], c[6], z16);
            }

            // Row 0
            h00.store(z00);
            h01.store(z01);
            h02.store(z02);
            h03.store(z03);
            h04.store(z04);
            h05.store(z05);
            h06.store(z06);
            h07.store(z07);
            h08.store(z08);

            // Row 1
            h11.store(z11);
            h12.store(z12);
            h13.store(z13);
            h14.store(z14);
            h15.store(z15);
            h16.store(z16);
        }

        {
            let mut z17 = h17.load();
            let mut z18 = h18.load();

            // Row 2
            let mut z22 = h22.load();
            let mut z23 = h23.load();
            let mut z24 = h24.load();
            let mut z25 = h25.load();
            let mut z26 = h26.load();
            let mut z27 = h27.load();
            let mut z28 = h28.load();

            // Row 3
            let mut z33 = h33.load();
            let mut z34 = h34.load();
            let mut z35 = h35.load();
            let mut z36 = h36.load();
            let mut z37 = h37.load();
            let mut z38 = h38.load();

            for block in scaled_column_buf.iter() {
                // 3 permanent + 1 temporary = 4
                let c1 = block[1].load();
                let c2 = block[2].load();
                let c3 = block[3].load();

                z22 = _mm512_fmadd_pd(c2, c2, z22);

                z23 = _mm512_fmadd_pd(c2, c3, z23);
                z33 = _mm512_fmadd_pd(c3, c3, z33);

                let c4 = block[4].load();
                z24 = _mm512_fmadd_pd(c2, c4, z24);
                z34 = _mm512_fmadd_pd(c3, c4, z34);

                let c5 = block[5].load();
                z25 = _mm512_fmadd_pd(c2, c5, z25);
                z35 = _mm512_fmadd_pd(c3, c5, z35);

                let c6 = block[6].load();
                z26 = _mm512_fmadd_pd(c2, c6, z26);
                z36 = _mm512_fmadd_pd(c3, c6, z36);

                let c7 = block[7].load();
                z17 = _mm512_fmadd_pd(c1, c7, z17);
                z27 = _mm512_fmadd_pd(c2, c7, z27);
                z37 = _mm512_fmadd_pd(c3, c7, z37);

                let c8 = block[8].load();
                z18 = _mm512_fmadd_pd(c1, c8, z18);
                z28 = _mm512_fmadd_pd(c2, c8, z28);
                z38 = _mm512_fmadd_pd(c3, c8, z38);

                // z17 = _mm512_fmadd_pd(c1, c7, z17);
                // z18 = _mm512_fmadd_pd(c1, c8, z18);

                // z22 = _mm512_fmadd_pd(c2, c2, z22);
                // z23 = _mm512_fmadd_pd(c2, c3, z23);
                // z24 = _mm512_fmadd_pd(c2, c4, z24);
                // z25 = _mm512_fmadd_pd(c2, c5, z25);
                // z26 = _mm512_fmadd_pd(c2, c6, z26);
                // z27 = _mm512_fmadd_pd(c2, c7, z27);
                // z28 = _mm512_fmadd_pd(c2, c8, z28);

                // z33 = _mm512_fmadd_pd(c3, c3, z33);
                // z34 = _mm512_fmadd_pd(c3, c4, z34);
                // z35 = _mm512_fmadd_pd(c3, c5, z35);
                // z36 = _mm512_fmadd_pd(c3, c6, z36);
                // z37 = _mm512_fmadd_pd(c3, c7, z37);
                // z38 = _mm512_fmadd_pd(c3, c8, z38);
            }

            h17.store(z17);
            h18.store(z18);

            // Row 2
            h22.store(z22);
            h23.store(z23);
            h24.store(z24);
            h25.store(z25);
            h26.store(z26);
            h27.store(z27);
            h28.store(z28);

            // Row 3
            h33.store(z33);
            h34.store(z34);
            h35.store(z35);
            h36.store(z36);
            h37.store(z37);
            h38.store(z38);
        }

        {
            // Row 4
            let mut z44 = _mm512_setzero_pd();
            let mut z45 = _mm512_setzero_pd();
            let mut z46 = _mm512_setzero_pd();
            let mut z47 = _mm512_setzero_pd();
            let mut z48 = _mm512_setzero_pd();

            // Row 5
            let mut z55 = _mm512_setzero_pd();
            let mut z56 = _mm512_setzero_pd();
            let mut z57 = _mm512_setzero_pd();
            let mut z58 = _mm512_setzero_pd();

            // Row 6
            let mut z66 = _mm512_setzero_pd();
            let mut z67 = _mm512_setzero_pd();
            let mut z68 = _mm512_setzero_pd();

            // Row 7
            let mut z77 = _mm512_setzero_pd();
            let mut z78 = _mm512_setzero_pd();

            // Row 8
            let mut z88 = _mm512_setzero_pd();

            for block in scaled_column_buf.iter() {
                // 5 permanent registers
                let c4 = block[4].load();
                let c5 = block[5].load();
                let c6 = block[6].load();
                let c7 = block[7].load();
                let c8 = block[8].load();

                z44 = _mm512_fmadd_pd(c4, c4, z44);
                z45 = _mm512_fmadd_pd(c4, c5, z45);
                z46 = _mm512_fmadd_pd(c4, c6, z46);
                z47 = _mm512_fmadd_pd(c4, c7, z47);
                z48 = _mm512_fmadd_pd(c4, c8, z48);

                z55 = _mm512_fmadd_pd(c5, c5, z55);
                z56 = _mm512_fmadd_pd(c5, c6, z56);
                z57 = _mm512_fmadd_pd(c5, c7, z57);
                z58 = _mm512_fmadd_pd(c5, c8, z58);

                z66 = _mm512_fmadd_pd(c6, c6, z66);
                z67 = _mm512_fmadd_pd(c6, c7, z67);
                z68 = _mm512_fmadd_pd(c6, c8, z68);

                z77 = _mm512_fmadd_pd(c7, c7, z77);
                z78 = _mm512_fmadd_pd(c7, c8, z78);

                z88 = _mm512_fmadd_pd(c8, c8, z88);
            }

            z44 = _mm512_add_pd(h44.load(), z44);
            z45 = _mm512_add_pd(h45.load(), z45);
            z46 = _mm512_add_pd(h46.load(), z46);
            z47 = _mm512_add_pd(h47.load(), z47);
            z48 = _mm512_add_pd(h48.load(), z48);

            // Row 5
            z55 = _mm512_add_pd(h55.load(), z55);
            z56 = _mm512_add_pd(h56.load(), z56);
            z57 = _mm512_add_pd(h57.load(), z57);
            z58 = _mm512_add_pd(h58.load(), z58);

            // Row 6
            z66 = _mm512_add_pd(h66.load(), z66);
            z67 = _mm512_add_pd(h67.load(), z67);
            z68 = _mm512_add_pd(h68.load(), z68);

            // Row 7
            z77 = _mm512_add_pd(h77.load(), z77);
            z78 = _mm512_add_pd(h78.load(), z78);

            // Row 8
            z88 = _mm512_add_pd(h88.load(), z88);

            // Row 4
            h44.store(z44);
            h45.store(z45);
            h46.store(z46);
            h47.store(z47);
            h48.store(z48);

            // Row 5
            h55.store(z55);
            h56.store(z56);
            h57.store(z57);
            h58.store(z58);

            // Row 6
            h66.store(z66);
            h67.store(z67);
            h68.store(z68);

            // Row 7
            h77.store(z77);
            h78.store(z78);

            // Row 8
            h88.store(z88);
        }
    }

    let mut grad = [0.0; 9];
    for col in 0..9 {
        grad[col] = _mm512_reduce_add_pd(g[col].load());
    }

    let mut hess = [[0.0; 9]; 9];

    // First row
    hess[0][0] = _mm512_reduce_add_pd(h00.load());
    hess[0][1] = _mm512_reduce_add_pd(h01.load());
    hess[0][2] = _mm512_reduce_add_pd(h02.load());
    hess[0][3] = _mm512_reduce_add_pd(h03.load());
    hess[0][4] = _mm512_reduce_add_pd(h04.load());
    hess[0][5] = _mm512_reduce_add_pd(h05.load());
    hess[0][6] = _mm512_reduce_add_pd(h06.load());
    hess[0][7] = _mm512_reduce_add_pd(h07.load());
    hess[0][8] = _mm512_reduce_add_pd(h08.load());

    // Second row
    hess[1][1] = _mm512_reduce_add_pd(h11.load());
    hess[1][2] = _mm512_reduce_add_pd(h12.load());
    hess[1][3] = _mm512_reduce_add_pd(h13.load());
    hess[1][4] = _mm512_reduce_add_pd(h14.load());
    hess[1][5] = _mm512_reduce_add_pd(h15.load());
    hess[1][6] = _mm512_reduce_add_pd(h16.load());
    hess[1][7] = _mm512_reduce_add_pd(h17.load());
    hess[1][8] = _mm512_reduce_add_pd(h18.load());

    hess[2][2] = _mm512_reduce_add_pd(h22.load());
    hess[2][3] = _mm512_reduce_add_pd(h23.load());
    hess[2][4] = _mm512_reduce_add_pd(h24.load());
    hess[2][5] = _mm512_reduce_add_pd(h25.load());
    hess[2][6] = _mm512_reduce_add_pd(h26.load());
    hess[2][7] = _mm512_reduce_add_pd(h27.load());
    hess[2][8] = _mm512_reduce_add_pd(h28.load());

    hess[3][3] = _mm512_reduce_add_pd(h33.load());
    hess[3][4] = _mm512_reduce_add_pd(h34.load());
    hess[3][5] = _mm512_reduce_add_pd(h35.load());
    hess[3][6] = _mm512_reduce_add_pd(h36.load());
    hess[3][7] = _mm512_reduce_add_pd(h37.load());
    hess[3][8] = _mm512_reduce_add_pd(h38.load());

    hess[4][4] = _mm512_reduce_add_pd(h44.load());
    hess[4][5] = _mm512_reduce_add_pd(h45.load());
    hess[4][6] = _mm512_reduce_add_pd(h46.load());
    hess[4][7] = _mm512_reduce_add_pd(h47.load());
    hess[4][8] = _mm512_reduce_add_pd(h48.load());

    hess[5][5] = _mm512_reduce_add_pd(h55.load());
    hess[5][6] = _mm512_reduce_add_pd(h56.load());
    hess[5][7] = _mm512_reduce_add_pd(h57.load());
    hess[5][8] = _mm512_reduce_add_pd(h58.load());

    hess[6][6] = _mm512_reduce_add_pd(h66.load());
    hess[6][7] = _mm512_reduce_add_pd(h67.load());
    hess[6][8] = _mm512_reduce_add_pd(h68.load());

    hess[7][7] = _mm512_reduce_add_pd(h77.load());
    hess[7][8] = _mm512_reduce_add_pd(h78.load());

    hess[8][8] = _mm512_reduce_add_pd(h88.load());

    for i in 0..9 {
        for j in i..9 {
            hess[j][i] = hess[i][j];
        }
    }

    (grad, hess)
}

#[target_feature(enable = "avx512f")]
fn fun_name(
    x: &[f64; 9],
    eps: f64,
    scaled_columns: &mut [[Lane8; 9]; 32],
    tile: &[[[f64; 8]; 9]; 32],
) {
    let one = _mm512_set1_pd(1.0);

    let xs = [
        _mm512_set1_pd(x[0]),
        _mm512_set1_pd(x[1]),
        _mm512_set1_pd(x[2]),
        _mm512_set1_pd(x[3]),
        _mm512_set1_pd(x[4]),
        _mm512_set1_pd(x[5]),
        _mm512_set1_pd(x[6]),
        _mm512_set1_pd(x[7]),
        _mm512_set1_pd(x[8]),
    ];
    for (buffer, block) in scaled_columns.iter_mut().zip(tile.iter()) {
        let mut c: [__m512d; 9] =
            std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

        // Calculate d
        // This computes a dot product between x and a row of p
        // TODO this could be manually unrolled a few times
        let mut d = _mm512_set1_pd(eps);
        for col in 0..9 {
            d = _mm512_fmadd_pd(xs[col], c[col], d);
        }

        // TODO investigate reciprocal
        d = _mm512_div_pd(one, d);

        // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

        for col in 0..9 {
            c[col] = _mm512_mul_pd(c[col], d);
        }

        for col in 0..9 {
            buffer[col].store(c[col]);
        }
    }
}

#[target_feature(enable = "avx512f")]
fn fun_name2(
    x: &[f64; 9],
    eps: f64,
    scaled_columns: &mut [[Lane8; 9]; 32],
    tile: &[[[f64; 8]; 9]; 32],
    g: &mut [Lane8; 9]
) {
}

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_row_major(
    rows: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {
    // We want to the outer product of each row with itself
    // and

    // Columns 1 to 8
    let mut h1to8 = [_mm512_setzero_pd(); 9];
    // Column 9
    let mut h9 = [0.0; 9];

    for row in rows.iter() {
        let r1to8 = unsafe { _mm512_loadu_pd(row.as_ptr()) };
        let r9 = row[8];
    }

    let mut h = [[0.0; 9]; 9];

    h
}

// New plan: calculate d * columns and store it in memory.
// (we need to read P from memory anyway again to calculate
// (g and h, so just store the whole dang thing once)
// (maybe?)
// Then

// #[target_feature(enable = "avx512f")]
// pub fn compute_pt_d2_p_avx512_three_pass_untiled(
//     blocks: &[Block<f64, 8, 9>],
//     remainder: &[[f64; 9]],
//     x: &[f64; 9],
//     eps: f64,
// ) -> [[f64; 9]; 9] {

//     let one = _mm512_set1_pd(1.0);

//     let xs = [
//         _mm512_set1_pd(x[0]),
//         _mm512_set1_pd(x[1]),
//         _mm512_set1_pd(x[2]),
//         _mm512_set1_pd(x[3]),
//         _mm512_set1_pd(x[4]),
//         _mm512_set1_pd(x[5]),
//         _mm512_set1_pd(x[6]),
//         _mm512_set1_pd(x[7]),
//         _mm512_set1_pd(x[8]),
//     ];

//     for block in blocks.iter() {

//         // Calculate d
//         // This computes a dot product between x and a row of p
//         // TODO this could be manually unrolled a few times
//         let mut d = _mm512_set1_pd(eps);
//         for col in 0..9 {
//             let c = _mm512_load_pd(block[col].as_ptr());
//             d = _mm512_fmadd_pd(xs[col], c, d);
//         }

//         // TODO investigate reciprocal
//         d = _mm512_div_pd(one, d);

//         // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

//         for col in 0.. 9 {
//             c[col] = _mm512_mul_pd(c[col], d);
//         }

//         {
//         // Row 0
//         let mut z00 = h00.load();
//         let mut z01 = h01.load();
//         let mut z02 = h02.load();
//         let mut z03 = h03.load();
//         let mut z04 = h04.load();
//         let mut z05 = h05.load();
//         let mut z06 = h06.load();
//         let mut z07 = h07.load();
//         let mut z08 = h08.load();

//         // Row 1
//         let mut z11 = h11.load();
//         let mut z12 = h12.load();
//         let mut z13 = h13.load();
//         let mut z14 = h14.load();
//         let mut z15 = h15.load();
//         let mut z16 = h16.load();

//         for block in scaled_column_buf.iter() {
//             let c: [__m512d; 9] = std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

//             z00 = _mm512_fmadd_pd(c[0], c[0], z00);
//             z01 = _mm512_fmadd_pd(c[0], c[1], z01);
//             z02 = _mm512_fmadd_pd(c[0], c[2], z02);
//             z03 = _mm512_fmadd_pd(c[0], c[3], z03);
//             z04 = _mm512_fmadd_pd(c[0], c[4], z04);
//             z05 = _mm512_fmadd_pd(c[0], c[5], z05);
//             z06 = _mm512_fmadd_pd(c[0], c[6], z06);
//             z07 = _mm512_fmadd_pd(c[0], c[7], z07);
//             z08 = _mm512_fmadd_pd(c[0], c[8], z08);

//             z11 = _mm512_fmadd_pd(c[1], c[1], z11);
//             z12 = _mm512_fmadd_pd(c[1], c[2], z12);
//             z13 = _mm512_fmadd_pd(c[1], c[3], z13);
//             z14 = _mm512_fmadd_pd(c[1], c[4], z14);
//             z15 = _mm512_fmadd_pd(c[1], c[5], z15);
//             z16 = _mm512_fmadd_pd(c[1], c[6], z16);
//         }
//     }

//         {

//         let mut z17 = h17.load();
//         let mut z18 = h18.load();

//         // Row 2
//         let mut z22 = h22.load();
//         let mut z23 = h23.load();
//         let mut z24 = h24.load();
//         let mut z25 = h25.load();
//         let mut z26 = h26.load();
//         let mut z27 = h27.load();
//         let mut z28 = h28.load();

//         // Row 3
//         let mut z33 = h33.load();
//         let mut z34 = h34.load();
//         let mut z35 = h35.load();
//         let mut z36 = h36.load();
//         let mut z37 = h37.load();
//         let mut z38 = h38.load();

//         for block in scaled_column_buf.iter() {
//             let c: [__m512d; 9] = std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

//             z17 = _mm512_fmadd_pd(c[1], c[7], z17);
//             z18 = _mm512_fmadd_pd(c[1], c[8], z18);

//             z22 = _mm512_fmadd_pd(c[2], c[2], z22);
//             z23 = _mm512_fmadd_pd(c[2], c[3], z23);
//             z24 = _mm512_fmadd_pd(c[2], c[4], z24);
//             z25 = _mm512_fmadd_pd(c[2], c[5], z25);
//             z26 = _mm512_fmadd_pd(c[2], c[6], z26);
//             z27 = _mm512_fmadd_pd(c[2], c[7], z27);
//             z28 = _mm512_fmadd_pd(c[2], c[8], z28);

//             z33 = _mm512_fmadd_pd(c[3], c[3], z33);
//             z34 = _mm512_fmadd_pd(c[3], c[4], z34);
//             z35 = _mm512_fmadd_pd(c[3], c[5], z35);
//             z36 = _mm512_fmadd_pd(c[3], c[6], z36);
//             z37 = _mm512_fmadd_pd(c[3], c[7], z37);
//             z38 = _mm512_fmadd_pd(c[3], c[8], z38);
//         }

//     }

//         {
//         // Row 4
//         let mut z44 = _mm512_setzero_pd();
//         let mut z45 = _mm512_setzero_pd();
//         let mut z46 = _mm512_setzero_pd();
//         let mut z47 = _mm512_setzero_pd();
//         let mut z48 = _mm512_setzero_pd();

//         // Row 5
//         let mut z55 = _mm512_setzero_pd();
//         let mut z56 = _mm512_setzero_pd();
//         let mut z57 = _mm512_setzero_pd();
//         let mut z58 = _mm512_setzero_pd();

//         // Row 6
//         let mut z66 = _mm512_setzero_pd();
//         let mut z67 = _mm512_setzero_pd();
//         let mut z68 = _mm512_setzero_pd();

//         // Row 7
//         let mut z77 = _mm512_setzero_pd();
//         let mut z78 = _mm512_setzero_pd();

//         // Row 8
//         let mut z88 = _mm512_setzero_pd();

//         for block in scaled_column_buf.iter() {
//             let c4 = block[4].load();
//             let c5 = block[5].load();
//             let c6 = block[6].load();
//             let c7 = block[7].load();
//             let c8 = block[8].load();

//             z44 = _mm512_fmadd_pd(c4, c4, z44);
//             z45 = _mm512_fmadd_pd(c4, c5, z45);
//             z46 = _mm512_fmadd_pd(c4, c6, z46);
//             z47 = _mm512_fmadd_pd(c4, c7, z47);
//             z48 = _mm512_fmadd_pd(c4, c8, z48);

//             z55 = _mm512_fmadd_pd(c5, c5, z55);
//             z56 = _mm512_fmadd_pd(c5, c6, z56);
//             z57 = _mm512_fmadd_pd(c5, c7, z57);
//             z58 = _mm512_fmadd_pd(c5, c8, z58);

//             z66 = _mm512_fmadd_pd(c6, c6, z66);
//             z67 = _mm512_fmadd_pd(c6, c7, z67);
//             z68 = _mm512_fmadd_pd(c6, c8, z68);

//             z77 = _mm512_fmadd_pd(c7, c7, z77);
//             z78 = _mm512_fmadd_pd(c7, c8, z78);

//             z88 = _mm512_fmadd_pd(c8, c8, z88);
//         }

//         z44 = _mm512_add_pd(h44.load(), z44);
//         z45 = _mm512_add_pd(h45.load(), z45);
//         z46 = _mm512_add_pd(h46.load(), z46);
//         z47 = _mm512_add_pd(h47.load(), z47);
//         z48 = _mm512_add_pd(h48.load(), z48);

//         // Row 5
//         z55 = _mm512_add_pd(h55.load(), z55);
//         z56 = _mm512_add_pd(h56.load(), z56);
//         z57 = _mm512_add_pd(h57.load(), z57);
//         z58 = _mm512_add_pd(h58.load(), z58);

//         // Row 6
//         z66 = _mm512_add_pd(h66.load(), z66);
//         z67 = _mm512_add_pd(h67.load(), z67);
//         z68 = _mm512_add_pd(h68.load(), z68);

//         // Row 7
//         z77 = _mm512_add_pd(h77.load(), z77);
//         z78 = _mm512_add_pd(h78.load(), z78);

//         // Row 8
//         z88 = _mm512_add_pd(h88.load(), z88);

//     }
//     }

//     let mut h = [[0.0; 9]; 9];

//     // First row
//     h[0][0] = _mm512_reduce_add_pd(h00.load());
//     h[0][1] = _mm512_reduce_add_pd(h01.load());
//     h[0][2] = _mm512_reduce_add_pd(h02.load());
//     h[0][3] = _mm512_reduce_add_pd(h03.load());
//     h[0][4] = _mm512_reduce_add_pd(h04.load());
//     h[0][5] = _mm512_reduce_add_pd(h05.load());
//     h[0][6] = _mm512_reduce_add_pd(h06.load());
//     h[0][7] = _mm512_reduce_add_pd(h07.load());
//     h[0][8] = _mm512_reduce_add_pd(h08.load());

//     // Second row
//     h[1][1] = _mm512_reduce_add_pd(h11.load());
//     h[1][2] = _mm512_reduce_add_pd(h12.load());
//     h[1][3] = _mm512_reduce_add_pd(h13.load());
//     h[1][4] = _mm512_reduce_add_pd(h14.load());
//     h[1][5] = _mm512_reduce_add_pd(h15.load());
//     h[1][6] = _mm512_reduce_add_pd(h16.load());
//     h[1][7] = _mm512_reduce_add_pd(h17.load());
//     h[1][8] = _mm512_reduce_add_pd(h18.load());

//     h[2][2] = _mm512_reduce_add_pd(h22.load());
//     h[2][3] = _mm512_reduce_add_pd(h23.load());
//     h[2][4] = _mm512_reduce_add_pd(h24.load());
//     h[2][5] = _mm512_reduce_add_pd(h25.load());
//     h[2][6] = _mm512_reduce_add_pd(h26.load());
//     h[2][7] = _mm512_reduce_add_pd(h27.load());
//     h[2][8] = _mm512_reduce_add_pd(h28.load());

//     h[3][3] = _mm512_reduce_add_pd(h33.load());
//     h[3][4] = _mm512_reduce_add_pd(h34.load());
//     h[3][5] = _mm512_reduce_add_pd(h35.load());
//     h[3][6] = _mm512_reduce_add_pd(h36.load());
//     h[3][7] = _mm512_reduce_add_pd(h37.load());
//     h[3][8] = _mm512_reduce_add_pd(h38.load());

//     h[4][4] = _mm512_reduce_add_pd(h44.load());
//     h[4][5] = _mm512_reduce_add_pd(h45.load());
//     h[4][6] = _mm512_reduce_add_pd(h46.load());
//     h[4][7] = _mm512_reduce_add_pd(h47.load());
//     h[4][8] = _mm512_reduce_add_pd(h48.load());

//     h[5][5] = _mm512_reduce_add_pd(h55.load());
//     h[5][6] = _mm512_reduce_add_pd(h56.load());
//     h[5][7] = _mm512_reduce_add_pd(h57.load());
//     h[5][8] = _mm512_reduce_add_pd(h58.load());

//     h[6][6] = _mm512_reduce_add_pd(h66.load());
//     h[6][7] = _mm512_reduce_add_pd(h67.load());
//     h[6][8] = _mm512_reduce_add_pd(h68.load());

//     h[7][7] = _mm512_reduce_add_pd(h77.load());
//     h[7][8] = _mm512_reduce_add_pd(h78.load());

//     h[8][8] = _mm512_reduce_add_pd(h88.load());

//     for i in 0..9 {
//         for j in i..9 {
//             h[j][i] = h[i][j];
//         }
//     }

//     h
// }

#[target_feature(enable = "avx512f")]
pub unsafe fn compute_pt_d2_p_avx512_three_pass_untiled(
    blocks: &[[[f64; 8]; 9]],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> [[f64; 9]; 9] {
    use std::arch::x86_64::*;

    let mut h = [[0.0; 9]; 9];
    let one = _mm512_set1_pd(1.0);

    // xs occupies 9 registers, but because they are loop invariants,
    // the compiler may fold them into memory broadcast operands if needed.
    let xs = [
        _mm512_set1_pd(x[0]),
        _mm512_set1_pd(x[1]),
        _mm512_set1_pd(x[2]),
        _mm512_set1_pd(x[3]),
        _mm512_set1_pd(x[4]),
        _mm512_set1_pd(x[5]),
        _mm512_set1_pd(x[6]),
        _mm512_set1_pd(x[7]),
        _mm512_set1_pd(x[8]),
    ];

    // ==========================================
    // PASS 1: Row 0 (9) + Row 1 partial (6) = 15
    // Peak Register Pressure: ~27 ZMMs
    // ==========================================
    {
        let mut z00 = _mm512_setzero_pd();
        let mut z01 = _mm512_setzero_pd();
        let mut z02 = _mm512_setzero_pd();
        let mut z03 = _mm512_setzero_pd();
        let mut z04 = _mm512_setzero_pd();
        let mut z05 = _mm512_setzero_pd();
        let mut z06 = _mm512_setzero_pd();
        let mut z07 = _mm512_setzero_pd();
        let mut z08 = _mm512_setzero_pd();

        let mut z11 = _mm512_setzero_pd();
        let mut z12 = _mm512_setzero_pd();
        let mut z13 = _mm512_setzero_pd();
        let mut z14 = _mm512_setzero_pd();
        let mut z15 = _mm512_setzero_pd();
        let mut z16 = _mm512_setzero_pd();

        for block in blocks.iter() {
            // PHASE 1: Compute d (Registers die immediately)
            let mut d = _mm512_set1_pd(eps);
            d = _mm512_fmadd_pd(xs[0], _mm512_loadu_pd(block[0].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[1], _mm512_loadu_pd(block[1].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[2], _mm512_loadu_pd(block[2].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[3], _mm512_loadu_pd(block[3].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[4], _mm512_loadu_pd(block[4].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[5], _mm512_loadu_pd(block[5].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[6], _mm512_loadu_pd(block[6].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[7], _mm512_loadu_pd(block[7].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[8], _mm512_loadu_pd(block[8].as_ptr()), d);
            d = _mm512_div_pd(one, d);

            // PHASE 2: Interleaved Double-Load Accumulation
            // Load, scale, accumulate, and let the register die.
            let sc0 = _mm512_mul_pd(_mm512_loadu_pd(block[0].as_ptr()), d);
            z00 = _mm512_fmadd_pd(sc0, sc0, z00);

            let sc1 = _mm512_mul_pd(_mm512_loadu_pd(block[1].as_ptr()), d);
            z01 = _mm512_fmadd_pd(sc0, sc1, z01);
            z11 = _mm512_fmadd_pd(sc1, sc1, z11);

            let sc2 = _mm512_mul_pd(_mm512_loadu_pd(block[2].as_ptr()), d);
            z02 = _mm512_fmadd_pd(sc0, sc2, z02);
            z12 = _mm512_fmadd_pd(sc1, sc2, z12);

            let sc3 = _mm512_mul_pd(_mm512_loadu_pd(block[3].as_ptr()), d);
            z03 = _mm512_fmadd_pd(sc0, sc3, z03);
            z13 = _mm512_fmadd_pd(sc1, sc3, z13);

            let sc4 = _mm512_mul_pd(_mm512_loadu_pd(block[4].as_ptr()), d);
            z04 = _mm512_fmadd_pd(sc0, sc4, z04);
            z14 = _mm512_fmadd_pd(sc1, sc4, z14);

            let sc5 = _mm512_mul_pd(_mm512_loadu_pd(block[5].as_ptr()), d);
            z05 = _mm512_fmadd_pd(sc0, sc5, z05);
            z15 = _mm512_fmadd_pd(sc1, sc5, z15);

            let sc6 = _mm512_mul_pd(_mm512_loadu_pd(block[6].as_ptr()), d);
            z06 = _mm512_fmadd_pd(sc0, sc6, z06);
            z16 = _mm512_fmadd_pd(sc1, sc6, z16);

            let sc7 = _mm512_mul_pd(_mm512_loadu_pd(block[7].as_ptr()), d);
            z07 = _mm512_fmadd_pd(sc0, sc7, z07);

            let sc8 = _mm512_mul_pd(_mm512_loadu_pd(block[8].as_ptr()), d);
            z08 = _mm512_fmadd_pd(sc0, sc8, z08);
        }

        h[0][0] = _mm512_reduce_add_pd(z00);
        h[0][1] = _mm512_reduce_add_pd(z01);
        h[0][2] = _mm512_reduce_add_pd(z02);
        h[0][3] = _mm512_reduce_add_pd(z03);
        h[0][4] = _mm512_reduce_add_pd(z04);
        h[0][5] = _mm512_reduce_add_pd(z05);
        h[0][6] = _mm512_reduce_add_pd(z06);
        h[0][7] = _mm512_reduce_add_pd(z07);
        h[0][8] = _mm512_reduce_add_pd(z08);

        h[1][1] = _mm512_reduce_add_pd(z11);
        h[1][2] = _mm512_reduce_add_pd(z12);
        h[1][3] = _mm512_reduce_add_pd(z13);
        h[1][4] = _mm512_reduce_add_pd(z14);
        h[1][5] = _mm512_reduce_add_pd(z15);
        h[1][6] = _mm512_reduce_add_pd(z16);
    }

    // =======================================================
    // PASS 2: Row 1 rem (2) + Row 2 (7) + Row 3 (6) = 15
    // Peak Register Pressure: ~29 ZMMs
    // =======================================================
    {
        let mut z17 = _mm512_setzero_pd();
        let mut z18 = _mm512_setzero_pd();

        let mut z22 = _mm512_setzero_pd();
        let mut z23 = _mm512_setzero_pd();
        let mut z24 = _mm512_setzero_pd();
        let mut z25 = _mm512_setzero_pd();
        let mut z26 = _mm512_setzero_pd();
        let mut z27 = _mm512_setzero_pd();
        let mut z28 = _mm512_setzero_pd();

        let mut z33 = _mm512_setzero_pd();
        let mut z34 = _mm512_setzero_pd();
        let mut z35 = _mm512_setzero_pd();
        let mut z36 = _mm512_setzero_pd();
        let mut z37 = _mm512_setzero_pd();
        let mut z38 = _mm512_setzero_pd();

        for block in blocks.iter() {
            let mut d = _mm512_set1_pd(eps);
            d = _mm512_fmadd_pd(xs[0], _mm512_loadu_pd(block[0].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[1], _mm512_loadu_pd(block[1].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[2], _mm512_loadu_pd(block[2].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[3], _mm512_loadu_pd(block[3].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[4], _mm512_loadu_pd(block[4].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[5], _mm512_loadu_pd(block[5].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[6], _mm512_loadu_pd(block[6].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[7], _mm512_loadu_pd(block[7].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[8], _mm512_loadu_pd(block[8].as_ptr()), d);
            d = _mm512_div_pd(one, d);

            // We must keep sc1, sc2, and sc3 alive as bases for the row accumulation
            let sc1 = _mm512_mul_pd(_mm512_loadu_pd(block[1].as_ptr()), d);
            let sc2 = _mm512_mul_pd(_mm512_loadu_pd(block[2].as_ptr()), d);
            z22 = _mm512_fmadd_pd(sc2, sc2, z22);

            let sc3 = _mm512_mul_pd(_mm512_loadu_pd(block[3].as_ptr()), d);
            z23 = _mm512_fmadd_pd(sc2, sc3, z23);
            z33 = _mm512_fmadd_pd(sc3, sc3, z33);

            let sc4 = _mm512_mul_pd(_mm512_loadu_pd(block[4].as_ptr()), d);
            z24 = _mm512_fmadd_pd(sc2, sc4, z24);
            z34 = _mm512_fmadd_pd(sc3, sc4, z34);

            let sc5 = _mm512_mul_pd(_mm512_loadu_pd(block[5].as_ptr()), d);
            z25 = _mm512_fmadd_pd(sc2, sc5, z25);
            z35 = _mm512_fmadd_pd(sc3, sc5, z35);

            let sc6 = _mm512_mul_pd(_mm512_loadu_pd(block[6].as_ptr()), d);
            z26 = _mm512_fmadd_pd(sc2, sc6, z26);
            z36 = _mm512_fmadd_pd(sc3, sc6, z36);

            let sc7 = _mm512_mul_pd(_mm512_loadu_pd(block[7].as_ptr()), d);
            z17 = _mm512_fmadd_pd(sc1, sc7, z17);
            z27 = _mm512_fmadd_pd(sc2, sc7, z27);
            z37 = _mm512_fmadd_pd(sc3, sc7, z37);

            let sc8 = _mm512_mul_pd(_mm512_loadu_pd(block[8].as_ptr()), d);
            z18 = _mm512_fmadd_pd(sc1, sc8, z18);
            z28 = _mm512_fmadd_pd(sc2, sc8, z28);
            z38 = _mm512_fmadd_pd(sc3, sc8, z38);
        }

        h[1][7] = _mm512_reduce_add_pd(z17);
        h[1][8] = _mm512_reduce_add_pd(z18);

        h[2][2] = _mm512_reduce_add_pd(z22);
        h[2][3] = _mm512_reduce_add_pd(z23);
        h[2][4] = _mm512_reduce_add_pd(z24);
        h[2][5] = _mm512_reduce_add_pd(z25);
        h[2][6] = _mm512_reduce_add_pd(z26);
        h[2][7] = _mm512_reduce_add_pd(z27);
        h[2][8] = _mm512_reduce_add_pd(z28);

        h[3][3] = _mm512_reduce_add_pd(z33);
        h[3][4] = _mm512_reduce_add_pd(z34);
        h[3][5] = _mm512_reduce_add_pd(z35);
        h[3][6] = _mm512_reduce_add_pd(z36);
        h[3][7] = _mm512_reduce_add_pd(z37);
        h[3][8] = _mm512_reduce_add_pd(z38);
    }

    // =================================================================
    // PASS 3: Row 4(5) + Row 5(4) + Row 6(3) + Row 7(2) + Row 8(1) = 15
    // Peak Register Pressure: ~30 ZMMs
    // =================================================================
    {
        let mut z44 = _mm512_setzero_pd();
        let mut z45 = _mm512_setzero_pd();
        let mut z46 = _mm512_setzero_pd();
        let mut z47 = _mm512_setzero_pd();
        let mut z48 = _mm512_setzero_pd();

        let mut z55 = _mm512_setzero_pd();
        let mut z56 = _mm512_setzero_pd();
        let mut z57 = _mm512_setzero_pd();
        let mut z58 = _mm512_setzero_pd();

        let mut z66 = _mm512_setzero_pd();
        let mut z67 = _mm512_setzero_pd();
        let mut z68 = _mm512_setzero_pd();

        let mut z77 = _mm512_setzero_pd();
        let mut z78 = _mm512_setzero_pd();

        let mut z88 = _mm512_setzero_pd();

        for block in blocks.iter() {
            let mut d = _mm512_set1_pd(eps);
            d = _mm512_fmadd_pd(xs[0], _mm512_loadu_pd(block[0].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[1], _mm512_loadu_pd(block[1].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[2], _mm512_loadu_pd(block[2].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[3], _mm512_loadu_pd(block[3].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[4], _mm512_loadu_pd(block[4].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[5], _mm512_loadu_pd(block[5].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[6], _mm512_loadu_pd(block[6].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[7], _mm512_loadu_pd(block[7].as_ptr()), d);
            d = _mm512_fmadd_pd(xs[8], _mm512_loadu_pd(block[8].as_ptr()), d);
            d = _mm512_div_pd(one, d);

            // Bases needed alive as we iterate down
            let sc4 = _mm512_mul_pd(_mm512_loadu_pd(block[4].as_ptr()), d);
            z44 = _mm512_fmadd_pd(sc4, sc4, z44);

            let sc5 = _mm512_mul_pd(_mm512_loadu_pd(block[5].as_ptr()), d);
            z45 = _mm512_fmadd_pd(sc4, sc5, z45);
            z55 = _mm512_fmadd_pd(sc5, sc5, z55);

            let sc6 = _mm512_mul_pd(_mm512_loadu_pd(block[6].as_ptr()), d);
            z46 = _mm512_fmadd_pd(sc4, sc6, z46);
            z56 = _mm512_fmadd_pd(sc5, sc6, z56);
            z66 = _mm512_fmadd_pd(sc6, sc6, z66);

            let sc7 = _mm512_mul_pd(_mm512_loadu_pd(block[7].as_ptr()), d);
            z47 = _mm512_fmadd_pd(sc4, sc7, z47);
            z57 = _mm512_fmadd_pd(sc5, sc7, z57);
            z67 = _mm512_fmadd_pd(sc6, sc7, z67);
            z77 = _mm512_fmadd_pd(sc7, sc7, z77);

            let sc8 = _mm512_mul_pd(_mm512_loadu_pd(block[8].as_ptr()), d);
            z48 = _mm512_fmadd_pd(sc4, sc8, z48);
            z58 = _mm512_fmadd_pd(sc5, sc8, z58);
            z68 = _mm512_fmadd_pd(sc6, sc8, z68);
            z78 = _mm512_fmadd_pd(sc7, sc8, z78);
            z88 = _mm512_fmadd_pd(sc8, sc8, z88);
        }

        h[4][4] = _mm512_reduce_add_pd(z44);
        h[4][5] = _mm512_reduce_add_pd(z45);
        h[4][6] = _mm512_reduce_add_pd(z46);
        h[4][7] = _mm512_reduce_add_pd(z47);
        h[4][8] = _mm512_reduce_add_pd(z48);

        h[5][5] = _mm512_reduce_add_pd(z55);
        h[5][6] = _mm512_reduce_add_pd(z56);
        h[5][7] = _mm512_reduce_add_pd(z57);
        h[5][8] = _mm512_reduce_add_pd(z58);

        h[6][6] = _mm512_reduce_add_pd(z66);
        h[6][7] = _mm512_reduce_add_pd(z67);
        h[6][8] = _mm512_reduce_add_pd(z68);

        h[7][7] = _mm512_reduce_add_pd(z77);
        h[7][8] = _mm512_reduce_add_pd(z78);

        h[8][8] = _mm512_reduce_add_pd(z88);
    }

    // ==========================================
    // Process Remainder (Scalar Fallback)
    // ==========================================
    for r in remainder {
        let mut d = eps;
        for col in 0..9 {
            d += x[col] * r[col];
        }

        d = 1.0 / d;

        let mut sc = [0.0; 9];
        for col in 0..9 {
            sc[col] = r[col] * d;
        }

        // Accumulate upper triangle
        for i in 0..9 {
            for j in i..9 {
                h[i][j] += sc[i] * sc[j];
            }
        }
    }

    // ==========================================
    // Mirror Upper Triangle to Lower Triangle
    // ==========================================
    for i in 0..9 {
        for j in (i + 1)..9 {
            h[j][i] = h[i][j];
        }
    }

    h
}



#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_three_passes3_d(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [[f64; 9]; 9]) {
    const BLOCKS: usize = 32;

    let mut g = [Lane8::zero(); 9];

    // First row
    let mut h00 = Lane8::zero();
    let mut h01 = Lane8::zero();
    let mut h02 = Lane8::zero();
    let mut h03 = Lane8::zero();
    let mut h04 = Lane8::zero();
    let mut h05 = Lane8::zero();
    let mut h06 = Lane8::zero();
    let mut h07 = Lane8::zero();
    let mut h08 = Lane8::zero();
    // Second row
    let mut h11 = Lane8::zero();
    let mut h12 = Lane8::zero();
    let mut h13 = Lane8::zero();
    let mut h14 = Lane8::zero();
    let mut h15 = Lane8::zero();
    let mut h16 = Lane8::zero();
    let mut h17 = Lane8::zero();
    let mut h18 = Lane8::zero();
    // Third row
    let mut h22 = Lane8::zero();
    let mut h23 = Lane8::zero();
    let mut h24 = Lane8::zero();
    let mut h25 = Lane8::zero();
    let mut h26 = Lane8::zero();
    let mut h27 = Lane8::zero();
    let mut h28 = Lane8::zero();
    // Fourth row
    let mut h33 = Lane8::zero();
    let mut h34 = Lane8::zero();
    let mut h35 = Lane8::zero();
    let mut h36 = Lane8::zero();
    let mut h37 = Lane8::zero();
    let mut h38 = Lane8::zero();
    // Fifth row
    let mut h44 = Lane8::zero();
    let mut h45 = Lane8::zero();
    let mut h46 = Lane8::zero();
    let mut h47 = Lane8::zero();
    let mut h48 = Lane8::zero();
    // Sixth row
    let mut h55 = Lane8::zero();
    let mut h56 = Lane8::zero();
    let mut h57 = Lane8::zero();
    let mut h58 = Lane8::zero();
    // Seventh row
    let mut h66 = Lane8::zero();
    let mut h67 = Lane8::zero();
    let mut h68 = Lane8::zero();
    // Eigth row
    let mut h77 = Lane8::zero();
    let mut h78 = Lane8::zero();
    // Ninth row
    let mut h88 = Lane8::zero();

    // Register allocation:
    // - 9 for x
    // - 1 for one
    // For calculation of d: 1 for d + 9 columns
    // For each Hessian pass: 15 accumulators + at most 5 columns = 20

    let one = _mm512_set1_pd(1.0);

    let xs = [
        _mm512_set1_pd(x[0]),
        _mm512_set1_pd(x[1]),
        _mm512_set1_pd(x[2]),
        _mm512_set1_pd(x[3]),
        _mm512_set1_pd(x[4]),
        _mm512_set1_pd(x[5]),
        _mm512_set1_pd(x[6]),
        _mm512_set1_pd(x[7]),
        _mm512_set1_pd(x[8]),
    ];

    // A block = [Lane8; 9]
    // This is one tile
    let mut d_buf = [Lane8::zero(); BLOCKS];

    let (tiles, rem) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        {
            let mut z = [_mm512_setzero_pd(); 9];

            for (buffer, block) in d_buf.iter_mut().zip(tile.iter()) {
                let mut c: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                // Calculate d
                // This computes a dot product between x and a row of p
                // TODO this could be manually unrolled a few times
                let mut d = _mm512_set1_pd(eps);
                for col in 0..9 {
                    d = _mm512_fmadd_pd(xs[col], c[col], d);
                }

                // TODO investigate reciprocal
                d = _mm512_div_pd(one, d);

                // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

                // for col in 0..9 {
                //     c[col] = _mm512_mul_pd(c[col], d);
                // }

                // TODO investigate only storing d
                buffer.store(d);

                for col in 0..9 {
                    z[col] = _mm512_fmadd_pd(c[col], d, z[col]);
                }
            }

            for col in 0..9 {
                z[col] = _mm512_add_pd(g[col].load(), z[col]);
            }

            for col in 0..9 {
                g[col].store(z[col]);
            }
        }


        {
            // Row 0
            let mut z00 = h00.load();
            let mut z01 = h01.load();
            let mut z02 = h02.load();
            let mut z03 = h03.load();
            let mut z04 = h04.load();
            let mut z05 = h05.load();
            let mut z06 = h06.load();
            let mut z07 = h07.load();
            let mut z08 = h08.load();

            // Row 1
            let mut z11 = h11.load();
            let mut z12 = h12.load();
            let mut z13 = h13.load();
            let mut z14 = h14.load();
            let mut z15 = h15.load();
            let mut z16 = h16.load();

            for (buffer, block) in d_buf.iter_mut().zip(tile.iter()) {
                let d = buffer.load();
                // 2 permanent + 1 temporary = 3 registers
                let c0 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[0].as_ptr()) }, d);
                let c1 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[1].as_ptr()) }, d);

                z00 = _mm512_fmadd_pd(c0, c0, z00);

                z01 = _mm512_fmadd_pd(c0, c1, z01);
                z11 = _mm512_fmadd_pd(c1, c1, z11);

                let c2 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[2].as_ptr()) }, d);
                z02 = _mm512_fmadd_pd(c0, c2, z02);
                z12 = _mm512_fmadd_pd(c1, c2, z12);

                let c3 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[3].as_ptr()) }, d);
                z03 = _mm512_fmadd_pd(c0, c3, z03);
                z13 = _mm512_fmadd_pd(c1, c3, z13);

                let c4 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[4].as_ptr()) }, d);
                z04 = _mm512_fmadd_pd(c0, c4, z04);
                z14 = _mm512_fmadd_pd(c1, c4, z14);

                let c5 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[5].as_ptr()) }, d);
                z05 = _mm512_fmadd_pd(c0, c5, z05);
                z15 = _mm512_fmadd_pd(c1, c5, z15);

                let c6 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[6].as_ptr()) }, d);
                z06 = _mm512_fmadd_pd(c0, c6, z06);
                z16 = _mm512_fmadd_pd(c1, c6, z16);

                let c7 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[7].as_ptr()) }, d);
                z07 = _mm512_fmadd_pd(c0, c7, z07);

                let c8 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[8].as_ptr()) }, d);
                z08 = _mm512_fmadd_pd(c0, c8, z08);
            }

            // Row 0
            h00.store(z00);
            h01.store(z01);
            h02.store(z02);
            h03.store(z03);
            h04.store(z04);
            h05.store(z05);
            h06.store(z06);
            h07.store(z07);
            h08.store(z08);

            // Row 1
            h11.store(z11);
            h12.store(z12);
            h13.store(z13);
            h14.store(z14);
            h15.store(z15);
            h16.store(z16);
        }

        {
            let mut z17 = h17.load();
            let mut z18 = h18.load();

            // Row 2
            let mut z22 = h22.load();
            let mut z23 = h23.load();
            let mut z24 = h24.load();
            let mut z25 = h25.load();
            let mut z26 = h26.load();
            let mut z27 = h27.load();
            let mut z28 = h28.load();

            // Row 3
            let mut z33 = h33.load();
            let mut z34 = h34.load();
            let mut z35 = h35.load();
            let mut z36 = h36.load();
            let mut z37 = h37.load();
            let mut z38 = h38.load();

            for (buffer, block) in d_buf.iter_mut().zip(tile.iter()) {
                let d = buffer.load();
                // 3 permanent + 1 temporary = 4
                let c1 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[1].as_ptr()) }, d);
                let c2 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[2].as_ptr()) }, d);
                let c3 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[3].as_ptr()) }, d);

                z22 = _mm512_fmadd_pd(c2, c2, z22);

                z23 = _mm512_fmadd_pd(c2, c3, z23);
                z33 = _mm512_fmadd_pd(c3, c3, z33);

                let c4 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[4].as_ptr()) }, d);
                z24 = _mm512_fmadd_pd(c2, c4, z24);
                z34 = _mm512_fmadd_pd(c3, c4, z34);

                let c5 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[5].as_ptr()) }, d);
                z25 = _mm512_fmadd_pd(c2, c5, z25);
                z35 = _mm512_fmadd_pd(c3, c5, z35);

                let c6 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[6].as_ptr()) }, d);
                z26 = _mm512_fmadd_pd(c2, c6, z26);
                z36 = _mm512_fmadd_pd(c3, c6, z36);

                let c7 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[7].as_ptr()) }, d);
                z17 = _mm512_fmadd_pd(c1, c7, z17);
                z27 = _mm512_fmadd_pd(c2, c7, z27);
                z37 = _mm512_fmadd_pd(c3, c7, z37);

                let c8 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[8].as_ptr()) }, d);
                z18 = _mm512_fmadd_pd(c1, c8, z18);
                z28 = _mm512_fmadd_pd(c2, c8, z28);
                z38 = _mm512_fmadd_pd(c3, c8, z38);
            }

            h17.store(z17);
            h18.store(z18);

            // Row 2
            h22.store(z22);
            h23.store(z23);
            h24.store(z24);
            h25.store(z25);
            h26.store(z26);
            h27.store(z27);
            h28.store(z28);

            // Row 3
            h33.store(z33);
            h34.store(z34);
            h35.store(z35);
            h36.store(z36);
            h37.store(z37);
            h38.store(z38);
        }

        {
            // Row 4
            let mut z44 = _mm512_setzero_pd();
            let mut z45 = _mm512_setzero_pd();
            let mut z46 = _mm512_setzero_pd();
            let mut z47 = _mm512_setzero_pd();
            let mut z48 = _mm512_setzero_pd();

            // Row 5
            let mut z55 = _mm512_setzero_pd();
            let mut z56 = _mm512_setzero_pd();
            let mut z57 = _mm512_setzero_pd();
            let mut z58 = _mm512_setzero_pd();

            // Row 6
            let mut z66 = _mm512_setzero_pd();
            let mut z67 = _mm512_setzero_pd();
            let mut z68 = _mm512_setzero_pd();

            // Row 7
            let mut z77 = _mm512_setzero_pd();
            let mut z78 = _mm512_setzero_pd();

            // Row 8
            let mut z88 = _mm512_setzero_pd();

            for (buffer, block) in d_buf.iter_mut().zip(tile.iter()) {
                let d = buffer.load();
                // 5 permanent registers
                let c4 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[4].as_ptr()) }, d);
                let c5 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[5].as_ptr()) }, d);
                let c6 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[6].as_ptr()) }, d);
                let c7 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[7].as_ptr()) }, d);
                let c8 = _mm512_mul_pd(unsafe { _mm512_load_pd(block[8].as_ptr()) }, d);

                z44 = _mm512_fmadd_pd(c4, c4, z44);
                z45 = _mm512_fmadd_pd(c4, c5, z45);
                z46 = _mm512_fmadd_pd(c4, c6, z46);
                z47 = _mm512_fmadd_pd(c4, c7, z47);
                z48 = _mm512_fmadd_pd(c4, c8, z48);

                z55 = _mm512_fmadd_pd(c5, c5, z55);
                z56 = _mm512_fmadd_pd(c5, c6, z56);
                z57 = _mm512_fmadd_pd(c5, c7, z57);
                z58 = _mm512_fmadd_pd(c5, c8, z58);

                z66 = _mm512_fmadd_pd(c6, c6, z66);
                z67 = _mm512_fmadd_pd(c6, c7, z67);
                z68 = _mm512_fmadd_pd(c6, c8, z68);

                z77 = _mm512_fmadd_pd(c7, c7, z77);
                z78 = _mm512_fmadd_pd(c7, c8, z78);

                z88 = _mm512_fmadd_pd(c8, c8, z88);
            }

            z44 = _mm512_add_pd(h44.load(), z44);
            z45 = _mm512_add_pd(h45.load(), z45);
            z46 = _mm512_add_pd(h46.load(), z46);
            z47 = _mm512_add_pd(h47.load(), z47);
            z48 = _mm512_add_pd(h48.load(), z48);

            // Row 5
            z55 = _mm512_add_pd(h55.load(), z55);
            z56 = _mm512_add_pd(h56.load(), z56);
            z57 = _mm512_add_pd(h57.load(), z57);
            z58 = _mm512_add_pd(h58.load(), z58);

            // Row 6
            z66 = _mm512_add_pd(h66.load(), z66);
            z67 = _mm512_add_pd(h67.load(), z67);
            z68 = _mm512_add_pd(h68.load(), z68);

            // Row 7
            z77 = _mm512_add_pd(h77.load(), z77);
            z78 = _mm512_add_pd(h78.load(), z78);

            // Row 8
            z88 = _mm512_add_pd(h88.load(), z88);

            // Row 4
            h44.store(z44);
            h45.store(z45);
            h46.store(z46);
            h47.store(z47);
            h48.store(z48);

            // Row 5
            h55.store(z55);
            h56.store(z56);
            h57.store(z57);
            h58.store(z58);

            // Row 6
            h66.store(z66);
            h67.store(z67);
            h68.store(z68);

            // Row 7
            h77.store(z77);
            h78.store(z78);

            // Row 8
            h88.store(z88);
        }
    }

    let mut grad = [0.0; 9];
    for col in 0..9 {
        grad[col] = _mm512_reduce_add_pd(g[col].load());
    }

    let mut hess = [[0.0; 9]; 9];

    // First row
    hess[0][0] = _mm512_reduce_add_pd(h00.load());
    hess[0][1] = _mm512_reduce_add_pd(h01.load());
    hess[0][2] = _mm512_reduce_add_pd(h02.load());
    hess[0][3] = _mm512_reduce_add_pd(h03.load());
    hess[0][4] = _mm512_reduce_add_pd(h04.load());
    hess[0][5] = _mm512_reduce_add_pd(h05.load());
    hess[0][6] = _mm512_reduce_add_pd(h06.load());
    hess[0][7] = _mm512_reduce_add_pd(h07.load());
    hess[0][8] = _mm512_reduce_add_pd(h08.load());

    // Second row
    hess[1][1] = _mm512_reduce_add_pd(h11.load());
    hess[1][2] = _mm512_reduce_add_pd(h12.load());
    hess[1][3] = _mm512_reduce_add_pd(h13.load());
    hess[1][4] = _mm512_reduce_add_pd(h14.load());
    hess[1][5] = _mm512_reduce_add_pd(h15.load());
    hess[1][6] = _mm512_reduce_add_pd(h16.load());
    hess[1][7] = _mm512_reduce_add_pd(h17.load());
    hess[1][8] = _mm512_reduce_add_pd(h18.load());

    hess[2][2] = _mm512_reduce_add_pd(h22.load());
    hess[2][3] = _mm512_reduce_add_pd(h23.load());
    hess[2][4] = _mm512_reduce_add_pd(h24.load());
    hess[2][5] = _mm512_reduce_add_pd(h25.load());
    hess[2][6] = _mm512_reduce_add_pd(h26.load());
    hess[2][7] = _mm512_reduce_add_pd(h27.load());
    hess[2][8] = _mm512_reduce_add_pd(h28.load());

    hess[3][3] = _mm512_reduce_add_pd(h33.load());
    hess[3][4] = _mm512_reduce_add_pd(h34.load());
    hess[3][5] = _mm512_reduce_add_pd(h35.load());
    hess[3][6] = _mm512_reduce_add_pd(h36.load());
    hess[3][7] = _mm512_reduce_add_pd(h37.load());
    hess[3][8] = _mm512_reduce_add_pd(h38.load());

    hess[4][4] = _mm512_reduce_add_pd(h44.load());
    hess[4][5] = _mm512_reduce_add_pd(h45.load());
    hess[4][6] = _mm512_reduce_add_pd(h46.load());
    hess[4][7] = _mm512_reduce_add_pd(h47.load());
    hess[4][8] = _mm512_reduce_add_pd(h48.load());

    hess[5][5] = _mm512_reduce_add_pd(h55.load());
    hess[5][6] = _mm512_reduce_add_pd(h56.load());
    hess[5][7] = _mm512_reduce_add_pd(h57.load());
    hess[5][8] = _mm512_reduce_add_pd(h58.load());

    hess[6][6] = _mm512_reduce_add_pd(h66.load());
    hess[6][7] = _mm512_reduce_add_pd(h67.load());
    hess[6][8] = _mm512_reduce_add_pd(h68.load());

    hess[7][7] = _mm512_reduce_add_pd(h77.load());
    hess[7][8] = _mm512_reduce_add_pd(h78.load());

    hess[8][8] = _mm512_reduce_add_pd(h88.load());

    for i in 0..9 {
        for j in i..9 {
            hess[j][i] = hess[i][j];
        }
    }

    (grad, hess)
}



#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_two_passes3(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [[f64; 9]; 9]) {
    const BLOCKS: usize = 32;

    let mut g = [Lane8::zero(); 9];

    // First row
    let mut h00 = Lane8::zero();
    let mut h01 = Lane8::zero();
    let mut h02 = Lane8::zero();
    let mut h03 = Lane8::zero();
    let mut h04 = Lane8::zero();
    let mut h05 = Lane8::zero();
    let mut h06 = Lane8::zero();
    let mut h07 = Lane8::zero();
    let mut h08 = Lane8::zero();
    // Second row
    let mut h11 = Lane8::zero();
    let mut h12 = Lane8::zero();
    let mut h13 = Lane8::zero();
    let mut h14 = Lane8::zero();
    let mut h15 = Lane8::zero();
    let mut h16 = Lane8::zero();
    let mut h17 = Lane8::zero();
    let mut h18 = Lane8::zero();
    // Third row
    let mut h22 = Lane8::zero();
    let mut h23 = Lane8::zero();
    let mut h24 = Lane8::zero();
    let mut h25 = Lane8::zero();
    let mut h26 = Lane8::zero();
    let mut h27 = Lane8::zero();
    let mut h28 = Lane8::zero();
    // Fourth row
    let mut h33 = Lane8::zero();
    let mut h34 = Lane8::zero();
    let mut h35 = Lane8::zero();
    let mut h36 = Lane8::zero();
    let mut h37 = Lane8::zero();
    let mut h38 = Lane8::zero();
    // Fifth row
    let mut h44 = Lane8::zero();
    let mut h45 = Lane8::zero();
    let mut h46 = Lane8::zero();
    let mut h47 = Lane8::zero();
    let mut h48 = Lane8::zero();
    // Sixth row
    let mut h55 = Lane8::zero();
    let mut h56 = Lane8::zero();
    let mut h57 = Lane8::zero();
    let mut h58 = Lane8::zero();
    // Seventh row
    let mut h66 = Lane8::zero();
    let mut h67 = Lane8::zero();
    let mut h68 = Lane8::zero();
    // Eigth row
    let mut h77 = Lane8::zero();
    let mut h78 = Lane8::zero();
    // Ninth row
    let mut h88 = Lane8::zero();

    // Register allocation:
    // - 9 for x
    // - 1 for one
    // For calculation of d: 1 for d + 9 columns + 9 gradient = 19
    // For each Hessian pass: 15 accumulators + at most 5 columns = 20

    // A block = [Lane8; 9]
    // This is one tile
    let mut scaled_column_buf = [[Lane8::zero(); 9]; BLOCKS];

    let (tiles, rem) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        {

            let one = _mm512_set1_pd(1.0);

            let xs = [
                _mm512_set1_pd(x[0]),
                _mm512_set1_pd(x[1]),
                _mm512_set1_pd(x[2]),
                _mm512_set1_pd(x[3]),
                _mm512_set1_pd(x[4]),
                _mm512_set1_pd(x[5]),
                _mm512_set1_pd(x[6]),
                _mm512_set1_pd(x[7]),
                _mm512_set1_pd(x[8]),
            ];

            let mut z = [_mm512_setzero_pd(); 9];

            for (buffer, block) in scaled_column_buf.iter_mut().zip(tile.iter()) {
                let mut c: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                // Calculate d
                // This computes a dot product between x and a row of p
                // TODO this could be manually unrolled a few times
                let mut d = _mm512_set1_pd(eps);
                for col in 0..9 {
                    d = _mm512_fmadd_pd(xs[col], c[col], d);
                }

                // TODO investigate reciprocal
                d = _mm512_div_pd(one, d);

                // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

                for col in 0..9 {
                    c[col] = _mm512_mul_pd(c[col], d);
                }

                for col in 0..9 {
                    buffer[col].store(c[col]);
                }

                for col in 0..9 {
                    z[col] = _mm512_fmadd_pd(c[col], d, z[col]);
                }
            }

            for col in 0..9 {
                z[col] = _mm512_add_pd(g[col].load(), z[col]);
            }

            for col in 0..9 {
                g[col].store(z[col]);
            }
        }


        // 24 accumulators + 4 = 28
        {
            // Row 0
            let mut z00 = h00.load();
            let mut z01 = h01.load();
            let mut z02 = h02.load();
            let mut z03 = h03.load();
            let mut z04 = h04.load();
            let mut z05 = h05.load();
            let mut z06 = h06.load();
            let mut z07 = h07.load();
            let mut z08 = h08.load();

            // Row 1
            let mut z11 = h11.load();
            let mut z12 = h12.load();
            let mut z13 = h13.load();
            let mut z14 = h14.load();
            let mut z15 = h15.load();
            let mut z16 = h16.load();
            let mut z17 = h17.load();
            let mut z18 = h18.load();

            // Row 2
            let mut z22 = h22.load();
            let mut z23 = h23.load();
            let mut z24 = h24.load();
            let mut z25 = h25.load();
            let mut z26 = h26.load();
            let mut z27 = h27.load();
            let mut z28 = h28.load();

            for block in scaled_column_buf.iter() {
                // 3 permanent + 1 temporary = 4 registers
                let c0 = block[0].load();
                let c1 = block[1].load();
                let c2 = block[2].load();

                z00 = _mm512_fmadd_pd(c0, c0, z00);

                z01 = _mm512_fmadd_pd(c0, c1, z01);
                z11 = _mm512_fmadd_pd(c1, c1, z11);

                z02 = _mm512_fmadd_pd(c0, c2, z02);
                z12 = _mm512_fmadd_pd(c1, c2, z12);
                z22 = _mm512_fmadd_pd(c2, c2, z22);

                let c3 = block[3].load();
                z03 = _mm512_fmadd_pd(c0, c3, z03);
                z13 = _mm512_fmadd_pd(c1, c3, z13);
                z23 = _mm512_fmadd_pd(c2, c3, z23);

                let c4 = block[4].load();
                z04 = _mm512_fmadd_pd(c0, c4, z04);
                z14 = _mm512_fmadd_pd(c1, c4, z14);
                z24 = _mm512_fmadd_pd(c2, c4, z24);

                let c5 = block[4].load();
                z05 = _mm512_fmadd_pd(c0, c5, z05);
                z15 = _mm512_fmadd_pd(c1, c5, z15);
                z25 = _mm512_fmadd_pd(c2, c5, z25);

                let c6 = block[4].load();
                z06 = _mm512_fmadd_pd(c0, c6, z06);
                z16 = _mm512_fmadd_pd(c1, c6, z16);
                z26 = _mm512_fmadd_pd(c2, c6, z26);

                let c7 = block[7].load();
                z07 = _mm512_fmadd_pd(c0, c7, z07);
                z17 = _mm512_fmadd_pd(c1, c7, z17);
                z27 = _mm512_fmadd_pd(c2, c7, z27);

                let c8 = block[8].load();
                z08 = _mm512_fmadd_pd(c0, c8, z08);
                z18 = _mm512_fmadd_pd(c1, c8, z18);
                z28 = _mm512_fmadd_pd(c2, c8, z28);

            }

            // Row 0
            h00.store(z00);
            h01.store(z01);
            h02.store(z02);
            h03.store(z03);
            h04.store(z04);
            h05.store(z05);
            h06.store(z06);
            h07.store(z07);
            h08.store(z08);

            // Row 1
            h11.store(z11);
            h12.store(z12);
            h13.store(z13);
            h14.store(z14);
            h15.store(z15);
            h16.store(z16);
            h17.store(z17);
            h18.store(z18);

            // Row 2
            h22.store(z22);
            h23.store(z23);
            h24.store(z24);
            h25.store(z25);
            h26.store(z26);
            h27.store(z27);
            h28.store(z28);
        }

        // 21 + 6 = 27
        {
            // Row 3
            let mut z33 = _mm512_setzero_pd();
            let mut z34 = _mm512_setzero_pd();
            let mut z35 = _mm512_setzero_pd();
            let mut z36 = _mm512_setzero_pd();
            let mut z37 = _mm512_setzero_pd();
            let mut z38 = _mm512_setzero_pd();

            // Row 4
            let mut z44 = _mm512_setzero_pd();
            let mut z45 = _mm512_setzero_pd();
            let mut z46 = _mm512_setzero_pd();
            let mut z47 = _mm512_setzero_pd();
            let mut z48 = _mm512_setzero_pd();

            // Row 5
            let mut z55 = _mm512_setzero_pd();
            let mut z56 = _mm512_setzero_pd();
            let mut z57 = _mm512_setzero_pd();
            let mut z58 = _mm512_setzero_pd();

            // Row 6
            let mut z66 = _mm512_setzero_pd();
            let mut z67 = _mm512_setzero_pd();
            let mut z68 = _mm512_setzero_pd();

            // Row 7
            let mut z77 = _mm512_setzero_pd();
            let mut z78 = _mm512_setzero_pd();

            // Row 8
            let mut z88 = _mm512_setzero_pd();

            for block in scaled_column_buf.iter() {
                // 6 permanent registers
                let c3 = block[4].load();
                let c4 = block[4].load();
                let c5 = block[5].load();
                let c6 = block[6].load();
                let c7 = block[7].load();
                let c8 = block[8].load();

                z33 = _mm512_fmadd_pd(c3, c3, z33);
                z34 = _mm512_fmadd_pd(c3, c4, z34);
                z35 = _mm512_fmadd_pd(c3, c5, z35);
                z36 = _mm512_fmadd_pd(c3, c6, z36);
                z37 = _mm512_fmadd_pd(c3, c7, z37);
                z38 = _mm512_fmadd_pd(c3, c8, z38);

                z44 = _mm512_fmadd_pd(c4, c4, z44);
                z45 = _mm512_fmadd_pd(c4, c5, z45);
                z46 = _mm512_fmadd_pd(c4, c6, z46);
                z47 = _mm512_fmadd_pd(c4, c7, z47);
                z48 = _mm512_fmadd_pd(c4, c8, z48);

                z55 = _mm512_fmadd_pd(c5, c5, z55);
                z56 = _mm512_fmadd_pd(c5, c6, z56);
                z57 = _mm512_fmadd_pd(c5, c7, z57);
                z58 = _mm512_fmadd_pd(c5, c8, z58);

                z66 = _mm512_fmadd_pd(c6, c6, z66);
                z67 = _mm512_fmadd_pd(c6, c7, z67);
                z68 = _mm512_fmadd_pd(c6, c8, z68);

                z77 = _mm512_fmadd_pd(c7, c7, z77);
                z78 = _mm512_fmadd_pd(c7, c8, z78);

                z88 = _mm512_fmadd_pd(c8, c8, z88);
            }

            z33 = _mm512_add_pd(h33.load(), z33);
            z34 = _mm512_add_pd(h34.load(), z34);
            z35 = _mm512_add_pd(h35.load(), z35);
            z36 = _mm512_add_pd(h36.load(), z36);
            z37 = _mm512_add_pd(h37.load(), z37);
            z38 = _mm512_add_pd(h38.load(), z38);

            z44 = _mm512_add_pd(h44.load(), z44);
            z45 = _mm512_add_pd(h45.load(), z45);
            z46 = _mm512_add_pd(h46.load(), z46);
            z47 = _mm512_add_pd(h47.load(), z47);
            z48 = _mm512_add_pd(h48.load(), z48);

            // Row 5
            z55 = _mm512_add_pd(h55.load(), z55);
            z56 = _mm512_add_pd(h56.load(), z56);
            z57 = _mm512_add_pd(h57.load(), z57);
            z58 = _mm512_add_pd(h58.load(), z58);

            // Row 6
            z66 = _mm512_add_pd(h66.load(), z66);
            z67 = _mm512_add_pd(h67.load(), z67);
            z68 = _mm512_add_pd(h68.load(), z68);

            // Row 7
            z77 = _mm512_add_pd(h77.load(), z77);
            z78 = _mm512_add_pd(h78.load(), z78);

            // Row 8
            z88 = _mm512_add_pd(h88.load(), z88);

            // Row 3
            h33.store(z33);
            h34.store(z34);
            h35.store(z35);
            h36.store(z36);
            h37.store(z37);
            h38.store(z38);

            // Row 4
            h44.store(z44);
            h45.store(z45);
            h46.store(z46);
            h47.store(z47);
            h48.store(z48);

            // Row 5
            h55.store(z55);
            h56.store(z56);
            h57.store(z57);
            h58.store(z58);

            // Row 6
            h66.store(z66);
            h67.store(z67);
            h68.store(z68);

            // Row 7
            h77.store(z77);
            h78.store(z78);

            // Row 8
            h88.store(z88);
        }
    }

    let mut grad = [0.0; 9];
    for col in 0..9 {
        grad[col] = _mm512_reduce_add_pd(g[col].load());
    }

    let mut hess = [[0.0; 9]; 9];

    // First row
    hess[0][0] = _mm512_reduce_add_pd(h00.load());
    hess[0][1] = _mm512_reduce_add_pd(h01.load());
    hess[0][2] = _mm512_reduce_add_pd(h02.load());
    hess[0][3] = _mm512_reduce_add_pd(h03.load());
    hess[0][4] = _mm512_reduce_add_pd(h04.load());
    hess[0][5] = _mm512_reduce_add_pd(h05.load());
    hess[0][6] = _mm512_reduce_add_pd(h06.load());
    hess[0][7] = _mm512_reduce_add_pd(h07.load());
    hess[0][8] = _mm512_reduce_add_pd(h08.load());

    // Second row
    hess[1][1] = _mm512_reduce_add_pd(h11.load());
    hess[1][2] = _mm512_reduce_add_pd(h12.load());
    hess[1][3] = _mm512_reduce_add_pd(h13.load());
    hess[1][4] = _mm512_reduce_add_pd(h14.load());
    hess[1][5] = _mm512_reduce_add_pd(h15.load());
    hess[1][6] = _mm512_reduce_add_pd(h16.load());
    hess[1][7] = _mm512_reduce_add_pd(h17.load());
    hess[1][8] = _mm512_reduce_add_pd(h18.load());

    hess[2][2] = _mm512_reduce_add_pd(h22.load());
    hess[2][3] = _mm512_reduce_add_pd(h23.load());
    hess[2][4] = _mm512_reduce_add_pd(h24.load());
    hess[2][5] = _mm512_reduce_add_pd(h25.load());
    hess[2][6] = _mm512_reduce_add_pd(h26.load());
    hess[2][7] = _mm512_reduce_add_pd(h27.load());
    hess[2][8] = _mm512_reduce_add_pd(h28.load());

    hess[3][3] = _mm512_reduce_add_pd(h33.load());
    hess[3][4] = _mm512_reduce_add_pd(h34.load());
    hess[3][5] = _mm512_reduce_add_pd(h35.load());
    hess[3][6] = _mm512_reduce_add_pd(h36.load());
    hess[3][7] = _mm512_reduce_add_pd(h37.load());
    hess[3][8] = _mm512_reduce_add_pd(h38.load());

    hess[4][4] = _mm512_reduce_add_pd(h44.load());
    hess[4][5] = _mm512_reduce_add_pd(h45.load());
    hess[4][6] = _mm512_reduce_add_pd(h46.load());
    hess[4][7] = _mm512_reduce_add_pd(h47.load());
    hess[4][8] = _mm512_reduce_add_pd(h48.load());

    hess[5][5] = _mm512_reduce_add_pd(h55.load());
    hess[5][6] = _mm512_reduce_add_pd(h56.load());
    hess[5][7] = _mm512_reduce_add_pd(h57.load());
    hess[5][8] = _mm512_reduce_add_pd(h58.load());

    hess[6][6] = _mm512_reduce_add_pd(h66.load());
    hess[6][7] = _mm512_reduce_add_pd(h67.load());
    hess[6][8] = _mm512_reduce_add_pd(h68.load());

    hess[7][7] = _mm512_reduce_add_pd(h77.load());
    hess[7][8] = _mm512_reduce_add_pd(h78.load());

    hess[8][8] = _mm512_reduce_add_pd(h88.load());

    for i in 0..9 {
        for j in i..9 {
            hess[j][i] = hess[i][j];
        }
    }

    (grad, hess)
}