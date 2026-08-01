use crate::{
    algebra::dot, blockbuffer::Block, buffer::Lane8, util::{Matrix9, Matrix9xN, Vector9},
};

use core::arch::x86_64::*;

pub fn compute_hess(p_mat_t: &Matrix9xN<f64>, x: &Vector9<f64>, eps: f64) -> Matrix9<f64> {
    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    let (p, _) = p_mat_t.as_slice().as_chunks::<9>();

    let ptd2p = compute_pt_d2_p_scalar(p, &x0, eps);
    // let ptd2p = compute_pt_d2_p_avx512_tiled_three_passes_kahan(blocks, remainder, x, eps)

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
            let mut z00 = _mm512_setzero_pd();
            let mut z01 = _mm512_setzero_pd();
            let mut z02 = _mm512_setzero_pd();
            let mut z03 = _mm512_setzero_pd();
            let mut z04 = _mm512_setzero_pd();
            let mut z05 = _mm512_setzero_pd();
            let mut z06 = _mm512_setzero_pd();
            let mut z07 = _mm512_setzero_pd();
            let mut z08 = _mm512_setzero_pd();

            // Row 1
            let mut z11 = _mm512_setzero_pd();
            let mut z12 = _mm512_setzero_pd();
            let mut z13 = _mm512_setzero_pd();
            let mut z14 = _mm512_setzero_pd();
            let mut z15 = _mm512_setzero_pd();
            let mut z16 = _mm512_setzero_pd();

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

                let c5 = block[5].load();
                z05 = _mm512_fmadd_pd(c0, c5, z05);
                z15 = _mm512_fmadd_pd(c1, c5, z15);

                let c6 = block[6].load();
                z06 = _mm512_fmadd_pd(c0, c6, z06);
                z16 = _mm512_fmadd_pd(c1, c6, z16);

                let c7 = block[7].load();
                z07 = _mm512_fmadd_pd(c0, c7, z07);

                let c8 = block[8].load();
                z08 = _mm512_fmadd_pd(c0, c8, z08);
            }

            z00 = _mm512_add_pd(h00.load(), z00);
            z01 = _mm512_add_pd(h01.load(), z01);
            z02 = _mm512_add_pd(h02.load(), z02);
            z03 = _mm512_add_pd(h03.load(), z03);
            z04 = _mm512_add_pd(h04.load(), z04);
            z05 = _mm512_add_pd(h05.load(), z05);
            z06 = _mm512_add_pd(h06.load(), z06);
            z07 = _mm512_add_pd(h07.load(), z07);
            z08 = _mm512_add_pd(h08.load(), z08);

            z11 = _mm512_add_pd(h11.load(), z11);
            z12 = _mm512_add_pd(h12.load(), z12);
            z13 = _mm512_add_pd(h13.load(), z13);
            z14 = _mm512_add_pd(h14.load(), z14);
            z15 = _mm512_add_pd(h15.load(), z15);
            z16 = _mm512_add_pd(h16.load(), z16);

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
            let mut z17 = _mm512_setzero_pd();
            let mut z18 = _mm512_setzero_pd();

            // Row 2
            let mut z22 = _mm512_setzero_pd();
            let mut z23 = _mm512_setzero_pd();
            let mut z24 = _mm512_setzero_pd();
            let mut z25 = _mm512_setzero_pd();
            let mut z26 = _mm512_setzero_pd();
            let mut z27 = _mm512_setzero_pd();
            let mut z28 = _mm512_setzero_pd();

            // Row 3
            let mut z33 = _mm512_setzero_pd();
            let mut z34 = _mm512_setzero_pd();
            let mut z35 = _mm512_setzero_pd();
            let mut z36 = _mm512_setzero_pd();
            let mut z37 = _mm512_setzero_pd();
            let mut z38 = _mm512_setzero_pd();

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
            }

            z17 = _mm512_add_pd(h17.load(), z17);
            z18 = _mm512_add_pd(h18.load(), z18);

            z22 = _mm512_add_pd(h22.load(), z22);
            z23 = _mm512_add_pd(h23.load(), z23);
            z24 = _mm512_add_pd(h24.load(), z24);
            z25 = _mm512_add_pd(h25.load(), z25);
            z26 = _mm512_add_pd(h26.load(), z26);
            z27 = _mm512_add_pd(h27.load(), z27);
            z28 = _mm512_add_pd(h28.load(), z28);

            z33 = _mm512_add_pd(h33.load(), z33);
            z34 = _mm512_add_pd(h34.load(), z34);
            z35 = _mm512_add_pd(h35.load(), z35);
            z36 = _mm512_add_pd(h36.load(), z36);
            z37 = _mm512_add_pd(h37.load(), z37);
            z38 = _mm512_add_pd(h38.load(), z38);

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
    g: &mut [Lane8; 9],
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
            let mut z00 = _mm512_setzero_pd();
            let mut z01 = _mm512_setzero_pd();
            let mut z02 = _mm512_setzero_pd();
            let mut z03 = _mm512_setzero_pd();
            let mut z04 = _mm512_setzero_pd();
            let mut z05 = _mm512_setzero_pd();
            let mut z06 = _mm512_setzero_pd();
            let mut z07 = _mm512_setzero_pd();
            let mut z08 = _mm512_setzero_pd();

            // Row 1
            let mut z11 = _mm512_setzero_pd();
            let mut z12 = _mm512_setzero_pd();
            let mut z13 = _mm512_setzero_pd();
            let mut z14 = _mm512_setzero_pd();
            let mut z15 = _mm512_setzero_pd();
            let mut z16 = _mm512_setzero_pd();
            let mut z17 = _mm512_setzero_pd();
            let mut z18 = _mm512_setzero_pd();

            // Row 2
            let mut z22 = _mm512_setzero_pd();
            let mut z23 = _mm512_setzero_pd();
            let mut z24 = _mm512_setzero_pd();
            let mut z25 = _mm512_setzero_pd();
            let mut z26 = _mm512_setzero_pd();
            let mut z27 = _mm512_setzero_pd();
            let mut z28 = _mm512_setzero_pd();

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

                let c5 = block[5].load();
                z05 = _mm512_fmadd_pd(c0, c5, z05);
                z15 = _mm512_fmadd_pd(c1, c5, z15);
                z25 = _mm512_fmadd_pd(c2, c5, z25);

                let c6 = block[6].load();
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

            z00 = _mm512_add_pd(h00.load(), z00);
            z01 = _mm512_add_pd(h01.load(), z01);
            z02 = _mm512_add_pd(h02.load(), z02);
            z03 = _mm512_add_pd(h03.load(), z03);
            z04 = _mm512_add_pd(h04.load(), z04);
            z05 = _mm512_add_pd(h05.load(), z05);
            z06 = _mm512_add_pd(h06.load(), z06);
            z07 = _mm512_add_pd(h07.load(), z07);
            z08 = _mm512_add_pd(h08.load(), z08);

            z11 = _mm512_add_pd(h11.load(), z11);
            z12 = _mm512_add_pd(h12.load(), z12);
            z13 = _mm512_add_pd(h13.load(), z13);
            z14 = _mm512_add_pd(h14.load(), z14);
            z15 = _mm512_add_pd(h15.load(), z15);
            z16 = _mm512_add_pd(h16.load(), z16);
            z17 = _mm512_add_pd(h17.load(), z17);
            z18 = _mm512_add_pd(h18.load(), z18);

            z22 = _mm512_add_pd(h22.load(), z22);
            z23 = _mm512_add_pd(h23.load(), z23);
            z24 = _mm512_add_pd(h24.load(), z24);
            z25 = _mm512_add_pd(h25.load(), z25);
            z26 = _mm512_add_pd(h26.load(), z26);
            z27 = _mm512_add_pd(h27.load(), z27);
            z28 = _mm512_add_pd(h28.load(), z28);

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
                let c3 = block[3].load();
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

// The current winner!
#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_three_passes_kahan(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [[f64; 9]; 9]) {
    const BLOCKS: usize = 32;

    let mut g = [Lane8::zero(); 9];
    let mut gc = [Lane8::zero(); 9];

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

    let mut c00 = Lane8::zero();
    let mut c01 = Lane8::zero();
    let mut c02 = Lane8::zero();
    let mut c03 = Lane8::zero();
    let mut c04 = Lane8::zero();
    let mut c05 = Lane8::zero();
    let mut c06 = Lane8::zero();
    let mut c07 = Lane8::zero();
    let mut c08 = Lane8::zero();
    // Second row
    let mut c11 = Lane8::zero();
    let mut c12 = Lane8::zero();
    let mut c13 = Lane8::zero();
    let mut c14 = Lane8::zero();
    let mut c15 = Lane8::zero();
    let mut c16 = Lane8::zero();
    let mut c17 = Lane8::zero();
    let mut c18 = Lane8::zero();
    // Tcird row
    let mut c22 = Lane8::zero();
    let mut c23 = Lane8::zero();
    let mut c24 = Lane8::zero();
    let mut c25 = Lane8::zero();
    let mut c26 = Lane8::zero();
    let mut c27 = Lane8::zero();
    let mut c28 = Lane8::zero();
    // Fourtc row
    let mut c33 = Lane8::zero();
    let mut c34 = Lane8::zero();
    let mut c35 = Lane8::zero();
    let mut c36 = Lane8::zero();
    let mut c37 = Lane8::zero();
    let mut c38 = Lane8::zero();
    // Fiftc row
    let mut c44 = Lane8::zero();
    let mut c45 = Lane8::zero();
    let mut c46 = Lane8::zero();
    let mut c47 = Lane8::zero();
    let mut c48 = Lane8::zero();
    // Sixtc row
    let mut c55 = Lane8::zero();
    let mut c56 = Lane8::zero();
    let mut c57 = Lane8::zero();
    let mut c58 = Lane8::zero();
    // Seventc row
    let mut c66 = Lane8::zero();
    let mut c67 = Lane8::zero();
    let mut c68 = Lane8::zero();
    // Eigtc row
    let mut c77 = Lane8::zero();
    let mut c78 = Lane8::zero();
    // Nintc row
    let mut c88 = Lane8::zero();

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

            // Gradient kahan
            for col in 0..9 {
                let mut sum = g[col].load();
                z[col] = _mm512_sub_pd(z[col], gc[col].load());
                let t = _mm512_add_pd(sum, z[col]);
                sum = _mm512_sub_pd(t, sum);
                sum = _mm512_sub_pd(sum, z[col]);

                g[col].store(t);
                gc[col].store(sum);
            }
        }

        {
            // Row 0
            let mut z00 = _mm512_setzero_pd();
            let mut z01 = _mm512_setzero_pd();
            let mut z02 = _mm512_setzero_pd();
            let mut z03 = _mm512_setzero_pd();
            let mut z04 = _mm512_setzero_pd();
            let mut z05 = _mm512_setzero_pd();
            let mut z06 = _mm512_setzero_pd();
            let mut z07 = _mm512_setzero_pd();
            let mut z08 = _mm512_setzero_pd();

            // Row 1
            let mut z11 = _mm512_setzero_pd();
            let mut z12 = _mm512_setzero_pd();
            let mut z13 = _mm512_setzero_pd();
            let mut z14 = _mm512_setzero_pd();
            let mut z15 = _mm512_setzero_pd();
            let mut z16 = _mm512_setzero_pd();

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

                let c5 = block[5].load();
                z05 = _mm512_fmadd_pd(c0, c5, z05);
                z15 = _mm512_fmadd_pd(c1, c5, z15);

                let c6 = block[6].load();
                z06 = _mm512_fmadd_pd(c0, c6, z06);
                z16 = _mm512_fmadd_pd(c1, c6, z16);

                let c7 = block[7].load();
                z07 = _mm512_fmadd_pd(c0, c7, z07);

                let c8 = block[8].load();
                z08 = _mm512_fmadd_pd(c0, c8, z08);
            }

            // Row 0
            let s00 = h00.load();
            let y00 = _mm512_sub_pd(z00, c00.load());
            let t00 = _mm512_add_pd(s00, y00);
            let mut tmp00 = _mm512_sub_pd(t00, s00);
            tmp00 = _mm512_sub_pd(tmp00, y00);
            c00.store(tmp00);
            h00.store(t00);

            let s01 = h01.load();
            let y01 = _mm512_sub_pd(z01, c01.load());
            let t01 = _mm512_add_pd(s01, y01);
            let mut tmp01 = _mm512_sub_pd(t01, s01);
            tmp01 = _mm512_sub_pd(tmp01, y01);
            c01.store(tmp01);
            h01.store(t01);

            let s02 = h02.load();
            let y02 = _mm512_sub_pd(z02, c02.load());
            let t02 = _mm512_add_pd(s02, y02);
            let mut tmp02 = _mm512_sub_pd(t02, s02);
            tmp02 = _mm512_sub_pd(tmp02, y02);
            c02.store(tmp02);
            h02.store(t02);

            let s03 = h03.load();
            let y03 = _mm512_sub_pd(z03, c03.load());
            let t03 = _mm512_add_pd(s03, y03);
            let mut tmp03 = _mm512_sub_pd(t03, s03);
            tmp03 = _mm512_sub_pd(tmp03, y03);
            c03.store(tmp03);
            h03.store(t03);

            let s04 = h04.load();
            let y04 = _mm512_sub_pd(z04, c04.load());
            let t04 = _mm512_add_pd(s04, y04);
            let mut tmp04 = _mm512_sub_pd(t04, s04);
            tmp04 = _mm512_sub_pd(tmp04, y04);
            c04.store(tmp04);
            h04.store(t04);

            let s05 = h05.load();
            let y05 = _mm512_sub_pd(z05, c05.load());
            let t05 = _mm512_add_pd(s05, y05);
            let mut tmp05 = _mm512_sub_pd(t05, s05);
            tmp05 = _mm512_sub_pd(tmp05, y05);
            c05.store(tmp05);
            h05.store(t05);

            let s06 = h06.load();
            let y06 = _mm512_sub_pd(z06, c06.load());
            let t06 = _mm512_add_pd(s06, y06);
            let mut tmp06 = _mm512_sub_pd(t06, s06);
            tmp06 = _mm512_sub_pd(tmp06, y06);
            c06.store(tmp06);
            h06.store(t06);

            let s07 = h07.load();
            let y07 = _mm512_sub_pd(z07, c07.load());
            let t07 = _mm512_add_pd(s07, y07);
            let mut tmp07 = _mm512_sub_pd(t07, s07);
            tmp07 = _mm512_sub_pd(tmp07, y07);
            c07.store(tmp07);
            h07.store(t07);

            let s08 = h08.load();
            let y08 = _mm512_sub_pd(z08, c08.load());
            let t08 = _mm512_add_pd(s08, y08);
            let mut tmp08 = _mm512_sub_pd(t08, s08);
            tmp08 = _mm512_sub_pd(tmp08, y08);
            c08.store(tmp08);
            h08.store(t08);

            // Row 1
            let s11 = h11.load();
            let y11 = _mm512_sub_pd(z11, c11.load());
            let t11 = _mm512_add_pd(s11, y11);
            let mut tmp11 = _mm512_sub_pd(t11, s11);
            tmp11 = _mm512_sub_pd(tmp11, y11);
            c11.store(tmp11);
            h11.store(t11);

            let s12 = h12.load();
            let y12 = _mm512_sub_pd(z12, c12.load());
            let t12 = _mm512_add_pd(s12, y12);
            let mut tmp12 = _mm512_sub_pd(t12, s12);
            tmp12 = _mm512_sub_pd(tmp12, y12);
            c12.store(tmp12);
            h12.store(t12);

            let s13 = h13.load();
            let y13 = _mm512_sub_pd(z13, c13.load());
            let t13 = _mm512_add_pd(s13, y13);
            let mut tmp13 = _mm512_sub_pd(t13, s13);
            tmp13 = _mm512_sub_pd(tmp13, y13);
            c13.store(tmp13);
            h13.store(t13);

            let s14 = h14.load();
            let y14 = _mm512_sub_pd(z14, c14.load());
            let t14 = _mm512_add_pd(s14, y14);
            let mut tmp14 = _mm512_sub_pd(t14, s14);
            tmp14 = _mm512_sub_pd(tmp14, y14);
            c14.store(tmp14);
            h14.store(t14);

            let s15 = h15.load();
            let y15 = _mm512_sub_pd(z15, c15.load());
            let t15 = _mm512_add_pd(s15, y15);
            let mut tmp15 = _mm512_sub_pd(t15, s15);
            tmp15 = _mm512_sub_pd(tmp15, y15);
            c15.store(tmp15);
            h15.store(t15);

            let s16 = h16.load();
            let y16 = _mm512_sub_pd(z16, c16.load());
            let t16 = _mm512_add_pd(s16, y16);
            let mut tmp16 = _mm512_sub_pd(t16, s16);
            tmp16 = _mm512_sub_pd(tmp16, y16);
            c16.store(tmp16);
            h16.store(t16);
        }

        {
            let mut z17 = _mm512_setzero_pd();
            let mut z18 = _mm512_setzero_pd();

            // Row 2
            let mut z22 = _mm512_setzero_pd();
            let mut z23 = _mm512_setzero_pd();
            let mut z24 = _mm512_setzero_pd();
            let mut z25 = _mm512_setzero_pd();
            let mut z26 = _mm512_setzero_pd();
            let mut z27 = _mm512_setzero_pd();
            let mut z28 = _mm512_setzero_pd();

            // Row 3
            let mut z33 = _mm512_setzero_pd();
            let mut z34 = _mm512_setzero_pd();
            let mut z35 = _mm512_setzero_pd();
            let mut z36 = _mm512_setzero_pd();
            let mut z37 = _mm512_setzero_pd();
            let mut z38 = _mm512_setzero_pd();

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
            }

            // Row 1 (continued)
            let s17 = h17.load();
            let y17 = _mm512_sub_pd(z17, c17.load());
            let t17 = _mm512_add_pd(s17, y17);
            let mut tmp17 = _mm512_sub_pd(t17, s17);
            tmp17 = _mm512_sub_pd(tmp17, y17);
            c17.store(tmp17);
            h17.store(t17);

            let s18 = h18.load();
            let y18 = _mm512_sub_pd(z18, c18.load());
            let t18 = _mm512_add_pd(s18, y18);
            let mut tmp18 = _mm512_sub_pd(t18, s18);
            tmp18 = _mm512_sub_pd(tmp18, y18);
            c18.store(tmp18);
            h18.store(t18);

            // Row 2
            let s22 = h22.load();
            let y22 = _mm512_sub_pd(z22, c22.load());
            let t22 = _mm512_add_pd(s22, y22);
            let mut tmp22 = _mm512_sub_pd(t22, s22);
            tmp22 = _mm512_sub_pd(tmp22, y22);
            c22.store(tmp22);
            h22.store(t22);

            let s23 = h23.load();
            let y23 = _mm512_sub_pd(z23, c23.load());
            let t23 = _mm512_add_pd(s23, y23);
            let mut tmp23 = _mm512_sub_pd(t23, s23);
            tmp23 = _mm512_sub_pd(tmp23, y23);
            c23.store(tmp23);
            h23.store(t23);

            let s24 = h24.load();
            let y24 = _mm512_sub_pd(z24, c24.load());
            let t24 = _mm512_add_pd(s24, y24);
            let mut tmp24 = _mm512_sub_pd(t24, s24);
            tmp24 = _mm512_sub_pd(tmp24, y24);
            c24.store(tmp24);
            h24.store(t24);

            let s25 = h25.load();
            let y25 = _mm512_sub_pd(z25, c25.load());
            let t25 = _mm512_add_pd(s25, y25);
            let mut tmp25 = _mm512_sub_pd(t25, s25);
            tmp25 = _mm512_sub_pd(tmp25, y25);
            c25.store(tmp25);
            h25.store(t25);

            let s26 = h26.load();
            let y26 = _mm512_sub_pd(z26, c26.load());
            let t26 = _mm512_add_pd(s26, y26);
            let mut tmp26 = _mm512_sub_pd(t26, s26);
            tmp26 = _mm512_sub_pd(tmp26, y26);
            c26.store(tmp26);
            h26.store(t26);

            let s27 = h27.load();
            let y27 = _mm512_sub_pd(z27, c27.load());
            let t27 = _mm512_add_pd(s27, y27);
            let mut tmp27 = _mm512_sub_pd(t27, s27);
            tmp27 = _mm512_sub_pd(tmp27, y27);
            c27.store(tmp27);
            h27.store(t27);

            let s28 = h28.load();
            let y28 = _mm512_sub_pd(z28, c28.load());
            let t28 = _mm512_add_pd(s28, y28);
            let mut tmp28 = _mm512_sub_pd(t28, s28);
            tmp28 = _mm512_sub_pd(tmp28, y28);
            c28.store(tmp28);
            h28.store(t28);

            // Row 3
            let s33 = h33.load();
            let y33 = _mm512_sub_pd(z33, c33.load());
            let t33 = _mm512_add_pd(s33, y33);
            let mut tmp33 = _mm512_sub_pd(t33, s33);
            tmp33 = _mm512_sub_pd(tmp33, y33);
            c33.store(tmp33);
            h33.store(t33);

            let s34 = h34.load();
            let y34 = _mm512_sub_pd(z34, c34.load());
            let t34 = _mm512_add_pd(s34, y34);
            let mut tmp34 = _mm512_sub_pd(t34, s34);
            tmp34 = _mm512_sub_pd(tmp34, y34);
            c34.store(tmp34);
            h34.store(t34);

            let s35 = h35.load();
            let y35 = _mm512_sub_pd(z35, c35.load());
            let t35 = _mm512_add_pd(s35, y35);
            let mut tmp35 = _mm512_sub_pd(t35, s35);
            tmp35 = _mm512_sub_pd(tmp35, y35);
            c35.store(tmp35);
            h35.store(t35);

            let s36 = h36.load();
            let y36 = _mm512_sub_pd(z36, c36.load());
            let t36 = _mm512_add_pd(s36, y36);
            let mut tmp36 = _mm512_sub_pd(t36, s36);
            tmp36 = _mm512_sub_pd(tmp36, y36);
            c36.store(tmp36);
            h36.store(t36);

            let s37 = h37.load();
            let y37 = _mm512_sub_pd(z37, c37.load());
            let t37 = _mm512_add_pd(s37, y37);
            let mut tmp37 = _mm512_sub_pd(t37, s37);
            tmp37 = _mm512_sub_pd(tmp37, y37);
            c37.store(tmp37);
            h37.store(t37);

            let s38 = h38.load();
            let y38 = _mm512_sub_pd(z38, c38.load());
            let t38 = _mm512_add_pd(s38, y38);
            let mut tmp38 = _mm512_sub_pd(t38, s38);
            tmp38 = _mm512_sub_pd(tmp38, y38);
            c38.store(tmp38);
            h38.store(t38);
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
            // Row 4
            let s44 = h44.load();
            let y44 = _mm512_sub_pd(z44, c44.load());
            let t44 = _mm512_add_pd(s44, y44);
            let mut tmp44 = _mm512_sub_pd(t44, s44);
            tmp44 = _mm512_sub_pd(tmp44, y44);
            c44.store(tmp44);
            h44.store(t44);

            let s45 = h45.load();
            let y45 = _mm512_sub_pd(z45, c45.load());
            let t45 = _mm512_add_pd(s45, y45);
            let mut tmp45 = _mm512_sub_pd(t45, s45);
            tmp45 = _mm512_sub_pd(tmp45, y45);
            c45.store(tmp45);
            h45.store(t45);

            let s46 = h46.load();
            let y46 = _mm512_sub_pd(z46, c46.load());
            let t46 = _mm512_add_pd(s46, y46);
            let mut tmp46 = _mm512_sub_pd(t46, s46);
            tmp46 = _mm512_sub_pd(tmp46, y46);
            c46.store(tmp46);
            h46.store(t46);

            let s47 = h47.load();
            let y47 = _mm512_sub_pd(z47, c47.load());
            let t47 = _mm512_add_pd(s47, y47);
            let mut tmp47 = _mm512_sub_pd(t47, s47);
            tmp47 = _mm512_sub_pd(tmp47, y47);
            c47.store(tmp47);
            h47.store(t47);

            let s48 = h48.load();
            let y48 = _mm512_sub_pd(z48, c48.load());
            let t48 = _mm512_add_pd(s48, y48);
            let mut tmp48 = _mm512_sub_pd(t48, s48);
            tmp48 = _mm512_sub_pd(tmp48, y48);
            c48.store(tmp48);
            h48.store(t48);

            // Row 5
            let s55 = h55.load();
            let y55 = _mm512_sub_pd(z55, c55.load());
            let t55 = _mm512_add_pd(s55, y55);
            let mut tmp55 = _mm512_sub_pd(t55, s55);
            tmp55 = _mm512_sub_pd(tmp55, y55);
            c55.store(tmp55);
            h55.store(t55);

            let s56 = h56.load();
            let y56 = _mm512_sub_pd(z56, c56.load());
            let t56 = _mm512_add_pd(s56, y56);
            let mut tmp56 = _mm512_sub_pd(t56, s56);
            tmp56 = _mm512_sub_pd(tmp56, y56);
            c56.store(tmp56);
            h56.store(t56);

            let s57 = h57.load();
            let y57 = _mm512_sub_pd(z57, c57.load());
            let t57 = _mm512_add_pd(s57, y57);
            let mut tmp57 = _mm512_sub_pd(t57, s57);
            tmp57 = _mm512_sub_pd(tmp57, y57);
            c57.store(tmp57);
            h57.store(t57);

            let s58 = h58.load();
            let y58 = _mm512_sub_pd(z58, c58.load());
            let t58 = _mm512_add_pd(s58, y58);
            let mut tmp58 = _mm512_sub_pd(t58, s58);
            tmp58 = _mm512_sub_pd(tmp58, y58);
            c58.store(tmp58);
            h58.store(t58);

            // Row 6
            let s66 = h66.load();
            let y66 = _mm512_sub_pd(z66, c66.load());
            let t66 = _mm512_add_pd(s66, y66);
            let mut tmp66 = _mm512_sub_pd(t66, s66);
            tmp66 = _mm512_sub_pd(tmp66, y66);
            c66.store(tmp66);
            h66.store(t66);

            let s67 = h67.load();
            let y67 = _mm512_sub_pd(z67, c67.load());
            let t67 = _mm512_add_pd(s67, y67);
            let mut tmp67 = _mm512_sub_pd(t67, s67);
            tmp67 = _mm512_sub_pd(tmp67, y67);
            c67.store(tmp67);
            h67.store(t67);

            let s68 = h68.load();
            let y68 = _mm512_sub_pd(z68, c68.load());
            let t68 = _mm512_add_pd(s68, y68);
            let mut tmp68 = _mm512_sub_pd(t68, s68);
            tmp68 = _mm512_sub_pd(tmp68, y68);
            c68.store(tmp68);
            h68.store(t68);

            // Row 7
            let s77 = h77.load();
            let y77 = _mm512_sub_pd(z77, c77.load());
            let t77 = _mm512_add_pd(s77, y77);
            let mut tmp77 = _mm512_sub_pd(t77, s77);
            tmp77 = _mm512_sub_pd(tmp77, y77);
            c77.store(tmp77);
            h77.store(t77);

            let s78 = h78.load();
            let y78 = _mm512_sub_pd(z78, c78.load());
            let t78 = _mm512_add_pd(s78, y78);
            let mut tmp78 = _mm512_sub_pd(t78, s78);
            tmp78 = _mm512_sub_pd(tmp78, y78);
            c78.store(tmp78);
            h78.store(t78);

            // Row 8
            let s88 = h88.load();
            let y88 = _mm512_sub_pd(z88, c88.load());
            let t88 = _mm512_add_pd(s88, y88);
            let mut tmp88 = _mm512_sub_pd(t88, s88);
            tmp88 = _mm512_sub_pd(tmp88, y88);
            c88.store(tmp88);
            h88.store(t88);
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
pub fn compute_pt_d2_p_avx512_tiled_two_passes_kahan(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [[f64; 9]; 9]) {
    const BLOCKS: usize = 32;

    let mut g = [Lane8::zero(); 9];
    let mut gc = [Lane8::zero(); 9];

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

    let mut c00 = Lane8::zero();
    let mut c01 = Lane8::zero();
    let mut c02 = Lane8::zero();
    let mut c03 = Lane8::zero();
    let mut c04 = Lane8::zero();
    let mut c05 = Lane8::zero();
    let mut c06 = Lane8::zero();
    let mut c07 = Lane8::zero();
    let mut c08 = Lane8::zero();
    // Second row
    let mut c11 = Lane8::zero();
    let mut c12 = Lane8::zero();
    let mut c13 = Lane8::zero();
    let mut c14 = Lane8::zero();
    let mut c15 = Lane8::zero();
    let mut c16 = Lane8::zero();
    let mut c17 = Lane8::zero();
    let mut c18 = Lane8::zero();
    // Tcird row
    let mut c22 = Lane8::zero();
    let mut c23 = Lane8::zero();
    let mut c24 = Lane8::zero();
    let mut c25 = Lane8::zero();
    let mut c26 = Lane8::zero();
    let mut c27 = Lane8::zero();
    let mut c28 = Lane8::zero();
    // Fourtc row
    let mut c33 = Lane8::zero();
    let mut c34 = Lane8::zero();
    let mut c35 = Lane8::zero();
    let mut c36 = Lane8::zero();
    let mut c37 = Lane8::zero();
    let mut c38 = Lane8::zero();
    // Fiftc row
    let mut c44 = Lane8::zero();
    let mut c45 = Lane8::zero();
    let mut c46 = Lane8::zero();
    let mut c47 = Lane8::zero();
    let mut c48 = Lane8::zero();
    // Sixtc row
    let mut c55 = Lane8::zero();
    let mut c56 = Lane8::zero();
    let mut c57 = Lane8::zero();
    let mut c58 = Lane8::zero();
    // Seventc row
    let mut c66 = Lane8::zero();
    let mut c67 = Lane8::zero();
    let mut c68 = Lane8::zero();
    // Eigtc row
    let mut c77 = Lane8::zero();
    let mut c78 = Lane8::zero();
    // Nintc row
    let mut c88 = Lane8::zero();

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

            // Gradient kahan
            for col in 0..9 {
                let mut sum = g[col].load();
                z[col] = _mm512_sub_pd(z[col], gc[col].load());
                let t = _mm512_add_pd(sum, z[col]);
                sum = _mm512_sub_pd(t, sum);
                sum = _mm512_sub_pd(sum, z[col]);

                g[col].store(t);
                gc[col].store(sum);
            }

            // for col in 0..9 {
            //     g[col].store(z[col]);
            // }
        }

        {
            // Row 0
            let mut z00 = _mm512_setzero_pd();
            let mut z01 = _mm512_setzero_pd();
            let mut z02 = _mm512_setzero_pd();
            let mut z03 = _mm512_setzero_pd();
            let mut z04 = _mm512_setzero_pd();
            let mut z05 = _mm512_setzero_pd();
            let mut z06 = _mm512_setzero_pd();
            let mut z07 = _mm512_setzero_pd();
            let mut z08 = _mm512_setzero_pd();

            // Row 1
            let mut z11 = _mm512_setzero_pd();
            let mut z12 = _mm512_setzero_pd();
            let mut z13 = _mm512_setzero_pd();
            let mut z14 = _mm512_setzero_pd();
            let mut z15 = _mm512_setzero_pd();
            let mut z16 = _mm512_setzero_pd();
            let mut z17 = _mm512_setzero_pd();
            let mut z18 = _mm512_setzero_pd();

            // Row 2
            let mut z22 = _mm512_setzero_pd();
            let mut z23 = _mm512_setzero_pd();
            let mut z24 = _mm512_setzero_pd();
            let mut z25 = _mm512_setzero_pd();
            let mut z26 = _mm512_setzero_pd();
            let mut z27 = _mm512_setzero_pd();
            let mut z28 = _mm512_setzero_pd();

            for block in scaled_column_buf.iter() {
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

                let c5 = block[5].load();
                z05 = _mm512_fmadd_pd(c0, c5, z05);
                z15 = _mm512_fmadd_pd(c1, c5, z15);
                z25 = _mm512_fmadd_pd(c2, c5, z25);

                let c6 = block[6].load();
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
            let s00 = h00.load();
            let y00 = _mm512_sub_pd(z00, c00.load());
            let t00 = _mm512_add_pd(s00, y00);
            let mut tmp00 = _mm512_sub_pd(t00, s00);
            tmp00 = _mm512_sub_pd(tmp00, y00);
            c00.store(tmp00);
            h00.store(t00);

            let s01 = h01.load();
            let y01 = _mm512_sub_pd(z01, c01.load());
            let t01 = _mm512_add_pd(s01, y01);
            let mut tmp01 = _mm512_sub_pd(t01, s01);
            tmp01 = _mm512_sub_pd(tmp01, y01);
            c01.store(tmp01);
            h01.store(t01);

            let s02 = h02.load();
            let y02 = _mm512_sub_pd(z02, c02.load());
            let t02 = _mm512_add_pd(s02, y02);
            let mut tmp02 = _mm512_sub_pd(t02, s02);
            tmp02 = _mm512_sub_pd(tmp02, y02);
            c02.store(tmp02);
            h02.store(t02);

            let s03 = h03.load();
            let y03 = _mm512_sub_pd(z03, c03.load());
            let t03 = _mm512_add_pd(s03, y03);
            let mut tmp03 = _mm512_sub_pd(t03, s03);
            tmp03 = _mm512_sub_pd(tmp03, y03);
            c03.store(tmp03);
            h03.store(t03);

            let s04 = h04.load();
            let y04 = _mm512_sub_pd(z04, c04.load());
            let t04 = _mm512_add_pd(s04, y04);
            let mut tmp04 = _mm512_sub_pd(t04, s04);
            tmp04 = _mm512_sub_pd(tmp04, y04);
            c04.store(tmp04);
            h04.store(t04);

            let s05 = h05.load();
            let y05 = _mm512_sub_pd(z05, c05.load());
            let t05 = _mm512_add_pd(s05, y05);
            let mut tmp05 = _mm512_sub_pd(t05, s05);
            tmp05 = _mm512_sub_pd(tmp05, y05);
            c05.store(tmp05);
            h05.store(t05);

            let s06 = h06.load();
            let y06 = _mm512_sub_pd(z06, c06.load());
            let t06 = _mm512_add_pd(s06, y06);
            let mut tmp06 = _mm512_sub_pd(t06, s06);
            tmp06 = _mm512_sub_pd(tmp06, y06);
            c06.store(tmp06);
            h06.store(t06);

            let s07 = h07.load();
            let y07 = _mm512_sub_pd(z07, c07.load());
            let t07 = _mm512_add_pd(s07, y07);
            let mut tmp07 = _mm512_sub_pd(t07, s07);
            tmp07 = _mm512_sub_pd(tmp07, y07);
            c07.store(tmp07);
            h07.store(t07);

            let s08 = h08.load();
            let y08 = _mm512_sub_pd(z08, c08.load());
            let t08 = _mm512_add_pd(s08, y08);
            let mut tmp08 = _mm512_sub_pd(t08, s08);
            tmp08 = _mm512_sub_pd(tmp08, y08);
            c08.store(tmp08);
            h08.store(t08);

            // Row 1
            let s11 = h11.load();
            let y11 = _mm512_sub_pd(z11, c11.load());
            let t11 = _mm512_add_pd(s11, y11);
            let mut tmp11 = _mm512_sub_pd(t11, s11);
            tmp11 = _mm512_sub_pd(tmp11, y11);
            c11.store(tmp11);
            h11.store(t11);

            let s12 = h12.load();
            let y12 = _mm512_sub_pd(z12, c12.load());
            let t12 = _mm512_add_pd(s12, y12);
            let mut tmp12 = _mm512_sub_pd(t12, s12);
            tmp12 = _mm512_sub_pd(tmp12, y12);
            c12.store(tmp12);
            h12.store(t12);

            let s13 = h13.load();
            let y13 = _mm512_sub_pd(z13, c13.load());
            let t13 = _mm512_add_pd(s13, y13);
            let mut tmp13 = _mm512_sub_pd(t13, s13);
            tmp13 = _mm512_sub_pd(tmp13, y13);
            c13.store(tmp13);
            h13.store(t13);

            let s14 = h14.load();
            let y14 = _mm512_sub_pd(z14, c14.load());
            let t14 = _mm512_add_pd(s14, y14);
            let mut tmp14 = _mm512_sub_pd(t14, s14);
            tmp14 = _mm512_sub_pd(tmp14, y14);
            c14.store(tmp14);
            h14.store(t14);

            let s15 = h15.load();
            let y15 = _mm512_sub_pd(z15, c15.load());
            let t15 = _mm512_add_pd(s15, y15);
            let mut tmp15 = _mm512_sub_pd(t15, s15);
            tmp15 = _mm512_sub_pd(tmp15, y15);
            c15.store(tmp15);
            h15.store(t15);

            let s16 = h16.load();
            let y16 = _mm512_sub_pd(z16, c16.load());
            let t16 = _mm512_add_pd(s16, y16);
            let mut tmp16 = _mm512_sub_pd(t16, s16);
            tmp16 = _mm512_sub_pd(tmp16, y16);
            c16.store(tmp16);
            h16.store(t16);

            let s17 = h17.load();
            let y17 = _mm512_sub_pd(z17, c17.load());
            let t17 = _mm512_add_pd(s17, y17);
            let mut tmp17 = _mm512_sub_pd(t17, s17);
            tmp17 = _mm512_sub_pd(tmp17, y17);
            c17.store(tmp17);
            h17.store(t17);

            let s18 = h18.load();
            let y18 = _mm512_sub_pd(z18, c18.load());
            let t18 = _mm512_add_pd(s18, y18);
            let mut tmp18 = _mm512_sub_pd(t18, s18);
            tmp18 = _mm512_sub_pd(tmp18, y18);
            c18.store(tmp18);
            h18.store(t18);

            // Row 2
            let s22 = h22.load();
            let y22 = _mm512_sub_pd(z22, c22.load());
            let t22 = _mm512_add_pd(s22, y22);
            let mut tmp22 = _mm512_sub_pd(t22, s22);
            tmp22 = _mm512_sub_pd(tmp22, y22);
            c22.store(tmp22);
            h22.store(t22);

            let s23 = h23.load();
            let y23 = _mm512_sub_pd(z23, c23.load());
            let t23 = _mm512_add_pd(s23, y23);
            let mut tmp23 = _mm512_sub_pd(t23, s23);
            tmp23 = _mm512_sub_pd(tmp23, y23);
            c23.store(tmp23);
            h23.store(t23);

            let s24 = h24.load();
            let y24 = _mm512_sub_pd(z24, c24.load());
            let t24 = _mm512_add_pd(s24, y24);
            let mut tmp24 = _mm512_sub_pd(t24, s24);
            tmp24 = _mm512_sub_pd(tmp24, y24);
            c24.store(tmp24);
            h24.store(t24);

            let s25 = h25.load();
            let y25 = _mm512_sub_pd(z25, c25.load());
            let t25 = _mm512_add_pd(s25, y25);
            let mut tmp25 = _mm512_sub_pd(t25, s25);
            tmp25 = _mm512_sub_pd(tmp25, y25);
            c25.store(tmp25);
            h25.store(t25);

            let s26 = h26.load();
            let y26 = _mm512_sub_pd(z26, c26.load());
            let t26 = _mm512_add_pd(s26, y26);
            let mut tmp26 = _mm512_sub_pd(t26, s26);
            tmp26 = _mm512_sub_pd(tmp26, y26);
            c26.store(tmp26);
            h26.store(t26);

            let s27 = h27.load();
            let y27 = _mm512_sub_pd(z27, c27.load());
            let t27 = _mm512_add_pd(s27, y27);
            let mut tmp27 = _mm512_sub_pd(t27, s27);
            tmp27 = _mm512_sub_pd(tmp27, y27);
            c27.store(tmp27);
            h27.store(t27);

            let s28 = h28.load();
            let y28 = _mm512_sub_pd(z28, c28.load());
            let t28 = _mm512_add_pd(s28, y28);
            let mut tmp28 = _mm512_sub_pd(t28, s28);
            tmp28 = _mm512_sub_pd(tmp28, y28);
            c28.store(tmp28);
            h28.store(t28);
        }

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
                let c3 = block[3].load();
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

            // Row 3
            let s33 = h33.load();
            let y33 = _mm512_sub_pd(z33, c33.load());
            let t33 = _mm512_add_pd(s33, y33);
            let mut tmp33 = _mm512_sub_pd(t33, s33);
            tmp33 = _mm512_sub_pd(tmp33, y33);
            c33.store(tmp33);
            h33.store(t33);

            let s34 = h34.load();
            let y34 = _mm512_sub_pd(z34, c34.load());
            let t34 = _mm512_add_pd(s34, y34);
            let mut tmp34 = _mm512_sub_pd(t34, s34);
            tmp34 = _mm512_sub_pd(tmp34, y34);
            c34.store(tmp34);
            h34.store(t34);

            let s35 = h35.load();
            let y35 = _mm512_sub_pd(z35, c35.load());
            let t35 = _mm512_add_pd(s35, y35);
            let mut tmp35 = _mm512_sub_pd(t35, s35);
            tmp35 = _mm512_sub_pd(tmp35, y35);
            c35.store(tmp35);
            h35.store(t35);

            let s36 = h36.load();
            let y36 = _mm512_sub_pd(z36, c36.load());
            let t36 = _mm512_add_pd(s36, y36);
            let mut tmp36 = _mm512_sub_pd(t36, s36);
            tmp36 = _mm512_sub_pd(tmp36, y36);
            c36.store(tmp36);
            h36.store(t36);

            let s37 = h37.load();
            let y37 = _mm512_sub_pd(z37, c37.load());
            let t37 = _mm512_add_pd(s37, y37);
            let mut tmp37 = _mm512_sub_pd(t37, s37);
            tmp37 = _mm512_sub_pd(tmp37, y37);
            c37.store(tmp37);
            h37.store(t37);

            let s38 = h38.load();
            let y38 = _mm512_sub_pd(z38, c38.load());
            let t38 = _mm512_add_pd(s38, y38);
            let mut tmp38 = _mm512_sub_pd(t38, s38);
            tmp38 = _mm512_sub_pd(tmp38, y38);
            c38.store(tmp38);
            h38.store(t38);

            // Row 4
            let s44 = h44.load();
            let y44 = _mm512_sub_pd(z44, c44.load());
            let t44 = _mm512_add_pd(s44, y44);
            let mut tmp44 = _mm512_sub_pd(t44, s44);
            tmp44 = _mm512_sub_pd(tmp44, y44);
            c44.store(tmp44);
            h44.store(t44);

            let s45 = h45.load();
            let y45 = _mm512_sub_pd(z45, c45.load());
            let t45 = _mm512_add_pd(s45, y45);
            let mut tmp45 = _mm512_sub_pd(t45, s45);
            tmp45 = _mm512_sub_pd(tmp45, y45);
            c45.store(tmp45);
            h45.store(t45);

            let s46 = h46.load();
            let y46 = _mm512_sub_pd(z46, c46.load());
            let t46 = _mm512_add_pd(s46, y46);
            let mut tmp46 = _mm512_sub_pd(t46, s46);
            tmp46 = _mm512_sub_pd(tmp46, y46);
            c46.store(tmp46);
            h46.store(t46);

            let s47 = h47.load();
            let y47 = _mm512_sub_pd(z47, c47.load());
            let t47 = _mm512_add_pd(s47, y47);
            let mut tmp47 = _mm512_sub_pd(t47, s47);
            tmp47 = _mm512_sub_pd(tmp47, y47);
            c47.store(tmp47);
            h47.store(t47);

            let s48 = h48.load();
            let y48 = _mm512_sub_pd(z48, c48.load());
            let t48 = _mm512_add_pd(s48, y48);
            let mut tmp48 = _mm512_sub_pd(t48, s48);
            tmp48 = _mm512_sub_pd(tmp48, y48);
            c48.store(tmp48);
            h48.store(t48);

            // Row 5
            let s55 = h55.load();
            let y55 = _mm512_sub_pd(z55, c55.load());
            let t55 = _mm512_add_pd(s55, y55);
            let mut tmp55 = _mm512_sub_pd(t55, s55);
            tmp55 = _mm512_sub_pd(tmp55, y55);
            c55.store(tmp55);
            h55.store(t55);

            let s56 = h56.load();
            let y56 = _mm512_sub_pd(z56, c56.load());
            let t56 = _mm512_add_pd(s56, y56);
            let mut tmp56 = _mm512_sub_pd(t56, s56);
            tmp56 = _mm512_sub_pd(tmp56, y56);
            c56.store(tmp56);
            h56.store(t56);

            let s57 = h57.load();
            let y57 = _mm512_sub_pd(z57, c57.load());
            let t57 = _mm512_add_pd(s57, y57);
            let mut tmp57 = _mm512_sub_pd(t57, s57);
            tmp57 = _mm512_sub_pd(tmp57, y57);
            c57.store(tmp57);
            h57.store(t57);

            let s58 = h58.load();
            let y58 = _mm512_sub_pd(z58, c58.load());
            let t58 = _mm512_add_pd(s58, y58);
            let mut tmp58 = _mm512_sub_pd(t58, s58);
            tmp58 = _mm512_sub_pd(tmp58, y58);
            c58.store(tmp58);
            h58.store(t58);

            // Row 6
            let s66 = h66.load();
            let y66 = _mm512_sub_pd(z66, c66.load());
            let t66 = _mm512_add_pd(s66, y66);
            let mut tmp66 = _mm512_sub_pd(t66, s66);
            tmp66 = _mm512_sub_pd(tmp66, y66);
            c66.store(tmp66);
            h66.store(t66);

            let s67 = h67.load();
            let y67 = _mm512_sub_pd(z67, c67.load());
            let t67 = _mm512_add_pd(s67, y67);
            let mut tmp67 = _mm512_sub_pd(t67, s67);
            tmp67 = _mm512_sub_pd(tmp67, y67);
            c67.store(tmp67);
            h67.store(t67);

            let s68 = h68.load();
            let y68 = _mm512_sub_pd(z68, c68.load());
            let t68 = _mm512_add_pd(s68, y68);
            let mut tmp68 = _mm512_sub_pd(t68, s68);
            tmp68 = _mm512_sub_pd(tmp68, y68);
            c68.store(tmp68);
            h68.store(t68);

            // Row 7
            let s77 = h77.load();
            let y77 = _mm512_sub_pd(z77, c77.load());
            let t77 = _mm512_add_pd(s77, y77);
            let mut tmp77 = _mm512_sub_pd(t77, s77);
            tmp77 = _mm512_sub_pd(tmp77, y77);
            c77.store(tmp77);
            h77.store(t77);

            let s78 = h78.load();
            let y78 = _mm512_sub_pd(z78, c78.load());
            let t78 = _mm512_add_pd(s78, y78);
            let mut tmp78 = _mm512_sub_pd(t78, s78);
            tmp78 = _mm512_sub_pd(tmp78, y78);
            c78.store(tmp78);
            h78.store(t78);

            // Row 8
            let s88 = h88.load();
            let y88 = _mm512_sub_pd(z88, c88.load());
            let t88 = _mm512_add_pd(s88, y88);
            let mut tmp88 = _mm512_sub_pd(t88, s88);
            tmp88 = _mm512_sub_pd(tmp88, y88);
            c88.store(tmp88);
            h88.store(t88);
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
pub fn compute_pt_d2_p_avx512_tiled_three_passes2_kahan_refactor(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [[f64; 9]; 9]) {
    const BLOCKS: usize = 32;

    let mut g = [Lane8::zero(); 9];
    let mut gc = [Lane8::zero(); 9];

    // Array allocating 45 slots for a 9x9 upper triangular matrix[cite: 1]
    let mut h = [Lane8::zero(); 45];
    let mut c = [Lane8::zero(); 45];

    // Register allocation:
    // - 9 for x
    // - 1 for one
    // For calculation of d: 1 for d + 9 columns + 9 gradient = 19
    // For each Hessian pass: 15 accumulators + at most 5 columns = 20[cite: 1]

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
    // This is one tile[cite: 1]
    let mut scaled_column_buf = [[Lane8::zero(); 9]; BLOCKS];

    let (tiles, rem) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        {
            let mut z_grad = [_mm512_setzero_pd(); 9];

            for (buffer, block) in scaled_column_buf.iter_mut().zip(tile.iter()) {
                let mut c_arr: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                let mut d = _mm512_set1_pd(eps);
                for col in 0..9 {
                    d = _mm512_fmadd_pd(xs[col], c_arr[col], d);
                }

                d = _mm512_div_pd(one, d);

                for col in 0..9 {
                    c_arr[col] = _mm512_mul_pd(c_arr[col], d);
                }

                for col in 0..9 {
                    buffer[col].store(c_arr[col]);
                }

                for col in 0..9 {
                    z_grad[col] = _mm512_fmadd_pd(c_arr[col], d, z_grad[col]);
                }
            }

            // Gradient kahan[cite: 1]
            for col in 0..9 {
                let mut sum = g[col].load();
                z_grad[col] = _mm512_sub_pd(z_grad[col], gc[col].load());
                let t = _mm512_add_pd(sum, z_grad[col]);
                sum = _mm512_sub_pd(t, sum);
                sum = _mm512_sub_pd(sum, z_grad[col]);

                g[col].store(t);
                gc[col].store(sum);
            }
        }

        {
            // Pass 1: Accumulators 0..15 (maps to h[0..15])
            let mut z = [_mm512_setzero_pd(); 15];

            for block in scaled_column_buf.iter() {
                let c0 = block[0].load();
                let c1 = block[1].load();

                z[0] = _mm512_fmadd_pd(c0, c0, z[0]); // z00[cite: 1]
                z[1] = _mm512_fmadd_pd(c0, c1, z[1]); // z01[cite: 1]
                z[9] = _mm512_fmadd_pd(c1, c1, z[9]); // z11[cite: 1]

                let c2 = block[2].load();
                z[2] = _mm512_fmadd_pd(c0, c2, z[2]); // z02[cite: 1]
                z[10] = _mm512_fmadd_pd(c1, c2, z[10]); // z12[cite: 1]

                let c3 = block[3].load();
                z[3] = _mm512_fmadd_pd(c0, c3, z[3]); // z03[cite: 1]
                z[11] = _mm512_fmadd_pd(c1, c3, z[11]); // z13[cite: 1]

                let c4 = block[4].load();
                z[4] = _mm512_fmadd_pd(c0, c4, z[4]); // z04[cite: 1]
                z[12] = _mm512_fmadd_pd(c1, c4, z[12]); // z14[cite: 1]

                let c5 = block[5].load();
                z[5] = _mm512_fmadd_pd(c0, c5, z[5]); // z05[cite: 1]
                z[13] = _mm512_fmadd_pd(c1, c5, z[13]); // z15[cite: 1]

                let c6 = block[6].load();
                z[6] = _mm512_fmadd_pd(c0, c6, z[6]); // z06[cite: 1]
                z[14] = _mm512_fmadd_pd(c1, c6, z[14]); // z16[cite: 1]

                let c7 = block[7].load();
                z[7] = _mm512_fmadd_pd(c0, c7, z[7]); // z07[cite: 1]

                let c8 = block[8].load();
                z[8] = _mm512_fmadd_pd(c0, c8, z[8]); // z08[cite: 1]
            }

            // Kahan Row 0 & 1 (part 1)[cite: 1]
            for i in 0..15 {
                let h_idx = i;
                let mut s = h[h_idx].load();
                z[i] = _mm512_sub_pd(z[i], c[h_idx].load());
                let t = _mm512_add_pd(s, z[i]);
                s = _mm512_sub_pd(t, s);
                s = _mm512_sub_pd(s, z[i]);
                h[h_idx].store(t);
                c[h_idx].store(s);
            }
        }

        {
            // Pass 2: Accumulators 0..15 (maps to h[15..30])
            let mut z = [_mm512_setzero_pd(); 15];

            for block in scaled_column_buf.iter() {
                let c1 = block[1].load();
                let c2 = block[2].load();
                let c3 = block[3].load();

                z[2] = _mm512_fmadd_pd(c2, c2, z[2]); // z22 (h_idx 17)[cite: 1]

                z[3] = _mm512_fmadd_pd(c2, c3, z[3]); // z23 (h_idx 18)[cite: 1]
                z[9] = _mm512_fmadd_pd(c3, c3, z[9]); // z33 (h_idx 24)[cite: 1]

                let c4 = block[4].load();
                z[4] = _mm512_fmadd_pd(c2, c4, z[4]); // z24 (h_idx 19)[cite: 1]
                z[10] = _mm512_fmadd_pd(c3, c4, z[10]); // z34 (h_idx 25)[cite: 1]

                let c5 = block[5].load();
                z[5] = _mm512_fmadd_pd(c2, c5, z[5]); // z25 (h_idx 20)[cite: 1]
                z[11] = _mm512_fmadd_pd(c3, c5, z[11]); // z35 (h_idx 26)[cite: 1]

                let c6 = block[6].load();
                z[6] = _mm512_fmadd_pd(c2, c6, z[6]); // z26 (h_idx 21)[cite: 1]
                z[12] = _mm512_fmadd_pd(c3, c6, z[12]); // z36 (h_idx 27)[cite: 1]

                let c7 = block[7].load();
                z[0] = _mm512_fmadd_pd(c1, c7, z[0]); // z17 (h_idx 15)[cite: 1]
                z[7] = _mm512_fmadd_pd(c2, c7, z[7]); // z27 (h_idx 22)[cite: 1]
                z[13] = _mm512_fmadd_pd(c3, c7, z[13]); // z37 (h_idx 28)[cite: 1]

                let c8 = block[8].load();
                z[1] = _mm512_fmadd_pd(c1, c8, z[1]); // z18 (h_idx 16)[cite: 1]
                z[8] = _mm512_fmadd_pd(c2, c8, z[8]); // z28 (h_idx 23)[cite: 1]
                z[14] = _mm512_fmadd_pd(c3, c8, z[14]); // z38 (h_idx 29)[cite: 1]
            }

            // Kahan Row 1 (continued), 2, & 3[cite: 1]
            for i in 0..15 {
                let h_idx = i + 15;

                let mut s = h[h_idx].load();
                z[i] = _mm512_sub_pd(z[i], c[h_idx].load());
                let t = _mm512_add_pd(s, z[i]);
                s = _mm512_sub_pd(t, s);
                s = _mm512_sub_pd(s, z[i]);
                h[h_idx].store(t);
                c[h_idx].store(s);
                // let s = h[h_idx].load();
                // let y = _mm512_sub_pd(z[i], c[h_idx].load());
                // let t = _mm512_add_pd(s, y);
                // let mut tmp = _mm512_sub_pd(t, s);
                // tmp = _mm512_sub_pd(tmp, y);
                // c[h_idx].store(tmp);
                // h[h_idx].store(t);
            }
        }

        {
            // Pass 3: Accumulators 0..15 (maps to h[30..45])
            let mut z = [_mm512_setzero_pd(); 15];

            for block in scaled_column_buf.iter() {
                let c4 = block[4].load();
                let c5 = block[5].load();
                let c6 = block[6].load();
                let c7 = block[7].load();
                let c8 = block[8].load();

                z[0] = _mm512_fmadd_pd(c4, c4, z[0]); // z44 (h_idx 30)[cite: 1]
                z[1] = _mm512_fmadd_pd(c4, c5, z[1]); // z45 (h_idx 31)[cite: 1]
                z[2] = _mm512_fmadd_pd(c4, c6, z[2]); // z46 (h_idx 32)[cite: 1]
                z[3] = _mm512_fmadd_pd(c4, c7, z[3]); // z47 (h_idx 33)[cite: 1]
                z[4] = _mm512_fmadd_pd(c4, c8, z[4]); // z48 (h_idx 34)[cite: 1]

                z[5] = _mm512_fmadd_pd(c5, c5, z[5]); // z55 (h_idx 35)[cite: 1]
                z[6] = _mm512_fmadd_pd(c5, c6, z[6]); // z56 (h_idx 36)[cite: 1]
                z[7] = _mm512_fmadd_pd(c5, c7, z[7]); // z57 (h_idx 37)[cite: 1]
                z[8] = _mm512_fmadd_pd(c5, c8, z[8]); // z58 (h_idx 38)[cite: 1]

                z[9] = _mm512_fmadd_pd(c6, c6, z[9]); // z66 (h_idx 39)[cite: 1]
                z[10] = _mm512_fmadd_pd(c6, c7, z[10]); // z67 (h_idx 40)[cite: 1]
                z[11] = _mm512_fmadd_pd(c6, c8, z[11]); // z68 (h_idx 41)[cite: 1]

                z[12] = _mm512_fmadd_pd(c7, c7, z[12]); // z77 (h_idx 42)[cite: 1]
                z[13] = _mm512_fmadd_pd(c7, c8, z[13]); // z78 (h_idx 43)[cite: 1]

                z[14] = _mm512_fmadd_pd(c8, c8, z[14]); // z88 (h_idx 44)[cite: 1]
            }

            // Kahan Row 4, 5, 6, 7, & 8[cite: 1]
            for i in 0..15 {
                let h_idx = i + 30;

                let mut s = h[h_idx].load();
                z[i] = _mm512_sub_pd(z[i], c[h_idx].load());
                let t = _mm512_add_pd(s, z[i]);
                s = _mm512_sub_pd(t, s);
                s = _mm512_sub_pd(s, z[i]);
                h[h_idx].store(t);
                c[h_idx].store(s);
                // let s = h[h_idx].load();
                // let y = _mm512_sub_pd(z[i], c[h_idx].load());
                // let t = _mm512_add_pd(s, y);
                // let mut tmp = _mm512_sub_pd(t, s);
                // tmp = _mm512_sub_pd(tmp, y);
                // c[h_idx].store(tmp);
                // h[h_idx].store(t);
            }
        }
    }

    let mut grad = [0.0; 9];
    for col in 0..9 {
        grad[col] = _mm512_reduce_add_pd(g[col].load());
    }

    let mut hess = [[0.0; 9]; 9];

    // Unload the 45-length array into the 9x9 Hessian[cite: 1]
    let mut flat_idx = 0;
    for i in 0..9 {
        for j in i..9 {
            hess[i][j] = _mm512_reduce_add_pd(h[flat_idx].load());
            flat_idx += 1;
        }
    }

    // Mirror to lower triangle[cite: 1]
    for i in 0..9 {
        for j in i..9 {
            hess[j][i] = hess[i][j];
        }
    }

    (grad, hess)
}

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d2_p_avx512_tiled_three_passes3_kahan_refactor(
    blocks: &[Block<f64, 8, 9>],
    remainder: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [[f64; 9]; 9]) {
    const BLOCKS: usize = 32;

    let mut g = [Lane8::zero(); 9];
    let mut gc = [Lane8::zero(); 9];

    // Array allocating 45 slots for a 9x9 upper triangular matrix[cite: 1]
    let mut h = [Lane8::zero(); 45];
    let mut c = [Lane8::zero(); 45];

    // Helper macro to condense the Kahan summation updates[cite: 1]
    macro_rules! kahan {
        ($idx:expr, $z:expr) => {
            let s = h[$idx].load();
            let y = _mm512_sub_pd($z, c[$idx].load());
            let t = _mm512_add_pd(s, y);
            let mut tmp = _mm512_sub_pd(t, s);
            tmp = _mm512_sub_pd(tmp, y);
            c[$idx].store(tmp);
            h[$idx].store(t);
        };
    }

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
                let mut c_arr: [__m512d; 9] =
                    std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });

                let mut d = _mm512_set1_pd(eps);
                for col in 0..9 {
                    d = _mm512_fmadd_pd(xs[col], c_arr[col], d);
                }

                d = _mm512_div_pd(one, d);

                for col in 0..9 {
                    c_arr[col] = _mm512_mul_pd(c_arr[col], d);
                }

                for col in 0..9 {
                    buffer[col].store(c_arr[col]);
                }

                for col in 0..9 {
                    z[col] = _mm512_fmadd_pd(c_arr[col], d, z[col]);
                }
            }

            // Gradient kahan
            for col in 0..9 {
                let mut sum = g[col].load();
                z[col] = _mm512_sub_pd(z[col], gc[col].load());
                let t = _mm512_add_pd(sum, z[col]);
                sum = _mm512_sub_pd(t, sum);
                sum = _mm512_sub_pd(sum, z[col]);

                g[col].store(t);
                gc[col].store(sum);
            }
        }

        {
            // Row 0
            let mut z00 = _mm512_setzero_pd();
            let mut z01 = _mm512_setzero_pd();
            let mut z02 = _mm512_setzero_pd();
            let mut z03 = _mm512_setzero_pd();
            let mut z04 = _mm512_setzero_pd();
            let mut z05 = _mm512_setzero_pd();
            let mut z06 = _mm512_setzero_pd();
            let mut z07 = _mm512_setzero_pd();
            let mut z08 = _mm512_setzero_pd();

            // Row 1
            let mut z11 = _mm512_setzero_pd();
            let mut z12 = _mm512_setzero_pd();
            let mut z13 = _mm512_setzero_pd();
            let mut z14 = _mm512_setzero_pd();
            let mut z15 = _mm512_setzero_pd();
            let mut z16 = _mm512_setzero_pd();

            for block in scaled_column_buf.iter() {
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

                let c5 = block[5].load();
                z05 = _mm512_fmadd_pd(c0, c5, z05);
                z15 = _mm512_fmadd_pd(c1, c5, z15);

                let c6 = block[6].load();
                z06 = _mm512_fmadd_pd(c0, c6, z06);
                z16 = _mm512_fmadd_pd(c1, c6, z16);

                let c7 = block[7].load();
                z07 = _mm512_fmadd_pd(c0, c7, z07);

                let c8 = block[8].load();
                z08 = _mm512_fmadd_pd(c0, c8, z08);
            }

            // Kahan Row 0[cite: 1]
            kahan!(0, z00);
            kahan!(1, z01);
            kahan!(2, z02);
            kahan!(3, z03);
            kahan!(4, z04);
            kahan!(5, z05);
            kahan!(6, z06);
            kahan!(7, z07);
            kahan!(8, z08);

            // Kahan Row 1 (first part)[cite: 1]
            kahan!(9, z11);
            kahan!(10, z12);
            kahan!(11, z13);
            kahan!(12, z14);
            kahan!(13, z15);
            kahan!(14, z16);
        }

        {
            let mut z17 = _mm512_setzero_pd();
            let mut z18 = _mm512_setzero_pd();

            // Row 2
            let mut z22 = _mm512_setzero_pd();
            let mut z23 = _mm512_setzero_pd();
            let mut z24 = _mm512_setzero_pd();
            let mut z25 = _mm512_setzero_pd();
            let mut z26 = _mm512_setzero_pd();
            let mut z27 = _mm512_setzero_pd();
            let mut z28 = _mm512_setzero_pd();

            // Row 3
            let mut z33 = _mm512_setzero_pd();
            let mut z34 = _mm512_setzero_pd();
            let mut z35 = _mm512_setzero_pd();
            let mut z36 = _mm512_setzero_pd();
            let mut z37 = _mm512_setzero_pd();
            let mut z38 = _mm512_setzero_pd();

            for block in scaled_column_buf.iter() {
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
            }

            // Kahan Row 1 (continued)[cite: 1]
            kahan!(15, z17);
            kahan!(16, z18);

            // Kahan Row 2[cite: 1]
            kahan!(17, z22);
            kahan!(18, z23);
            kahan!(19, z24);
            kahan!(20, z25);
            kahan!(21, z26);
            kahan!(22, z27);
            kahan!(23, z28);

            // Kahan Row 3[cite: 1]
            kahan!(24, z33);
            kahan!(25, z34);
            kahan!(26, z35);
            kahan!(27, z36);
            kahan!(28, z37);
            kahan!(29, z38);
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

            // Kahan Row 4[cite: 1]
            kahan!(30, z44);
            kahan!(31, z45);
            kahan!(32, z46);
            kahan!(33, z47);
            kahan!(34, z48);

            // Kahan Row 5[cite: 1]
            kahan!(35, z55);
            kahan!(36, z56);
            kahan!(37, z57);
            kahan!(38, z58);

            // Kahan Row 6[cite: 1]
            kahan!(39, z66);
            kahan!(40, z67);
            kahan!(41, z68);

            // Kahan Row 7[cite: 1]
            kahan!(42, z77);
            kahan!(43, z78);

            // Kahan Row 8[cite: 1]
            kahan!(44, z88);
        }
    }

    let mut grad = [0.0; 9];
    for col in 0..9 {
        grad[col] = _mm512_reduce_add_pd(g[col].load());
    }

    let mut hess = [[0.0; 9]; 9];

    // Unload the 45-length array into the 9x9 Hessian[cite: 1]
    let mut flat_idx = 0;
    for i in 0..9 {
        for j in i..9 {
            hess[i][j] = _mm512_reduce_add_pd(h[flat_idx].load());
            flat_idx += 1;
        }
    }

    // Mirror to lower triangle[cite: 1]
    for i in 0..9 {
        for j in i..9 {
            hess[j][i] = hess[i][j];
        }
    }

    (grad, hess)
}
