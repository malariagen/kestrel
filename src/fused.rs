use core::arch::x86_64::*;

use crate::{
    algebra::{Matrix, Vector, dot, scale_div_mut}, blockbuffer::{Block, BlockBuffer}, buffer::Lane8};

// h - 8*8*45 = 2880 bytes
// column buffer - 8 * 8 * 9 * 32 = 18432 bytes (TODO avoid memset)

const BLOCKS: usize = 32;

pub fn compute_grad_hess(
    p_mat: &BlockBuffer<f64, 8, 9>,
    x: &Vector<9>,
    eps: f64,
) -> (Vector<9>, Matrix<9>) {
    let (g, mut h) = unsafe { compute_g_h_fused_avx512(p_mat, x, eps) };

    let n = p_mat.num_rows() as f64;

    // let grad: Vector<9> = std::array::from_fn(|i| 1.0 - g[i] / n);
    let grad: Vector<9> = std::array::from_fn(|i| - g[i] / n);

    scale_div_mut(&mut h, n);

    (grad, h)
}

fn compute_g_h_fused_scalar(
    p_mat: &[[f64; 9]],
    x: &[f64; 9],
    eps: f64,
    g: &mut [f64; 9],
    h: &mut [[f64; 9]; 9],
) {
    for row in p_mat.iter() {
        let prod = dot(row, x);

        // d = 1 / (P x + eps)
        let d = 1.0 / (prod + eps);

        // g = P^T d
        for i in 0..9 {
            g[i] = row[i].mul_add(d, g[i]);
        }

        let mut scaled_row = [0.0; 9];
        for i in 0..9 {
            scaled_row[i] = d * row[i];
        }

        // h = P^T D^D P
        for i in 0..9 {
            // This generates much nicer assembly by iterating from 0
            // And in practice has the same performance as iterating from i
            for j in 0..9 {
                h[i][j] = scaled_row[i].mul_add(scaled_row[j], h[i][j]);
            }
        }
    }
}

#[target_feature(enable = "avx512f")]
pub fn compute_g_h_fused_avx512(
    p_mat: &BlockBuffer<f64, 8, 9>,
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [[f64; 9]; 9]) {
    let mut g = [Lane8::zero(); 9];

    // Upper triangular Hessian accumulators stored in row-major order:
    // Row 0: h[0..9]   (h00..h08)
    // Row 1: h[9..17]  (h11..h18)
    // Row 2: h[17..24] (h22..h28)
    // Row 3: h[24..30] (h33..h38)
    // Row 4: h[30..35] (h44..h48)
    // Row 5: h[35..39] (h55..h58)
    // Row 6: h[39..42] (h66..h68)
    // Row 7: h[42..44] (h77..h78)
    // Row 8: h[44]     (h88)
    let mut h = [Lane8::zero(); 45];

    // Register allocation:
    // - 1 for one (though this can be broadcasted as a constant)
    // - 1 for eps (can be broadcasted from stack)
    // - 9 for x
    // For calculation of d: 1 for d + 9 columns + 9 gradient = 19
    // For each Hessian pass: 15 accumulators + at most 5 columns = 20

    // A block = [Lane8; 9]
    // This is one tile
    let mut scaled_column_buf = [[Lane8::zero(); 9]; BLOCKS];

    let (blocks, remainder) = p_mat.as_blocks();
    let (tiles, partial_tile) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        tile_loop(tile, x, eps, &mut g, &mut h, &mut scaled_column_buf);
    }

    tile_loop(partial_tile, x, eps, &mut g, &mut h, &mut scaled_column_buf);

    let mut grad = [0.0; 9];
    for col in 0..9 {
        grad[col] = _mm512_reduce_add_pd(g[col].load());
    }

    let mut hess = [[0.0; 9]; 9];

    let mut h_id = 0;
    for i in 0..9 {
        for j in i..9 {
            let val = _mm512_reduce_add_pd(h[h_id].load());
            hess[i][j] = val;
            hess[j][i] = val;
            h_id += 1;
        }
    }

    compute_g_h_fused_scalar(remainder, x, eps, &mut grad, &mut hess);

    (grad, hess)
}

#[target_feature(enable = "avx512f")]
pub fn tile_loop(
    tile: &[Block<f64, 8, 9>],
    x: &[f64; 9],
    eps: f64,
    g: &mut [Lane8; 9],
    h: &mut [Lane8; 45],
    scaled_column_buf: &mut [[Lane8; 9]; BLOCKS],
) {
    let one = _mm512_set1_pd(1.0);
    let ze = _mm512_set1_pd(eps);

    let zx: [__m512d; 9] = std::array::from_fn(|i| _mm512_set1_pd(x[i]));

    // This will always be <= BLOCKS
    let blocks = tile.len();

    {
        let mut zg = [_mm512_setzero_pd(); 9];

        for i in 0..blocks {
            let tile_block = unsafe { tile.get_unchecked(i) };
            let buffer_block = unsafe { scaled_column_buf.get_unchecked_mut(i) };

            let mut c: [__m512d; 9] =
                std::array::from_fn(|i| unsafe { _mm512_load_pd(tile_block[i].as_ptr()) });

            // Calculate d
            // This computes a dot product between x and a row of p
            // TODO this could be manually unrolled a few times
            // let mut d = _mm512_set1_pd(eps);
            // for col in 0..9 {
            //     d = _mm512_fmadd_pd(zx[col], c[col], d);
            // }

            let mut d0 = _mm512_fmadd_pd(zx[0], c[0], ze);
            let mut d1 = _mm512_mul_pd(zx[1], c[1]);
            let mut d2 = _mm512_mul_pd(zx[2], c[2]);

            d0 = _mm512_fmadd_pd(zx[3], c[3], d0);
            d1 = _mm512_fmadd_pd(zx[4], c[4], d1);
            d2 = _mm512_fmadd_pd(zx[5], c[5], d2);

            d0 = _mm512_fmadd_pd(zx[6], c[6], d0);
            d1 = _mm512_fmadd_pd(zx[7], c[7], d1);
            d2 = _mm512_fmadd_pd(zx[8], c[8], d2);

            let mut d = _mm512_add_pd(d0, d1);
            d = _mm512_add_pd(d, d2);

            d = _mm512_div_pd(one, d);

            for col in 0..9 {
                zg[col] = _mm512_fmadd_pd(c[col], d, zg[col]);
            }

            // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

            for col in 0..9 {
                c[col] = _mm512_mul_pd(c[col], d);
            }

            for col in 0..9 {
                buffer_block[col].store(c[col]);
            }
        }

        for col in 0..9 {
            zg[col] = _mm512_add_pd(g[col].load(), zg[col]);
        }

        for col in 0..9 {
            g[col].store(zg[col]);
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

        // for block in scaled_column_buf.iter() {
        for i in 0..blocks {
            let block = unsafe { scaled_column_buf.get_unchecked(i) };

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

        z00 = _mm512_add_pd(h[0].load(), z00);
        z01 = _mm512_add_pd(h[1].load(), z01);
        z02 = _mm512_add_pd(h[2].load(), z02);
        z03 = _mm512_add_pd(h[3].load(), z03);
        z04 = _mm512_add_pd(h[4].load(), z04);
        z05 = _mm512_add_pd(h[5].load(), z05);
        z06 = _mm512_add_pd(h[6].load(), z06);
        z07 = _mm512_add_pd(h[7].load(), z07);
        z08 = _mm512_add_pd(h[8].load(), z08);

        z11 = _mm512_add_pd(h[9].load(), z11);
        z12 = _mm512_add_pd(h[10].load(), z12);
        z13 = _mm512_add_pd(h[11].load(), z13);
        z14 = _mm512_add_pd(h[12].load(), z14);
        z15 = _mm512_add_pd(h[13].load(), z15);
        z16 = _mm512_add_pd(h[14].load(), z16);

        // Row 0
        h[0].store(z00);
        h[1].store(z01);
        h[2].store(z02);
        h[3].store(z03);
        h[4].store(z04);
        h[5].store(z05);
        h[6].store(z06);
        h[7].store(z07);
        h[8].store(z08);

        // Row 1
        h[9].store(z11);
        h[10].store(z12);
        h[11].store(z13);
        h[12].store(z14);
        h[13].store(z15);
        h[14].store(z16);
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

        // for block in scaled_column_buf.iter() {
        for i in 0..blocks {
            let block = unsafe { scaled_column_buf.get_unchecked(i) };

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

        z17 = _mm512_add_pd(h[15].load(), z17);
        z18 = _mm512_add_pd(h[16].load(), z18);

        z22 = _mm512_add_pd(h[17].load(), z22);
        z23 = _mm512_add_pd(h[18].load(), z23);
        z24 = _mm512_add_pd(h[19].load(), z24);
        z25 = _mm512_add_pd(h[20].load(), z25);
        z26 = _mm512_add_pd(h[21].load(), z26);
        z27 = _mm512_add_pd(h[22].load(), z27);
        z28 = _mm512_add_pd(h[23].load(), z28);

        z33 = _mm512_add_pd(h[24].load(), z33);
        z34 = _mm512_add_pd(h[25].load(), z34);
        z35 = _mm512_add_pd(h[26].load(), z35);
        z36 = _mm512_add_pd(h[27].load(), z36);
        z37 = _mm512_add_pd(h[28].load(), z37);
        z38 = _mm512_add_pd(h[29].load(), z38);

        h[15].store(z17);
        h[16].store(z18);

        // Row 2
        h[17].store(z22);
        h[18].store(z23);
        h[19].store(z24);
        h[20].store(z25);
        h[21].store(z26);
        h[22].store(z27);
        h[23].store(z28);

        // Row 3
        h[24].store(z33);
        h[25].store(z34);
        h[26].store(z35);
        h[27].store(z36);
        h[28].store(z37);
        h[29].store(z38);
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

        // for block in scaled_column_buf.iter() {
        for i in 0..blocks {
            let block = unsafe { scaled_column_buf.get_unchecked(i) };

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

        z44 = _mm512_add_pd(h[30].load(), z44);
        z45 = _mm512_add_pd(h[31].load(), z45);
        z46 = _mm512_add_pd(h[32].load(), z46);
        z47 = _mm512_add_pd(h[33].load(), z47);
        z48 = _mm512_add_pd(h[34].load(), z48);

        // Row 5
        z55 = _mm512_add_pd(h[35].load(), z55);
        z56 = _mm512_add_pd(h[36].load(), z56);
        z57 = _mm512_add_pd(h[37].load(), z57);
        z58 = _mm512_add_pd(h[38].load(), z58);

        // Row 6
        z66 = _mm512_add_pd(h[39].load(), z66);
        z67 = _mm512_add_pd(h[40].load(), z67);
        z68 = _mm512_add_pd(h[41].load(), z68);

        // Row 7
        z77 = _mm512_add_pd(h[42].load(), z77);
        z78 = _mm512_add_pd(h[43].load(), z78);

        // Row 8
        z88 = _mm512_add_pd(h[44].load(), z88);

        // Row 4
        h[30].store(z44);
        h[31].store(z45);
        h[32].store(z46);
        h[33].store(z47);
        h[34].store(z48);

        // Row 5
        h[35].store(z55);
        h[36].store(z56);
        h[37].store(z57);
        h[38].store(z58);

        // Row 6
        h[39].store(z66);
        h[40].store(z67);
        h[41].store(z68);

        // Row 7
        h[42].store(z77);
        h[43].store(z78);

        // Row 8
        h[44].store(z88);
    }
}
