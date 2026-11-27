use crate::{
    algebra::{Matrix, Vector},
    lanevector::LaneVector,
    simd::{Arith, Lane, LoadStore},
};

// h - 8*8*45 = 2880 bytes
// column buffer - 8 * 8 * 9 * 32 = 18432 bytes (TODO avoid memset)

const BLOCKS: usize = 32;

#[inline(always)]
pub fn compute_jacquard_grad_hess<L: Lane>(
    p_mat: &LaneVector<L, 9>,
    x: &Vector<9>,
    eps: f64,
) -> (Vector<9>, Matrix<9>) where L::Reg: Arith {
    let (bg, bh) = compute_grad_hess_lane(p_mat.lanes(), x, eps);
    let (rg, rh) = compute_grad_hess_lane(p_mat.rem(), x, eps);

    let n = p_mat.len() as f64;

    let mut g = [0.0f64; 9];
    let mut h = [[0.0f64; 9]; 9];

    for i in 0..9 {
        g[i] = -(bg[i] + rg[i]) / n;
    }

    let mut h_id = 0;
    for i in 0..9 {
        for j in i..9 {
            let val = (bh[h_id] + rh[h_id]) / n;
            h[i][j] = val;
            h[j][i] = val;
            h_id += 1;
        }
    }

    (g, h)
}

#[inline(always)]
fn compute_grad_hess_lane<L: Lane>(
    blocks: &[[L; 9]],
    x: &[f64; 9],
    eps: f64,
) -> ([f64; 9], [f64; 45]) where L::Reg: Arith {
    // Hmm, in theory we could make individual variables for each g, h element
    let mut g = [L::default(); 9];

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
    let mut h = [L::default(); 45];

    // Register allocation:
    // - 1 for one (though this can be broadcasted as a constant)
    // - 1 for eps (can be broadcasted from stack)
    // - 9 for x
    // For calculation of d: 1 for d + 9 columns + 9 gradient = 19
    // For each Hessian pass: 15 accumulators + at most 5 columns = 20

    // A block = [Lane; 9]
    // This is one tile
    let mut scaled_column_buf = [[L::default(); 9]; BLOCKS];

    let (tiles, partial_tile) = blocks.as_chunks::<BLOCKS>();

    for tile in tiles.iter() {
        tile_loop(tile, x, eps, &mut g, &mut h, &mut scaled_column_buf);
    }

    tile_loop(partial_tile, x, eps, &mut g, &mut h, &mut scaled_column_buf);

    let grad = std::array::from_fn(|i| L::Reg::load(&g[i]).radd());
    let hess = std::array::from_fn(|i| L::Reg::load(&h[i]).radd());

    (grad, hess)
}

#[inline(always)]
fn tile_loop<L: Lane>(
    tile: &[[L; 9]],
    x: &[f64; 9],
    eps: f64,
    g: &mut [L; 9],
    h: &mut [L; 45],
    scaled_column_buf: &mut [[L; 9]; BLOCKS],
) where L::Reg: Arith {
    let one = L::Reg::set(1.0);
    let ze = L::Reg::set(eps);

    let zx: [L::Reg; 9] = std::array::from_fn(|i| L::Reg::set(x[i]));

    let blocks = tile.len();
    debug_assert!(blocks <= BLOCKS);

    // Fill buffer and compute gradient
    {
        let mut zg = [L::Reg::zero(); 9];

        for i in 0..blocks {
            let tile_block = unsafe { tile.get_unchecked(i) };
            let buffer_block = unsafe { scaled_column_buf.get_unchecked_mut(i) };

            let mut c: [L::Reg; 9] = std::array::from_fn(|i| L::Reg::load(&tile_block[i]));

            // This computes a dot product between x and a row of p
            let mut d = ze;
            for col in 0..9 {
                d = zx[col].fma(c[col], d);
            }

            d = one.div(d);

            for col in 0..9 {
                zg[col] = c[col].fma(d, zg[col]);
            }

            // c[i]*d^2*c[j] = (c[i]*d) * (c[j]*d)

            for col in 0..9 {
                c[col] = c[col].mul(d);
            }

            for col in 0..9 {
                c[col].store(&mut buffer_block[col]);
            }
        }

        for col in 0..9 {
            zg[col] = L::Reg::load(&g[col]).add(zg[col]);
        }

        for col in 0..9 {
            zg[col].store(&mut g[col]);
        }
    }

    // Compute first 15 elements of Hessian
    {
        // Row 0
        let mut z00 = L::Reg::zero();
        let mut z01 = L::Reg::zero();
        let mut z02 = L::Reg::zero();
        let mut z03 = L::Reg::zero();
        let mut z04 = L::Reg::zero();
        let mut z05 = L::Reg::zero();
        let mut z06 = L::Reg::zero();
        let mut z07 = L::Reg::zero();
        let mut z08 = L::Reg::zero();

        // Row 1
        let mut z11 = L::Reg::zero();
        let mut z12 = L::Reg::zero();
        let mut z13 = L::Reg::zero();
        let mut z14 = L::Reg::zero();
        let mut z15 = L::Reg::zero();
        let mut z16 = L::Reg::zero();

        for i in 0..blocks {
            let block = unsafe { scaled_column_buf.get_unchecked(i) };

            // 2 permanent + 1 temporary = 3 registers
            let c0 = L::Reg::load(&block[0]);
            let c1 = L::Reg::load(&block[1]);

            z00 = c0.fma(c0, z00);

            z01 = c0.fma(c1, z01);
            z11 = c1.fma(c1, z11);

            let c2 = L::Reg::load(&block[2]);
            z02 = c0.fma(c2, z02);
            z12 = c1.fma(c2, z12);

            let c3 = L::Reg::load(&block[3]);
            z03 = c0.fma(c3, z03);
            z13 = c1.fma(c3, z13);

            let c4 = L::Reg::load(&block[4]);
            z04 = c0.fma(c4, z04);
            z14 = c1.fma(c4, z14);

            let c5 = L::Reg::load(&block[5]);
            z05 = c0.fma(c5, z05);
            z15 = c1.fma(c5, z15);

            let c6 = L::Reg::load(&block[6]);
            z06 = c0.fma(c6, z06);
            z16 = c1.fma(c6, z16);

            let c7 = L::Reg::load(&block[7]);
            z07 = c0.fma(c7, z07);

            let c8 = L::Reg::load(&block[8]);
            z08 = c0.fma(c8, z08);
        }

        z00 = L::Reg::load(&h[0]).add(z00);
        z01 = L::Reg::load(&h[1]).add(z01);
        z02 = L::Reg::load(&h[2]).add(z02);
        z03 = L::Reg::load(&h[3]).add(z03);
        z04 = L::Reg::load(&h[4]).add(z04);
        z05 = L::Reg::load(&h[5]).add(z05);
        z06 = L::Reg::load(&h[6]).add(z06);
        z07 = L::Reg::load(&h[7]).add(z07);
        z08 = L::Reg::load(&h[8]).add(z08);

        z11 = L::Reg::load(&h[9]).add(z11);
        z12 = L::Reg::load(&h[10]).add(z12);
        z13 = L::Reg::load(&h[11]).add(z13);
        z14 = L::Reg::load(&h[12]).add(z14);
        z15 = L::Reg::load(&h[13]).add(z15);
        z16 = L::Reg::load(&h[14]).add(z16);

        // Row 0
        z00.store(&mut h[0]);
        z01.store(&mut h[1]);
        z02.store(&mut h[2]);
        z03.store(&mut h[3]);
        z04.store(&mut h[4]);
        z05.store(&mut h[5]);
        z06.store(&mut h[6]);
        z07.store(&mut h[7]);
        z08.store(&mut h[8]);

        // Row 1
        z11.store(&mut h[9]);
        z12.store(&mut h[10]);
        z13.store(&mut h[11]);
        z14.store(&mut h[12]);
        z15.store(&mut h[13]);
        z16.store(&mut h[14]);
    }

    // Compute next 15 elements of Hessian
    {
        let mut z17 = L::Reg::zero();
        let mut z18 = L::Reg::zero();

        // Row 2
        let mut z22 = L::Reg::zero();
        let mut z23 = L::Reg::zero();
        let mut z24 = L::Reg::zero();
        let mut z25 = L::Reg::zero();
        let mut z26 = L::Reg::zero();
        let mut z27 = L::Reg::zero();
        let mut z28 = L::Reg::zero();

        // Row 3
        let mut z33 = L::Reg::zero();
        let mut z34 = L::Reg::zero();
        let mut z35 = L::Reg::zero();
        let mut z36 = L::Reg::zero();
        let mut z37 = L::Reg::zero();
        let mut z38 = L::Reg::zero();

        for i in 0..blocks {
            let block = unsafe { scaled_column_buf.get_unchecked(i) };

            // 3 permanent + 1 temporary = 4
            let c1 = L::Reg::load(&block[1]);
            let c2 = L::Reg::load(&block[2]);
            let c3 = L::Reg::load(&block[3]);

            z22 = c2.fma(c2, z22);

            z23 = c2.fma(c3, z23);
            z33 = c3.fma(c3, z33);

            let c4 = L::Reg::load(&block[4]);
            z24 = c2.fma(c4, z24);
            z34 = c3.fma(c4, z34);

            let c5 = L::Reg::load(&block[5]);
            z25 = c2.fma(c5, z25);
            z35 = c3.fma(c5, z35);

            let c6 = L::Reg::load(&block[6]);
            z26 = c2.fma(c6, z26);
            z36 = c3.fma(c6, z36);

            let c7 = L::Reg::load(&block[7]);
            z17 = c1.fma(c7, z17);
            z27 = c2.fma(c7, z27);
            z37 = c3.fma(c7, z37);

            let c8 = L::Reg::load(&block[8]);
            z18 = c1.fma(c8, z18);
            z28 = c2.fma(c8, z28);
            z38 = c3.fma(c8, z38);
        }

        z17 = L::Reg::load(&h[15]).add(z17);
        z18 = L::Reg::load(&h[16]).add(z18);

        z22 = L::Reg::load(&h[17]).add(z22);
        z23 = L::Reg::load(&h[18]).add(z23);
        z24 = L::Reg::load(&h[19]).add(z24);
        z25 = L::Reg::load(&h[20]).add(z25);
        z26 = L::Reg::load(&h[21]).add(z26);
        z27 = L::Reg::load(&h[22]).add(z27);
        z28 = L::Reg::load(&h[23]).add(z28);

        z33 = L::Reg::load(&h[24]).add(z33);
        z34 = L::Reg::load(&h[25]).add(z34);
        z35 = L::Reg::load(&h[26]).add(z35);
        z36 = L::Reg::load(&h[27]).add(z36);
        z37 = L::Reg::load(&h[28]).add(z37);
        z38 = L::Reg::load(&h[29]).add(z38);

        z17.store(&mut h[15]);
        z18.store(&mut h[16]);

        // Row 2
        z22.store(&mut h[17]);
        z23.store(&mut h[18]);
        z24.store(&mut h[19]);
        z25.store(&mut h[20]);
        z26.store(&mut h[21]);
        z27.store(&mut h[22]);
        z28.store(&mut h[23]);

        // Row 3
        z33.store(&mut h[24]);
        z34.store(&mut h[25]);
        z35.store(&mut h[26]);
        z36.store(&mut h[27]);
        z37.store(&mut h[28]);
        z38.store(&mut h[29]);
    }

    // Compute last 15 elements of Hessian
    {
        // Row 4
        let mut z44 = L::Reg::zero();
        let mut z45 = L::Reg::zero();
        let mut z46 = L::Reg::zero();
        let mut z47 = L::Reg::zero();
        let mut z48 = L::Reg::zero();

        // Row 5
        let mut z55 = L::Reg::zero();
        let mut z56 = L::Reg::zero();
        let mut z57 = L::Reg::zero();
        let mut z58 = L::Reg::zero();

        // Row 6
        let mut z66 = L::Reg::zero();
        let mut z67 = L::Reg::zero();
        let mut z68 = L::Reg::zero();

        // Row 7
        let mut z77 = L::Reg::zero();
        let mut z78 = L::Reg::zero();

        // Row 8
        let mut z88 = L::Reg::zero();

        for i in 0..blocks {
            let block = unsafe { scaled_column_buf.get_unchecked(i) };

            // 5 permanent registers
            let c4 = L::Reg::load(&block[4]);
            let c5 = L::Reg::load(&block[5]);
            let c6 = L::Reg::load(&block[6]);
            let c7 = L::Reg::load(&block[7]);
            let c8 = L::Reg::load(&block[8]);

            z44 = c4.fma(c4, z44);
            z45 = c4.fma(c5, z45);
            z46 = c4.fma(c6, z46);
            z47 = c4.fma(c7, z47);
            z48 = c4.fma(c8, z48);

            z55 = c5.fma(c5, z55);
            z56 = c5.fma(c6, z56);
            z57 = c5.fma(c7, z57);
            z58 = c5.fma(c8, z58);

            z66 = c6.fma(c6, z66);
            z67 = c6.fma(c7, z67);
            z68 = c6.fma(c8, z68);

            z77 = c7.fma(c7, z77);
            z78 = c7.fma(c8, z78);

            z88 = c8.fma(c8, z88);
        }

        z44 = L::Reg::load(&h[30]).add(z44);
        z45 = L::Reg::load(&h[31]).add(z45);
        z46 = L::Reg::load(&h[32]).add(z46);
        z47 = L::Reg::load(&h[33]).add(z47);
        z48 = L::Reg::load(&h[34]).add(z48);

        // Row 5
        z55 = L::Reg::load(&h[35]).add(z55);
        z56 = L::Reg::load(&h[36]).add(z56);
        z57 = L::Reg::load(&h[37]).add(z57);
        z58 = L::Reg::load(&h[38]).add(z58);

        // Row 6
        z66 = L::Reg::load(&h[39]).add(z66);
        z67 = L::Reg::load(&h[40]).add(z67);
        z68 = L::Reg::load(&h[41]).add(z68);

        // Row 7
        z77 = L::Reg::load(&h[42]).add(z77);
        z78 = L::Reg::load(&h[43]).add(z78);

        // Row 8
        z88 = L::Reg::load(&h[44]).add(z88);

        // Row 4
        z44.store(&mut h[30]);
        z45.store(&mut h[31]);
        z46.store(&mut h[32]);
        z47.store(&mut h[33]);
        z48.store(&mut h[34]);

        // Row 5
        z55.store(&mut h[35]);
        z56.store(&mut h[36]);
        z57.store(&mut h[37]);
        z58.store(&mut h[38]);

        // Row 6
        z66.store(&mut h[39]);
        z67.store(&mut h[40]);
        z68.store(&mut h[41]);

        // Row 7
        z77.store(&mut h[42]);
        z78.store(&mut h[43]);

        // Row 8
        z88.store(&mut h[44]);
    }
}
