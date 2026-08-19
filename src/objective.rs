use crate::{
    algebra::{Vector, dot},
    blockbuffer::{Block, BlockBuffer},
    log::Log,
};
use core::arch::x86_64::*;

pub fn compute_obj(p_mat: &BlockBuffer<f64, 8, 9>, x: &Vector<9>, eps: f64) -> f64 {
    let (blocks, remainder) = p_mat.as_blocks();

    let b = compute_obj_blocks(blocks, x, eps);

    let r = compute_obj_remainder(remainder, x, eps);

    let n = p_mat.num_rows();

    return -(b + r) / (n as f64);
}

fn compute_obj_blocks(blocks: &[Block<f64, 8, 9>], x: &Vector<9>, eps: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
        return unsafe { compute_blocks_avx512(blocks, x, eps) };
    }

    compute_blocks_scalar(blocks, x, eps)
}

fn compute_obj_remainder(remainder: &[Vector<9>], x: &Vector<9>, eps: f64) -> f64 {
    let mut s = 0.0;
    for row in remainder.iter() {
        let prod = dot(row, x);
        let t = Log::log(prod + eps);
        s += t;
    }
    s
}

fn compute_blocks_scalar<const L: usize>(blocks: &[Block<f64, L, 9>], x: &[f64; 9], eps: f64) -> f64 {
    let mut s = 0.0;
    for block in blocks.iter() {
        for i in 0..L {
            let mut row = [0.0; 9];
            for j in 0..9 {
                row[j] = block[j][i];
            }
            let prod = dot(&row, x);
            let t = Log::log(prod + eps);
            // TODO kahan?
            s += t;
        }
    }

    s
}

#[target_feature(enable = "avx512f,avx512dq")]
pub fn compute_blocks_avx512(blocks: &[Block<f64, 8, 9>], x: &[f64; 9], eps: f64) -> f64 {
    // 9
    let zx: [__m512d; 9] = std::array::from_fn(|i| _mm512_set1_pd(x[i]));

    // 12
    let mut zs = _mm512_setzero_pd();
    let mut zc = _mm512_setzero_pd();
    let ze = _mm512_set1_pd(eps);

    // This still loads 3 constants using 1to8...
    for block in blocks.iter() {
        // This computes a dot product between x and a row of p
        // In theory this could be manually unrolled a few times
        // But in practice that's slower.

        // 13
        let mut d = ze;
        for col in 0..9 {
            let c = unsafe { _mm512_load_pd(block[col].as_ptr()) };
            d = _mm512_fmadd_pd(zx[col], c, d);
        }

        let l = Log::log(d);

        // Kahan summation
        let y = _mm512_sub_pd(l, zc);
        let t = _mm512_add_pd(zs, y);
        let b = _mm512_sub_pd(t, zs);
        zc = _mm512_sub_pd(b, y);
        zs = t;
    }

    // Perhaps do something with zc too...
    _mm512_reduce_add_pd(zs)
}

// MUST look at assembly. Essential!
// Making a function call requires spilling registers so they can be restored after.

#[target_feature(enable = "avx512f,avx512dq")]
pub fn compute_obj_2avx512(p_mat: &BlockBuffer<f64, 8, 9>, x: &[f64; 9], y: &[f64; 9], eps: f64) -> (f64, f64) {
    // 18
    // let zx: [__m512d; 9] = std::array::from_fn(|i| _mm512_set1_pd(x[i]));
    // let zy: [__m512d; 9] = std::array::from_fn(|i| _mm512_set1_pd(y[i]));

    let mut zxs = _mm512_setzero_pd();
    let mut zys = _mm512_setzero_pd();

    let (blocks, remainder) = p_mat.as_blocks();

    for block in blocks.iter() {
        // let c: [__m512d; 9] =
        // std::array::from_fn(|i| unsafe { _mm512_load_pd(block[i].as_ptr()) });
        // Calculate d
        // This computes a dot product between x and a row of p
        // TODO this could be manually unrolled a few times
        let mut dx = _mm512_set1_pd(eps);
        let mut dy = _mm512_set1_pd(eps);
        for col in 0..9 {
            let zx = _mm512_set1_pd(x[col]);
            let zy = _mm512_set1_pd(y[col]);
            let c = unsafe { _mm512_load_pd(block[col].as_ptr()) };
            dx = _mm512_fmadd_pd(zx, c, dx);
            dy = _mm512_fmadd_pd(zy, c, dy);

            // dx = _mm512_fmadd_pd(zx[col], c, dx);
            // dy = _mm512_fmadd_pd(zy[col], c, dy);
        }

        let lx = dx.log();
        let ly = dy.log();

        // TODO kahan
        zxs = _mm512_add_pd(lx, zxs);
        zys = _mm512_add_pd(ly, zys);
    }

    let mut xs = _mm512_reduce_add_pd(zxs);
    let mut ys = _mm512_reduce_add_pd(zys);

    xs += compute_obj_remainder(remainder, x, eps);
    ys += compute_obj_remainder(remainder, y, eps);

    (xs, ys)
}
