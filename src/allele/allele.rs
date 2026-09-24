use std::arch::x86_64::*;

use ndarray::{Array2, Array3, Array4};

use crate::{
    algebra::{Matrix, Vector, dot, mul, outer, scale_div}, allele::objective, blockbuffer::Block, lane::Lane8, log::Log, sqp::{self, Tuneables},
};

pub fn calculate_allele_frequencies(likelihoods: &Array3<f64>) -> Array2<f64> {
    let num_variants = likelihoods.shape()[0];
    let num_samples = likelihoods.shape()[1];

    let mut af = Array2::zeros((num_variants, 4));

    let mut multi = 0;
    for (variant, likelihood) in likelihoods.iter().enumerate() {
        let x0 = [0.25; 4];

        let obj = |x: &Vector<4>, eps| objective::compute_objective(&likelihood, &x, eps);
        let grad_hess = |x: &Vector<4>, eps| calculate_grad_hess2(&likelihood, &x, eps);

        let (x, _) = sqp::solve_sqp(obj, grad_hess, &x0, &Tuneables::new());

        if x.iter().filter(|&i| *i > 0.0).count() >= 3 {
            multi += 1;
            // println!("MULTI");
        }

        for i in 0..4 {
            af[[variant, i]] = x[i];
        }

        // println!("{:?}", x);
    }
    println!("Multi {multi}");

    af
}

// pub fn calculate_allele_prob(
//     sample_likelihoods: &Array3<f64>
// ) -> Vector<4> {

// }

fn compute_grad_hess_blocks(blocks: &[[Lane8; 10]], x: &Vector<4>, eps: f64) -> (Vector<4>, Matrix<4>) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return unsafe { compute_grad_hess_avx512(blocks, x, eps) };
    }

    unimplemented!("SIMD intrinsics haven't been written for your platform yet")
}

// Instead of a matrix with 1 elem in each spot
// You have a matrix with 4 elems in each spot
// Need to make a LanedMatrix and LanedVector class
// Then you just manually unroll the accumulator loops
// To simplify life, we could just use the Lane8 class
// And store the leftovers separately. I think that makes sense.
// Then we don't need a custom allocator class anymore.
// In fact we could just have a vector of length 10 to store the
// elements, and yeah, nice
// TODO expand arrays for h and g in other func to variables
// TODO make generic versions of these algorithms? For scalars, etc?

// For big functions you need to manually re-order instructions to
// get the compiler to generate different code (e.g. avoid spillage)
// even though in theory they're the equivalent.

#[target_feature(enable = "avx512f")]
pub fn compute_grad_hess_avx512(blocks: &[[Lane8; 10]], x: &Vector<4>, eps: f64) -> ([f64; 4], [f64; 10]) {
    // 4
    let mut zg0 = _mm512_setzero_pd();
    let mut zg1 = _mm512_setzero_pd();
    let mut zg2 = _mm512_setzero_pd();
    let mut zg3 = _mm512_setzero_pd();

    // 10
    let mut zh0 = _mm512_setzero_pd();
    let mut zh1 = _mm512_setzero_pd();
    let mut zh2 = _mm512_setzero_pd();
    let mut zh3 = _mm512_setzero_pd();
    let mut zh4 = _mm512_setzero_pd();
    let mut zh5 = _mm512_setzero_pd();
    let mut zh6 = _mm512_setzero_pd();
    let mut zh7 = _mm512_setzero_pd();
    let mut zh8 = _mm512_setzero_pd();
    let mut zh9 = _mm512_setzero_pd();

    // 3
    let one = _mm512_set1_pd(1.0);
    let two = _mm512_set1_pd(2.0);
    let ze = _mm512_set1_pd(eps);

    // 4
    let x0 = _mm512_set1_pd(x[0]);
    let x1 = _mm512_set1_pd(x[1]);
    let x2 = _mm512_set1_pd(x[2]);
    let x3 = _mm512_set1_pd(x[3]);

    for block in blocks.iter() {
        let l00 = block[0].load();
        let l01 = block[1].load();
        let l11 = block[2].load();
        let l02 = block[3].load();
        let l12 = block[4].load();
        let l22 = block[5].load();
        let l03 = block[6].load();
        let l13 = block[7].load();
        let l23 = block[8].load();
        let l33 = block[9].load();

        let mut lx0 = _mm512_mul_pd(x0, l00);
        lx0 = _mm512_fmadd_pd(x1, l01, lx0);
        lx0 = _mm512_fmadd_pd(x2, l02, lx0);
        lx0 = _mm512_fmadd_pd(x3, l03, lx0);

        let mut lx1 = _mm512_mul_pd(x0, l01);
        lx1 = _mm512_fmadd_pd(x1, l11, lx1);
        lx1 = _mm512_fmadd_pd(x2, l12, lx1);
        lx1 = _mm512_fmadd_pd(x3, l13, lx1);

        let mut lx2 = _mm512_mul_pd(x0, l02);
        lx2 = _mm512_fmadd_pd(x1, l12, lx2);
        lx2 = _mm512_fmadd_pd(x2, l22, lx2);
        lx2 = _mm512_fmadd_pd(x3, l23, lx2);

        let mut lx3 = _mm512_mul_pd(x0, l03);
        lx3 = _mm512_fmadd_pd(x1, l13, lx3);
        lx3 = _mm512_fmadd_pd(x2, l23, lx3);
        lx3 = _mm512_fmadd_pd(x3, l33, lx3);

        let mut d = ze;
        d = _mm512_fmadd_pd(x0, lx0, d);
        d = _mm512_fmadd_pd(x1, lx1, d);
        d = _mm512_fmadd_pd(x2, lx2, d);
        d = _mm512_fmadd_pd(x3, lx3, d);

        let inv_d = _mm512_div_pd(one, d);

        let lxd0 = _mm512_mul_pd(lx0, inv_d);
        let lxd1 = _mm512_mul_pd(lx1, inv_d);
        let lxd2 = _mm512_mul_pd(lx2, inv_d);
        let lxd3 = _mm512_mul_pd(lx3, inv_d);

        let lxd00 = _mm512_mul_pd(lxd0, lxd0);
        zh0 = _mm512_fmadd_pd(two, lxd00, zh0);
        zh0 = _mm512_fnmadd_pd(l00, inv_d, zh0);
        let lxd01 = _mm512_mul_pd(lxd0, lxd1);
        zh1 = _mm512_fmadd_pd(two, lxd01, zh1);
        zh1 = _mm512_fnmadd_pd(l01, inv_d, zh1);
        let lxd11 = _mm512_mul_pd(lxd1, lxd1);
        zh2 = _mm512_fmadd_pd(two, lxd11, zh2);
        zh2 = _mm512_fnmadd_pd(l11, inv_d, zh2);
        let lxd02 = _mm512_mul_pd(lxd0, lxd2);
        zh3 = _mm512_fmadd_pd(two, lxd02, zh3);
        zh3 = _mm512_fnmadd_pd(l02, inv_d, zh3);
        let lxd12 = _mm512_mul_pd(lxd1, lxd2);
        zh4 = _mm512_fmadd_pd(two, lxd12, zh4);
        zh4 = _mm512_fnmadd_pd(l12, inv_d, zh4);
        let lxd22 = _mm512_mul_pd(lxd2, lxd2);
        zh5 = _mm512_fmadd_pd(two, lxd22, zh5);
        zh5 = _mm512_fnmadd_pd(l22, inv_d, zh5);
        let lxd03 = _mm512_mul_pd(lxd0, lxd3);
        zh6 = _mm512_fmadd_pd(two, lxd03, zh6);
        zh6 = _mm512_fnmadd_pd(l03, inv_d, zh6);
        let lxd13 = _mm512_mul_pd(lxd1, lxd3);
        zh7 = _mm512_fmadd_pd(two, lxd13, zh7);
        zh7 = _mm512_fnmadd_pd(l13, inv_d, zh7);
        let lxd23 = _mm512_mul_pd(lxd2, lxd3);
        zh8 = _mm512_fmadd_pd(two, lxd23, zh8);
        zh8 = _mm512_fnmadd_pd(l23, inv_d, zh8);
        let lxd33 = _mm512_mul_pd(lxd3, lxd3);
        zh9 = _mm512_fmadd_pd(two, lxd33, zh9);
        zh9 = _mm512_fnmadd_pd(l33, inv_d, zh9);

        zg0 = _mm512_fmadd_pd(lx0, inv_d, zg0);
        zg1 = _mm512_fmadd_pd(lx1, inv_d, zg1);
        zg2 = _mm512_fmadd_pd(lx2, inv_d, zg2);
        zg3 = _mm512_fmadd_pd(lx3, inv_d, zg3);
    }

    let g = [
        _mm512_reduce_add_pd(zg0),
        _mm512_reduce_add_pd(zg1),
        _mm512_reduce_add_pd(zg2),
        _mm512_reduce_add_pd(zg3),
    ];

    let h = [
        _mm512_reduce_add_pd(zh0),
        _mm512_reduce_add_pd(zh1),
        _mm512_reduce_add_pd(zh2),
        _mm512_reduce_add_pd(zh3),
        _mm512_reduce_add_pd(zh4),
        _mm512_reduce_add_pd(zh5),
        _mm512_reduce_add_pd(zh6),
        _mm512_reduce_add_pd(zh7),
        _mm512_reduce_add_pd(zh8),
        _mm512_reduce_add_pd(zh9),
    ];

    (g, h)
}


pub fn calculate_grad_hess2(likelihoods: &[Matrix<4>], x: &Vector<4>, eps: f64) -> (Vector<4>, Matrix<4>) {
    let mut g = [0.0; 4];
    let mut h = [[0.0; 4]; 4];

    for l in likelihoods.iter() {
        let lx = mul(l, x);
        let d = dot(x, &lx) + eps;

        let lxd = scale_div(d, &lx);

        for i in 0..4 {
            g[i] += lxd[i];
        }

        for i in 0..4 {
            for j in 0..4 {
                h[i][j] += 2.0 * lxd[i] * lxd[j] - l[i][j] / d;
            }
        }
    }

    let n = likelihoods.len();

    let grad = std::array::from_fn(|i| -2.0 * g[i] / (n as f64));

    let mut hess = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            hess[i][j] = 2.0 * h[i][j] / (n as f64);
        }
    }

    (grad, hess)
}
