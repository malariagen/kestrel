use crate::{
    algebra::{Matrix, Vector},
    lanevector::LaneVector,
    simd::{Arith, Lane, LoadStore},
};

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::Avx2Lane;
#[cfg(target_arch = "x86_64")]
use crate::simd::avx512::Avx512Lane;
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::NeonLane;

pub trait AlleleGradHess<const A: usize, const G: usize>: Lane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, G>, x: &Vector<A>, eps: f64) -> (Vector<A>, Matrix<A>);
}

// ==========================================
// AVX-512
// ==========================================

#[cfg(target_arch = "x86_64")]
impl AlleleGradHess<2, 3> for Avx512Lane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 3>, x: &Vector<2>, eps: f64) -> (Vector<2>, Matrix<2>) {
        use std::arch::x86_64::__m512d;
        compute_grad_hess_generic2::<Self, __m512d>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "x86_64")]
impl AlleleGradHess<3, 6> for Avx512Lane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 6>, x: &Vector<3>, eps: f64) -> (Vector<3>, Matrix<3>) {
        use std::arch::x86_64::__m512d;
        compute_grad_hess_generic3::<Self, __m512d>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "x86_64")]
impl AlleleGradHess<4, 10> for Avx512Lane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 10>, x: &Vector<4>, eps: f64) -> (Vector<4>, Matrix<4>) {
        use std::arch::x86_64::__m512d;
        compute_grad_hess_generic4::<Self, __m512d>(likelihood_mats, x, eps)
    }
}

// ==========================================
// AVX2
// ==========================================

#[cfg(target_arch = "x86_64")]
impl AlleleGradHess<2, 3> for Avx2Lane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 3>, x: &Vector<2>, eps: f64) -> (Vector<2>, Matrix<2>) {
        use std::arch::x86_64::__m256d;
        compute_grad_hess_generic2::<Self, __m256d>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "x86_64")]
impl AlleleGradHess<3, 6> for Avx2Lane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 6>, x: &Vector<3>, eps: f64) -> (Vector<3>, Matrix<3>) {
        use std::arch::x86_64::__m256d;
        compute_grad_hess_generic3::<Self, __m256d>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "x86_64")]
impl AlleleGradHess<4, 10> for Avx2Lane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 10>, x: &Vector<4>, eps: f64) -> (Vector<4>, Matrix<4>) {
        use std::arch::x86_64::__m256d;
        compute_grad_hess_generic4::<Self, __m256d>(likelihood_mats, x, eps)
    }
}

// ==========================================
// Neon
// ==========================================

#[cfg(target_arch = "aarch64")]
impl AlleleGradHess<2, 3> for NeonLane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 3>, x: &Vector<2>, eps: f64) -> (Vector<2>, Matrix<2>) {
        use std::arch::aarch64::float64x2_t;
        compute_grad_hess_generic2::<Self, float64x2_t>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "aarch64")]
impl AlleleGradHess<3, 6> for NeonLane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 6>, x: &Vector<3>, eps: f64) -> (Vector<3>, Matrix<3>) {
        use std::arch::aarch64::float64x2_t;
        compute_grad_hess_generic3::<Self, float64x2_t>(likelihood_mats, x, eps)
    }
}

#[cfg(target_arch = "aarch64")]
impl AlleleGradHess<4, 10> for NeonLane {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 10>, x: &Vector<4>, eps: f64) -> (Vector<4>, Matrix<4>) {
        use std::arch::aarch64::float64x2_t;
        compute_grad_hess_generic4::<Self, float64x2_t>(likelihood_mats, x, eps)
    }
}

// ==========================================
// Scalar
// ==========================================

impl AlleleGradHess<2, 3> for f64 {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 3>, x: &Vector<2>, eps: f64) -> (Vector<2>, Matrix<2>) {
        compute_grad_hess_generic2::<Self, f64>(likelihood_mats, x, eps)
    }
}

impl AlleleGradHess<3, 6> for f64 {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 6>, x: &Vector<3>, eps: f64) -> (Vector<3>, Matrix<3>) {
        compute_grad_hess_generic3::<Self, f64>(likelihood_mats, x, eps)
    }
}

impl AlleleGradHess<4, 10> for f64 {
    fn compute_grad_hess(likelihood_mats: &LaneVector<Self, 10>, x: &Vector<4>, eps: f64) -> (Vector<4>, Matrix<4>) {
        compute_grad_hess_generic4::<Self, f64>(likelihood_mats, x, eps)
    }
}

fn compute_grad_hess_generic2<L: Lane, S: Arith + LoadStore<L>>(
    likelihood_mats: &LaneVector<L, 3>,
    x: &Vector<2>,
    eps: f64,
) -> (Vector<2>, Matrix<2>) {
    let (bg, bh) = compute_grad_hess_lane2::<L, S>(likelihood_mats.lanes(), x, eps);
    let (rg, rh) = compute_grad_hess_lane2::<f64, f64>(likelihood_mats.rem(), x, eps);

    let n = likelihood_mats.len() as f64;

    let (g, h) = build_grad_hess(bg, bh, rg, rh, n);

    (g, h)
}

fn compute_grad_hess_generic3<L: Lane, S: Arith + LoadStore<L>>(
    likelihood_mats: &LaneVector<L, 6>,
    x: &Vector<3>,
    eps: f64,
) -> (Vector<3>, Matrix<3>) {
    let (bg, bh) = compute_grad_hess_lane3::<L, S>(likelihood_mats.lanes(), x, eps);
    let (rg, rh) = compute_grad_hess_lane3::<f64, f64>(likelihood_mats.rem(), x, eps);

    let n = likelihood_mats.len() as f64;

    let (g, h) = build_grad_hess(bg, bh, rg, rh, n);

    (g, h)
}

fn compute_grad_hess_generic4<L: Lane, S: Arith + LoadStore<L>>(
    likelihood_mats: &LaneVector<L, 10>,
    x: &Vector<4>,
    eps: f64,
) -> (Vector<4>, Matrix<4>) {
    let (bg, bh) = compute_grad_hess_lane4::<L, S>(likelihood_mats.lanes(), x, eps);
    let (rg, rh) = compute_grad_hess_lane4::<f64, f64>(likelihood_mats.rem(), x, eps);

    let n = likelihood_mats.len() as f64;

    let (g, h) = build_grad_hess(bg, bh, rg, rh, n);

    (g, h)
}

fn build_grad_hess<const A: usize, const G: usize>(
    bg: [f64; A],
    bh: [f64; G],
    rg: [f64; A],
    rh: [f64; G],
    n: f64,
) -> ([f64; A], [[f64; A]; A]) {
    let mut g = [0.0f64; A];
    let mut h = [[0.0f64; A]; A];

    for i in 0..A {
        g[i] = -2.0 * (bg[i] + rg[i]) / n;
    }

    for j in 0..A {
        for i in 0..=j {
            // The index of (i, j) where i <= j (see the VCF spec)
            let index = j * (j + 1) / 2 + i;
            let val = 2.0 * (bh[index] + rh[index]) / n;
            h[i][j] = val;
            h[j][i] = val;
        }
    }
    (g, h)
}

fn compute_grad_hess_lane2<L: Lane, S: Arith + LoadStore<L>>(
    blocks: &[[L; 3]],
    x: &Vector<2>,
    eps: f64,
) -> ([f64; 2], [f64; 3]) {
    // 4
    let mut zg0 = S::zero();
    let mut zg1 = S::zero();

    // 10
    let mut zh0 = S::zero();
    let mut zh1 = S::zero();
    let mut zh2 = S::zero();

    // 3
    let one = S::set(1.0);
    let two = S::set(2.0);
    let ze = S::set(eps);

    // 4
    let x0 = S::set(x[0]);
    let x1 = S::set(x[1]);

    for block in blocks.iter() {
        let l00 = S::load(&block[0]);
        let l01 = S::load(&block[1]);
        let l11 = S::load(&block[2]);

        let mut lx0 = x0.mul(l00);
        lx0 = x1.fma(l01, lx0);

        let mut lx1 = x0.mul(l01);
        lx1 = x1.fma(l11, lx1);

        let mut d = ze;
        d = x0.fma(lx0, d);
        d = x1.fma(lx1, d);

        let inv_d = one.div(d);

        let lxd0 = lx0.mul(inv_d);
        let lxd1 = lx1.mul(inv_d);

        let lxd00 = lxd0.mul(lxd0);
        zh0 = two.fma(lxd00, zh0);
        zh0 = l00.fnma(inv_d, zh0);
        let lxd01 = lxd0.mul(lxd1);
        zh1 = two.fma(lxd01, zh1);
        zh1 = l01.fnma(inv_d, zh1);
        let lxd11 = lxd1.mul(lxd1);
        zh2 = two.fma(lxd11, zh2);
        zh2 = l11.fnma(inv_d, zh2);

        zg0 = lx0.fma(inv_d, zg0);
        zg1 = lx1.fma(inv_d, zg1);
    }

    let g = [zg0.radd(), zg1.radd()];

    let h = [zh0.radd(), zh1.radd(), zh2.radd()];

    (g, h)
}

fn compute_grad_hess_lane3<L: Lane, S: Arith + LoadStore<L>>(
    blocks: &[[L; 6]],
    x: &Vector<3>,
    eps: f64,
) -> ([f64; 3], [f64; 6]) {
    // 4
    let mut zg0 = S::zero();
    let mut zg1 = S::zero();
    let mut zg2 = S::zero();

    // 10
    let mut zh0 = S::zero();
    let mut zh1 = S::zero();
    let mut zh2 = S::zero();
    let mut zh3 = S::zero();
    let mut zh4 = S::zero();
    let mut zh5 = S::zero();

    // 3
    let one = S::set(1.0);
    let two = S::set(2.0);
    let ze = S::set(eps);

    // 4
    let x0 = S::set(x[0]);
    let x1 = S::set(x[1]);
    let x2 = S::set(x[2]);

    for block in blocks.iter() {
        let l00 = S::load(&block[0]);
        let l01 = S::load(&block[1]);
        let l11 = S::load(&block[2]);
        let l02 = S::load(&block[3]);
        let l12 = S::load(&block[4]);
        let l22 = S::load(&block[5]);

        let mut lx0 = x0.mul(l00);
        lx0 = x1.fma(l01, lx0);
        lx0 = x2.fma(l02, lx0);

        let mut lx1 = x0.mul(l01);
        lx1 = x1.fma(l11, lx1);
        lx1 = x2.fma(l12, lx1);

        let mut lx2 = x0.mul(l02);
        lx2 = x1.fma(l12, lx2);
        lx2 = x2.fma(l22, lx2);

        let mut d = ze;
        d = x0.fma(lx0, d);
        d = x1.fma(lx1, d);
        d = x2.fma(lx2, d);

        let inv_d = one.div(d);

        let lxd0 = lx0.mul(inv_d);
        let lxd1 = lx1.mul(inv_d);
        let lxd2 = lx2.mul(inv_d);

        let lxd00 = lxd0.mul(lxd0);
        zh0 = two.fma(lxd00, zh0);
        zh0 = l00.fnma(inv_d, zh0);
        let lxd01 = lxd0.mul(lxd1);
        zh1 = two.fma(lxd01, zh1);
        zh1 = l01.fnma(inv_d, zh1);
        let lxd11 = lxd1.mul(lxd1);
        zh2 = two.fma(lxd11, zh2);
        zh2 = l11.fnma(inv_d, zh2);
        let lxd02 = lxd0.mul(lxd2);
        zh3 = two.fma(lxd02, zh3);
        zh3 = l02.fnma(inv_d, zh3);
        let lxd12 = lxd1.mul(lxd2);
        zh4 = two.fma(lxd12, zh4);
        zh4 = l12.fnma(inv_d, zh4);
        let lxd22 = lxd2.mul(lxd2);
        zh5 = two.fma(lxd22, zh5);
        zh5 = l22.fnma(inv_d, zh5);

        zg0 = lx0.fma(inv_d, zg0);
        zg1 = lx1.fma(inv_d, zg1);
        zg2 = lx2.fma(inv_d, zg2);
    }

    let g = [zg0.radd(), zg1.radd(), zg2.radd()];

    let h = [zh0.radd(), zh1.radd(), zh2.radd(), zh3.radd(), zh4.radd(), zh5.radd()];

    (g, h)
}

fn compute_grad_hess_lane4<L: Lane, S: Arith + LoadStore<L>>(
    blocks: &[[L; 10]],
    x: &Vector<4>,
    eps: f64,
) -> ([f64; 4], [f64; 10]) {
    // 4
    let mut zg0 = S::zero();
    let mut zg1 = S::zero();
    let mut zg2 = S::zero();
    let mut zg3 = S::zero();

    // 10
    let mut zh0 = S::zero();
    let mut zh1 = S::zero();
    let mut zh2 = S::zero();
    let mut zh3 = S::zero();
    let mut zh4 = S::zero();
    let mut zh5 = S::zero();
    let mut zh6 = S::zero();
    let mut zh7 = S::zero();
    let mut zh8 = S::zero();
    let mut zh9 = S::zero();

    // 3
    let one = S::set(1.0);
    let two = S::set(2.0);
    let ze = S::set(eps);

    // 4
    let x0 = S::set(x[0]);
    let x1 = S::set(x[1]);
    let x2 = S::set(x[2]);
    let x3 = S::set(x[3]);

    for block in blocks.iter() {
        let l00 = S::load(&block[0]);
        let l01 = S::load(&block[1]);
        let l11 = S::load(&block[2]);
        let l02 = S::load(&block[3]);
        let l12 = S::load(&block[4]);
        let l22 = S::load(&block[5]);
        let l03 = S::load(&block[6]);
        let l13 = S::load(&block[7]);
        let l23 = S::load(&block[8]);
        let l33 = S::load(&block[9]);

        let mut lx0 = x0.mul(l00);
        lx0 = x1.fma(l01, lx0);
        lx0 = x2.fma(l02, lx0);
        lx0 = x3.fma(l03, lx0);

        let mut lx1 = x0.mul(l01);
        lx1 = x1.fma(l11, lx1);
        lx1 = x2.fma(l12, lx1);
        lx1 = x3.fma(l13, lx1);

        let mut lx2 = x0.mul(l02);
        lx2 = x1.fma(l12, lx2);
        lx2 = x2.fma(l22, lx2);
        lx2 = x3.fma(l23, lx2);

        let mut lx3 = x0.mul(l03);
        lx3 = x1.fma(l13, lx3);
        lx3 = x2.fma(l23, lx3);
        lx3 = x3.fma(l33, lx3);

        let mut d = ze;
        d = x0.fma(lx0, d);
        d = x1.fma(lx1, d);
        d = x2.fma(lx2, d);
        d = x3.fma(lx3, d);

        let inv_d = one.div(d);

        let lxd0 = lx0.mul(inv_d);
        let lxd1 = lx1.mul(inv_d);
        let lxd2 = lx2.mul(inv_d);
        let lxd3 = lx3.mul(inv_d);

        let lxd00 = lxd0.mul(lxd0);
        zh0 = two.fma(lxd00, zh0);
        zh0 = l00.fnma(inv_d, zh0);
        let lxd01 = lxd0.mul(lxd1);
        zh1 = two.fma(lxd01, zh1);
        zh1 = l01.fnma(inv_d, zh1);
        let lxd11 = lxd1.mul(lxd1);
        zh2 = two.fma(lxd11, zh2);
        zh2 = l11.fnma(inv_d, zh2);
        let lxd02 = lxd0.mul(lxd2);
        zh3 = two.fma(lxd02, zh3);
        zh3 = l02.fnma(inv_d, zh3);
        let lxd12 = lxd1.mul(lxd2);
        zh4 = two.fma(lxd12, zh4);
        zh4 = l12.fnma(inv_d, zh4);
        let lxd22 = lxd2.mul(lxd2);
        zh5 = two.fma(lxd22, zh5);
        zh5 = l22.fnma(inv_d, zh5);
        let lxd03 = lxd0.mul(lxd3);
        zh6 = two.fma(lxd03, zh6);
        zh6 = l03.fnma(inv_d, zh6);
        let lxd13 = lxd1.mul(lxd3);
        zh7 = two.fma(lxd13, zh7);
        zh7 = l13.fnma(inv_d, zh7);
        let lxd23 = lxd2.mul(lxd3);
        zh8 = two.fma(lxd23, zh8);
        zh8 = l23.fnma(inv_d, zh8);
        let lxd33 = lxd3.mul(lxd3);
        zh9 = two.fma(lxd33, zh9);
        zh9 = l33.fnma(inv_d, zh9);

        zg0 = lx0.fma(inv_d, zg0);
        zg1 = lx1.fma(inv_d, zg1);
        zg2 = lx2.fma(inv_d, zg2);
        zg3 = lx3.fma(inv_d, zg3);
    }

    let g = [zg0.radd(), zg1.radd(), zg2.radd(), zg3.radd()];

    let h = [
        zh0.radd(),
        zh1.radd(),
        zh2.radd(),
        zh3.radd(),
        zh4.radd(),
        zh5.radd(),
        zh6.radd(),
        zh7.radd(),
        zh8.radd(),
        zh9.radd(),
    ];

    (g, h)
}
