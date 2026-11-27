use crate::{
    algebra::Vector,
    lanevector::LaneVector,
    log::Log,
    simd::{Arith, Lane, LoadStore},
};

pub trait AlleleObjective<const A: usize, const G: usize>: Lane {
    fn compute_objective(likelihood_mats: &LaneVector<Self, G>, x: &Vector<A>, eps: f64) -> f64;
}

impl<T: Lane> AlleleObjective<2, 3> for T where T::Reg: Arith + Log {
    #[inline(always)]
    fn compute_objective(likelihood_mats: &LaneVector<T, 3>, x: &Vector<2>, eps: f64) -> f64 {
        compute_obj_generic2(likelihood_mats, x, eps)
    }
}

impl<T: Lane> AlleleObjective<3, 6> for T where T::Reg: Arith + Log {
    #[inline(always)]
    fn compute_objective(likelihood_mats: &LaneVector<T, 6>, x: &Vector<3>, eps: f64) -> f64 {
        compute_obj_generic3(likelihood_mats, x, eps)
    }
}

impl<T: Lane> AlleleObjective<4, 10> for T where T::Reg: Arith + Log {
    #[inline(always)]
    fn compute_objective(likelihood_mats: &LaneVector<T, 10>, x: &Vector<4>, eps: f64) -> f64 {
        compute_obj_generic4(likelihood_mats, x, eps)
    }
}

#[inline(always)]
fn compute_obj_generic2<L: Lane>(
    likelihood_mats: &LaneVector<L, 3>,
    x: &Vector<2>,
    eps: f64,
) -> f64 where L::Reg: Arith + Log {
    let blocks = likelihood_mats.lanes();
    let remainder = likelihood_mats.rem();

    let b = compute_obj_lane2(blocks, x, eps);
    let r = compute_obj_lane2(remainder, x, eps);

    let n = likelihood_mats.len() as f64;

    -(b + r) / n
}

#[inline(always)]
fn compute_obj_generic3<L: Lane>(
    likelihood_mats: &LaneVector<L, 6>,
    x: &Vector<3>,
    eps: f64,
) -> f64 where L::Reg: Arith + Log {
    let blocks = likelihood_mats.lanes();
    let remainder = likelihood_mats.rem();

    let b = compute_obj_lane3(blocks, x, eps);
    let r = compute_obj_lane3(remainder, x, eps);

    let n = likelihood_mats.len() as f64;

    -(b + r) / n
}

#[inline(always)]
fn compute_obj_generic4<L: Lane>(
    likelihood_mats: &LaneVector<L, 10>,
    x: &Vector<4>,
    eps: f64,
) -> f64 where L::Reg: Arith + Log {
    let blocks = likelihood_mats.lanes();
    let remainder = likelihood_mats.rem();

    let b = compute_obj_lane4(blocks, x, eps);
    let r = compute_obj_lane4(remainder, x, eps);

    let n = likelihood_mats.len() as f64;

    -(b + r) / n
}

#[inline(always)]
fn compute_obj_lane2<L: Lane>(blocks: &[[L; 3]], x: &Vector<2>, eps: f64) -> f64 where L::Reg: Arith + Log {
    // 4
    let mut zs = L::Reg::zero();
    let ze = L::Reg::set(eps);

    let x0 = L::Reg::set(x[0]);
    let x1 = L::Reg::set(x[1]);

    // 4
    let x00 = x0.mul(x0);
    let x11 = x1.mul(x1);
    let x01 = x0.mul(x1);
    let tx01 = x01.add(x01);

    for block in blocks.iter() {
        let mut p = ze;

        p = x00.fma(L::Reg::load(&block[0]), p);
        p = tx01.fma(L::Reg::load(&block[1]), p);
        p = x11.fma(L::Reg::load(&block[2]), p);

        let l = Log::log(p);

        zs = l.add(zs);
    }

    zs.radd()
}

#[inline(always)]
fn compute_obj_lane3<L: Lane>(blocks: &[[L; 6]], x: &Vector<3>, eps: f64) -> f64 where L::Reg: Arith + Log {
    // 2
    let mut zs = L::Reg::zero();
    let ze = L::Reg::set(eps);

    let x0 = L::Reg::set(x[0]);
    let x1 = L::Reg::set(x[1]);
    let x2 = L::Reg::set(x[2]);

    // 10
    let x00 = x0.mul(x0);
    let x11 = x1.mul(x1);
    let x22 = x2.mul(x2);

    let x01 = x0.mul(x1);
    let x02 = x0.mul(x2);
    let x12 = x1.mul(x2);

    let tx01 = x01.add(x01);
    let tx02 = x02.add(x02);
    let tx12 = x12.add(x12);

    // No stack spillage, very good
    for block in blocks.iter() {
        let mut p = ze;

        p = x00.fma(L::Reg::load(&block[0]), p);
        p = tx01.fma(L::Reg::load(&block[1]), p);
        p = x11.fma(L::Reg::load(&block[2]), p);
        p = tx02.fma(L::Reg::load(&block[3]), p);
        p = tx12.fma(L::Reg::load(&block[4]), p);
        p = x22.fma(L::Reg::load(&block[5]), p);

        let l = Log::log(p);

        zs = l.add(zs);
    }

    zs.radd()
}

#[inline(always)]
fn compute_obj_lane4<L: Lane>(blocks: &[[L; 10]], x: &Vector<4>, eps: f64) -> f64
where L::Reg: Arith + Log {
    // 2
    let mut zs = L::Reg::zero();
    let ze = L::Reg::set(eps);

    let x0 = L::Reg::set(x[0]);
    let x1 = L::Reg::set(x[1]);
    let x2 = L::Reg::set(x[2]);
    let x3 = L::Reg::set(x[3]);

    // 10
    let x00 = x0.mul(x0);
    let x11 = x1.mul(x1);
    let x22 = x2.mul(x2);
    let x33 = x3.mul(x3);

    let x01 = x0.mul(x1);
    let x02 = x0.mul(x2);
    let x03 = x0.mul(x3);
    let x12 = x1.mul(x2);
    let x13 = x1.mul(x3);
    let x23 = x2.mul(x3);

    let tx01 = x01.add(x01);
    let tx02 = x02.add(x02);
    let tx03 = x03.add(x03);
    let tx12 = x12.add(x12);
    let tx13 = x13.add(x13);
    let tx23 = x23.add(x23);

    // No stack spillage, very good
    for block in blocks.iter() {
        let mut p = ze;

        p = x00.fma(L::Reg::load(&block[0]), p);
        p = tx01.fma(L::Reg::load(&block[1]), p);
        p = x11.fma(L::Reg::load(&block[2]), p);
        p = tx02.fma(L::Reg::load(&block[3]), p);
        p = tx12.fma(L::Reg::load(&block[4]), p);
        p = x22.fma(L::Reg::load(&block[5]), p);
        p = tx03.fma(L::Reg::load(&block[6]), p);
        p = tx13.fma(L::Reg::load(&block[7]), p);
        p = tx23.fma(L::Reg::load(&block[8]), p);
        p = x33.fma(L::Reg::load(&block[9]), p);

        let l = Log::log(p);

        zs = l.add(zs);
    }

    zs.radd()
}
