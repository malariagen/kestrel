use crate::simd::Arith;

type IisTable = [[[[[[Option<Iis>; 2]; 2]; 2]; 2]; 2]; 2];

const IIS_TABLE: IisTable = generate_iis_lookup_table();

#[derive(Clone, Copy, Debug)]
pub enum Iis {
    M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15
}

const fn generate_iis_lookup_table() -> IisTable {
    // For 4 items, there are 4 choose 2 = 6 different equality combinations
    // between items. Each combination has 2 outcomes, so there are 2^6 = 64
    // possible outcomes. However, only 15 of the possibilities define valid
    // equivalence relations.
    let mut table = [[[[[[None; 2]; 2]; 2]; 2]; 2]; 2];

    // 1
    // all equal
    // ii-ii
    table[1][1][1][1][1][1] = Some(Iis::M1);

    // 2
    // i == j, k == l
    // ii-kk
    table[1][0][0][0][0][1] = Some(Iis::M2);

    // 3
    // i == j, i == k, j == k
    // ii-il
    table[1][1][0][1][0][0] = Some(Iis::M3);
    // i == j, i == l, j == l
    // ii-ki
    table[1][0][1][0][1][0] = Some(Iis::M4);

    // 4
    // i == j
    // ii-kl
    table[1][0][0][0][0][0] = Some(Iis::M5);

    // 5
    // i == k, i == l, k == l
    // ij-ii
    table[0][1][1][0][0][1] = Some(Iis::M6);
    // j == k, j == l, k == l
    // ij-jj
    table[0][0][0][1][1][1] = Some(Iis::M7);

    // 6
    // k == l
    // ij-kk
    table[0][0][0][0][0][1] = Some(Iis::M8);

    // 7
    // i == k, j == l
    // ij-ij
    table[0][1][0][0][1][0] = Some(Iis::M9);
    // i == l, j == k
    // ij-ji
    table[0][0][1][1][0][0] = Some(Iis::M10);

    // 8
    // i == k, ij-il
    table[0][1][0][0][0][0] = Some(Iis::M11);
    // i == l, ij-ki
    table[0][0][1][0][0][0] = Some(Iis::M12);
    // j == k, ij-jl
    table[0][0][0][1][0][0] = Some(Iis::M13);
    // j == l, ij-kj
    table[0][0][0][0][1][0] = Some(Iis::M14);

    // 9
    // none equal, ij-kl
    table[0][0][0][0][0][0] = Some(Iis::M15);

    table
}

pub const fn calc_iis_mode(i: usize, j: usize, k: usize, l: usize) -> Iis {
    // usize::from isn't const, so cast manually
    let c1 = (i == j) as usize;
    let c2 = (i == k) as usize;
    let c3 = (i == l) as usize;
    let c4 = (j == k) as usize;
    let c5 = (j == l) as usize;
    let c6 = (k == l) as usize;

    IIS_TABLE[c1][c2][c3][c4][c5][c6].unwrap()
}

#[inline]
pub fn accumulate_iis1<S: Arith>(a: &mut [S; 9], g: S, pi: S, _pj: S, _pk: S, _pl: S) {
    let pi2 = pi.mul(pi);
    let pi3 = pi2.mul(pi);

    a[0] = g.fma(pi, a[0]);
    a[1] = g.fma(pi2, a[1]);
    a[2] = g.fma(pi2, a[2]);
    a[3] = g.fma(pi3, a[3]);
    a[4] = g.fma(pi2, a[4]);
    a[5] = g.fma(pi3, a[5]);
    a[6] = g.fma(pi2, a[6]);
    a[7] = g.fma(pi3, a[7]);
    a[8] = g.fma(pi2.mul(pi2), a[8]);
}

#[inline]
pub fn accumulate_iis2<S: Arith>(a: &mut [S; 9], g: S, pi: S, _pj: S, pk: S, _pl: S) {
    let pipk = pi.mul(pk);

    a[1] = g.fma(pipk, a[1]);
    a[3] = g.fma(pipk.mul(pk), a[3]);
    a[5] = g.fma(pi.mul(pipk), a[5]);
    a[8] = g.fma(pipk.mul(pipk), a[8]);
}

#[inline]
pub fn accumulate_iis3<S: Arith>(a: &mut [S; 9], g: S, pi: S, _pj: S, _pk: S, pl: S) {
    let pipl = pi.mul(pl);
    let pi2pl = pi.mul(pipl);
    let twopi2pl = pi2pl.add(pi2pl);
    let twopi3pl = twopi2pl.mul(pi);

    a[2] = g.fma(pipl, a[2]);
    a[3] = g.fma(twopi2pl, a[3]);
    a[7] = g.fma(pi2pl, a[7]);
    a[8] = g.fma(twopi3pl, a[8]);
}

#[inline]
pub fn accumulate_iis4<S: Arith>(a: &mut [S; 9], g: S, pi: S, _pj: S, pk: S, _pl: S) {
    let pipk = pi.mul(pk);
    let pi2pk = pi.mul(pipk);
    let twopi2pk = pi2pk.add(pi2pk);
    let twopi3pk = twopi2pk.mul(pi);

    a[2] = g.fma(pipk, a[2]);
    a[3] = g.fma(twopi2pk, a[3]);
    a[7] = g.fma(pi2pk, a[7]);
    a[8] = g.fma(twopi3pk, a[8]);
}

#[inline]
pub fn accumulate_iis5<S: Arith>(a: &mut [S; 9], g: S, pi: S, _pj: S, pk: S, pl: S) {
    let pipkpl = pi.mul(pk).mul(pl);
    let twopipkpl = pipkpl.add(pipkpl);

    a[3] = g.fma(twopipkpl, a[3]);
    a[8] = g.fma(twopipkpl.mul(pi), a[8]);
}

#[inline]
pub fn accumulate_iis6<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, _pk: S, _pl: S) {
    let pipj = pi.mul(pj);
    let pi2pj = pi.mul(pipj);
    let twopi2pj = pi2pj.add(pi2pj);

    a[4] = g.fma(pipj, a[4]);
    a[5] = g.fma(twopi2pj, a[5]);
    a[7] = g.fma(pi2pj, a[7]);
    a[8] = g.fma(twopi2pj.mul(pi), a[8]);
}

#[inline]
pub fn accumulate_iis7<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, _pk: S, _pl: S) {
    let pipj = pi.mul(pj);
    let pj2pi = pj.mul(pipj);
    let twopj2pi = pj2pi.add(pj2pi);

    a[4] = g.fma(pipj, a[4]);
    a[5] = g.fma(twopj2pi, a[5]);
    a[7] = g.fma(pj2pi, a[7]);
    a[8] = g.fma(twopj2pi.mul(pj), a[8]);
}

#[inline]
pub fn accumulate_iis8<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, pk: S, _pl: S) {
    let pipjpk = pi.mul(pj).mul(pk);
    let twopipjpk = pipjpk.add(pipjpk);

    a[5] = g.fma(twopipjpk, a[5]);
    a[8] = g.fma(twopipjpk.mul(pk), a[8]);
}

#[inline]
pub fn accumulate_iis9<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, _pk: S, _pl: S) {
    let pipj = pi.mul(pj);
    let twopipj = pipj.add(pipj);
    let fourpi2pj2 = twopipj.mul(twopipj);

    a[6] = g.fma(twopipj, a[6]);
    a[7] = g.fma(pipj.mul(pi.add(pj)), a[7]);
    a[8] = g.fma(fourpi2pj2, a[8]);
}

#[inline]
pub fn accumulate_iis10<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, pk: S, pl: S) {
    // Mode 10 (ij-ji) is mathematically identical to Mode 9 (ij-ij)
    accumulate_iis9(a, g, pi, pj, pk, pl);
}

#[inline]
pub fn accumulate_iis11<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, _pk: S, pl: S) {
    let pipjpl = pi.mul(pj).mul(pl);

    a[7] = g.fma(pipjpl, a[7]);
    a[8] = g.fma(S::set(4.0).mul(pipjpl).mul(pi), a[8]);
}

#[inline]
pub fn accumulate_iis12<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, pk: S, _pl: S) {
    let pipjpk = pi.mul(pj).mul(pk);

    a[7] = g.fma(pipjpk, a[7]);
    a[8] = g.fma(S::set(4.0).mul(pipjpk).mul(pi), a[8]);
}

#[inline]
pub fn accumulate_iis13<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, _pk: S, pl: S) {
    let pipjpl = pi.mul(pj).mul(pl);

    a[7] = g.fma(pipjpl, a[7]);
    a[8] = g.fma(S::set(4.0).mul(pipjpl).mul(pj), a[8]);
}

#[inline]
pub fn accumulate_iis14<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, pk: S, _pl: S) {
    let pipjpk = pi.mul(pj).mul(pk);

    a[7] = g.fma(pipjpk, a[7]);
    a[8] = g.fma(S::set(4.0).mul(pipjpk).mul(pj), a[8]);
}

#[inline]
pub fn accumulate_iis15<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, pk: S, pl: S) {
    let pipj = pi.mul(pj);
    let pkpl = pk.mul(pl);
    let pipjpkpl = pipj.mul(pkpl);

    a[8] = g.fma(S::set(4.0).mul(pipjpkpl), a[8]);
}

pub fn accumulate_iis<S: Arith>(a: &mut [S; 9], g: S, pi: S, pj: S, pk: S, pl: S, iis: Iis) {
    match iis {
        Iis::M1 => accumulate_iis1(a, g, pi, pj, pk, pl),
        Iis::M2 => accumulate_iis2(a, g, pi, pj, pk, pl),
        Iis::M3 => accumulate_iis3(a, g, pi, pj, pk, pl),
        Iis::M4 => accumulate_iis4(a, g, pi, pj, pk, pl),
        Iis::M5 => accumulate_iis5(a, g, pi, pj, pk, pl),
        Iis::M6 => accumulate_iis6(a, g, pi, pj, pk, pl),
        Iis::M7 => accumulate_iis7(a, g, pi, pj, pk, pl),
        Iis::M8 => accumulate_iis8(a, g, pi, pj, pk, pl),
        Iis::M9 => accumulate_iis9(a, g, pi, pj, pk, pl),
        Iis::M10 => accumulate_iis10(a, g, pi, pj, pk, pl),
        Iis::M11 => accumulate_iis11(a, g, pi, pj, pk, pl),
        Iis::M12 => accumulate_iis12(a, g, pi, pj, pk, pl),
        Iis::M13 => accumulate_iis13(a, g, pi, pj, pk, pl),
        Iis::M14 => accumulate_iis14(a, g, pi, pj, pk, pl),
        Iis::M15 => accumulate_iis15(a, g, pi, pj, pk, pl),
    }
}

pub fn conditional_probabilities<S: Arith>(pi: S, pj: S, pk: S, pl: S, iis: Iis) -> [S; 9] {

    let zero = S::zero();
    let two = S::set(2.0);
    let four = S::set(4.0);

    let pi2 = pi.mul(pi);
    let pi3 = pi2.mul(pi);
    let pj2 = pj.mul(pj);
    let pk2 = pk.mul(pk);

    match iis {
        // case 1: ii-ii
        Iis::M1 => [ pi, pi2, pi2, pi3, pi2, pi3, pi2, pi3, pi2.mul(pi2), ],

        // case 2: ii-kk
        Iis::M2 => [ zero, pi.mul(pk), zero, pi.mul(pk2), zero, pi2.mul(pk), zero, zero, pi2.mul(pk2), ],

        // case 3: ii-il
        Iis::M3 => [ zero, zero, pi.mul(pl), two.mul(pi2).mul(pl), zero, zero, zero, pi2.mul(pl), two.mul(pi3).mul(pl), ],

        // case 4: ii-ki
        Iis::M4 => [ zero, zero, pi.mul(pk), two.mul(pi2).mul(pk), zero, zero, zero, pi2.mul(pk), two.mul(pi3).mul(pk), ],

        // case 5: ii-kl
        Iis::M5 => [ zero, zero, zero, two.mul(pi).mul(pk).mul(pl), zero, zero, zero, zero, two.mul(pi2).mul(pk).mul(pl), ],

        // case 6: ij-ii
        Iis::M6 => [ zero, zero, zero, zero, pi.mul(pj), two.mul(pi2).mul(pj), zero, pi2.mul(pj), two.mul(pi3).mul(pj), ],

        // case 7: ij-jj
        Iis::M7 => [ zero, zero, zero, zero, pi.mul(pj), two.mul(pj2).mul(pi), zero, pj2.mul(pi), two.mul(pj2.mul(pj)).mul(pi), ],

        // case 8: ij-kk
        Iis::M8 => [ zero, zero, zero, zero, zero, two.mul(pi).mul(pj).mul(pk), zero, zero, two.mul(pk2).mul(pi).mul(pj), ],

        // case 9 | 10: ij-ij and ij-ji
        Iis::M9 | Iis::M10 => [ zero, zero, zero, zero, zero, zero, two.mul(pi).mul(pj), pi.mul(pj).mul(pi.add(pj)), four.mul(pi2).mul(pj2), ],

        // case 11: ij-il
        Iis::M11 => [ zero, zero, zero, zero, zero, zero, zero, pi.mul(pj).mul(pl), four.mul(pi2).mul(pj).mul(pl), ],

        // case 12: ij-ki
        Iis::M12 => [ zero, zero, zero, zero, zero, zero, zero, pi.mul(pj).mul(pk), four.mul(pi2).mul(pj).mul(pk), ],

        // case 13: ij-jl
        Iis::M13 => [ zero, zero, zero, zero, zero, zero, zero, pi.mul(pj).mul(pl), four.mul(pi).mul(pj2).mul(pl), ],

        // case 14: ij-kj
        Iis::M14 => [ zero, zero, zero, zero, zero, zero, zero, pi.mul(pj).mul(pk), four.mul(pi).mul(pj2).mul(pk), ],

        // case 15: ij-kl
        Iis::M15 => [ zero, zero, zero, zero, zero, zero, zero, zero, four.mul(pi).mul(pj).mul(pk).mul(pl), ],
    }
}