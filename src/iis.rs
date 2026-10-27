use crate::simd::Arith;

type IisTable = [[[[[[Option<Iis>; 2]; 2]; 2]; 2]; 2]; 2];

const IIS_TABLE: IisTable = generate_iis_lookup_table();

#[derive(Clone, Copy)]
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