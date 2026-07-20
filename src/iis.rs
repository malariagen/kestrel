type IisTable = [[[[[[usize; 2]; 2]; 2]; 2]; 2]; 2];

const IIS_TABLE: IisTable = generate_iis_lookup_table();

const fn generate_iis_lookup_table() -> IisTable {
    // For 4 items, there are 4 choose 2 = 6 different equality combinations
    // between items. Each combination has 2 outcomes, so there are 2^6 = 64
    // possible outcomes. However, only 15 of the possibilities define valid
    // equivalence relations.
    let mut table = [[[[[[0; 2]; 2]; 2]; 2]; 2]; 2];

    // 1
    // all equal
    // ii-ii
    table[1][1][1][1][1][1] = 1;

    // 2
    // i == j, k == l
    // ii-kk
    table[1][0][0][0][0][1] = 2;

    // 3
    // i == j, i == k, j == k
    // ii-il
    table[1][1][0][1][0][0] = 3;
    // i == j, i == l, j == l
    // ii-ki
    table[1][0][1][0][1][0] = 4;

    // 4
    // i == j
    // ii-kl
    table[1][0][0][0][0][0] = 5;

    // 5
    // i == k, i == l, k == l
    // ij-ii
    table[0][1][1][0][0][1] = 6;
    // j == k, j == l, k == l
    // ij-jj
    table[0][0][0][1][1][1] = 7;

    // 6
    // k == l
    // ij-kk
    table[0][0][0][0][0][1] = 8;

    // 7
    // i == k, j == l
    // ij-ij
    table[0][1][0][0][1][0] = 9;
    // i == l, j == k
    // ij-ji
    table[0][0][1][1][0][0] = 10;

    // 8
    // i == k, ij-il
    table[0][1][0][0][0][0] = 11;
    // i == l, ij-ki
    table[0][0][1][0][0][0] = 12;
    // j == k, ij-jl
    table[0][0][0][1][0][0] = 13;
    // j == l, ij-kj
    table[0][0][0][0][1][0] = 14;

    // 9
    // none equal, ij-kl
    table[0][0][0][0][0][0] = 15;

    table
}

pub fn calc_iis_mode(i: usize, j: usize, k: usize, l: usize) -> usize {
    let c1 = usize::from(i == j);
    let c2 = usize::from(i == k);
    let c3 = usize::from(i == l);
    let c4 = usize::from(j == k);
    let c5 = usize::from(j == l);
    let c6 = usize::from(k == l);

    let iis_mode = IIS_TABLE[c1][c2][c3][c4][c5][c6];
    // TODO change the table to return zero indices, we can prove
    // it will never be mis-called since the array is private
    debug_assert_ne!(iis_mode, 0);
    iis_mode
}

pub fn conditional_probability(pi: f64, pj: f64, pk: f64, pl: f64, iis: usize, ibd: usize) -> f64 {
    debug_assert!(iis >= 1 && iis <= 15, "iis must be between 1 and 15");
    debug_assert!(ibd >= 1 && ibd <= 9, "ibd must be between 1 and 9");

    match (iis, ibd) {
        // case 1: ii-ii
        (1, 1) => pi,
        (1, 2 | 3 | 5 | 7) => pi.powi(2),
        (1, 4 | 6 | 8) => pi.powi(3),
        (1, 9) => pi.powi(4),

        // case 2: ii-kk
        (2, 2) => pi * pk,
        (2, 4) => pi * pk.powi(2),
        (2, 6) => pi.powi(2) * pk,
        (2, 9) => pi.powi(2) * pk.powi(2),

        // case 3: ii-il
        (3, 3) => pi * pl,
        (3, 4) => 2.0 * pi.powi(2) * pl,
        (3, 8) => pi.powi(2) * pl,
        (3, 9) => 2.0 * pi.powi(3) * pl,

        // case 4: ii-ki
        (4, 3) => pi * pk,
        (4, 4) => 2.0 * pi.powi(2) * pk,
        (4, 8) => pi.powi(2) * pk,
        (4, 9) => 2.0 * pi.powi(3) * pk,

        // case 5: ii-kl
        (5, 4) => 2.0 * pi * pk * pl,
        (5, 9) => 2.0 * pi.powi(2) * pk * pl,

        // case 6: ij-ii
        (6, 5) => pi * pj,
        (6, 6) => 2.0 * pi.powi(2) * pj,
        (6, 8) => pi.powi(2) * pj,
        (6, 9) => 2.0 * pi.powi(3) * pj,

        // case 7: ij-jj
        (7, 5) => pi * pj,
        (7, 6) => 2.0 * pj.powi(2) * pi,
        (7, 8) => pj.powi(2) * pi,
        (7, 9) => 2.0 * pj.powi(3) * pi,

        // case 8: ij-kk
        (8, 6) => 2.0 * pi * pj * pk,
        (8, 9) => 2.0 * pk.powi(2) * pi * pj,

        // case 9 | 10: ij-ij and ij-ji
        (9 | 10, 7) => 2.0 * pi * pj,
        (9 | 10, 8) => pi * pj * (pi + pj),
        (9 | 10, 9) => 4.0 * pi.powi(2) * pj.powi(2),

        // case 11: ij-il
        (11, 8) => pi * pj * pl,
        (11, 9) => 4.0 * pi.powi(2) * pj * pl,

        // case 12: ij-ki
        (12, 8) => pi * pj * pk,
        (12, 9) => 4.0 * pi.powi(2) * pj * pk,

        // case 13: ij-jl
        (13, 8) => pi * pj * pl,
        (13, 9) => 4.0 * pi * pj.powi(2) * pl,

        // case 14: ij-kj
        (14, 8) => pi * pj * pk,
        (14, 9) => 4.0 * pi * pj.powi(2) * pk,

        // case 15: ij-kl
        (15, 9) => 4.0 * pi * pj * pk * pl,

        // Fallback catch-all
        _ => 0.0,
    }
}
