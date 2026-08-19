use crate::algebra::{Matrix, Vector, scale_div, sub};
use crate::iis;
use ndarray::{Array4, ArrayRef2, ArrayView2};

pub fn calculate_stacked_m(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    allele_frequencies: &ArrayRef2<f64>,
) -> Vec<Vector<9>> {
    let num_v = allele_frequencies.shape()[0];
    // TODO make this locus-specific
    let num_g = all_joint_genotypes.len();

    let mut stacked_m = Vec::with_capacity(num_v * num_g);

    for v in 0..num_v {
        for g in 0..num_g {
            let ((i, j), (k, l), iis_mode) = all_joint_genotypes[g];
            let pi = allele_frequencies[[v, i]];
            let pj = allele_frequencies[[v, j]];
            let pk = allele_frequencies[[v, k]];
            let pl = allele_frequencies[[v, l]];

            let row = std::array::from_fn(|ibd_mode| iis::conditional_probability(pi, pj, pk, pl, iis_mode, ibd_mode + 1));

            stacked_m.push(row);
        }
    }

    stacked_m
}

pub fn calculate_quadratic_q_mat(stacked_m: &[Vector<9>], num_v: usize) -> Matrix<9> {
    let scale = 1.0 / num_v as f64;
    ata(stacked_m, scale)
}

fn ata(a_mat: &[Vector<9>], scale: f64) -> Matrix<9> {
    let rows = a_mat.len();

    let mut h = [[0.0; 9]; 9];

    let n: i32 = 9;
    let k = i32::try_from(rows).unwrap();

    let lda = n;
    let ldc = n;

    unsafe {
        cblas::dsyrk(
            cblas::Layout::RowMajor,
            cblas::Part::Lower,
            cblas::Transpose::Ordinary,
            n,
            k,
            scale,
            a_mat.as_flattened(),
            lda,
            0.0,
            h.as_flattened_mut(),
            ldc,
        );
    }

    // Make symmetric
    for i in 0..9 {
        for j in (i+1)..9 {
            h[i][j] = h[j][i];
        }
    }

    h
}

pub fn calculate_quadratic_c(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    stacked_m: &[Vector<9>],
    genotypes_x: ArrayView2<i8>,
    genotypes_y: ArrayView2<i8>,
    lookup_table: &Array4<usize>,
) -> Vector<9> {
    let num_g = all_joint_genotypes.len();
    let num_v = genotypes_x.shape()[0];

    let mut c = [0.0; 9];

    let iter_x = genotypes_x.as_slice().unwrap().chunks_exact(2);
    let iter_y = genotypes_y.as_slice().unwrap().chunks_exact(2);

    // for (locus, (geno_x, geno_y)) in chunks_x.iter().copied().zip(chunks_y.iter().copied()).enumerate() {
    for (locus, (geno_x, geno_y)) in iter_x.zip(iter_y).enumerate() {
        // let [i, j] = geno_x;
        // let [k, l] = geno_y;
        let (i, j) = (geno_x[0], geno_x[1]);
        let (k, l) = (geno_y[0], geno_y[1]);

        // TODO do a check here for missing data
        // if i < 0 || j < 0 || k < 0 || l < 0 {
        //     continue;
        // }

        let g = unsafe { lookup_table.uget((i as usize, j as usize, k as usize, l as usize)) };
        // let g = lookup_table[(i as usize, j as usize, k as usize, l as usize)];

        let row = unsafe { stacked_m.get_unchecked(locus.unchecked_mul(num_g).unchecked_add(*g)) };
        c = sub(&c, row);
    }

    scale_div(num_v as f64, &c)
}

pub fn calculate_all_joint_genotypes(num_a: usize) -> Vec<((usize, usize), (usize, usize), usize)> {
    let num_single_genotypes = (num_a * (num_a + 1)) / 2;
    let num_joint_genotypes = num_single_genotypes * num_single_genotypes;
    let mut joint_genotypes = Vec::with_capacity(num_joint_genotypes);

    for i in 0..num_a {
        for j in i..num_a {
            for k in 0..num_a {
                for l in k..num_a {
                    let iis_mode = iis::calc_iis_mode(i, j, k, l);
                    joint_genotypes.push(((i, j), (k, l), iis_mode));
                }
            }
        }
    }

    joint_genotypes
}

pub fn calculate_joint_genotype_lookup_table(
    all_joint_genotypes: &[((usize, usize), (usize, usize), usize)],
    num_a: usize,
) -> Array4<usize> {
    let mut lookup = Array4::<usize>::zeros((num_a, num_a, num_a, num_a));

    for (g, ((i, j), (k, l), _)) in all_joint_genotypes.iter().enumerate() {
        lookup[(*i, *j, *k, *l)] = g;
    }

    lookup
}
