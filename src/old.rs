

pub fn calculate_stacked_m(all_joint_genotypes: &[(i8, i8, i8, i8, i8)], allele_frequencies: &ArrayRef2<f64>) -> Mat<f64> {
    let num_v = allele_frequencies.shape()[0];
    let num_g = all_joint_genotypes.len();

    let mut stacked_m = Mat::<f64>::zeros(num_v * num_g, 9);

    // Faer stores its matrices in column-major format, so one might think that
    // calculating the transpose of the matrix is faster since the writes would
    // be adjacent in memory. Turns out it is not.
    for v in 0..num_v {
        for g in 0..num_g {
            let (i, j, k, l, iis_mode) = all_joint_genotypes[g];
            let pi = allele_frequencies[[v, i as usize]];
            let pj = allele_frequencies[[v, j as usize]];
            let pk = allele_frequencies[[v, k as usize]];
            let pl = allele_frequencies[[v, l as usize]];

            for ibd_mode in 0..9 {
                stacked_m[(v * num_g + g, ibd_mode)] = iis::conditional_probability(pi, pj, pk, pl, iis_mode, ibd_mode as i8 + 1);
            }
        }
    }

    stacked_m
}

pub fn calculate_quadratic_q(stacked_m: &Mat<f64>, num_v: usize) -> Mat<f64> {
    let cols = stacked_m.ncols();

    let mut q = Mat::<f64>::zeros(cols, cols);

    // TODO should we bother with this? Idk
    let scale = 1.0 / num_v as f64;

    // matmul_with_conj(
    //     &mut q,
    //     BlockStructure::TriangularLower,
    //     Accum::Replace,
    //     stacked_m,
    //     BlockStructure::Rectangular,
    //     Conj::Yes,
    //     stacked_m,
    //     BlockStructure::Rectangular,
    //     Conj::No,
    //     scale,
    //     Par::Seq
    // );

    matmul(
        &mut q,
        BlockStructure::TriangularLower,
        Accum::Replace,
        stacked_m.transpose(),
        BlockStructure::Rectangular,
        stacked_m,
        BlockStructure::Rectangular,
        scale,
        Par::Seq
    );

    // Mirror the matrix, since it's symmetric
    for i in 0..cols {
        for j in (i+1)..cols {
            q[(i, j)] = q[(j, i)];
        }
    }

    q
}
