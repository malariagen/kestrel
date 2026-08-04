use crate::{algebra::{Matrix, Vector}, util::{DMatrixView9, DMatrixViewMut9, DVectorViewMut9}};

pub fn modify_gmw(
    a: DMatrixView9<f64>,
    l: &mut DMatrixViewMut9<f64>,
    d: &mut DVectorViewMut9<f64>,
) {
    let n = a.nrows();

    // TODO make this column-major friendly? There are dot products here
    // We can get a lower bound on the condition number from d

    let mut gamma = 0.0f64;
    for i in 0..n {
        gamma = gamma.max(a[(i, i)].abs());
    }

    let mut xi = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            xi = xi.max(a[(i, j)].abs());
        }
    }

    let delta = f64::EPSILON * 1.0f64.max(gamma + xi);

    let t = ((n as f64) * (n as f64) - 1.0).sqrt();

    // If n == 1 then xi == 0
    let ratio = if n == 1 { 0.0 } else { xi / t };

    let beta2 = gamma.max(ratio).max(f64::EPSILON);

    for j in 0..n {
        let mut s_jj = 0.0;
        for s in 0..j {
            s_jj += d[s] * l[(j, s)] * l[(j, s)];
        }

        let c_jj = a[(j, j)] - s_jj;

        l[(j, j)] = 1.0;

        let mut theta_j = 0.0;

        for i in (j + 1)..n {
            let mut s_ij = 0.0;
            for s in 0..j {
                s_ij += d[s] * l[(i, s)] * l[(j, s)];
            }
            let c_ij = a[(i, j)] - s_ij;

            if c_ij.abs() > theta_j {
                theta_j = c_ij.abs();
            }

            l[(i, j)] = c_ij;
        }

        let d_j = c_jj.abs().max(theta_j * theta_j / beta2).max(delta);

        for i in (j + 1)..n {
            l[(i, j)] /= d_j;
        }

        d[j] = d_j;
    }
}

pub fn unmodified(
    a: DMatrixView9<f64>,
    l: &mut DMatrixViewMut9<f64>,
    d: &mut DVectorViewMut9<f64>,
) {
    let n = a.nrows();

    for j in 0..n {
        l[(j, j)] = 1.0;

        let mut s_jj = 0.0;
        for s in 0..j {
            s_jj += d[s] * l[(j, s)] * l[(j, s)];
        }

        let c_jj = a[(j, j)] - s_jj;
        let d_j = c_jj;

        for i in (j + 1)..n {
            let mut s_ij = 0.0;
            for s in 0..j {
                s_ij += d[s] * l[(i, s)] * l[(j, s)];
            }
            let c_ij = a[(i, j)] - s_ij;

            l[(i, j)] = c_ij / d_j;
        }

        d[j] = d_j;
    }
}

pub fn modify_gmw_n<const N: usize>(
    n: usize,
    a: &Matrix<N>,
    l: &mut Matrix<N>,
    d: &mut Vector<N>,
) {
    assert!(n <= N);

    // We can get a lower bound on the condition number from d

    let mut gamma = 0.0f64;
    for i in 0..n {
        gamma = gamma.max(a[i][i].abs());
    }

    let mut xi = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            xi = xi.max(a[i][j].abs());
        }
    }

    let delta = f64::EPSILON * 1.0f64.max(gamma + xi);

    let t = ((n as f64) * (n as f64) - 1.0).sqrt();

    // If n <= 1 then xi == 0
    let ratio = if n <= 1 { 0.0 } else { xi / t };

    let beta2 = gamma.max(ratio).max(f64::EPSILON);

    let mut p = [0.0; N];

    for j in 0..n {
        let mut s_jj = 0.0;
        for s in 0..j {
            p[s] = d[s] * l[j][s];
            s_jj = p[s].mul_add(l[j][s], s_jj);
        }

        let c_jj = a[j][j] - s_jj;

        l[j][j] = 1.0;

        let mut theta_j = 0.0f64;

        for i in (j + 1)..n {
            let mut s_ij = 0.0;
            for s in 0..j {
                s_ij = p[s].mul_add(l[i][s], s_ij);
            }
            let c_ij = a[i][j] - s_ij;

            theta_j = theta_j.max(c_ij.abs());

            l[i][j] = c_ij;
        }

        let d_j = c_jj.abs().max(theta_j * theta_j / beta2).max(delta);

        for i in (j + 1)..n {
            l[i][j] /= d_j;
        }

        d[j] = d_j;
    }
}

pub fn unmodified_n<const N: usize>(
    n: usize,
    a: &Matrix<N>,
    l: &mut Matrix<N>,
    d: &mut Vector<N>,
) {
    // This removes the bounds checks in the assembly
    assert!(n <= N);

    let mut p = [0.0; N];

    for j in 0..n {
        l[j][j] = 1.0;

        let mut s_jj = 0.0;
        for s in 0..j {
            p[s] = d[s] * l[j][s];
            s_jj = p[s].mul_add(l[j][s], s_jj);
        }

        let c_jj = a[j][j] - s_jj;
        let d_j = c_jj;

        for i in (j + 1)..n {
            let mut s_ij = 0.0;
            for s in 0..j {
                s_ij = p[s].mul_add(l[i][s], s_ij);
            }
            let c_ij = a[i][j] - s_ij;

            l[i][j] = c_ij / d_j;
        }

        d[j] = d_j;
    }
}

// This solves LDL^T x = b and returns it in b
pub fn solve_ldl_mut_n<const N: usize>(n: usize, l: &Matrix<N>, d: &Vector<N>, b: &mut Vector<N>) {

    assert!(n <= N);

    // Forward substitution
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..i {
            s = b[j].mul_add(l[i][j], s);
        }
        // Normally we also divide by m[i][i], but here we know that is 1
        b[i] -= s;
    }

    // Diagonal scaling
    for i in 0..n {
        b[i] /= d[i];
    }

    // Backward substitution
    for i in (0..n).rev() {
        let mut s = 0.0;
        for j in (i+1)..n {
            s = b[j].mul_add(l[j][i], s);
        }
        // Normally we also divide by m[i][i], but here we know that is 1
        b[i] -= s;
    }
}


pub fn unmodified_9(
    n: usize,
    a: &Matrix<9>,
    l: &mut Matrix<9>,
    d: &mut Vector<9>,
) {
    unmodified_n(n, a, l, d);
    modify_gmw_n(n, a, l, d);
}