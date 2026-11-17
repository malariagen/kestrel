use crate::algebra::{Matrix, Vector};

// A modified Cholesky factorization, based on GMW81.
// This will add a small diagonal correction matrix to a
// in order to make it positive definite.
pub fn modify_gmw_n<const N: usize>(n: usize, a: &Matrix<N>, l: &mut Matrix<N>, d: &mut Vector<N>) {
    assert!(n <= N);

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

// An unmodified Cholesky factorization of a symmetric matrix a.
// This will succeed iff a is positive definite.
// In theory, one can use complete pivoting to handle the case when a is PSD,
// which will give some zero d values.
// For symmetric indefinite matrices, one can use an LBL^T factorization instead
pub fn unmodified_n<const N: usize>(n: usize, a: &Matrix<N>, l: &mut Matrix<N>, d: &mut Vector<N>) {
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

    forward_substitution(n, l, b);

    // Diagonal scaling
    for i in 0..n {
        b[i] /= d[i];
    }

    backward_substitution(n, l, b);
}

// Solve L x = b and store it in b, where L is lower triangular and has 1 along diagonal
pub fn forward_substitution<const N: usize>(n: usize, l: &Matrix<N>, b: &mut Vector<N>) {
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..i {
            s = b[j].mul_add(l[i][j], s);
        }
        // Normally we also divide by l[i][i], but here we know that is 1
        b[i] -= s;
    }
}

// Solve L^T x = b and store it in b, where L is lower triangular and has 1 along diagonal
pub fn backward_substitution<const N: usize>(n: usize, l: &Matrix<N>, b: &mut Vector<N>) {
    for i in (0..n).rev() {
        let mut s = 0.0;
        for j in (i + 1)..n {
            s = b[j].mul_add(l[j][i], s);
        }
        // Normally we also divide by l[i][i], but here we know that is 1
        b[i] -= s;
    }
}
