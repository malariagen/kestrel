use crate::util::{DMatrixView9, DMatrixViewMut9, DVectorViewMut9};

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
            xi = xi.max(a[(i, j)]);
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
    assert!(n <= 9);

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

pub fn unmodified2(
    n: usize,
    a: &[[f64; 9]; 9],
    l: &mut [[f64; 9]; 9],
    d: &mut [f64; 9],
) {

    for j in 0..n {
        l[j][j] = 1.0;

        let mut s_jj = 0.0;
        for s in 0..j {
            s_jj += d[s] * l[j][s] * l[j][s];
        }

        let c_jj = a[j][j] - s_jj;
        let d_j = c_jj;

        for i in (j + 1)..n {
            let mut s_ij = 0.0;
            for s in 0..j {
                s_ij += d[s] * l[i][s] * l[j][s];
            }
            let c_ij = a[i][j] - s_ij;

            l[i][j] = c_ij / d_j;
        }

        d[j] = d_j;
    }
}
