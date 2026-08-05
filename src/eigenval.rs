use crate::algebra::{Matrix, Vector};

// See Numerical Recipes, Chapter 11
pub fn eigenvals_jacobi<const N: usize>(m: &Matrix<N>, max_iter: u64) -> Vector<N> {
    let mut d = [0.0; N];

    // A 2x2 matrix can be diagonalized in a single iteration
    if N == 2 {
        let a = m[0][0];
        let b = m[0][1];
        let c = m[1][1];

        if b.abs() <= f64::EPSILON*(a.abs() + c.abs()) {
            d[0] = a;
            d[1] = c;
        } else {
            let theta = (c - a)/(2.0 * b);
            let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
            d[0] = (-t).mul_add(b, a);
            d[1] = t.mul_add(b, c);
        }

        return d;
    }

    let mut a = *m;
    for i in 0..N {
        d[i] = a[i][i];
    }

    for _ in 0..max_iter {
        // This is a temporary accumulutator that we keep to reduce numerical errors.
        // The changes to the diagonal are accumulated for a full sweep, and then we
        // increment the diagonal afterward.
        let mut z = [0.0; N];

        let mut did_a_sweep = false;

        // Sweep over every off-diagonal element in the upper triangle
        for p in 0..N {
            for q in (p+1)..N {
                let a_qq = a[q][q];
                let a_pp = a[p][p];
                let a_pq = a[p][q];

                // This is the convergence check that nalgebra uses for its eigenval algorithm
                if a_pq.abs() <= f64::EPSILON * (a_pp.abs() + a_qq.abs()) {
                    continue;
                }

                did_a_sweep = true;

                let theta = (a_qq - a_pp) / (2.0 * a_pq);
                let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                let tau = s / (1.0 + c);

                a[p][q] = 0.0;

                z[p] = (-t).mul_add(a_pq, z[p]);
                z[q] = t.mul_add(a_pq, z[q]);

                a[p][p] = (-t).mul_add(a_pq, a_pp);
                a[q][q] = t.mul_add(a_pq, a_qq);

                // Now we need to update the rows and columns

                // 0 <= j < p
                for j in 0..p {
                    let a_jp = a[j][p];
                    let a_jq = a[j][q];
                    a[j][p] = (-s).mul_add(tau.mul_add(a_jp, a_jq), a_jp);
                    a[j][q] = s.mul_add((-tau).mul_add(a_jq, a_jp), a_jq);
                }

                // p < j < q
                for j in (p+1)..q {
                    let a_pj = a[p][j];
                    let a_jq = a[j][q];
                    a[p][j] = (-s).mul_add(tau.mul_add(a_pj, a_jq), a_pj);
                    a[j][q] = s.mul_add((-tau).mul_add(a_jq, a_pj), a_jq);
                }

                // q < j < N
                for j in (q+1)..N {
                    let a_pj = a[p][j];
                    let a_qj = a[q][j];
                    a[p][j] = (-s).mul_add(tau.mul_add(a_pj, a_qj), a_pj);
                    a[q][j] = s.mul_add((-tau).mul_add(a_qj, a_pj), a_qj);
                }
            }
        }

        for i in 0..N {
            d[i] += z[i];
            a[i][i] = d[i];
        }

        if !did_a_sweep {
            break;
        }
    }

    d
}

pub fn test() -> Vector<4> {
    eigenvals_jacobi(&[[10.0; 4]; 4], 50)
}

pub fn tes2t() -> Vector<2> {
    eigenvals_jacobi(&[[10.0; 2]; 2], 50)
}