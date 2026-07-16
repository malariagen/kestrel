use nalgebra::{Cholesky, SMatrix};

use crate::{cholesky, util::{Matrix9, Vector9}};

pub struct QpTune {
    max_iter: u64,
    conv_tol: f64,
    zero_threshold: f64,
    zero_search_tol: f64
}

impl QpTune {
    pub fn new() -> QpTune {
        QpTune { max_iter: 10, conv_tol: 1e-10, zero_threshold: 1e-8, zero_search_tol: 1e-14 }
    }
}

// Hmm, they use a pseudo-inverse for the constrained least squares problem. Perhaps we could too?
// We already have Q = M^T M, where M is the big stacked matrix. Maybe we can do something with this
// (an SVD?) and use that to our advantage. (Uhh prolly not, M is big boi and Q is not).
// However, we may be able to regularize Q first and then use a standard Cholesky algorithm. Faster, perhaps.
// Can do loop-unrolling for small matrices to calculate Cholesky of submatrices.

// Minimize 1/2 y^T Q y + c^T y, subject to y >= 0 and sum(y) = 1
pub fn solve_qp_active_set(q_mat: &Matrix9<f64>, c: &Vector9<f64>, y0: &Vector9<f64>, sum_to_one: bool, tune: &QpTune) -> (Vector9<f64>, u64) {

    let mut working_set = [false; 9];

    let mut y = y0.clone();

    for row in 0..9 {
        if y[row] <= tune.zero_threshold {
            y[row] = 0.0;
            working_set[row] = true;
        }
    }

    let mut iter = 0;

    while iter < tune.max_iter {

        if sum_to_one {
            let sum = y.sum();
            y /= sum;
        }

        // let mut free_indices = [0_usize; 9];
        // let mut free_count = 0;
        // for i in 0..9 {
        //     if !working_set[i] {
        //         free_indices[free_count] = i;
        //         free_count += 1;
        //     }
        // }

        // TODO we could move all the buffers outside of the loop in theory
        let mut y_free_buf = Vector9::<f64>::zeros();
        let mut free_count = 0;
        for row in 0..9 {
            if !working_set[row] {
                y_free_buf[free_count] = y[row];
                free_count += 1;
            }
        }
        let y_free = y_free_buf.rows(0, free_count);

        let mut q_mat_free_buf = Matrix9::<f64>::zeros();
        let mut nonzero_col = 0;
        for col in 0..9 {
            if !working_set[col] {
                let mut nonzero_row = 0;
                for row in 0..9 {
                    if !working_set[row] {
                        q_mat_free_buf[(nonzero_row, nonzero_col)] = q_mat[(row, col)];
                        nonzero_row += 1;
                    }
                }
                nonzero_col += 1;
            }
        }

        debug_assert_eq!(free_count, nonzero_col);

        let q_mat_free = q_mat_free_buf.view((0, 0), (nonzero_col, nonzero_col));

        let mut c_free_buf = Vector9::<f64>::zeros();
        let mut free_count = 0;
        for row in 0..9 {
            if !working_set[row] {
                c_free_buf[free_count] = c[row];
                free_count += 1;
            }
        }

        let mut g_free = c_free_buf.rows_mut(0, free_count);
        g_free.gemv(1.0, &q_mat_free, &y_free, 1.0);

        let mut l_buf = Matrix9::<f64>::zeros();
        let mut sub_l = l_buf.view_mut((0, 0), (q_mat_free.nrows(), q_mat_free.ncols()));

        let mut d_buf = Vector9::<f64>::zeros();
        let mut sub_d = d_buf.rows_mut(0, q_mat_free.nrows());

        // In theory this factorization can be updated using Givens rotations
        // instead of calculating it from scratch
        cholesky::modify_gmw(q_mat_free, &mut sub_l, &mut sub_d);

        // We now want to solve L D L^T q = g
        // Forward substitution
        sub_l.solve_lower_triangular_mut(&mut g_free);
        // Diagonal scaling
        g_free.component_div_assign(&sub_d);
        // Backward substitution
        sub_l.tr_solve_lower_triangular_mut(&mut g_free);

        let lambda = if sum_to_one {
            let mut ones_buf = Vector9::<f64>::from_element(1.0);
            let mut ones = ones_buf.rows_mut(0, g_free.nrows());

            sub_l.solve_lower_triangular_mut(&mut ones);
            ones.component_div_assign(&sub_d);
            sub_l.tr_solve_lower_triangular_mut(&mut ones);
            let lambda = g_free.sum() / ones.sum();

            // g_free = lambda * ones - g_free
            g_free.axpy(lambda, &ones, -1.0);
            lambda
        } else {
            g_free.neg_mut();
            0.0
        };

        if g_free.amax() <= tune.zero_search_tol {
            // q is roughly zero, so check KKT to see if we are at an optimum solution

            // The free set is full, aka the working set is empty
            if g_free.nrows() == 9 {
                return (y, iter)
            }

            // This is the gradient of the QP
            let mut g = c.clone();
            // g = c + Qy
            g.gemv(1.0, &q_mat, &y, 1.0);

            let smallest_multiplier_index = (0..9).filter(|row| working_set[*row]).min_by(|i, j| {
                g[*i].partial_cmp(&g[*j]).expect("Found a NaN")
            }).expect("Working set is full!");

            let smallest_muliplier = g[smallest_multiplier_index] - lambda;

            if smallest_muliplier >= -tune.conv_tol {
                return (y, iter)
            }

            working_set[smallest_multiplier_index] = false;

        } else {
            let mut p = Vector9::<f64>::zeros();
            let mut free_count = 0;
            for row in 0..9 {
                if !working_set[row] {
                    p[row] = g_free[free_count];
                    free_count += 1;
                }
            }
            // In theory we should use y[free_set] and q[free_set] to calculate this.
            // However, we know that q[working_set] == 0.0, so this will be equivalent
            // as long as there is at least one element in the free set. The current
            // approach also means the indexes will match.
            let (alpha, blocking_index) = feasible_step_size(&y, &p);

            // y += alpha * p
            y.axpy(alpha, &p, 1.0);
            // This could be slightly below 0 in some cases due to rounding, so clamp
            y.apply(|x| *x = x.max(0.0));

            // TODO have a thing where this only adds to the blocking set if two or more non-zero

            if let Some(index) = blocking_index {
                // print("Adding", blocking_index, "to working set")
                working_set[index] = true;
                y[index] = 0.0;
            }
        }

        iter += 1;
    }

    // If we run out of QP iterations, just return the latest value
    // (we do not have to solve the QP perfectly).
    (y, iter)
}

fn feasible_step_size(y: &Vector9<f64>, q: &Vector9<f64>) -> (f64, Option<usize>) {
    let mut alpha = 1.0;
    let mut blocking_index = None;
    for i in 0..9 {
        if q[i] < 0.0 {
            // We want to solve y[i] + a*q[i] = 0
            let a = -y[i] / q[i];
            // In theory we could use <= because of exact blockage
            if a < alpha {
                alpha = a;
                blocking_index = Some(i);
            }
        }
    }

    (alpha, blocking_index)
}