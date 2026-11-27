use std::num::NonZeroU64;

use crate::{
    algebra::{Matrix, Vector, add, add_n, dot, l2_norm_n, mul, mul_n, scale_mul, sub, sum_n, sum_to_one},
    cholesky::{self, backward_substitution, unmodified_n},
};

pub trait Objective<const N: usize> {
    fn objective(&self, x: &Vector<N>, eps: f64) -> f64;
}

pub trait GradHess<const N: usize> {
    fn grad_hess(&self, x: &Vector<N>, eps: f64) -> (Vector<N>, Matrix<N>);
}

pub struct Tuneables {
    pub sqp_max_iter: NonZeroU64,
    pub sqp_conv_tol: f64,

    pub qp_max_iter: NonZeroU64,
    pub qp_conv_tol: f64,
    pub qp_zero_search_tol: f64,

    pub bls_max_iter: NonZeroU64,
    pub bls_sufficient_decrease: f64,
    pub bls_step_size_reduce: f64,

    pub epsilon: f64,
}

impl Tuneables {
    pub fn new(sqp_max_iter: NonZeroU64, qp_max_iter: NonZeroU64) -> Tuneables {
        Tuneables {
            sqp_max_iter,
            sqp_conv_tol: 1e-8,
            qp_max_iter,
            qp_conv_tol: 1e-10,
            // This was originally 1e-14, but that caused cycling in the QP solver
            // (likely due to floating point noise)
            qp_zero_search_tol: 1e-12,
            bls_max_iter: NonZeroU64::new(10).unwrap(),
            bls_sufficient_decrease: 1e-4,
            bls_step_size_reduce: 0.9,
            epsilon: 1e-8,
        }
    }
}

pub fn solve_sqp<const N: usize, D: Objective<N> + GradHess<N>>(
    data: &D,
    x0: &Vector<N>,
    tune: &Tuneables,
) -> (f64, Vector<N>, u64, f64) {
    let mut x = *x0;

    // let mut qp = 0;
    // let mut bl = 0;

    let mut f = 0.0;

    for iter in 0..tune.sqp_max_iter.into() {
        x = sum_to_one(&x);

        let (g, h) = data.grad_hess(&x, tune.epsilon);

        // let eigs = eigenvals_jacobi(&h, 50).unwrap();
        // for e in eigs.iter() {
        //     let tol = -1e-5;
        //     if *e < tol {
        //         println!("Not PSD: {:?} {}", x, e);
        //     }
        // }

        // println!("{iter} {x:?} {g:?} {qp} {bl}");

        if check_convergence(&x, &g, tune.sqp_conv_tol) {
            let cond = estimate_condition_number(&x, &g, &h, tune.sqp_conv_tol);
            return (f, x, iter, cond);
        }

        // c = g - H x
        let c = sub(&g, &mul(&h, &x));

        let (y, _qp_iter) = solve_qp_active_set(&h, &c, &x, true, tune);

        let (fnew, xnew, _bls_iter) = backtracking_line_search(data, &x, &y, &g, tune);

        // qp = _qp_iter;
        // bl = _bls_iter;

        x = xnew;
        f = fnew;
    }

    (f, x, tune.sqp_max_iter.into(), f64::NAN)
}

fn check_convergence<const N: usize>(x: &Vector<N>, g: &Vector<N>, tol: f64) -> bool {
    // This is the Lagrange multiplier for the sum-to-one constraint
    let lambda = dot(x, g);

    // If x == 0 then gx >= lambda
    // And if x > 0.0 then gx == lambda

    x.iter().zip(g).all(|(&xi, &gi)| {
        if xi == 0.0 {
            gi >= lambda - tol
        } else {
            (gi - lambda).abs() <= tol
        }
    })
}

// Minimize 1/2 y^T Q y + c^T y, subject to y >= 0 and sum(y) = 1
pub fn solve_qp_active_set<const N: usize>(
    q_mat: &Matrix<N>,
    c: &Vector<N>,
    y0: &Vector<N>,
    modify: bool,
    tune: &Tuneables,
) -> (Vector<N>, u64) {
    let mut y = y0.clone();

    let mut working_set = [false; N];

    for row in 0..N {
        if y[row] == 0.0 {
            working_set[row] = true;
        }
    }

    let mut iter = 0;

    while iter < tune.qp_max_iter.into() {
        y = sum_to_one(&y);

        let mut free_indices = [0; N];
        let mut free_count = 0;
        for i in 0..N {
            if !working_set[i] {
                free_indices[free_count] = i;
                free_count += 1;
            }
        }

        // Should Never Happen(TM)
        assert!(free_count != 0);
        // Removes some bounds checks
        assert!(free_count <= N);

        let mut y_free = [0.0; N];
        let mut c_free = [0.0; N];
        let mut q_mat_free = [[0.0; N]; N];
        for i in 0..free_count {
            let free_i = free_indices[i];
            y_free[i] = y[free_i];
            c_free[i] = c[free_i];
            for j in 0..free_count {
                let free_j = free_indices[j];
                q_mat_free[i][j] = q_mat[free_i][free_j];
            }
        }

        // g_f = c_f + Q_f y_f
        let mut g_free = add_n(free_count, &c_free, &mul_n(free_count, &q_mat_free, &y_free));

        let mut sub_l = [[0.0; N]; N];
        let mut sub_d = [0.0; N];

        if modify {
            // Hmmmm, should we modify this here, or once at the start?
            cholesky::modify_gmw_n(free_count, &q_mat_free, &mut sub_l, &mut sub_d);
        } else {
            cholesky::unmodified_n(free_count, &q_mat_free, &mut sub_l, &mut sub_d);
        }

        // We now want to solve L D L^T q = g
        cholesky::solve_ldl_mut_n(free_count, &sub_l, &sub_d, &mut g_free);

        let mut ones_free = [0.0; N];
        for i in 0..free_count {
            ones_free[i] = 1.0;
        }

        cholesky::solve_ldl_mut_n(free_count, &sub_l, &sub_d, &mut ones_free);

        let lambda = sum_n(free_count, &g_free) / sum_n(free_count, &ones_free);

        // g_free = lambda * ones - g_free
        // TODO check
        g_free = sub(&scale_mul(lambda, &ones_free), &g_free);

        // TODO check
        let m = g_free.iter().map(|x| x.abs()).max_by(|a, b| a.total_cmp(b)).unwrap();

        if m <= tune.qp_zero_search_tol {
            // q is roughly zero, so check KKT to see if we are at an optimum solution

            // The free set is full, aka the working set is empty
            if free_count == N {
                return (y, iter);
            }

            // This is the gradient of the QP
            // g = c + Qy
            let g = add(c, &mul(&q_mat, &y));

            let smallest_multiplier_index = (0..N)
                .filter(|row| working_set[*row])
                .min_by(|i, j| g[*i].total_cmp(&g[*j]))
                .expect("Working set is full!");

            let smallest_muliplier = g[smallest_multiplier_index] - lambda;

            // println!("m {m} smallest mult {smallest_muliplier}");
            if smallest_muliplier >= -tune.qp_conv_tol {
                return (y, iter);
            }

            // println!("Removing {smallest_multiplier_index} from working set");
            working_set[smallest_multiplier_index] = false;
        } else {
            let mut p = [0.0; N];
            let mut free_count = 0;
            for row in 0..N {
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
            // This could be slightly below 0 in some cases due to rounding, so clamp
            y = std::array::from_fn(|i| (y[i] + alpha * p[i]).max(0.0));

            if let Some(index) = blocking_index {
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

fn feasible_step_size<const N: usize>(y: &Vector<N>, q: &Vector<N>) -> (f64, Option<usize>) {
    let mut alpha = 1.0;
    let mut blocking_index = None;
    for i in 0..N {
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

fn backtracking_line_search<const N: usize, D: Objective<N>>(
    data: &D,
    x: &Vector<N>,
    y: &Vector<N>,
    g: &Vector<N>,
    tune: &Tuneables,
) -> (f64, Vector<N>, u64) {
    let f = data.objective(&x, tune.epsilon);

    let p = sub(y, x);
    let t = tune.bls_sufficient_decrease * dot(g, &p);

    // We know that y = x + p is already feasible, so this is a feasible starting point
    let mut alpha = 1.0;
    let mut iter = 0;

    // bls_max_iter is NonZeroU64, so this loop will always terminate
    loop {
        // Numerically this is always feasible (>= 0) for floats
        let xnew = add(x, &scale_mul(alpha, &p));
        // In theory we could store an intermediate array that makes
        // this more efficient, a la N&W
        let fnew = data.objective(&xnew, tune.epsilon);
        if fnew <= f + alpha * t {
            return (fnew, xnew, iter);
        }

        iter += 1;

        // If we hit the maximum number of backtracks, then just
        // return the last one. This could happen because of floating
        // point problems, and it's better to be robust instead of
        // throwing errors (let the main loop handle it).
        if iter == tune.bls_max_iter.get() {
            return (fnew, xnew, iter);
        }

        alpha *= tune.bls_step_size_reduce;
    }
}

// Practical Optimization, 8.3.3.2
pub fn estimate_condition_number<const N: usize>(x: &Vector<N>, g: &Vector<N>, h: &Matrix<N>, tol: f64) -> f64 {
    // We need to project the Hessian matrix along the active constraints
    // (a la the null space method) to get the condition number of the reduced Hessian.
    // If a constraint is in strict complementarity then the gradient is pushing
    // against the constraint, so that is unique even if the curvature is flat in that
    // direction (think about a downward sloping line).
    // (In theory we could also reduce the Hessian for the sum-to-one constraint...)

    // This is the Lagrange multiplier for the sum-to-one constraintt
    let lambda = dot(x, g);

    // Find the free indices
    let mut free_indices = [0; N];
    let mut free_count = 0;
    for i in 0..N {
        // Strict complementarity holds if x == 0 and gx > lambda
        let strict_complementarity = x[i] == 0.0 && g[i] >= lambda + tol;
        if !strict_complementarity {
            free_indices[free_count] = i;
            free_count += 1;
        }
    }

    // Should Never Happen(TM)
    assert!(free_count != 0);
    // Removes some bounds checks
    assert!(free_count <= N);

    // Extract the reduced Hessian
    let mut h_free = [[0.0; N]; N];
    for i in 0..free_count {
        let free_i = free_indices[i];
        for j in 0..free_count {
            let free_j = free_indices[j];
            h_free[i][j] = h[free_i][free_j];
        }
    }

    let mut sub_l = [[0.0; N]; N];
    let mut sub_d = [0.0; N];

    unmodified_n(free_count, &h_free, &mut sub_l, &mut sub_d);

    // Any negative or zero entry along d indicates a zero eigenvalue, so not invertible
    let all_positive = sub_d.iter().take(free_count).all(|d| *d > 0.0);

    if !all_positive {
        return f64::INFINITY;
    }

    let (max_index, max) = sub_d
        .iter()
        .enumerate()
        .take(free_count)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let (min_index, min) = sub_d
        .iter()
        .enumerate()
        .take(free_count)
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();

    let mut e_max = [0.0; N];
    e_max[max_index] = 1.0;

    let mut e_min = [0.0; N];
    e_min[min_index] = 1.0;

    let w_max = mul_n(free_count, &sub_l, &e_max);
    backward_substitution(free_count, &sub_l, &mut e_min);

    let norm_max = l2_norm_n(free_count, &w_max);
    let norm_min = l2_norm_n(free_count, &e_min);

    // This is a lower bound on the condition number
    // It should be approximately correct to the same order of magnitude
    norm_max * norm_min * (max / min)
}
