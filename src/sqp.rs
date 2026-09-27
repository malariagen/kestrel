use crate::{
    algebra::{Matrix, Vector, add, add_n, dot, mul, mul_n, scale_mul, sub, sum_n, sum_to_one}, cholesky, eigenval::eigenvals_jacobi,
};

pub struct Tuneables {
    pub sqp_max_iter: u64,
    pub sqp_conv_tol: f64,

    // qp_max_iter: u64,
    pub qp_conv_tol: f64,
    pub qp_zero_search_tol: f64,

    pub bls_max_iter: u64,
    pub bls_sufficient_decrease: f64,
    pub bls_step_size_reduce: f64,

    pub epsilon: f64,
}

impl Tuneables {
    pub fn new() -> Tuneables {
        Tuneables {
            sqp_max_iter: 100,
            sqp_conv_tol: 1e-8,
            // qp_max_iter: 10,
            qp_conv_tol: 1e-10,
            qp_zero_search_tol: 1e-14,
            bls_max_iter: 10,
            bls_sufficient_decrease: 1e-4,
            bls_step_size_reduce: 0.9,
            epsilon: 1e-8,
        }
    }
}

pub fn solve_sqp<const N: usize, Obj, GradHess>(
    obj: Obj,
    grad_hess: GradHess,
    x0: &Vector<N>,
    tune: &Tuneables,
) -> (Vector<N>, u64)
where
    Obj: Fn(&Vector<N>, f64) -> f64,
    GradHess: Fn(&Vector<N>, f64) -> (Vector<N>, Matrix<N>),
{
    let mut x = *x0;

    let mut qp = 0;
    let mut bl = 0;

    for iter in 0..tune.sqp_max_iter {
        x = sum_to_one(&x);

        let (g, h) = grad_hess(&x, tune.epsilon);

        // let eigs = eigenvals_jacobi(&h, 50).unwrap();
        // for e in eigs.iter() {
        //     let tol = -1e-5;
        //     if *e < tol {
        //         println!("Not PSD: {:?} {}", x, e);
        //     }
        // }

        // println!("{iter} {x:?} {g:?} {qp} {bl}");

        if check_convergence(&x, &g, tune.sqp_conv_tol) {
            return (x, iter);
        }

        // c = g - H x
        let c = sub(&g, &mul(&h, &x));

        let (y, _qp_iter) = solve_qp_active_set(&h, &c, &x, true, tune);

        let (xnew, _bls_iter) = backtracking_line_search(&obj, &x, &y, &g, tune);

        qp = _qp_iter;
        bl = _bls_iter;

        // println!("{iter} {x:?} {y:?} {g:?} {_qp_iter} {_bls_iter}");
        // println!("{iter} {x:?} {g:?} {_qp_iter} {_bls_iter}");

        x = xnew;

        // println!("{iter} {x:?} {g:?} {qp_iter} {bls_iter}");
    }

    (x, tune.sqp_max_iter)
}

// TODO also print a warning when the algorithm doesn't converge within the iterations

fn check_convergence<const N: usize>(x: &Vector<N>, g: &Vector<N>, tol: f64) -> bool {
    // This is the Lagrange multiplier
    let lambda = dot(x, g);

    // If x == 0 then gx >= lambda
    // And if x > 0.0 then gx == lambda

    x.iter().zip(g.iter()).all(|(&xi, &gi)| {
        if xi == 0.0 {
            gi >= lambda - tol
        } else {
            (gi - lambda).abs() <= tol
        }
    })
}

// Hmm, they use a pseudo-inverse for the constrained least squares problem. Perhaps we could too?
// We already have Q = M^T M, where M is the big stacked matrix. Maybe we can do something with this
// (an SVD?) and use that to our advantage. (Uhh prolly not, M is big boi and Q is not).
// However, we may be able to regularize Q first and then use a standard Cholesky algorithm. Faster, perhaps.
// Can do loop-unrolling for small matrices to calculate Cholesky of submatrices.

// Minimize 1/2 y^T Q y + c^T y, subject to y >= 0 and sum(y) = 1
pub fn solve_qp_active_set<const N: usize>(
    q_mat: &Matrix<N>,
    c: &Vector<N>,
    y0: &Vector<N>,
    modify: bool,
    tune: &Tuneables,
) -> (Vector<N>, usize) {
    let mut y = y0.clone();

    let mut working_set = [false; N];

    for row in 0..N {
        if y[row] == 0.0 {
            working_set[row] = true;
        }
    }

    let mut iter = 0;

    while iter < N + 1 {
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

fn backtracking_line_search<const N: usize, Obj>(
    obj: Obj,
    x: &Vector<N>,
    y: &Vector<N>,
    g: &Vector<N>,
    tune: &Tuneables,
) -> (Vector<N>, u64)
where
    Obj: Fn(&Vector<N>, f64) -> f64,
{
    let f = obj(&x, tune.epsilon);

    let p = sub(y, x);
    let t = tune.bls_sufficient_decrease * dot(g, &p);

    // We know that y = x + p is already feasible, so this is a feasible starting point
    let mut alpha = 1.0;
    for iter in 0..tune.bls_max_iter {
        // Numerically this is always feasible (>= 0) for floats
        let xnew = add(x, &scale_mul(alpha, &p));
        // In theory we could store an intermediate array that makes
        // this more efficient, a la N&W
        let fnew = obj(&xnew, tune.epsilon);
        if fnew <= f + alpha * t {
            return (xnew, iter);
        }

        alpha *= tune.bls_step_size_reduce;
    }

    // If we exceed the maximum number of backtracks, then just
    // return the last one. This could happen because of floating
    // point problems, and it's better to be robust instead of
    // throwing errors (let the main loop handle it).
    return (add(x, &scale_mul(alpha, &p)), tune.bls_max_iter);
}
