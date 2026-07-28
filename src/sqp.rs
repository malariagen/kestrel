use nalgebra::DVector;

use crate::{
    cholesky, fused, gradient, hessian, matrix::BlockBuffer, objective, util::{Matrix9, Matrix9xN, MatrixNx9, Vector9},
};

pub struct Tuneables {
    sqp_max_iter: u64,
    sqp_conv_tol: f64,
    sqp_zero_threshold: f64,

    qp_max_iter: u64,
    qp_conv_tol: f64,
    qp_zero_search_tol: f64,

    bls_max_iter: u64,
    bls_sufficient_decrease: f64,
    bls_step_size_reduce: f64,

    epsilon: f64,
}

impl Tuneables {
    pub fn new() -> Tuneables {
        Tuneables {
            sqp_max_iter: 50,
            sqp_conv_tol: 1e-8,
            qp_max_iter: 10,
            qp_conv_tol: 1e-10,
            qp_zero_search_tol: 1e-14,
            bls_max_iter: 10,
            bls_sufficient_decrease: 1e-4,
            bls_step_size_reduce: 0.9,
            epsilon: 1e-8,
            sqp_zero_threshold: 1e-8,
        }
    }
}

pub fn compute_obj_scalar(p_mat: &MatrixNx9<f64>, x: &Vector9<f64>, eps: f64) -> f64 {
    let num_v = p_mat.nrows();

    let a = p_mat
        .row_iter()
        .map(|row| (row.dot(&x.transpose()) + eps).ln())
        .sum::<f64>();

    x.sum() - a / (num_v as f64) - 1.0
}

pub fn compute_obj_scalar_t(p_mat_t: &Matrix9xN<f64>, x: &Vector9<f64>, eps: f64) -> f64 {
    let num_v = p_mat_t.ncols();

    let a = p_mat_t
        .column_iter()
        .map(|col| (col.dot(x) + eps).ln())
        .sum::<f64>();

    x.sum() - a / (num_v as f64) - 1.0
}

pub fn compute_obj_old(
    p_mat: &MatrixNx9<f64>,
    x: &Vector9<f64>,
    d: &mut DVector<f64>,
    eps: f64,
) -> f64 {
    let num_v = p_mat.nrows();

    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    // unsafe { mat_vec_mul_col_major_avx512(p_mat.as_slice(), &x0, num_v, d.as_mut_slice());}
    p_mat.mul_to(x, d);
    d.apply(|x| *x += eps);

    let scale = 1.0 / (num_v as f64);

    let f = x.sum() - scale * d.iter().map(|x| x.ln()).sum::<f64>() - 1.0;

    f
}

pub fn compute_grad_old(p_mat: &MatrixNx9<f64>, x: &Vector9<f64>, eps: f64) -> Vector9<f64> {
    let n = p_mat.nrows();

    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    let ptd = compute_pt_d(p_mat, x, eps);

    // g = 1.0 - (P^T d) / L
    let mut g = Vector9::<f64>::zeros();
    for i in 0..9 {
        g[i] = 1.0 - ptd[i] / (n as f64);
    }

    g
}

fn compute_pt_d(p_mat: &MatrixNx9<f64>, x: &Vector9<f64>, eps: f64) -> Vector9<f64> {
    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        let ptd = unsafe {
            crate::simd::compute_pt_d_avx512_column_major(p_mat.nrows(), p_mat.as_slice(), &x0, eps)
        };
        // let ptd = unsafe { compute_pt_d_wide(p_mat.nrows(), p_mat.as_slice(), &x0, eps) };

        let mut tmp = Vector9::<f64>::zeros();

        for i in 0..9 {
            tmp[i] = ptd[i];
        }

        return tmp;
    } else {
        let ptd = unsafe {
            crate::simd::compute_pt_d_scalar_column(p_mat.nrows(), p_mat.as_slice(), &x0, eps)
        };

        let mut tmp = Vector9::<f64>::zeros();

        for i in 0..9 {
            tmp[i] = ptd[i];
        }

        return tmp;
    }
}

// TODO test this in row major format

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d_scalar_row(
    rows: usize,
    p_mat: &[f64], // rows x 9, row major
    x: &[f64; 9],
    eps: f64,
) -> [f64; 9] {
    assert!(p_mat.len() == rows * 9);

    let mut g = [0.0; 9];

    let mut row_ptr = p_mat.as_ptr();

    for _ in 0..rows {
        let mut d = 0.0;
        // d = P x
        for col in 0..9 {
            let c = unsafe { *row_ptr.add(col) };
            d = c.mul_add(x[col], d);
        }

        d = 1.0 / (d + eps);

        // g = P^T d
        for col in 0..9 {
            let c = unsafe { *row_ptr.add(col) };
            g[col] = c.mul_add(d, g[col]);
        }

        row_ptr = unsafe { row_ptr.add(9) };
    }

    g
}

pub fn compute_grad_d_nalgebra(
    p_mat: &MatrixNx9<f64>,
    x: &Vector9<f64>,
    d: &mut DVector<f64>,
    eps: f64,
) -> Vector9<f64> {
    let num_v = p_mat.nrows();

    p_mat.mul_to(x, d);

    d.apply(|x| *x = 1.0 / (*x + eps));

    let scale = 1.0 / (num_v as f64);

    // g = 1.0 - (P^T d) / L
    let mut g = Vector9::<f64>::from_element(1.0);
    g.gemv_tr(-scale, p_mat, &d, 1.0);

    g
}

pub fn compute_grad_d(
    p_mat: &MatrixNx9<f64>,
    x: &Vector9<f64>,
    d: &mut DVector<f64>,
    eps: f64,
) -> Vector9<f64> {
    let num_v = p_mat.nrows();

    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    // unsafe { mat_vec_mul_col_major_avx512(p_mat.as_slice(), &x0, num_v, d.as_mut_slice());}
    p_mat.mul_to(x, d);

    d.apply(|x| *x += eps);

    d.apply(|x| *x = 1.0 / *x);

    let scale = 1.0 / (num_v as f64);

    // g = 1.0 - (P^T d) / L
    let mut g = Vector9::<f64>::from_element(1.0);
    g.gemv_tr(-scale, p_mat, &d, 1.0);
    // let b = unsafe { p_d_simd(num_v, p_mat.as_slice(), d.as_slice()) };
    // let b = unsafe { p_d_simd3(num_v, p_mat.as_slice(), d.as_slice()) };
    // let mut g = Vector9::<f64>::zeros();
    // for i in 0..9 {
    //     g[i] = 1.0 - b[i] * scale;
    // }

    g
}

fn compute_obj_grad_d(
    p_mat: &MatrixNx9<f64>,
    p_mat_t: &Matrix9xN<f64>,
    x: &Vector9<f64>,
    d: &mut DVector<f64>,
    eps: f64,
) -> (f64, Vector9<f64>) {
    let num_v = p_mat.nrows();

    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    // unsafe { matrix_vector_mul_nx9_avx512(p_mat_t.as_slice(), &x0, num_v, d.as_mut_slice());}
    // unsafe { mat_vec_mul_col_major_avx512(p_mat.as_slice(), &x0, num_v, d.as_mut_slice());}
    p_mat.mul_to(x, d);

    d.apply(|x| *x += eps);

    let scale = 1.0 / (num_v as f64);

    let f = x.sum() - scale * d.iter().map(|x| x.ln()).sum::<f64>() - 1.0;

    d.apply(|x| *x = 1.0 / *x);

    // g = 1.0 - (P^T d) / L
    let mut g = Vector9::<f64>::from_element(1.0);
    g.gemv_tr(-scale, p_mat, &d, 1.0);
    // let b = unsafe { p_d_simd(num_v, p_mat.as_slice(), d.as_slice()) };
    // let b = unsafe { p_d_simd3(num_v, p_mat.as_slice(), d.as_slice()) };
    // let mut g = Vector9::<f64>::zeros();
    // for i in 0..9 {
    //     g[i] = 1.0 - b[i] * scale;
    // }

    (f, g)
}

pub fn compute_hessian(
    p_mat: &MatrixNx9<f64>,
    d: &DVector<f64>,
    a_mat: &mut MatrixNx9<f64>,
) -> Matrix9<f64> {
    a_mat.copy_from(p_mat);

    for mut col in a_mat.column_iter_mut() {
        col.component_mul_assign(d);
    }

    // let num_v = p_mat.nrows();
    // let scale = 1.0 / (num_v as f64);
    // let mut h = Matrix9::<f64>::zeros();
    // a_mat.tr_mul(&_mat) * scale;
    // h.gemm_tr(scale, &a_mat, &a_mat, 0.0);
    // h

    ata(a_mat)
}

// fn compute_hessian2(
//     p_mat: &MatrixNx9<f64>,
//     d: &DVector<f64>,
// ) -> Matrix9<f64> {
//     let num_v = p_mat.nrows();

//     let h = unsafe { at_d2_a_col_major_fast_avx512(p_mat.as_slice(), d.into(), num_v) };

//     let scale = 1.0 / (num_v as f64);
//     Matrix9::from_fn(|i, j| h[i][j] * scale)
// }

// fn compute_hessian_t(
//     p_mat_t: &Matrix9xN<f64>,
//     d: &DVector<f64>,
// ) -> Matrix9<f64> {

//     let num_v = p_mat_t.ncols();

//     let slice = unsafe {
//         std::slice::from_raw_parts(
//             p_mat_t.as_slice().as_ptr().cast::<[f64; 9]>(),
//             num_v
//         )
//     };

//     let h = unsafe { pt_d2_p_simd(num_v, d.into(), slice) };

//     let scale = 1.0 / (num_v as f64);
//     Matrix9::from_fn(|i, j| h[i][j] * scale)
// }

fn ata(a_mat: &MatrixNx9<f64>) -> Matrix9<f64> {
    let rows = a_mat.nrows();
    let scale = 1.0 / (rows as f64);

    let mut h = Matrix9::<f64>::zeros();

    let n: i32 = 9;
    let k = i32::try_from(rows).unwrap();

    let lda = k;
    let ldc = n;

    unsafe {
        cblas::dsyrk(
            cblas::Layout::ColumnMajor,
            cblas::Part::Lower,
            cblas::Transpose::Ordinary,
            n,
            k,
            scale,
            a_mat.as_slice(),
            lda,
            0.0,
            h.as_mut_slice(),
            ldc,
        );
    }

    h.fill_upper_triangle_with_lower_triangle();

    h
}

pub fn solve_sqp(
    p_mat: &BlockBuffer<f64, 8, 9>,
    x0: &Vector9<f64>,
    tune: &Tuneables,
) -> (Vector9<f64>, u64) {
    let mut x = x0.clone();

    for iter in 0..tune.sqp_max_iter {
        // We manually truncate some values to zero, which will put them in the active set
        // If we guessed wrong this will be corrected later
        x.apply(|val| {
            if *val <= tune.sqp_zero_threshold {
                *val = 0.0;
            }
        });

        let sum = x.sum();
        x /= sum;

        // let (f, g) = compute_obj_grad_d(p_mat, p_mat_t, &x, d, tune.epsilon);
        // let g = compute_grad_d(p_mat, &x, d, tune.epsilon);
        let (g, h) = fused::compute_grad_hess(p_mat, &x, tune.epsilon);

        // For a convex function, a point x is optimal iff nab
        // For optimization subject to x >= 0, we have (Bertsekas Nonlinear Programming p. 238):
        // At the optimal point, all partial derivatives are >= 0, and:
        // If x == 0 then the partial derivative is >= 0
        // And if x > 0.0, then the partial derivative is equal to zero.
        //  However, because of the special structure of the objective function
        //  we only need to check if it's >= -conv tol in this case (I think)

        let gmin = g
            .iter()
            .enumerate()
            .filter(|(row, _)| x[*row] > 0.0)
            .map(|(_, gx)| *gx)
            .min_by(|i, j| i.partial_cmp(j).expect("Found a NaN"))
            .expect("Working set is full!");

        if gmin >= -tune.sqp_conv_tol {
            return (x, iter);
        }

        // let h = hessian::compute_hess(p_mat_t, &x, tune.epsilon);

        // c = g - H x
        let mut c = g.clone();
        c.gemv(-1.0, &h, &x, 1.0);

        let (y, qp_iter) = solve_qp_active_set(&h, &c, &x, false, true, tune);

        let f = objective::compute_obj_avx(p_mat, &x, tune.epsilon);

        let (xnew, bls_iter) = backtracking_line_search(p_mat, &x, &y, f, &g, tune);

        x = xnew;
    }

    (x, tune.sqp_max_iter)
}

// Hmm, they use a pseudo-inverse for the constrained least squares problem. Perhaps we could too?
// We already have Q = M^T M, where M is the big stacked matrix. Maybe we can do something with this
// (an SVD?) and use that to our advantage. (Uhh prolly not, M is big boi and Q is not).
// However, we may be able to regularize Q first and then use a standard Cholesky algorithm. Faster, perhaps.
// Can do loop-unrolling for small matrices to calculate Cholesky of submatrices.

// Minimize 1/2 y^T Q y + c^T y, subject to y >= 0 and sum(y) = 1
pub fn solve_qp_active_set(
    q_mat: &Matrix9<f64>,
    c: &Vector9<f64>,
    y0: &Vector9<f64>,
    sum_to_one: bool,
    modify: bool,
    tune: &Tuneables,
) -> (Vector9<f64>, u64) {
    let mut working_set = [false; 9];

    let mut y = y0.clone();

    for row in 0..9 {
        if y[row] == 0.0 {
            working_set[row] = true;
        }
    }

    let mut iter = 0;

    while iter < tune.qp_max_iter {
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

        if modify {
            cholesky::modify_gmw(q_mat_free, &mut sub_l, &mut sub_d);
        } else {
            cholesky::unmodified(q_mat_free, &mut sub_l, &mut sub_d);
        }

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

        if g_free.amax() <= tune.qp_zero_search_tol {
            // q is roughly zero, so check KKT to see if we are at an optimum solution

            // The free set is full, aka the working set is empty
            if g_free.nrows() == 9 {
                return (y, iter);
            }

            // This is the gradient of the QP
            let mut g = c.clone();
            // g = c + Qy
            g.gemv(1.0, &q_mat, &y, 1.0);

            let smallest_multiplier_index = (0..9)
                .filter(|row| working_set[*row])
                .min_by(|i, j| g[*i].partial_cmp(&g[*j]).expect("Found a NaN"))
                .expect("Working set is full!");

            let smallest_muliplier = g[smallest_multiplier_index] - lambda;

            if smallest_muliplier >= -tune.qp_conv_tol {
                return (y, iter);
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

fn backtracking_line_search(
    p_mat: &BlockBuffer<f64, 8, 9>,
    x: &Vector9<f64>,
    y: &Vector9<f64>,
    f: f64,
    g: &Vector9<f64>,
    tune: &Tuneables,
) -> (Vector9<f64>, u64) {
    let p = y - x;
    let t = tune.bls_sufficient_decrease * g.dot(&p);

    // We know that y = x + p is already feasible, so this is a feasible starting point
    let mut alpha = 1.0;
    for iter in 0..tune.bls_max_iter {
        // Numerically this is always feasible (>= 0) for floats
        let xnew = x + alpha * p;
        // TODO this can be made more efficient a la N&W
        // let fnew = objective::compute_obj_avx(p_mat, &xnew, tune.epsilon);
        let fnew = objective::compute_obj_avx(p_mat, &xnew, tune.epsilon);
        if fnew <= f + alpha * t {
            return (xnew, iter);
        }

        alpha *= tune.bls_step_size_reduce;
    }

    // If we exceed the maximum number of backtracks, then just
    // return the last one. This could happen because of floating
    // point problems, and it's better to be robust instead of
    // throwing errors (let the main loop handle it).
    return (x + alpha * p, tune.bls_max_iter);
}
