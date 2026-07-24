use nalgebra::DVector;

use crate::{
    cholesky, gradient, hessian, objective, util::{Matrix9, Matrix9xN, MatrixNx9, Vector9},
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

    let a = p_mat.row_iter().map(|row| (row.dot(&x.transpose()) + eps).ln()).sum::<f64>();

    x.sum() - a / (num_v as f64) - 1.0
}

pub fn compute_obj_scalar_t(p_mat_t: &Matrix9xN<f64>, x: &Vector9<f64>, eps: f64) -> f64 {
    let num_v = p_mat_t.ncols();

    let a = p_mat_t.column_iter().map(|col| (col.dot(x) + eps).ln()).sum::<f64>();

    x.sum() - a / (num_v as f64) - 1.0
}

pub fn compute_obj_old(p_mat: &MatrixNx9<f64>, x: &Vector9<f64>, d: &mut DVector<f64>, eps: f64) -> f64 {
    let num_v = p_mat.nrows();

    let mut x0 = [0.0; 9];
    for i in 0..9 {
        x0[i] = x[i];
    }

    unsafe { mat_vec_mul_col_major_avx512(p_mat.as_slice(), &x0, num_v, d.as_mut_slice());}
    // p_mat.mul_to(x, d);
    d.apply(|x| *x += eps);

    let scale = 1.0 / (num_v as f64);

    let f = x.sum() - scale * d.iter().map(|x| x.ln()).sum::<f64>() - 1.0;

    f
}

pub fn compute_grad_old(
    p_mat: &MatrixNx9<f64>,
    x: &Vector9<f64>,
    eps: f64,
) -> Vector9<f64> {
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
        let ptd = unsafe { crate::simd::compute_pt_d_avx512(p_mat.nrows(), p_mat.as_slice(), &x0, eps) };
        // let ptd = unsafe { compute_pt_d_wide(p_mat.nrows(), p_mat.as_slice(), &x0, eps) };

        let mut tmp = Vector9::<f64>::zeros();

        for i in 0..9 {
            tmp[i] = ptd[i];
        }

        return tmp;
    } else {
        let ptd = unsafe { crate::simd::compute_pt_d_scalar_column(p_mat.nrows(), p_mat.as_slice(), &x0, eps) };

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

use wide::f64x8;

#[target_feature(enable = "avx512f")]
pub fn compute_pt_d_wide2(
    rows: usize,
    p_mat: &[f64],
    x: &[f64; 9],
    eps: f64,
) -> [f64; 9] {
    // 1. Single invariant check at the top
    assert_eq!(p_mat.len(), rows * 9, "p_mat must have length rows * 9");

    let num_chunks = rows / 8;
    let rem_len = rows % 8;

    // 2. Column slices & chunk views
    let cols: [&[f64]; 9] = std::array::from_fn(|c| &p_mat[c * rows..(c + 1) * rows]);
    let chunks: [&[[f64; 8]]; 9] = std::array::from_fn(|c| cols[c].as_chunks::<8>().0);
    let remainders: [&[f64]; 9] = std::array::from_fn(|c| cols[c].as_chunks::<8>().1);

    // 3. Pre-splat SIMD constants
    let mut gradients = [f64x8::splat(0.0); 9];
    let x_simd: [f64x8; 9] = std::array::from_fn(|c| f64x8::splat(x[c]));
    let ones = f64x8::splat(1.0);

    // --- HOT SIMD LOOP (Guaranteed 0 bounds checks) ---
    for i in 0..num_chunks {
        // SAFETY: p_mat length is rows*9, so each column has `rows` elements.
        // `chunks[c]` has length `rows / 8` = `num_chunks`. Therefore `i < chunks[c].len()`.
        let mut denom = f64x8::splat(eps);

        for c in 0..9 {
            denom = x_simd[c].mul_add(f64x8::from(unsafe { *chunks[c].get_unchecked(i) }), denom);
        }

        let d = ones / denom;

        for c in 0..9 {
            gradients[c] = d.mul_add(f64x8::from(unsafe { *chunks[c].get_unchecked(i) }), gradients[c]);
        }
    }

    let mut g = [0.0; 9];

    // Accumulate vector reductions into output g
    for c in 0..9 {
        g[c] = gradients[c].reduce_add();
    }

    // --- SCALAR TAIL LOOP (Guaranteed 0 bounds checks) ---
    for r in 0..rem_len {
        // SAFETY: remainders[c] has length `rows % 8` = `rem_len`. Therefore `r < remainders[c].len()`.
        let mut denom = eps;
        for c in 0..9 {
            denom += unsafe { x[c] * remainders[c].get_unchecked(r) };
        }

        let d = 1.0 / denom;

        for c in 0..9 {
            g[c] += unsafe { remainders[c].get_unchecked(r) } * d;
        }
    }

    g
}

pub fn compute_pt_d_wide(
    rows: usize,
    p_mat: &[f64], // rows x 9, column major
    x: &[f64; 9],
    eps: f64,
) -> [f64; 9] {

    assert!(p_mat.len() == rows * 9);

    let p = p_mat.as_ptr();

    // In the matrix, P[r, c] = c * rows + r
    // This could be calculated on the fly, but LLVM likes the pointers to each column better
    // since it avoids arithmetic.
    let mut col_ptrs: [*const f64; 9] = std::array::from_fn(|col| unsafe { p.add(col.unchecked_mul(rows)) });

    let xs: [f64x8; 9] = std::array::from_fn(|col| f64x8::splat(x[col]));

    let mut columns = [wide::f64x8::ZERO; 9];
    let mut gradients = [wide::f64x8::ZERO; 9];

    let one = wide::f64x8::splat(1.0);

    let chunks = rows / 8;
    let remainder = rows % 8;

    for _ in 0..chunks {
        // Load columns of P from memory
        for col in 0..9 {
            // TODO align the memory
            columns[col] = f64x8::from(unsafe { col_ptrs[col].cast::<[f64; 8]>().read_unaligned() });
        }

        // Calculate d
        // This computes a dot product between x and a row of p
        let mut d = f64x8::splat(eps);
        for col in 0..9 {
            // mul_add(a, b) computes (self * a) + b using hardware FMA where available
            d = xs[col].mul_add(columns[col], d);
        }

        d = one / d;

        // 3. Accumulate P^T * d into gradients
        for col in 0..9 {
            gradients[col] = columns[col].mul_add(d, gradients[col]);
        }

        for col in 0..9 {
            col_ptrs[col] = unsafe { col_ptrs[col].add(8) };
        }
    }

    let mut g = [0.0; 9];
    for col in 0..9 {
        g[col] = gradients[col].reduce_add();
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

    unsafe { mat_vec_mul_col_major_avx512(p_mat.as_slice(), &x0, num_v, d.as_mut_slice());}
    // p_mat.mul_to(x, d);

    d.apply(|x| *x += eps);

    d.apply(|x| *x = 1.0 / *x);

    let scale = 1.0 / (num_v as f64);

    // g = 1.0 - (P^T d) / L
    // let mut g = Vector9::<f64>::from_element(1.0);
    // g.gemv_tr(-scale, p_mat, &d, 1.0);
    // let b = unsafe { p_d_simd(num_v, p_mat.as_slice(), d.as_slice()) };
    let b = unsafe { p_d_simd3(num_v, p_mat.as_slice(), d.as_slice()) };
    let mut g = Vector9::<f64>::zeros();
    for i in 0..9 {
        g[i] = 1.0 - b[i] * scale;
    }

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
    unsafe { mat_vec_mul_col_major_avx512(p_mat.as_slice(), &x0, num_v, d.as_mut_slice());}
    // p_mat.mul_to(x, d);

    d.apply(|x| *x += eps);

    let scale = 1.0 / (num_v as f64);

    let f = x.sum() - scale * d.iter().map(|x| x.ln()).sum::<f64>() - 1.0;

    d.apply(|x| *x = 1.0 / *x);

    // g = 1.0 - (P^T d) / L
    // let mut g = Vector9::<f64>::from_element(1.0);
    // g.gemv_tr(-scale, p_mat, &d, 1.0);
    // let b = unsafe { p_d_simd(num_v, p_mat.as_slice(), d.as_slice()) };
    let b = unsafe { p_d_simd3(num_v, p_mat.as_slice(), d.as_slice()) };
    let mut g = Vector9::<f64>::zeros();
    for i in 0..9 {
        g[i] = 1.0 - b[i] * scale;
    }

    (f, g)
}

fn compute_hessian(
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

fn compute_hessian2(
    p_mat: &MatrixNx9<f64>,
    d: &DVector<f64>,
) -> Matrix9<f64> {
    let num_v = p_mat.nrows();

    let h = unsafe { at_d2_a_col_major_fast_avx512(p_mat.as_slice(), d.into(), num_v) };

    let scale = 1.0 / (num_v as f64);
    Matrix9::from_fn(|i, j| h[i][j] * scale)
}

fn compute_hessian_t(
    p_mat_t: &Matrix9xN<f64>,
    d: &DVector<f64>,
) -> Matrix9<f64> {

    let num_v = p_mat_t.ncols();

    let slice = unsafe {
        std::slice::from_raw_parts(
            p_mat_t.as_slice().as_ptr().cast::<[f64; 9]>(),
            num_v
        )
    };

    let h = unsafe { pt_d2_p_simd(num_v, d.into(), slice) };

    let scale = 1.0 / (num_v as f64);
    Matrix9::from_fn(|i, j| h[i][j] * scale)
}

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
            ldc
        );
    }

    h.fill_upper_triangle_with_lower_triangle();

    h
}

use std::arch::x86_64::*;

#[target_feature(enable = "avx512f,fma")]
pub unsafe fn pt_d2_p_simd(
    n: usize,
    d: &[f64],
    p: &[[f64; 9]],
) -> [[f64; 9]; 9] {
    // 9 rows of 8-wide AVX-512 accumulators
    let mut h8 = [_mm512_setzero_pd(); 9];
    // 9 rows of 1-wide scalar accumulators
    let mut h1 = [0.0; 9];

    for k in 0..n {
        // 1. Square the diagonal element for D^2
        let d_val = *d.get_unchecked(k);
        let d2 = d_val * d_val;
        let d2_vec = _mm512_set1_pd(d2);

        // 2. Load memory DIRECTLY into vector registers
        let p_row = p.get_unchecked(k);
        let p8 = _mm512_loadu_pd(p_row.as_ptr());
        let p1 = *p_row.get_unchecked(8);

        // 3. Scale the row by D^2
        let s8 = _mm512_mul_pd(p8, d2_vec);
        let s1 = p1 * d2;

        // 4. FMA Outer Product
        // LLVM natively unrolls small fixed-size loops (0..9) automatically in release mode;
        // no #pragma needed.
        for i in 0..9 {
            let p_i = *p_row.get_unchecked(i);

            // Generate the `vfmadd231pd zmm, zmm, [mem]{1to8}` instruction
            h8[i] = _mm512_fmadd_pd(_mm512_set1_pd(p_i), s8, h8[i]);

            h1[i] = p_i.mul_add(s1, h1[i]);

            // 1-wide scalar FMA
            // h1[i] += p_i * s1;
        }
    }

    let mut c = [[0.0; 9]; 9];

    // 5. Clean Extraction
    for i in 0..9 {
        _mm512_storeu_pd(c.get_unchecked_mut(i).as_mut_ptr(), h8[i]);
        *c.get_unchecked_mut(i).get_unchecked_mut(8) = h1[i];
    }

    c
}

/// AVX-512 implementation of C = A^T * D^2 * A
///
/// # Safety
/// Target CPU must support the `avx512f` instruction set.
#[target_feature(enable = "avx512f")]
pub unsafe fn at_d2_a_col_major_avx512(
    a: &[f64],
    d: &[f64],
    n: usize,
) -> [[f64; 9]; 9] {
    let mut c_out = [[0.0; 9]; 9];

    let a_ptr = a.as_ptr();
    let d_ptr = d.as_ptr();

    // 2048 doubles = 16 KB (fits cleanly in L1 cache frame)
    const TILE_SIZE: usize = 512;
    let mut scaled_col_i = [0.0f64; TILE_SIZE];

    let mut t = 0;
    while t < n {
        let t_end = (t + TILE_SIZE).min(n);
        let tile_len = t_end - t;
        let chunks_8 = tile_len / 8;
        let rem_8 = tile_len % 8;

        for i in 0..9 {
            let col_i_ptr = a_ptr.add(i * n + t);

            // -----------------------------------------------------------------
            // STEP 1: Pre-scale column i for this tile: scaled_col_i = d^2 * col_i
            // -----------------------------------------------------------------
            for k in 0..chunks_8 {
                let offset = k * 8;
                let vd = _mm512_loadu_pd(d_ptr.add(t + offset));
                let vd2 = _mm512_mul_pd(vd, vd); // d_k^2 (Square diagonal entry)
                let va = _mm512_loadu_pd(col_i_ptr.add(offset));

                let v_scaled = _mm512_mul_pd(va, vd2);
                _mm512_storeu_pd(scaled_col_i.as_mut_ptr().add(offset), v_scaled);
            }

            // Scalar remainder for pre-scaling (< 8 elements)
            let start_rem = chunks_8 * 8;
            for r in start_rem..tile_len {
                let dk = *d_ptr.add(t + r);
                scaled_col_i[r] = (dk * dk) * (*col_i_ptr.add(r));
            }

            // -----------------------------------------------------------------
            // STEP 2: Compute dot product of scaled_col_i with col_j (for j >= i)
            // -----------------------------------------------------------------
            for j in i..9 {
                let col_j_ptr = a_ptr.add(j * n + t);
                let mut acc = _mm512_setzero_pd();

                for k in 0..chunks_8 {
                    let offset = k * 8;
                    // Fast L1 cache load of scaled column i
                    let v_scaled = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset));
                    let vb = _mm512_loadu_pd(col_j_ptr.add(offset));

                    acc = _mm512_fmadd_pd(v_scaled, vb, acc);
                }

                // AVX-512 zero-masking for remainder elements
                if rem_8 > 0 {
                    let offset = chunks_8 * 8;
                    let mask = ((1u16 << rem_8) - 1) as u8;
                    let v_scaled = _mm512_maskz_loadu_pd(mask, scaled_col_i.as_ptr().add(offset));
                    let vb = _mm512_maskz_loadu_pd(mask, col_j_ptr.add(offset));

                    acc = _mm512_fmadd_pd(v_scaled, vb, acc);
                }

                let partial_sum = _mm512_reduce_add_pd(acc);
                c_out[i][j] += partial_sum;
            }
        }

        t = t_end;
    }

    // Mirror upper triangle to lower triangle (Symmetry: C[j][i] = C[i][j])
    for i in 0..9 {
        for j in (i + 1)..9 {
            c_out[j][i] = c_out[i][j];
        }
    }

    c_out
}

#[target_feature(enable = "avx512f")]
pub unsafe fn at_d2_a_col_major_fast_avx512(
    a: &[f64],
    d: &[f64],
    n: usize,
) -> [[f64; 9]; 9] {
    let mut c_out = [[0.0; 9]; 9];

    let a_ptr = a.as_ptr();
    let d_ptr = d.as_ptr();

    // 512 elements * 8 bytes = 4 KB per buffer (100% L1D residency on Zen 5)
    const TILE_SIZE: usize = 512;
    let mut d2_tile = [0.0f64; TILE_SIZE];
    let mut scaled_col_i = [0.0f64; TILE_SIZE];

    let mut t = 0;
    while t < n {
        let t_end = (t + TILE_SIZE).min(n);
        let tile_len = t_end - t;

        let chunks_32 = tile_len / 32;
        let chunks_8 = tile_len / 8;
        let rem_8 = tile_len % 8;

        // -----------------------------------------------------------------
        // OPTIMIZATION 1: Pre-calculate D^2 ONCE per tile
        // -----------------------------------------------------------------
        for k in 0..chunks_8 {
            let offset = k * 8;
            let vd = _mm512_loadu_pd(d_ptr.add(t + offset));
            let vd2 = _mm512_mul_pd(vd, vd);
            _mm512_storeu_pd(d2_tile.as_mut_ptr().add(offset), vd2);
        }
        let start_rem = chunks_8 * 8;
        for r in start_rem..tile_len {
            let dk = *d_ptr.add(t + r);
            d2_tile[r] = dk * dk;
        }

        // Process matrix B
        for i in 0..9 {
            let col_i_ptr = a_ptr.add(i * n + t);

            // Pre-scale column i using d2_tile
            for k in 0..chunks_8 {
                let offset = k * 8;
                let vd2 = _mm512_loadu_pd(d2_tile.as_ptr().add(offset));
                let va = _mm512_loadu_pd(col_i_ptr.add(offset));
                _mm512_storeu_pd(scaled_col_i.as_mut_ptr().add(offset), _mm512_mul_pd(va, vd2));
            }
            for r in start_rem..tile_len {
                scaled_col_i[r] = d2_tile[r] * (*col_i_ptr.add(r));
            }

            // Compute dot product with col_j
            for j in i..9 {
                let col_j_ptr = a_ptr.add(j * n + t);

                // ---------------------------------------------------------
                // OPTIMIZATION 2: 4-Way Accumulator Unrolling (32 elements)
                // ---------------------------------------------------------
                let mut acc0 = _mm512_setzero_pd();
                let mut acc1 = _mm512_setzero_pd();
                let mut acc2 = _mm512_setzero_pd();
                let mut acc3 = _mm512_setzero_pd();

                for k in 0..chunks_32 {
                    let offset = k * 32;

                    let vs0 = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset));
                    let vb0 = _mm512_loadu_pd(col_j_ptr.add(offset));
                    acc0 = _mm512_fmadd_pd(vs0, vb0, acc0);

                    let vs1 = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset + 8));
                    let vb1 = _mm512_loadu_pd(col_j_ptr.add(offset + 8));
                    acc1 = _mm512_fmadd_pd(vs1, vb1, acc1);

                    let vs2 = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset + 16));
                    let vb2 = _mm512_loadu_pd(col_j_ptr.add(offset + 16));
                    acc2 = _mm512_fmadd_pd(vs2, vb2, acc2);

                    let vs3 = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset + 24));
                    let vb3 = _mm512_loadu_pd(col_j_ptr.add(offset + 24));
                    acc3 = _mm512_fmadd_pd(vs3, vb3, acc3);
                }

                let mut total_acc = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));

                // Cleanup remaining 8-element blocks
                let rem_32_start = chunks_32 * 4;
                for k in rem_32_start..chunks_8 {
                    let offset = k * 8;
                    let vs = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset));
                    let vb = _mm512_loadu_pd(col_j_ptr.add(offset));
                    total_acc = _mm512_fmadd_pd(vs, vb, total_acc);
                }

                // AVX-512 zero-masking for final scalar tail (< 8 elements)
                if rem_8 > 0 {
                    let offset = chunks_8 * 8;
                    let mask = ((1u16 << rem_8) - 1) as u8;
                    let vs = _mm512_maskz_loadu_pd(mask, scaled_col_i.as_ptr().add(offset));
                    let vb = _mm512_maskz_loadu_pd(mask, col_j_ptr.add(offset));
                    total_acc = _mm512_fmadd_pd(vs, vb, total_acc);
                }

                c_out[i][j] += _mm512_reduce_add_pd(total_acc);
            }
        }

        t = t_end;
    }

    // Mirror upper triangle to lower triangle
    for i in 0..9 {
        for j in (i + 1)..9 {
            c_out[j][i] = c_out[i][j];
        }
    }
    c_out
}

#[target_feature(enable = "avx512f")]
pub unsafe fn at_d2_a_col_major_untiled_avx512(
    a: &[f64],
    d: &[f64],
    n: usize,
) -> [[f64; 9]; 9] {
    let mut c_out = [[0.0; 9]; 9];

    let a_ptr = a.as_ptr();
    let d_ptr = d.as_ptr();

    let chunks_8 = n / 8;
    let rem_8 = n % 8;

    // Loop over the 45 upper-triangular pairs
    for i in 0..9 {
        let col_i_ptr = a_ptr.add(i * n);

        for j in i..9 {
            let col_j_ptr = a_ptr.add(j * n);
            let mut acc = _mm512_setzero_pd();

            // Single continuous sweep over all N elements in 8-element SIMD vectors
            for k in 0..chunks_8 {
                let offset = k * 8;

                // 1. Load d_k and square it: d_k^2
                let vd = _mm512_loadu_pd(d_ptr.add(offset));
                let vd2 = _mm512_mul_pd(vd, vd);

                // 2. Load A_k,i and A_k,j
                let va = _mm512_loadu_pd(col_i_ptr.add(offset));
                let vb = _mm512_loadu_pd(col_j_ptr.add(offset));

                // 3. acc += (A_k,i * d_k^2) * A_k,j
                let v_scaled_a = _mm512_mul_pd(va, vd2);
                acc = _mm512_fmadd_pd(v_scaled_a, vb, acc);
            }

            // Remainder handling happens ONCE at the very end of N (if N % 8 != 0)
            if rem_8 > 0 {
                let offset = chunks_8 * 8;
                let mask = ((1u16 << rem_8) - 1) as u8;

                let vd = _mm512_maskz_loadu_pd(mask, d_ptr.add(offset));
                let vd2 = _mm512_mul_pd(vd, vd);

                let va = _mm512_maskz_loadu_pd(mask, col_i_ptr.add(offset));
                let vb = _mm512_maskz_loadu_pd(mask, col_j_ptr.add(offset));

                let v_scaled_a = _mm512_mul_pd(va, vd2);
                acc = _mm512_fmadd_pd(v_scaled_a, vb, acc);
            }

            // Reduce 8-lane SIMD vector to a single scalar sum
            c_out[i][j] = _mm512_reduce_add_pd(acc);
        }
    }

    // Mirror upper triangle to lower triangle (Symmetry: C[j][i] = C[i][j])
    for i in 0..9 {
        for j in (i + 1)..9 {
            c_out[j][i] = c_out[i][j];
        }
    }

    c_out
}

#[target_feature(enable = "avx512f")]
pub unsafe fn at_d2_a_col_major_blocked_avx512(a: &[f64], d: &[f64], n: usize) -> [[f64; 9]; 9] {
    let mut c_out = [[0.0f64; 9]; 9];

    let a_ptr = a.as_ptr();
    let d_ptr = d.as_ptr();

    // 512 elements * 8 bytes = 4 KB per buffer (100% L1D cache residency)
    const TILE_SIZE: usize = 512;
    let mut d2_tile = [0.0f64; TILE_SIZE];
    let mut scaled_col_i = [0.0f64; TILE_SIZE];

    let mut t = 0;
    while t < n {
        let t_end = (t + TILE_SIZE).min(n);
        let tile_len = t_end - t;

        let chunks_8 = tile_len / 8;
        let rem_8 = tile_len % 8;

        // -----------------------------------------------------------------
        // STEP 1: Pre-calculate D^2 once per tile
        // -----------------------------------------------------------------
        for k in 0..chunks_8 {
            let offset = k * 8;
            let vd = _mm512_loadu_pd(d_ptr.add(t + offset));
            let vd2 = _mm512_mul_pd(vd, vd);
            _mm512_storeu_pd(d2_tile.as_mut_ptr().add(offset), vd2);
        }
        let start_rem = chunks_8 * 8;
        for r in start_rem..tile_len {
            let dk = *d_ptr.add(t + r);
            d2_tile[r] = dk * dk;
        }

        // -----------------------------------------------------------------
        // STEP 2: Outer loop over Column i
        // -----------------------------------------------------------------
        for i in 0..9 {
            let col_i_ptr = a_ptr.add(i * n + t);

            // Pre-scale column i: scaled_col_i = A_i * D^2
            for k in 0..chunks_8 {
                let offset = k * 8;
                let vd2 = _mm512_loadu_pd(d2_tile.as_ptr().add(offset));
                let va = _mm512_loadu_pd(col_i_ptr.add(offset));
                _mm512_storeu_pd(scaled_col_i.as_mut_ptr().add(offset), _mm512_mul_pd(va, vd2));
            }
            for r in start_rem..tile_len {
                scaled_col_i[r] = d2_tile[r] * (*col_i_ptr.add(r));
            }

            // -------------------------------------------------------------
            // STEP 3: Multi-Column Blocking for Column j (j >= i)
            // -------------------------------------------------------------
            let mut j = i;
            while j < 9 {
                let rem_j = 9 - j;

                if rem_j >= 4 {
                    // 4-Column Microkernel (j, j+1, j+2, j+3)
                    let col_j0_ptr = a_ptr.add((j + 0) * n + t);
                    let col_j1_ptr = a_ptr.add((j + 1) * n + t);
                    let col_j2_ptr = a_ptr.add((j + 2) * n + t);
                    let col_j3_ptr = a_ptr.add((j + 3) * n + t);

                    let mut acc0 = _mm512_setzero_pd();
                    let mut acc1 = _mm512_setzero_pd();
                    let mut acc2 = _mm512_setzero_pd();
                    let mut acc3 = _mm512_setzero_pd();

                    for k in 0..chunks_8 {
                        let offset = k * 8;
                        let vs = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset));

                        let vb0 = _mm512_loadu_pd(col_j0_ptr.add(offset));
                        let vb1 = _mm512_loadu_pd(col_j1_ptr.add(offset));
                        let vb2 = _mm512_loadu_pd(col_j2_ptr.add(offset));
                        let vb3 = _mm512_loadu_pd(col_j3_ptr.add(offset));

                        acc0 = _mm512_fmadd_pd(vs, vb0, acc0);
                        acc1 = _mm512_fmadd_pd(vs, vb1, acc1);
                        acc2 = _mm512_fmadd_pd(vs, vb2, acc2);
                        acc3 = _mm512_fmadd_pd(vs, vb3, acc3);
                    }

                    if rem_8 > 0 {
                        let offset = chunks_8 * 8;
                        let mask = ((1u16 << rem_8) - 1) as u8;
                        let vs = _mm512_maskz_loadu_pd(mask, scaled_col_i.as_ptr().add(offset));

                        let vb0 = _mm512_maskz_loadu_pd(mask, col_j0_ptr.add(offset));
                        let vb1 = _mm512_maskz_loadu_pd(mask, col_j1_ptr.add(offset));
                        let vb2 = _mm512_maskz_loadu_pd(mask, col_j2_ptr.add(offset));
                        let vb3 = _mm512_maskz_loadu_pd(mask, col_j3_ptr.add(offset));

                        acc0 = _mm512_fmadd_pd(vs, vb0, acc0);
                        acc1 = _mm512_fmadd_pd(vs, vb1, acc1);
                        acc2 = _mm512_fmadd_pd(vs, vb2, acc2);
                        acc3 = _mm512_fmadd_pd(vs, vb3, acc3);
                    }

                    c_out[i][j + 0] += _mm512_reduce_add_pd(acc0);
                    c_out[i][j + 1] += _mm512_reduce_add_pd(acc1);
                    c_out[i][j + 2] += _mm512_reduce_add_pd(acc2);
                    c_out[i][j + 3] += _mm512_reduce_add_pd(acc3);

                    j += 4;
                } else if rem_j >= 2 {
                    // 2-Column Microkernel (j, j+1)
                    let col_j0_ptr = a_ptr.add((j + 0) * n + t);
                    let col_j1_ptr = a_ptr.add((j + 1) * n + t);

                    let mut acc0 = _mm512_setzero_pd();
                    let mut acc1 = _mm512_setzero_pd();

                    for k in 0..chunks_8 {
                        let offset = k * 8;
                        let vs = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset));

                        let vb0 = _mm512_loadu_pd(col_j0_ptr.add(offset));
                        let vb1 = _mm512_loadu_pd(col_j1_ptr.add(offset));

                        acc0 = _mm512_fmadd_pd(vs, vb0, acc0);
                        acc1 = _mm512_fmadd_pd(vs, vb1, acc1);
                    }

                    if rem_8 > 0 {
                        let offset = chunks_8 * 8;
                        let mask = ((1u16 << rem_8) - 1) as u8;
                        let vs = _mm512_maskz_loadu_pd(mask, scaled_col_i.as_ptr().add(offset));

                        let vb0 = _mm512_maskz_loadu_pd(mask, col_j0_ptr.add(offset));
                        let vb1 = _mm512_maskz_loadu_pd(mask, col_j1_ptr.add(offset));

                        acc0 = _mm512_fmadd_pd(vs, vb0, acc0);
                        acc1 = _mm512_fmadd_pd(vs, vb1, acc1);
                    }

                    c_out[i][j + 0] += _mm512_reduce_add_pd(acc0);
                    c_out[i][j + 1] += _mm512_reduce_add_pd(acc1);

                    j += 2;
                } else {
                    // 1-Column Tail Microkernel (j)
                    let col_j0_ptr = a_ptr.add(j * n + t);
                    let mut acc0 = _mm512_setzero_pd();

                    for k in 0..chunks_8 {
                        let offset = k * 8;
                        let vs = _mm512_loadu_pd(scaled_col_i.as_ptr().add(offset));
                        let vb0 = _mm512_loadu_pd(col_j0_ptr.add(offset));
                        acc0 = _mm512_fmadd_pd(vs, vb0, acc0);
                    }

                    if rem_8 > 0 {
                        let offset = chunks_8 * 8;
                        let mask = ((1u16 << rem_8) - 1) as u8;
                        let vs = _mm512_maskz_loadu_pd(mask, scaled_col_i.as_ptr().add(offset));
                        let vb0 = _mm512_maskz_loadu_pd(mask, col_j0_ptr.add(offset));
                        acc0 = _mm512_fmadd_pd(vs, vb0, acc0);
                    }

                    c_out[i][j] += _mm512_reduce_add_pd(acc0);

                    j += 1;
                }
            }
        }

        t = t_end;
    }

    // Mirror upper triangle to lower triangle (Symmetry: C[j][i] = C[i][j])
    for i in 0..9 {
        for j in (i + 1)..9 {
            c_out[j][i] = c_out[i][j];
        }
    }

    c_out
}

#[target_feature(enable = "avx512f")]
pub unsafe fn p_d_simd(
    n: usize,
    a: &[f64],
    d: &[f64],
) -> [f64; 9] {

    // 1. Initialize 9 AVX-512 registers (512-bit / 8 double-precision floats each)
    let mut acc = [_mm512_setzero_pd(); 9];

    let chunks = n / 8;
    let remainder = n % 8;

    let d_ptr = d.as_ptr();
    let a_ptr = a.as_ptr();

    // 2. Main Loop: Step through columns in blocks of 8
    for c in 0..chunks {
        let col = c * 8;

        // Load 8 contiguous f64 elements from vector d into register once
        let vd = _mm512_loadu_pd(d_ptr.add(col));

        // Multiply-accumulate across all 9 rows
        for r in 0..9 {
            let va = _mm512_loadu_pd(a_ptr.add(r * n + col));
            // acc[r] += va * vd
            acc[r] = _mm512_fmadd_pd(va, vd, acc[r]);
        }
    }

    // 3. Tail Loop: Handle remaining N % 8 elements using AVX-512 zero-masking
    if remainder > 0 {
        let col = chunks * 8;
        // Generate bitmask for leftover elements (e.g., 3 remaining -> 0b00000111)
        let mask = ((1u16 << remainder) - 1) as u8;

        // Load remaining elements; unselected lanes are automatically set to 0.0
        let vd = _mm512_maskz_loadu_pd(mask, d_ptr.add(col));

        for r in 0..9 {
            let va = _mm512_maskz_loadu_pd(mask, a_ptr.add(r * n + col));
            acc[r] = _mm512_fmadd_pd(va, vd, acc[r]);
        }
    }

    let mut out = [0.0; 9];

    // 4. Horizontal Reduction: Sum 8 lanes in each register into scalar f64
    for r in 0..9 {
        out[r] = _mm512_reduce_add_pd(acc[r]);
    }

    out
}

#[target_feature(enable = "avx512f")]
pub unsafe fn p_d_simd2(
    n: usize,
    a: &[f64],
    d: &[f64],
) -> [f64; 9] {
    let chunks_32 = n / 32;
    let rem_32 = n % 32;

    let mut out = [0.0; 9];

    for r in 0..9 {
        let row_ptr = a.as_ptr().add(r * n);
        let d_ptr = d.as_ptr();

        // 4 independent accumulators to break the 4-cycle FMA dependency chain
        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut acc2 = _mm512_setzero_pd();
        let mut acc3 = _mm512_setzero_pd();

        // 1. Process blocks of 32 double-precision floats (4 x ZMM registers)
        for i in 0..chunks_32 {
            let offset = i * 32;

            acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(row_ptr.add(offset)),    _mm512_loadu_pd(d_ptr.add(offset)),    acc0);
            acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(row_ptr.add(offset+8)),  _mm512_loadu_pd(d_ptr.add(offset+8)),  acc1);
            acc2 = _mm512_fmadd_pd(_mm512_loadu_pd(row_ptr.add(offset+16)), _mm512_loadu_pd(d_ptr.add(offset+16)), acc2);
            acc3 = _mm512_fmadd_pd(_mm512_loadu_pd(row_ptr.add(offset+24)), _mm512_loadu_pd(d_ptr.add(offset+24)), acc3);
        }

        // Combine the 4 accumulators into one
        let mut acc = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));

        // 2. Handle remaining elements (< 32) in 8-element chunks
        let mut rem_offset = chunks_32 * 32;
        let rem_8_chunks = rem_32 / 8;
        let final_rem = rem_32 % 8;

        for _ in 0..rem_8_chunks {
            let va = _mm512_loadu_pd(row_ptr.add(rem_offset));
            let vd = _mm512_loadu_pd(d_ptr.add(rem_offset));
            acc = _mm512_fmadd_pd(va, vd, acc);
            rem_offset += 8;
        }

        // 3. Handle leftover tail (< 8) using zero-masking
        if final_rem > 0 {
            let mask = ((1u16 << final_rem) - 1) as u8;
            let va = _mm512_maskz_loadu_pd(mask, row_ptr.add(rem_offset));
            let vd = _mm512_maskz_loadu_pd(mask, d_ptr.add(rem_offset));
            acc = _mm512_fmadd_pd(va, vd, acc);
        }

        // Horizontal sum for this row
        out[r] = _mm512_reduce_add_pd(acc);
    }

    out
}

#[target_feature(enable = "avx512f")]
pub unsafe fn p_d_simd3(
    n: usize,
    a: &[f64],
    d: &[f64],
) -> [f64; 9] {
    // Zero initialize the output array
    let mut out = [0.0; 9];

    // 4096 double-precision floats = 32 KB (fits cleanly inside L1 data cache)
    const TILE_SIZE: usize = 4096;

    let mut t_start = 0;
    while t_start < n {
        let t_end = (t_start + TILE_SIZE).min(n);
        let tile_len = t_end - t_start;

        let chunks_32 = tile_len / 32;
        let rem_32 = tile_len % 32;

        // Process all 9 rows for the current L1 cache tile
        for r in 0..9 {
            let row_ptr = a.as_ptr().add(r * n + t_start);
            let d_ptr = d.as_ptr().add(t_start);

            // 4 independent accumulators to break FMA instruction dependency latency
            let mut acc0 = _mm512_setzero_pd();
            let mut acc1 = _mm512_setzero_pd();
            let mut acc2 = _mm512_setzero_pd();
            let mut acc3 = _mm512_setzero_pd();

            // 1. Unrolled loop: 32 elements (4 x 512-bit ZMM registers) per iteration
            for i in 0..chunks_32 {
                let offset = i * 32;
                acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(row_ptr.add(offset)),    _mm512_loadu_pd(d_ptr.add(offset)),    acc0);
                acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(row_ptr.add(offset+8)),  _mm512_loadu_pd(d_ptr.add(offset+8)),  acc1);
                acc2 = _mm512_fmadd_pd(_mm512_loadu_pd(row_ptr.add(offset+16)), _mm512_loadu_pd(d_ptr.add(offset+16)), acc2);
                acc3 = _mm512_fmadd_pd(_mm512_loadu_pd(row_ptr.add(offset+24)), _mm512_loadu_pd(d_ptr.add(offset+24)), acc3);
            }

            let mut acc = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));

            // 2. Handle 8-element chunks in tile remainder
            let mut rem_offset = chunks_32 * 32;
            let rem_8 = rem_32 / 8;
            let final_rem = rem_32 % 8;

            for _ in 0..rem_8 {
                let va = _mm512_loadu_pd(row_ptr.add(rem_offset));
                let vd = _mm512_loadu_pd(d_ptr.add(rem_offset));
                acc = _mm512_fmadd_pd(va, vd, acc);
                rem_offset += 8;
            }

            // 3. Handle tail elements (< 8) using zero-masking
            if final_rem > 0 {
                let mask = ((1u16 << final_rem) - 1) as u8;
                let va = _mm512_maskz_loadu_pd(mask, row_ptr.add(rem_offset));
                let vd = _mm512_maskz_loadu_pd(mask, d_ptr.add(rem_offset));
                acc = _mm512_fmadd_pd(va, vd, acc);
            }

            // Accumulate partial tile result into output vector
            out[r] += _mm512_reduce_add_pd(acc);
        }

        t_start = t_end;
    }
    out
}

#[target_feature(enable = "avx512f")]
unsafe fn matrix_vector_mul_nx9_avx512(b: &[f64], x: &[f64; 9], n: usize, y: &mut [f64]) {
    // 1. Pre-load x[0..8] into a 512-bit register ONCE for all N rows
    let vx_0_7 = _mm512_loadu_pd(x.as_ptr());
    let x8 = x[8]; // Keep 9th element in a scalar register

    let chunks_4 = n / 4;
    let remainder = n % 4;

    let b_ptr = b.as_ptr();
    let y_ptr = y.as_mut_ptr();

    // 2. Main Loop: Process 4 rows at a time
    for c in 0..chunks_4 {
        let r = c * 4;

        let row0 = b_ptr.add(r * 9);
        let row1 = b_ptr.add((r + 1) * 9);
        let row2 = b_ptr.add((r + 2) * 9);
        let row3 = b_ptr.add((r + 3) * 9);

        // Load first 8 elements (512-bit) for 4 consecutive rows
        let vb0 = _mm512_loadu_pd(row0);
        let vb1 = _mm512_loadu_pd(row1);
        let vb2 = _mm512_loadu_pd(row2);
        let vb3 = _mm512_loadu_pd(row3);

        // Load 9th scalar element for each row
        let b0_8 = *row0.add(8);
        let b1_8 = *row1.add(8);
        let b2_8 = *row2.add(8);
        let b3_8 = *row3.add(8);

        // Multiply B[i, 0..8] by x[0..8]
        let p0 = _mm512_mul_pd(vb0, vx_0_7);
        let p1 = _mm512_mul_pd(vb1, vx_0_7);
        let p2 = _mm512_mul_pd(vb2, vx_0_7);
        let p3 = _mm512_mul_pd(vb3, vx_0_7);

        // Reduce 8 vector lanes into a scalar and add (B[i, 8] * x[8])
        *y_ptr.add(r)     = _mm512_reduce_add_pd(p0) + (b0_8 * x8);
        *y_ptr.add(r + 1) = _mm512_reduce_add_pd(p1) + (b1_8 * x8);
        *y_ptr.add(r + 2) = _mm512_reduce_add_pd(p2) + (b2_8 * x8);
        *y_ptr.add(r + 3) = _mm512_reduce_add_pd(p3) + (b3_8 * x8);
    }

    // 3. Tail Loop: Process remaining rows (N % 4)
    let start_rem = chunks_4 * 4;
    for r in start_rem..n {
        let row = b_ptr.add(r * 9);
        let vb = _mm512_loadu_pd(row);
        let b8 = *row.add(8);
        let p = _mm512_mul_pd(vb, vx_0_7);
        *y_ptr.add(r) = _mm512_reduce_add_pd(p) + (b8 * x8);
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn mat_vec_mul_col_major_avx512(b: &[f64], x: &[f64; 9], n: usize, y: &mut [f64]) {
    let b_ptr = b.as_ptr();
    let y_ptr = y.as_mut_ptr();

    // 1. Initialize output array y with x[0] * col_0
    let vx0 = _mm512_set1_pd(x[0]);
    let col0_ptr = b_ptr; // Column 0 starts at index 0

    let chunks_8 = n / 8;
    for i in 0..chunks_8 {
        let offset = i * 8;
        let vb = _mm512_loadu_pd(col0_ptr.add(offset));
        _mm512_storeu_pd(y_ptr.add(offset), _mm512_mul_pd(vb, vx0));
    }

    // 2. Accumulate x[j] * col_j for columns 1..8
    for j in 1..9 {
        let vxj = _mm512_set1_pd(x[j]);
        let col_j_ptr = b_ptr.add(j * n); // Column j starts at j * N

        for i in 0..chunks_8 {
            let offset = i * 8;
            let vy = _mm512_loadu_pd(y_ptr.add(offset));
            let vb = _mm512_loadu_pd(col_j_ptr.add(offset));
            // y[i] += b_j[i] * x[j]
            let res = _mm512_fmadd_pd(vb, vxj, vy);
            _mm512_storeu_pd(y_ptr.add(offset), res);
        }
    }

    // Tail loop for remaining N % 8 elements
    let start_rem = chunks_8 * 8;
    for i in start_rem..n {
        let mut sum = 0.0;
        for j in 0..9 {
            sum += b[j * n + i] * x[j];
        }
        y[i] = sum;
    }
}

pub fn solve_sqp(
    p_mat: &MatrixNx9<f64>,
    p_mat_t: &Matrix9xN<f64>,
    x0: &Vector9<f64>,
    d: &mut DVector<f64>,
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
        let g = gradient::compute_grad(p_mat_t, &x, tune.epsilon);

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

        // let h = compute_hessian(p_mat, d, a_mat);
        // let h = compute_hessian_t(p_mat_t, d);
        // let h = compute_hessian2(p_mat, d);
        let h = hessian::compute_hess(p_mat_t, &x, tune.epsilon);

        // c = g - H x
        let mut c = g.clone();
        c.gemv(-1.0, &h, &x, 1.0);

        let (y, qp_iter) = solve_qp_active_set(&h, &c, &x, false, true, tune);

        let f = objective::compute_obj(p_mat_t, &x, tune.epsilon);

        let (xnew, bls_iter) = backtracking_line_search(p_mat, p_mat_t, &x, &y, f, &g, d, tune);

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
    p_mat: &MatrixNx9<f64>,
    p_mat_t: &Matrix9xN<f64>,
    x: &Vector9<f64>,
    y: &Vector9<f64>,
    f: f64,
    g: &Vector9<f64>,
    d: &mut DVector<f64>,
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
        // let fnew = compute_obj(p_mat, &xnew, d, tune.epsilon);
        // let fnew = compute_obj_scalar(p_mat, &xnew, tune.epsilon);
        let fnew = objective::compute_obj(p_mat_t, &xnew, tune.epsilon);
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
