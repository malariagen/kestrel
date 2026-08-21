use ndarray::{Array2, Array4};

use crate::{
    algebra::{Matrix, Vector, dot, mul, outer, scale_div, },
    sqp::{self, Tuneables},
};

pub fn calculate_allele_probabilities(likelihoods: &[Vec<Matrix<4>>]) {
    for likel in likelihoods.iter() {
        let x0 = [0.25; 4];

        let obj = |x: &Vector<4>, eps| calculate_objective(&likel, &x, eps);
        let grad_hess = |x: &Vector<4>, eps| calculate_grad_hess(&likel, &x, eps);

        let (x, _) = sqp::solve_sqp(obj, grad_hess, &x0, &Tuneables::new());

        if x.iter().filter(|&i| *i > 0.0).count() >= 3 {
            println!("MULTI");
        }

        println!("{:?}", x);
    }
}

// pub fn calculate_allele_prob(
//     sample_likelihoods: &Array3<f64>
// ) -> Vector<4> {

// }

pub fn calculate_objective(likelihoods: &[Matrix<4>], x: &Vector<4>, eps: f64) -> f64 {
    let mut s = 0.0;

    for mat in likelihoods.iter() {
        let p = dot(x, &mul(mat, x));
        s += (p + eps).ln();
    }

    let n = likelihoods.len();

    -s / (n as f64)
}

// Instead of a matrix with 1 elem in each spot
// You have a matrix with 4 elems in each spot
// Need to make a LanedMatrix and LanedVector class
// Then you just manually unroll the accumulator loops
// To simplify life, we could just use the Lane8 class
// And store the leftovers separately. I think that makes sense.
// Then we don't need a custom allocator class anymore.
// In fact we could just have a vector of length 10 to store the
// elements, and yeah, nice

pub fn calculate_grad_hess(likelihoods: &[Matrix<4>], x: &Vector<4>, eps: f64) -> (Vector<4>, Matrix<4>) {
    let mut g = [0.0; 4];
    let mut h = [[0.0; 4]; 4];

    for l in likelihoods.iter() {
        let lx = mul(l, x);
        let d = dot(x, &lx) + eps;

        let lxd = scale_div(d, &lx);

        for i in 0..4 {
            g[i] += lxd[i];
        }

        let o = outer(&lxd, &lxd);

        for i in 0..4 {
            for j in 0..4 {
                h[i][j] += 2.0 * o[i][j] - l[i][j] / d;
            }
        }
    }

    let n = likelihoods.len();

    let grad = std::array::from_fn(|i| -2.0 * g[i] / (n as f64));

    let mut hess = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            hess[i][j] = 2.0 * h[i][j] / (n as f64);
        }
    }

    (grad, hess)
}
