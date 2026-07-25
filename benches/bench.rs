use kestrel::{matrix::BlockArray, util::{Matrix9xN, MatrixNx9, Vector9}};
use nalgebra::DVector;
use ndarray::Axis;
use rand::{Rng, RngExt};

fn main() {
    divan::main();
}

// #[divan::bench]
// fn bench(bencher: divan::Bencher) {
//     let file = Path::new("../relatedness/sims/human/first_cousin_n50.vcf.gz");
//     let (gt, af) = kestrel::vcf::parse_vcf(file).unwrap();

//     // let v = vec![af.view(); 10];
//     // let af = ndarray::concatenate(Axis(0), &v).unwrap();

//     bencher.bench(|| {
//         kestrel::jacquard::calculate_relatedness_coefficients(&gt, &af);
//     })
// }

const N: usize = 1000000;

fn generate_mat() -> MatrixNx9<f64> {

    let mat = MatrixNx9::<f64>::new_random(N);

    mat
}

fn generate_mat_t() -> Matrix9xN<f64> {

    let mat = Matrix9xN::<f64>::new_random(N);

    mat
}

pub fn generate_random_block() -> BlockArray<f64> {
    let mut rng = rand::rng();

    let mut v = vec![[0.0; 9]; N];
    for i in 0..N {
        for j in 0..9 {
            v[i][j] = rng.random();
        }
        // rng.fill(&mut v[i]);
    }

    BlockArray::from_rows::<8, 9>(v.iter().copied(), v.len())
}

// #[divan::bench]
// fn bench_h(bencher: divan::Bencher) {

//     let eps = 1e-8;

//     let x = [1.0/9.0; 9];

//     bencher
//     .with_inputs(generate_mat_t)
//     .bench_local_refs(move |mat| {
//         let (p, _) = mat.as_slice().as_chunks::<9>();
//         kestrel::hessian::compute_pt_d2_p_scalar(&p, &x, eps);
//     });
// }

// #[divan::bench]
// fn bench_h2(bencher: divan::Bencher) {

//     let eps = 1e-8;

//     let x = [1.0/9.0; 9];

//     bencher
//     .with_inputs(generate_mat_t)
//     .bench_local_refs(move |mat| {
//         let (p, _) = mat.as_slice().as_chunks::<9>();
//         kestrel::hessian::compute_pt_d2_p_scalar2(&p, &x, eps);
//     });
// }

#[divan::bench]
fn bench_column(bencher: divan::Bencher) {

    let eps = 1e-8;

    let x = [1.0/9.0; 9];

    let mat = generate_mat();

    bencher
    // .with_inputs(generate_mat())
    .bench_local(|| {
        let p = mat.as_slice();
        unsafe { kestrel::simd::compute_pt_d_scalar_column(N, &p, &x, eps) }
    });
}

#[divan::bench]
fn bench_blocked(bencher: divan::Bencher) {

    let eps = 1e-8;

    let x = [1.0/9.0; 9];

    let block = generate_random_block();

    bencher
    // .with_inputs(generate_random_block())
    .bench_local(|| {
        let (blocks, r) = block.as_blocks::<8, 9>();
        unsafe { kestrel::gradient::compute_pt_d_avx512_blocked(blocks, r, &x, eps) }
    });
}


#[divan::bench]
fn bench_grad_d(bencher: divan::Bencher) {
    let mut d = DVector::<f64>::zeros(N);

    let eps = 1e-8;

    let x = Vector9::from_element(1.0 / 9.0);

    bencher
    .with_inputs(generate_mat)
    .bench_local_refs(move |mat| {
        kestrel::sqp::compute_grad_d(&mat, &x, &mut d, eps);
    });
}

// #[divan::bench]
// fn bench_grad(bencher: divan::Bencher) {

//     let eps = 1e-8;

//     let x = Vector9::from_element(1.0 / 9.0);

//     bencher
//     .with_inputs(generate_mat)
//     .bench_local_refs(move |mat| {
//         kestrel::sqp::compute_grad(&mat, &x, eps);
//     });
// }

#[divan::bench]
fn bench_grad_nalgebra(bencher: divan::Bencher) {
    let mut d = DVector::<f64>::zeros(N);

    let eps = 1e-8;

    let x = Vector9::from_element(1.0 / 9.0);

    bencher
    .with_inputs(generate_mat)
    .bench_local_refs(move |mat| {
        kestrel::sqp::compute_grad_d_nalgebra(&mat, &x, &mut d, eps);
    });
}

// #[divan::bench]
// fn bench_column(bencher: divan::Bencher) {

//     let eps = 1e-8;

//     let x = [1.0 / 9.0; 9];

//     bencher
//     .with_inputs(generate_mat)
//     .bench_local_refs(move |mat| {
//         unsafe { kestrel::simd::compute_pt_d_scalar(mat.nrows(), &mat.as_slice(), &x, eps) };
//     });
// }

// #[divan::bench]
// fn bench_column_row(bencher: divan::Bencher) {

//     let eps = 1e-8;

//     let x = [1.0 / 9.0; 9];

//     bencher
//     .with_inputs(generate_mat_t)
//     .bench_local_refs(move |mat| {
//         unsafe { kestrel::sqp::compute_pt_d_scalar_row(mat.ncols(), &mat.as_slice(), &x, eps) };
//     });
// }