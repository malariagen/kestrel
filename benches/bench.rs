use kestrel::util::{MatrixNx9, Vector9};
use nalgebra::DVector;
use ndarray::Axis;

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

const N: usize = 100000;

fn generate_mat() -> MatrixNx9<f64> {

    let mat = MatrixNx9::<f64>::new_random(N);

    mat
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

#[divan::bench]
fn bench_grad(bencher: divan::Bencher) {

    let eps = 1e-8;

    let x = Vector9::from_element(1.0 / 9.0);

    bencher
    .with_inputs(generate_mat)
    .bench_local_refs(move |mat| {
        kestrel::sqp::compute_grad(&mat, &x, eps);
    });
}

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