use std::path::Path;

use ndarray::Axis;

fn main() {
    divan::main();
}

#[divan::bench]
fn bench(bencher: divan::Bencher) {
    let file = Path::new("../relatedness/sims/human/first_cousin_n50.vcf.gz");
    let (gt, af) = kestrel::vcf::parse_vcf(file).unwrap();

    // let v = vec![af.view(); 10];
    // let af = ndarray::concatenate(Axis(0), &v).unwrap();

    bencher.bench(|| {
        kestrel::jacquard::calculate_relatedness_coefficients(&gt, &af);
    })
}
