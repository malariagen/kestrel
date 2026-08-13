// mod sqp;
// pub mod vcf;

use std::io::Write;
use std::{fs::File, io::BufWriter, path::Path};

use anyhow::Result;
use ndarray::{Axis, concatenate};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let vcf_file = Path::new(&args[1]);

    let gl = kestrel::vcf::parse_vcf_gl(vcf_file)?;
    kestrel::allele::calculate_allele_probabilities(&gl);
    return Ok(());

    let (gt, af) = kestrel::vcf::parse_vcf(vcf_file)?;

    // let gt = concatenate(Axis(0), &[gt.view(), gt.view(), gt.view()]).unwrap();
    // let af = concatenate(Axis(0), &[af.view(), af.view(), af.view()]).unwrap();

    let kinship = kestrel::jacquard::calculate_relatedness_coefficients(&gt, &af);

    println!("sum {}", kinship.sum());

    let out_file = File::create(&args[2])?;

    let mut writer = BufWriter::new(out_file);
    writeln!(writer, "ID1 ID2 kinship")?;
    let s = kinship.shape()[0] / 2;
    for i in 0..s {
        let a = 2 * i;
        let b = a + 1;
        writeln!(writer, "{} {} {}", a, b, kinship[(a, b)])?;
    }

    writer.flush()?;

    Ok(())
}
