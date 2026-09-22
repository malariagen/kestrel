// mod sqp;
// pub mod vcf;

use std::io::Write;
use std::{fs::File, io::BufWriter, path::Path};

use anyhow::Result;
use ndarray::{Axis, concatenate};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let vcf_file = Path::new(&args[1]);

    // let gl = kestrel::vcf::parse_vcf_gl(vcf_file)?;
    // kestrel::allele::calculate_allele_probabilities(&gl);
    // return Ok(());

    let (samples, gt, af) = kestrel::vcf::parse_vcf(vcf_file)?;

    // let gt = concatenate(Axis(0), &[gt.view(), gt.view(), gt.view()]).unwrap();
    // let af = concatenate(Axis(0), &[af.view(), af.view(), af.view()]).unwrap();

    // let kinship = kestrel::coefficients::calculate_relatedness_coefficients(&gt, &af);
    let kinship = kestrel::coefficients::calculate_relatedness_coefficients_no_freq(gt.view().into());

    println!("sum {}", kinship.sum());

    let out_file = File::create(&args[2])?;

    let mut writer = BufWriter::new(out_file);
    writeln!(writer, "sample1 sample2 kinship")?;
    let s = kinship.shape()[0];
    for i in 0..s {
        for j in i..s {
            writeln!(writer, "{} {} {}", samples[i], samples[j], kinship[(i, j)])?;
        }
    }

    writer.flush()?;

    Ok(())
}
