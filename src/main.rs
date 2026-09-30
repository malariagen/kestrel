// mod sqp;
// pub mod vcf;

use std::{num::NonZeroUsize, path::Path};

use anyhow::Result;
use csv::WriterBuilder;
use kestrel::{algebra::dot, arith::simd::Simd};
use paralight::threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let simd = Simd::detect();

    let vcf_file = Path::new(&args[1]);

    println!("Parsing VCF {:?}", vcf_file);

    let (samples, gl) = kestrel::vcf::parse_vcf_gl(vcf_file)?;

    let mut thread_pool = ThreadPoolBuilder {
        num_threads: ThreadCount::Count(NonZeroUsize::new(10).unwrap()),
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::No,
    }
    .build();

    let af = kestrel::allele::calculate_allele_frequencies(&gl, &mut thread_pool, simd);

    let outputs = kestrel::coefficients::calculate_relatedness_coefficients_gl(gl, &af, &mut thread_pool, simd);

    // return Ok(());

    // let (samples, gt, af) = kestrel::vcf::parse_vcf(vcf_file)?;

    // let gt = concatenate(Axis(0), &[gt.view(), gt.view(), gt.view()]).unwrap();
    // let af = concatenate(Axis(0), &[af.view(), af.view(), af.view()]).unwrap();

    // let kinship = kestrel::coefficients::calculate_relatedness_coefficients(&gt, &af);
    // let kinship = kestrel::coefficients::calculate_relatedness_coefficients_no_freq(gt.view().into());

    // println!("sum {}", kinship.sum());

    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(&args[2])?;

    writer.write_record([
        "sample1",
        "sample2",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "delta5",
        "delta6",
        "delta7",
        "delta8",
        "delta9",
        "kinship",
        "convergence",
        "iterations",
        "fit",
        "perplexity",
    ])?;

    let kinship_vec = [1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0];

    for out in outputs.iter() {
        let sample1 = &samples[out.x];
        let sample2 = &samples[out.y];

        let delta1 = out.jacquard[0];
        let delta2 = out.jacquard[1];
        let delta3 = out.jacquard[2];
        let delta4 = out.jacquard[3];
        let delta5 = out.jacquard[4];
        let delta6 = out.jacquard[5];
        let delta7 = out.jacquard[6];
        let delta8 = out.jacquard[7];
        let delta9 = out.jacquard[8];

        let kinship = dot(&out.jacquard, &kinship_vec);

        let iterations = out.iters;

        // TODO check this
        let convergence = iterations < 100;
        // TODO warning about perplexity > 8?
        let perplexity = (out.obj).exp();
        // TODO condition number of matrix
        let fit = if perplexity < 8.0 { "good" } else { "poor" };

        writer.serialize((
            sample1,
            sample2,
            delta1,
            delta2,
            delta3,
            delta4,
            delta5,
            delta6,
            delta7,
            delta8,
            delta9,
            kinship,
            convergence,
            iterations,
            fit,
            perplexity,
        ))?;
    }

    writer.flush()?;

    Ok(())
}
