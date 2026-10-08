use std::{num::NonZeroUsize, path::Path};
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use clap::{Parser, ValueEnum};
use csv::WriterBuilder;
use crate::{algebra::dot, arith::simd::Simd};
use paralight::threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

#[derive(Parser)]
#[command(name = "kestrel", version, about)]
struct Args {
    /// VCF file with genotype likelihoods (GL) or genotypes (GT)
    input: PathBuf,

    /// Tab-separated file to write the coefficients to
    output: PathBuf,

    /// Where the allele frequencies come from
    #[arg(short = 'f', long, value_enum, default_value_t = AlleleFreqs::Estimate)]
    allele_freqs: AlleleFreqs,

    /// Which FORMAT tag to compute kinship from
    #[arg(short = 't', long, value_enum, default_value_t = Tag::Gl)]
    tag: Tag,
}

#[derive(Clone, Copy, ValueEnum)]
enum Tag {
    /// Genotype likelihoods
    Gl,
    /// Hard-called genotypes
    Gt,
}

#[derive(Clone, Copy, ValueEnum)]
enum AlleleFreqs {
    /// Estimate the frequencies on the flyfrom the samples in the VCF
    Estimate,
    /// Read pre-computed frequencies from the AF INFO field
    Info,
}

pub fn run_cli(args: &[String]) -> Result<()> {
    let args = Args::try_parse_from(args).unwrap_or_else(|e| e.exit());

    if let AlleleFreqs::Info = args.allele_freqs {
        bail!("--allele-freqs info is not implemented yet");
    }

    if let Tag::Gt = args.tag {
        return run_gt(args.input.as_path(), args.output.as_path());
    }

    let simd = Simd::detect();

    // let vcf_file = Path::new(&args[1]);
    let vcf_file: &Path = args.input.as_path();

    println!("Parsing VCF {:?}", vcf_file);

    let (samples, gl) = crate::vcf::parse_vcf_gl(vcf_file)?;

    let mut thread_pool = ThreadPoolBuilder {
        num_threads: ThreadCount::Count(NonZeroUsize::new(10).unwrap()),
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::No,
    }
    .build();

    let af = crate::allele::calculate_allele_frequencies(&gl, &mut thread_pool, simd);

    let outputs = crate::coefficients::calculate_relatedness_coefficients_gl(gl, &af, &mut thread_pool, simd);

    // return Ok(());

    // let (samples, gt, af) = kestrel::vcf::parse_vcf(vcf_file)?;

    // let gt = concatenate(Axis(0), &[gt.view(), gt.view(), gt.view()]).unwrap();
    // let af = concatenate(Axis(0), &[af.view(), af.view(), af.view()]).unwrap();

    // let kinship = kestrel::coefficients::calculate_relatedness_coefficients(&gt, &af);
    // let kinship = kestrel::coefficients::calculate_relatedness_coefficients_no_freq(gt.view().into());

    // println!("sum {}", kinship.sum());

    // let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(&args[2])?;
    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(&args.output)?;

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
        "perplexity", // I am often perplexed
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
        let perplexity = (out.obj).unwrap().exp();
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

// Same output as the GL route, using the hard-called genotype solver
fn run_gt(vcf_file: &Path, output: &Path) -> Result<()> {
    println!("Parsing VCF {:?}", vcf_file);

    let (samples, gt) = crate::vcf::parse_vcf_gt(vcf_file)?;

    // S x S x 9, only filled where the first sample index <= the second
    let jacquard_mat = crate::coefficients::calculate_relatedness_coefficients_gt(gt.view());

    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(output)?;

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

    for x in 0..samples.len() {
        for y in x..samples.len() {
            let sample1 = &samples[x];
            let sample2 = &samples[y];

            let jacquard: [f64; 9] = std::array::from_fn(|i| jacquard_mat[(x, y, i)]);

            let kinship = dot(&jacquard, &kinship_vec);

            // The genotype solver only returns the coefficients. Fit deets are onlty defined for GLs
            writer.serialize((sample1, sample2, jacquard, kinship, "NA", "NA", "NA", "NA"))?;
        }
    }

    writer.flush()?;

    Ok(())
}
