use std::path::PathBuf;

use anyhow::Result;
use csv::Trim::All;
use csv::WriterBuilder;
use clap::{Parser, ValueEnum};
use paralight::threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

use crate::coefficients::Output;
use crate::{algebra::dot, arith::simd::Simd};

#[derive(Parser)]
#[command(name = "kestrel", version, about)]
struct Args {
    /// VCF file with genotype likelihoods (GL, PL) or genotypes (GT)
    input: PathBuf,

    /// Tab-separated file to write the coefficients to
    output: PathBuf,

    /// Where the allele frequencies come from
    #[arg(short = 'f', long, value_enum, default_value_t = AlleleFreqs::Estimate)]
    allele_freqs: AlleleFreqs,

    /// Which FORMAT tag to compute kinship from
    #[arg(short = 't', long, value_enum, default_value_t = Tag::GL)]
    tag: Tag,
}

#[derive(Clone, Copy, ValueEnum)]
#[value(rename_all = "UPPERCASE")]
enum Tag {
    /// Log-scaled genotype likelihoods
    GL,
    /// Phred-scaled genotype likelihoods
    PL,
    /// Hard-called genotypes
    GT,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum AlleleFreqs {
    /// Estimate the frequencies on the fly from the samples in the VCF
    Estimate,
    /// Read pre-computed frequencies from the AF INFO field
    Info,
}

// TODO switch println to logger

pub fn run_cli(args: &[String]) -> Result<()> {

    let args = Args::parse_from(args);

    let simd = Simd::detect();

    let threads = std::thread::available_parallelism()?;

    println!("Using thread pool with {threads} threads");

    let mut thread_pool = ThreadPoolBuilder {
        num_threads: ThreadCount::Count(threads),
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::No,
    }
    .build();

    let vcf_file = &args.input;

    println!("Parsing VCF {:?}", vcf_file);

    let parse_af = args.allele_freqs == AlleleFreqs::Info;

    let (samples, outputs) = match args.tag {
        Tag::GL => {
            let (samples, gl) = crate::vcf::parse_vcf_gl(vcf_file)?;
            let af = crate::allele::calculate_allele_frequencies(&gl, &mut thread_pool, simd);
            let outputs = crate::coefficients::calculate_relatedness_coefficients_gl(gl, &af, &mut thread_pool, simd);
            (samples, outputs)
        },
        Tag::PL => {
            let (samples, gl) = crate::vcf::parse_vcf_pl(vcf_file)?;
            let af = crate::allele::calculate_allele_frequencies(&gl, &mut thread_pool, simd);
            let outputs = crate::coefficients::calculate_relatedness_coefficients_gl(gl, &af, &mut thread_pool, simd);
            (samples, outputs)
        },
        Tag::GT => {
            let (samples, gt, af) = crate::vcf::parse_vcf_gt(vcf_file, parse_af)?;
            let af = af.unwrap_or_else(|| crate::coefficients::calculate_allele_frequencies(gt.view()));
            let outputs = crate::coefficients::calculate_coefficients_gt(gt.view(), af.view(), &mut thread_pool, simd);
            (samples, outputs)
        },
    };

    write_output(args.output, &samples, &outputs)?;

    Ok(())
}

fn write_output(ofile: PathBuf, samples: &[String], outputs: &[Output]) -> Result<()> {

    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(ofile)?;

    writer.write_record([
        "sample1",
        "sample2",
        "kinship",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "delta5",
        "delta6",
        "delta7",
        "delta8",
        "delta9",
        "convergence",
        "iterations",
        "objective",
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
        let objective = out.obj;
        // TODO condition number of matrix
        // let fit = if perplexity < 8.0 { "good" } else { "poor" };

        writer.serialize((
            sample1,
            sample2,
            kinship,
            delta1,
            delta2,
            delta3,
            delta4,
            delta5,
            delta6,
            delta7,
            delta8,
            delta9,
            convergence,
            iterations,
            objective,
        ))?;
    }

    writer.flush()?;

    Ok(())
}
