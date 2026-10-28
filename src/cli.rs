use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use csv::WriterBuilder;
use env_logger::{Builder, Env};

use crate::algebra::dot;
use crate::coefficients::Output;
use crate::conditional::IIS_3;

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

    /// Number of threads used for parallelism, defaults to all available threads
    #[arg(short = 'p', long)]
    threads: Option<NonZeroUsize>,

    /// Maximum number of iterations for minimization algorithm
    #[arg(short = 'i', long, default_value = "100")]
    iterations: NonZeroU64,

    /// Allele frequencies below this threshold are set to zero
    #[arg(short = 'm', long)]
    minimum_allele_frequency: Option<f64>,
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

pub fn run_cli(args: &[String]) -> Result<()> {
    Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .format_target(false)
        .init();

    for (i, a) in IIS_3.iter().enumerate() {
        for (j, b) in a.iter().enumerate() {
            println!("{:?} {:?} - {i} {j} - {:?}", b.0, b.1, b.2);
        }
    }

    panic!();

    let args = Args::parse_from(args);

    let vcf_file = &args.input;

    log::info!("Parsing VCF {:?}", vcf_file);

    let parse_af = args.allele_freqs == AlleleFreqs::Info;

    let iters = args.iterations;
    let maf = args.minimum_allele_frequency;

    let (samples, outputs) = match args.tag {
        Tag::GL => {
            let (samples, gl, af) = crate::vcf::parse_vcf_gl(vcf_file, parse_af)?;
            let af = af.as_ref().map(|a| a.view());
            let outputs =
                crate::wrapper::calculate_relatedness_coefficients_gl(gl.view(), af, args.threads, iters, maf);
            (samples, outputs)
        }
        Tag::PL => {
            let (samples, gl, af) = crate::vcf::parse_vcf_pl(vcf_file, parse_af)?;
            let af = af.as_ref().map(|a| a.view());
            let outputs =
                crate::wrapper::calculate_relatedness_coefficients_gl(gl.view(), af, args.threads, iters, maf);
            (samples, outputs)
        }
        Tag::GT => {
            let (samples, gt, af) = crate::vcf::parse_vcf_gt(vcf_file, parse_af)?;
            let af = af.as_ref().map(|a| a.view());
            let outputs =
                crate::wrapper::calculate_relatedness_coefficients_gt(gt.view(), af, args.threads, args.iterations);
            (samples, outputs)
        }
    };

    write_output(args.output, &samples, &outputs)?;

    Ok(())
}

fn write_output(ofile: PathBuf, samples: &[String], outputs: &[Vec<Output>]) -> Result<()> {
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
        "iterations",
        "objective",
    ])?;

    let kinship_vec = [1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0];

    for out in outputs.iter().flatten() {
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
        let objective = out.obj;
        // TODO condition number of matrix

        writer.serialize((
            sample1, sample2, kinship, delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta9,
            iterations, objective,
        ))?;
    }

    writer.flush()?;

    Ok(())
}
