use std::path::Path;

use anyhow::{Context, Result, bail};
use itertools::Itertools;
use ndarray::{Array2, Array3};
use noodles::vcf::variant::record::info::field::Value as InfoValue;
use noodles::vcf::variant::record::info::field::value::Array as InfoArray;
use noodles::vcf::variant::record::samples::Series;
use noodles::vcf::variant::record::samples::series::value::Value as SeriesValue;

pub fn parse_vcf(file: &Path) -> Result<(Array3<i8>, Array2<f64>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let num_samples = header.sample_names().len();

    // println!("{:?}", header);

    let maf = 0.01;

    let mut genotypes = Vec::<Vec<(i8, i8)>>::new(); // V x S x 2
    let mut allele_frequencies = Vec::<(f64, f64)>::new(); // V x 2 (bi-allelic)
    // let mut genotypes: Vec<(i8, i8)> = Vec::new();

    for result in reader.records() {
        let record = result?;

        let info = record.info();
        let af_info = info
            .get(&header, "AF")
            .context("No AF data found")??
            .context("No AF data found")?;

        if let InfoValue::Array(af_array) = af_info {
            if let InfoArray::Float(af_float) = af_array {
                if let Some((af,)) = af_float.iter().collect_tuple() {
                    let af: f64 = af?.context("No AF data found")?.into();
                    if af < maf || 1.0 - af < maf {
                        continue;
                    }
                    allele_frequencies.push((1.0 - af, af));
                } else {
                    // Not bi-allelic
                    continue;
                }
            } else {
                bail!("Value {:?} is not an array", af_array);
            }
        } else {
            bail!("Value {:?} is not an array", af_info);
        }

        let samples = record.samples();
        let gt_series = samples.select("GT").context("No GT data found")?;

        let mut variants = Vec::with_capacity(num_samples);

        // TODO remove unknown data, and deal with MNPs and in/dels

        for result in gt_series.iter(&header) {
            let value = result?.context("No genotype for sample found")?;
            if let SeriesValue::Genotype(gt) = value {
                if let Some((a0, a1)) = gt.iter().collect_tuple() {
                    let a0 = a0?
                        .0
                        .map_or(Ok(-1), |i| i8::try_from(i))
                        .context("Allele variant is greater than 127")?;
                    let a1 = a1?
                        .0
                        .map_or(Ok(-1), |i| i8::try_from(i))
                        .context("Allele variant is greater than 127")?;
                    variants.push((a0, a1))
                    // genotypes.push((a0, a1))
                } else {
                    bail!("Genotype {:?} is not diploid", gt);
                }
            } else {
                bail!("Value {:?} is not a genotype", value);
            }
        }

        genotypes.push(variants);
    }

    // println!("{:?}", genotypes.len());
    // println!("{:?}", genotypes.len());
    // println!("{:?}", allele_frequencies.len());

    let num_variants = genotypes.len();
    let mut gt = Array3::<i8>::zeros((num_variants, num_samples, 2));
    for v in 0..num_variants {
        let samples = &genotypes[v];
        for s in 0..samples.len() {
            let (a0, a1) = samples[s];
            gt[[v, s, 0]] = a0;
            gt[[v, s, 1]] = a1;
        }
    }

    let mut af = Array2::<f64>::zeros((num_variants, 2));
    for v in 0..num_variants {
        let (f0, f1) = allele_frequencies[v];
        af[[v, 0]] = f0;
        af[[v, 1]] = f1;
    }

    Ok((gt, af))
}
