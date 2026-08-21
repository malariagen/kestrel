use std::path::Path;

use anyhow::{Context, Result, bail};
use itertools::Itertools;
use ndarray::{Array2, Array3, Array4};
use noodles::vcf::variant::record::info::field::Value as InfoValue;
use noodles::vcf::variant::record::info::field::value::Array as InfoArray;
use noodles::vcf::variant::record::samples::Series;
use noodles::vcf::variant::record::samples::series::value::Array as SeriesArray;
use noodles::vcf::variant::record::samples::series::value::Value as SeriesValue;

use crate::algebra::Matrix;

pub fn parse_vcf(file: &Path) -> Result<(Vec<String>, Array3<i8>, Array2<f64>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = samples.len();

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

    Ok((samples, gt, af))
}

// pub fn parse_vcf_gl(file: &Path) -> Result<Array4<f64>> {
pub fn parse_vcf_gl(file: &Path) -> Result<Vec<Vec<Matrix<4>>>> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let num_samples = header.sample_names().len();

    let mut skipped = 0;

    let mut likelihoods = Vec::<Vec<[f64; 10]>>::new(); // V x S x 10

    'variant: for result in reader.records() {
        let record = result?;

        let samples = record.samples();
        let gl_series = samples.select("GL").context("No GL data found")?;

        let mut variant_gls = Vec::with_capacity(num_samples);

        // TODO remove unknown data, and deal with MNPs and in/dels

        for result in gl_series.iter(&header) {
            let value = result?.context("No genotype likelihood for sample found")?;
            if let SeriesValue::Array(gl_array) = value {
                if let SeriesArray::Float(gl_float) = gl_array {
                    if let Some(option_gl) = gl_float.iter().collect_array::<10>() {
                        let mut gl = [0.0; 10];
                        for (out, item) in gl.iter_mut().zip(option_gl) {
                            // let val = item?.context("Error reading genotype likelihood");
                            let val = item.unwrap();
                            // vcfgl produced 0:.,.,.,.,.,.,.,.,.,. at a site for one sample
                            // so just skip that site
                            match val {
                                Some(v) => *out = v.into(),
                                None => {
                                    skipped += 1;
                                    continue 'variant;
                                }
                            }
                        }
                        // println!("{} {:?}", record.variant_start().unwrap().unwrap(), gl);
                        // // TODO error handling
                        // let gl: [f64; 10] = option_gl.map(|item| item.transpose());
                        variant_gls.push(gl)
                    } else {
                        // Not 4-allelic
                        continue 'variant;
                    }
                } else {
                    bail!("Array {:?} does not contain floats", gl_array);
                }
            } else {
                bail!("Value {:?} is not an array", value);
            }
        }

        likelihoods.push(variant_gls);
    }

    let num_variants = likelihoods.len();

    println!("Parsed {} variants, skipped {}", num_variants, skipped);

    // let mut gls = Array4::<f64>::zeros((num_variants, num_samples, 4, 4));
    // for v in 0..num_variants {
    //     for s in 0..num_samples {
    //         let sample_gls = likelihoods[v][s];
    //         // Normalize by the maximum GL to avoid possible underflow
    //         // (This matches what PL does)
    //         let max_gl = sample_gls.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    //         for i in 0..4 {
    //             for j in i..4 {
    //                 // The index of (i, j) where i <= j (see the VCF spec)
    //                 let index = j*(j+1)/2 + i;
    //                 let gl = sample_gls[index];
    //                 let prob = 10.0f64.powf(gl - max_gl);
    //                 gls[[v, s, i, j]] = prob;
    //                 gls[[v, s, j, i]] = prob;
    //             }
    //         }
    //     }
    // }

    let mut gls = Vec::with_capacity(num_variants);
    for v in 0..num_variants {
        let mut a = Vec::with_capacity(num_samples);
        for s in 0..num_samples {
            let sample_gls = likelihoods[v][s];
            // Normalize by the maximum GL to avoid possible underflow
            // (This matches what PL does)
            let max_gl = sample_gls.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
            let mut m = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in i..4 {
                    // The index of (i, j) where i <= j (see the VCF spec)
                    let index = j * (j + 1) / 2 + i;
                    let gl = sample_gls[index];
                    let prob = 10.0f64.powf(gl - max_gl);
                    m[i][j] = prob;
                    m[j][i] = prob;
                }
            }
            a.push(m);
        }
        gls.push(a);
    }

    for v in gls.iter() {
        for mat in v.iter() {
            for i in 0..4 {
                for j in 0..4 {
                    if mat[i][j] * mat[i][j] < mat[i][i] * mat[j][j] {
                        println!("Check did not work for {:?}", mat);
                    }
                }
            }
        }
    }

    Ok(gls)
}
