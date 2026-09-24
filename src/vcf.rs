use std::path::Path;

use anyhow::{Context, Result, bail};
use itertools::Itertools;
use ndarray::{Array2, Array3, Array4};
use noodles::vcf::Record;
use noodles::vcf::variant::record::AlternateBases;
use noodles::vcf::variant::record::info::field::Value as InfoValue;
use noodles::vcf::variant::record::info::field::value::Array as InfoArray;
use noodles::vcf::variant::record::samples::Series;
use noodles::vcf::variant::record::samples::series::value::Array as SeriesArray;
use noodles::vcf::variant::record::samples::series::value::Value as SeriesValue;

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

        // parse af
        if false {
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
                        // Not diallelic
                        continue;
                    }
                } else {
                    bail!("Value {:?} is not an array", af_array);
                }
            } else {
                bail!("Value {:?} is not an array", af_info);
            }
        }

        let samples = record.samples();
        // println!("{:?}", samples.keys());
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
    // for v in 0..num_variants {
    //     let (f0, f1) = allele_frequencies[v];
    //     af[[v, 0]] = f0;
    //     af[[v, 1]] = f1;
    // }

    Ok((samples, gt, af))
}

pub fn parse_vcf_gl(file: &Path) -> Result<(Vec<String>, Array3<f64>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = samples.len();

    let mut skipped_missing = 0;
    let mut not_snp = 0;
    let mut total_variants = 0;

    let mut likelihoods = Vec::<Vec<[f64; 10]>>::new(); // V x S x 10

    let mut gl_buf = Vec::with_capacity(10);

    'variant: for result in reader.records() {
        let record = result.context("Error reading record")?;

        total_variants += 1;

        if !is_snp(&record)? {
            not_snp += 1;
            continue;
        }

        let samples = record.samples();
        let gl_series = samples.select("GL").context("No GL data found")?;

        let mut variant_gls = Vec::with_capacity(num_samples);

        for result in gl_series.iter(&header) {
            let value = result?.context("No genotype likelihood for sample found")?;
            if let SeriesValue::Array(gl_array) = value {
                if let SeriesArray::Float(gl_float) = gl_array {
                    gl_buf.clear();

                    for result_gl in gl_float.iter() {
                        let option_gl = result_gl.context("Error reading genotype likelihood")?;
                        // This is where we could have a missing GL
                        // e.g. vcfgl produced 0:.,.,.,.,.,.,.,.,.,. at a site for one sample
                        // Just skip the site/variant if that happens
                        match option_gl {
                            Some(gl) => gl_buf.push(gl),
                            None => {
                                skipped_missing += 1;
                                continue 'variant;
                            }
                        }
                    }

                    let mut gls = [0.0f64; 10];

                    if gl_buf.len() > 10 {
                        continue 'variant;
                    }

                    for (out_gl, in_gl) in gls.iter_mut().zip(gl_buf.iter().copied()) {
                        *out_gl = in_gl.into();
                    }

                    variant_gls.push(gls);
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

    println!("Parsed {} total variants", total_variants);
    println!("Skipped {} variants with missing data", skipped_missing);
    println!("Skipped {} variants that were not SNPs", not_snp);
    println!("Kept {} final variants", num_variants);

    let mut gls = Array3::zeros((num_variants, num_samples, 10));

    // TODO probably faster way to do this with zip or some such
    for v in 0..num_variants {
        for s in 0..num_samples {
            let sample_gls = likelihoods[v][s];
            // Normalize by the maximum GL to avoid possible underflow
            // (This matches what PL does)
            let max_gl = sample_gls.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

            for i in 0..10 {
                let gl = sample_gls[i];
                let prob = 10.0f64.powf(gl - max_gl);
                gls[[v, s, i]] = prob;
            }
        }
    }

    // let mut gls = Vec::with_capacity(num_variants);
    // for v in 0..num_variants {
    //     let mut a = Vec::with_capacity(num_samples);
    //     for s in 0..num_samples {
    //         let sample_gls = likelihoods[v][s];
    //         // Normalize by the maximum GL to avoid possible underflow
    //         // (This matches what PL does)
    //         let max_gl = sample_gls.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    //         let mut m = [[0.0; 4]; 4];
    //         for i in 0..4 {
    //             for j in i..4 {
    //                 // The index of (i, j) where i <= j (see the VCF spec)
    //                 let index = j * (j + 1) / 2 + i;
    //                 let gl = sample_gls[index];
    //                 let prob = 10.0f64.powf(gl - max_gl);
    //                 m[i][j] = prob;
    //                 m[j][i] = prob;
    //             }
    //         }
    //         a.push(m);
    //     }
    //     gls.push(a);
    // }

    // for v in gls.iter() {
    //     for mat in v.iter() {
    //         for i in 0..4 {
    //             for j in 0..4 {
    //                 // TODO check the eigenvalues here
    //                 if mat[i][j] * mat[i][j] < mat[i][i] * mat[j][j] {
    //                     println!("Check did not work for {:?}", mat);
    //                 }
    //             }
    //         }
    //     }
    // }

    Ok((samples, gls))
}

// VCF spec says this must be A, C, G, T, or N (case insensitive)
fn is_snp(record: &Record) -> Result<bool> {

    let ref_bases = record.reference_bases();

    if !matches!(ref_bases, "A" | "C" | "G" | "T" | "a" | "c" | "g" | "t") {
        return Ok(false);
    }

    let num_alts = record.alternate_bases().len();

    if num_alts < 1 || num_alts > 3 {
        return Ok(false);
    }

    for result_alt in record.alternate_bases().iter() {
        let alt = result_alt.context("Error reading alternate bases")?;

        if !matches!(alt, "A" | "C" | "G" | "T" | "a" | "c" | "g" | "t" ) {
            return Ok(false);
        }
    }

    Ok(true)
}