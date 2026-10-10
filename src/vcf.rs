use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array2, Array3};
use noodles::vcf::Header;
use noodles::vcf::Record;
use noodles::vcf::variant::record::AlternateBases;
use noodles::vcf::variant::record::info::field::Value as InfoValue;
use noodles::vcf::variant::record::info::field::value::Array as InfoArray;
use noodles::vcf::variant::record::samples::Series;
use noodles::vcf::variant::record::samples::series::value::Array as SeriesArray;
use noodles::vcf::variant::record::samples::series::value::Value as SeriesValue;

// TODO: possibly split into af2, af3, af4, etc...

pub fn parse_vcf_gt(file: &Path, parse_af: bool) -> Result<(Vec<String>, Array3<u8>, Option<Array2<f64>>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = samples.len();

    let mut skipped_missing = 0;
    let mut not_snp = 0;
    let mut total_variants = 0;
    let mut num_variants = 0;

    let mut genotypes = Vec::<u8>::new(); // V x S x 2
    let mut sample_gts = Vec::with_capacity(num_samples * 2);
    let mut gt_buf = Vec::with_capacity(2);

    let mut allele_frequencies = Vec::<f64>::new(); // V x 4
    let mut af_buf = Vec::<f64>::with_capacity(4);

    'variant: for result in reader.records() {
        let record = result.context("Error reading record")?;

        total_variants += 1;

        if !is_snp(&record)? {
            not_snp += 1;
            continue;
        }

        let record_samples = record.samples();
        let gt_series = record_samples.select("GT").context("No GT data found")?;

        sample_gts.clear();

        for result in gt_series.iter(&header) {
            let value = result?.context("No genotype for sample found")?;

            if let SeriesValue::Genotype(gt) = value {
                gt_buf.clear();

                for result_allele in gt.iter() {
                    let option_allele = result_allele.context("Error reading genotype")?;
                    match option_allele.0 {
                        Some(a) => {
                            let allele = u8::try_from(a).context("Allele is greater than 255")?;
                            gt_buf.push(allele);
                        }
                        // Skip this site if some of the data is missing
                        None => {
                            skipped_missing += 1;
                            continue 'variant;
                        }
                    }
                }

                if gt_buf.len() == 2 {
                    sample_gts.extend_from_slice(&gt_buf);
                } else {
                    bail!("Genotype {:?} is not diploid", gt);
                }
            } else {
                bail!("Value {:?} is not a genotype", value);
            }
        }

        num_variants += 1;
        genotypes.extend_from_slice(&sample_gts);

        if parse_af {
            read_variant_af(&header, &record, &mut allele_frequencies, &mut af_buf)?;
        }
    }

    log::info!("Parsed {} total variants", total_variants);
    log::info!("Skipped {} variants that were not SNPs", not_snp);
    log::info!("Skipped {} variants with missing data", skipped_missing);
    log::info!("Kept {} final variants", num_variants);

    let gts = Array3::<u8>::from_shape_vec((num_variants, num_samples, 2), genotypes).unwrap();

    let afs = if parse_af {
        let afs = Array2::<f64>::from_shape_vec((num_variants, 4), allele_frequencies).unwrap();
        Some(afs)
    } else {
        None
    };

    Ok((samples, gts, afs))
}

pub fn parse_vcf_gl(file: &Path, parse_af: bool) -> Result<(Vec<String>, Array3<f64>, Option<Array2<f64>>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let record_samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = record_samples.len();

    let mut skipped_missing = 0;
    let mut not_snp = 0;
    let mut total_variants = 0;
    let mut num_variants = 0;

    let mut likelihoods = Vec::<f64>::new(); // V x S x 10
    let mut samples_buf = Vec::<f64>::with_capacity(num_samples * 10);
    let mut gl_buf = Vec::<f64>::with_capacity(10);

    let mut allele_frequencies = Vec::<f64>::new(); // V x 4
    let mut af_buf = Vec::<f64>::with_capacity(4);

    'variant: for result in reader.records() {
        let record = result.context("Error reading record")?;

        total_variants += 1;

        if !is_snp(&record)? {
            not_snp += 1;
            continue;
        }

        let samples = record.samples();
        let gl_series = samples.select("GL").context("No GL data found")?;

        samples_buf.clear();
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
                            Some(gl) => gl_buf.push(gl.into()),
                            None => {
                                skipped_missing += 1;
                                continue 'variant;
                            }
                        }
                    }

                    // We checked that this site is a SNP, so at most 10 GLs
                    assert!(gl_buf.len() <= 10);

                    // Normalize by the maximum GL to avoid possible underflow
                    // (This matches what PL does)
                    let max_gl = gl_buf.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

                    for gl in gl_buf.iter() {
                        let prob = 10.0f64.powf(gl - max_gl);
                        samples_buf.push(prob)
                    }

                    // TODO: padding a Lorentzian matrix with zeros will make it non-L
                    // No longer invertible, hence some zero eigenvalues
                    // So need to record the arity of each site
                    for _ in gl_buf.len()..10 {
                        samples_buf.push(0.0);
                    }

                } else {
                    bail!("Array {:?} does not contain floats", gl_array);
                }
            } else {
                bail!("Value {:?} is not an array", value);
            }
        }

        num_variants += 1;
        likelihoods.extend_from_slice(&samples_buf);

        if parse_af {
            read_variant_af(&header, &record, &mut allele_frequencies, &mut af_buf)?;
        }
    }

    log::info!("Parsed {} total variants", total_variants);
    log::info!("Skipped {} variants that were not SNPs", not_snp);
    log::info!("Skipped {} variants with missing data", skipped_missing);
    log::info!("Kept {} final variants", num_variants);

    let gls = Array3::from_shape_vec((num_variants, num_samples, 10), likelihoods).unwrap();

    let afs = if parse_af {
        let afs = Array2::<f64>::from_shape_vec((num_variants, 4), allele_frequencies).unwrap();
        Some(afs)
    } else {
        None
    };

    Ok((record_samples, gls, afs))
}

pub fn parse_vcf_pl(file: &Path, parse_af: bool) -> Result<(Vec<String>, Array3<f64>, Option<Array2<f64>>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let record_samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = record_samples.len();

    let mut skipped_missing = 0;
    let mut not_snp = 0;
    let mut total_variants = 0;
    let mut num_variants = 0;

    let mut likelihoods = Vec::<f64>::new(); // V x S x 10
    let mut samples_buf = Vec::<f64>::with_capacity(num_samples * 10);
    let mut pl_buf = Vec::<f64>::with_capacity(10);

    let mut allele_frequencies = Vec::<f64>::new(); // V x 4
    let mut af_buf = Vec::<f64>::with_capacity(4);

    'variant: for result in reader.records() {
        let record = result.context("Error reading record")?;

        total_variants += 1;

        if !is_snp(&record)? {
            not_snp += 1;
            continue;
        }

        let samples = record.samples();
        let pl_series = samples.select("PL").context("No PL data found")?;

        samples_buf.clear();
        for result in pl_series.iter(&header) {
            let value = result?.context("No genotype likelihood for sample found")?;
            if let SeriesValue::Array(pl_array) = value {
                if let SeriesArray::Integer(pl_int) = pl_array {

                    pl_buf.clear();
                    for result_pl in pl_int.iter() {
                        let option_pl = result_pl.context("Error reading genotype likelihood")?;
                        // If missing PL, skip this site
                        match option_pl {
                            Some(pl) => pl_buf.push(pl.into()),
                            None => {
                                skipped_missing += 1;
                                continue 'variant;
                            }
                        }
                    }

                    assert!(pl_buf.len() <= 10);

                    for pl in pl_buf.iter() {
                        let prob = 10.0f64.powf(- *pl / 10.0);
                        samples_buf.push(prob);
                    }

                    for _ in pl_buf.len()..10 {
                        samples_buf.push(0.0);
                    }

                } else {
                    bail!("Array {:?} does not contain integers", pl_array);
                }
            } else {
                bail!("Value {:?} is not an array", value);
            }
        }

        num_variants += 1;
        likelihoods.extend_from_slice(&samples_buf);

        if parse_af {
            read_variant_af(&header, &record, &mut allele_frequencies, &mut af_buf)?;
        }
    }

    log::info!("Parsed {} total variants", total_variants);
    log::info!("Skipped {} variants that were not SNPs", not_snp);
    log::info!("Skipped {} variants with missing data", skipped_missing);
    log::info!("Kept {} final variants", num_variants);

    let pls = Array3::from_shape_vec((num_variants, num_samples, 10), likelihoods).unwrap();

    let afs = if parse_af {
        let afs = Array2::<f64>::from_shape_vec((num_variants, 4), allele_frequencies).unwrap();
        Some(afs)
    } else {
        None
    };

    Ok((record_samples, pls, afs))
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

        if !matches!(alt, "A" | "C" | "G" | "T" | "a" | "c" | "g" | "t") {
            return Ok(false);
        }
    }

    Ok(true)
}

fn read_variant_af(header: &Header, record: &Record, allele_frequencies: &mut Vec<f64>, af_buf: &mut Vec<f64>) -> Result<()> {
    let info = record.info();
    let af_info = info
        .get(header, "AF")
        .context("No AF info found")?
        .context("Error reading AF info")?
        .context("No AF info found")?;

    if let InfoValue::Array(af_array) = af_info {
        if let InfoArray::Float(af_float) = af_array {

            af_buf.clear();
            for af in af_float.iter() {
                let af = af.context("Error reading AF")?.context("No AF data found")?;
                af_buf.push(af.into());
            }

            // We checked this is a SNP, so there are at most 3 ALTs
            assert!(af_buf.len() <= 3);

            let alt_sum = af_buf.iter().sum::<f64>().clamp(0.0, 1.0);
            let ref_af = 1.0 - alt_sum;

            allele_frequencies.push(ref_af);
            allele_frequencies.extend_from_slice(&*af_buf);

            // Remaining unobserved alleles have frequency 0
            for _ in af_buf.len()..3 {
                allele_frequencies.push(0.0)
            }

        } else {
            bail!("Array {:?} is does not contain floats", af_array);
        }
    } else {
        bail!("Value {:?} is not an array", af_info);
    }

    Ok(())
}
