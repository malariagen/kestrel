use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array2, Array3};
use noodles::vcf::Record;
use noodles::vcf::variant::record::AlternateBases;
use noodles::vcf::variant::record::info::field::Value as InfoValue;
use noodles::vcf::variant::record::info::field::value::Array as InfoArray;
use noodles::vcf::variant::record::samples::Series;
use noodles::vcf::variant::record::samples::series::value::Array as SeriesArray;
use noodles::vcf::variant::record::samples::series::value::Value as SeriesValue;

// TODO use Array3::from_shape_vec() to make the whole dang thing faster

// We only support up to tetra-allelic sites, so we could-hardcode A = 4 at most.
// Well, could also have less than that for bi-allelelics.
// In that case we can split into af2, af3, af4, etc...

pub fn parse_vcf_gt(file: &Path, parse_af: bool) -> Result<(Vec<String>, Array3<u8>, Option<Array2<f64>>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = samples.len();

    let mut skipped_missing = 0;
    let mut not_snp = 0;
    let mut total_variants = 0;

    let mut genotypes = Vec::<Vec<[u8; 2]>>::new(); // V x S x 2

    let mut allele_frequencies = Vec::<Vec<f32>>::new(); // V x A

    let mut gt_buf = Vec::with_capacity(2);

    'variant: for result in reader.records() {
        let record = result.context("Error reading record")?;

        total_variants += 1;

        if !is_snp(&record)? {
            not_snp += 1;
            continue;
        }

        let record_samples = record.samples();
        let gt_series = record_samples.select("GT").context("No GT data found")?;

        let mut sample_gts = Vec::with_capacity(num_samples);

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
                        },
                        // Skip this site if some of the data is missing
                        None => {
                            skipped_missing += 1;
                            continue 'variant;
                        }
                    }
                }

                if gt_buf.len() == 2 {
                    sample_gts.push([gt_buf[0], gt_buf[1]]);
                } else {
                    bail!("Genotype {:?} is not diploid", gt);
                }

            } else {
                bail!("Value {:?} is not a genotype", value);
            }

        }

        genotypes.push(sample_gts);

        if parse_af {
            let info = record.info();
            let af_info = info
                .get(&header, "AF")
                .context("No AF info found")?
                .context("Error reading AF info")?
                .context("No AF info found")?;

            if let InfoValue::Array(af_array) = af_info {
                if let InfoArray::Float(af_float) = af_array {

                    // Most sites are diallelic
                    let mut af_buf = Vec::<f32>::with_capacity(2);
                    for af in af_float.iter() {
                        let af = af.context("Error reading AF")?.context("No AF data found")?;
                        af_buf.push(af);
                    }

                    allele_frequencies.push(af_buf);
                } else {
                    bail!("Array {:?} is does not contain floats", af_array);
                }
            } else {
                bail!("Value {:?} is not an array", af_info);
            }
        }
    }

    let num_variants = genotypes.len();

    println!("Parsed {} total variants", total_variants);
    println!("Skipped {} variants that were not SNPs", not_snp);
    println!("Skipped {} variants with missing data", skipped_missing);
    println!("Kept {} final variants", num_variants);

    let mut gts = Array3::<u8>::zeros((num_variants, num_samples, 2));
    for v in 0..num_variants {
        let samples = &genotypes[v];
        for s in 0..samples.len() {
            let [a0, a1] = samples[s];
            gts[[v, s, 0]] = a0;
            gts[[v, s, 1]] = a1;
        }
    }

    let afs = if parse_af {
        let max_alleles = allele_frequencies.iter().map(|v| v.len()).max().unwrap() + 1;

        let mut afs = Array2::<f64>::zeros((num_variants, max_alleles));

        for v in 0..num_variants {
            let variant_afs = &allele_frequencies[v];
            let alt_sum = variant_afs.iter().map(|af| f64::from(*af)).sum::<f64>().clamp(0.0, 1.0);

            afs[[v, 0]] = 1.0 - alt_sum;
            for (i, af) in variant_afs.iter().enumerate() {
                afs[[v, i+1]] = f64::from(*af);
            }
        }

        Some(afs)
    } else {
        None
    };

    Ok((samples, gts, afs))
}


pub fn parse_vcf_gl(file: &Path) -> Result<(Vec<String>, Array3<f64>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let record_samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = record_samples.len();

    let mut skipped_missing = 0;
    let mut not_snp = 0;
    let mut total_variants = 0;

    let mut likelihoods = Vec::<Vec<[f32; 10]>>::new(); // V x S x 10

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

                    if gl_buf.len() > 10 {
                        continue 'variant;
                    }

                    // Alleles that are not present have a likelihood of 0, which
                    // is negative infinity in log space
                    // TODO: padding a Lorentzian matrix with zeros will make it non-L
                    // No longer invertible, hence some zero eigenvalues
                    // So need to record the arity of each site
                    let mut gls = [-f32::INFINITY; 10];

                    for (out_gl, in_gl) in gls.iter_mut().zip(gl_buf.iter().copied()) {
                        *out_gl = in_gl;
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
    println!("Skipped {} variants that were not SNPs", not_snp);
    println!("Skipped {} variants with missing data", skipped_missing);
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
                let prob = 10.0f64.powf(f64::from(gl) - f64::from(*max_gl));
                gls[[v, s, i]] = prob;
            }
        }
    }

    Ok((record_samples, gls))
}

pub fn parse_vcf_pl(file: &Path) -> Result<(Vec<String>, Array3<f64>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let record_samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = record_samples.len();

    let mut skipped_missing = 0;
    let mut not_snp = 0;
    let mut total_variants = 0;

    let mut likelihoods = Vec::<Vec<[i32; 10]>>::new(); // V x S x 10

    let mut pl_buf = Vec::with_capacity(10);

    'variant: for result in reader.records() {
        let record = result.context("Error reading record")?;

        total_variants += 1;

        if !is_snp(&record)? {
            not_snp += 1;
            continue;
        }

        let samples = record.samples();
        let pl_series = samples.select("PL").context("No PL data found")?;

        let mut variant_pls = Vec::with_capacity(num_samples);

        for result in pl_series.iter(&header) {
            let value = result?.context("No genotype likelihood for sample found")?;
            if let SeriesValue::Array(pl_array) = value {
                if let SeriesArray::Integer(pl_int) = pl_array {
                    pl_buf.clear();

                    for result_pl in pl_int.iter() {
                        let option_pl = result_pl.context("Error reading genotype likelihood")?;
                        // If missing PL, skip this site
                        match option_pl {
                            Some(pl) => pl_buf.push(pl),
                            None => {
                                skipped_missing += 1;
                                continue 'variant;
                            }
                        }
                    }

                    if pl_buf.len() > 10 {
                        continue 'variant;
                    }

                    // TODO also change the max thing so we explicitly get zero
                    let mut pls = [i32::MAX; 10];

                    for (out_pl, in_pl) in pls.iter_mut().zip(pl_buf.iter().copied()) {
                        *out_pl = in_pl.into();
                    }

                    variant_pls.push(pls);
                } else {
                    bail!("Array {:?} does not contain integers", pl_array);
                }
            } else {
                bail!("Value {:?} is not an array", value);
            }
        }

        likelihoods.push(variant_pls);
    }

    let num_variants = likelihoods.len();

    println!("Parsed {} total variants", total_variants);
    println!("Skipped {} variants that were not SNPs", not_snp);
    println!("Skipped {} variants with missing data", skipped_missing);
    println!("Kept {} final variants", num_variants);

    let mut gls = Array3::zeros((num_variants, num_samples, 10));

    // TODO probably faster way to do this with zip or some such
    for v in 0..num_variants {
        for s in 0..num_samples {
            let sample_pls = likelihoods[v][s];
            for i in 0..10 {
                let pl = sample_pls[i];
                let prob = 10.0f64.powf(-f64::from(pl) / 10.0);
                gls[[v, s, i]] = prob;
            }
        }
    }

    Ok((record_samples, gls))
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
