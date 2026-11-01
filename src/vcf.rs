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

pub fn parse_vcf_gt(file: &Path, parse_af: bool) -> Result<(Vec<String>, Array3<u8>, Option<Array2<f64>>)> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = samples.len();

    let mut skipped_missing = 0;
    let mut not_snp = 0;
    let mut total_variants = 0;
    let mut num_variants = 0;

    // TODO clean up this stuff too
    // Perhaps for GT we can allow arbitrary number, not just 4 bases?
    // I think that could work in future

    let mut genotypes = Vec::<u8>::new(); // V x S x 2
    let mut sample_gts = Vec::with_capacity(num_samples * 2);
    let mut gt_buf = Vec::with_capacity(2);

    let mut allele_frequencies = Vec::<f64>::new(); // V x 4
    let mut af_buf = Vec::<f64>::with_capacity(4);

    'variant: for result in reader.records() {
        let record = result.context("Error reading record")?;

        total_variants += 1;

        let num_bases = count_snp_bases(&record)?;

        if let Some(num_bases) = num_bases {
            if num_bases < 2 || num_bases > 4 {
                not_snp += 1;
                continue;
            }
        } else {
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

        // TODO fix later
        // if parse_af {
        //     parse_variant_af(&header, &record)?;
        // }
    }

    log::info!("Parsed {} total sites", total_variants);
    log::info!("Skipped {} sites that were not SNPs", not_snp);
    log::info!("Skipped {} sites with missing data", skipped_missing);
    log::info!("Kept {} final sites", num_variants);

    let gts = Array3::<u8>::from_shape_vec((num_variants, num_samples, 2), genotypes).unwrap();

    let afs = if parse_af {
        let afs = Array2::<f64>::from_shape_vec((num_variants, 4), allele_frequencies).unwrap();
        Some(afs)
    } else {
        None
    };

    Ok((samples, gts, afs))
}

pub struct ParseGL {
    pub samples: Vec<String>,
    pub gl: GenotypeLikelihoods,
    pub af: Option<AlleleFrequencies>
}

pub struct AlleleFrequencies {
    pub af2: Vec<[f64; 2]>,
    pub af3: Vec<[f64; 3]>,
    pub af4: Vec<[f64; 4]>,
}

pub struct GenotypeLikelihoods {
    pub gl2: Array2<[f64; 3]>,
    pub gl3: Array2<[f64; 6]>,
    pub gl4: Array2<[f64; 10]>,
    pub samples: usize,
    pub variants: usize,
}

impl GenotypeLikelihoods {
    fn new(gl2: Array2<[f64; 3]>, gl3: Array2<[f64; 6]>, gl4: Array2<[f64; 10]>) -> Self {
        let (num_v2, num_s2) = gl2.dim();
        let (num_v3, num_s3) = gl3.dim();
        let (num_v4, num_s4) = gl4.dim();

        let variants = num_v2 + num_v3 + num_v4;

        assert!(variants > 0, "Must have at least one site");
        assert!(num_s2 == num_s3 && num_s3 == num_s4 && num_s2 > 0, "Must have at least one sample");

        Self { gl2, gl3, gl4, variants, samples: num_s2 }
    }
}

pub fn parse_vcf_gl(file: &Path, parse_af: bool, gl: bool) -> Result<ParseGL> {
    let mut reader = noodles::vcf::io::reader::Builder::default().build_from_path(file)?;
    let header = reader.read_header()?;

    let samples = header.sample_names().iter().map(|s| s.to_owned()).collect::<Vec<_>>();

    let num_samples = samples.len();

    let mut not_snp = 0;
    let mut total_variants = 0;

    let mut num_variants2 = 0;
    let mut num_variants3 = 0;
    let mut num_variants4 = 0;

    let mut gl2 = Vec::<[f64; 3]>::new(); // V2 x S
    let mut gl3 = Vec::<[f64; 6]>::new(); // V3 x S
    let mut gl4 = Vec::<[f64; 10]>::new(); // V4 x S

    let mut af2 = Vec::<[f64; 2]>::new(); // V2
    let mut af3 = Vec::<[f64; 3]>::new(); // V3
    let mut af4 = Vec::<[f64; 4]>::new(); // V4

    for result in reader.records() {
        let record = result.context("Error reading record")?;

        total_variants += 1;

        let num_bases = count_snp_bases(&record)?;

        if let Some(num_bases) = num_bases {
            match num_bases {
                2 => {
                    if gl {
                        parse_samples_gl(&header, &record, &mut gl2)?;
                    } else {
                        parse_samples_pl(&header, &record, &mut gl2)?;
                    }
                    if parse_af {
                        let af = parse_variant_af(&header, &record)?;
                        af2.push(af);
                    }
                    num_variants2 += 1;
                },
                3 => {
                    if gl {
                        parse_samples_gl(&header, &record, &mut gl3)?;
                    } else {
                        parse_samples_pl(&header, &record, &mut gl3)?;
                    }
                    if parse_af {
                        let af = parse_variant_af(&header, &record)?;
                        af3.push(af);
                    }
                    num_variants3 += 1;
                },
                4 => {
                    if gl {
                        parse_samples_gl(&header, &record, &mut gl4)?;
                    } else {
                        parse_samples_pl(&header, &record, &mut gl4)?;
                    }
                    if parse_af {
                        let af = parse_variant_af(&header, &record)?;
                        af4.push(af);
                    }
                    num_variants4 += 1;
                },
                _ => {
                    not_snp += 1;
                    continue;
                }
            }
        } else {
            not_snp += 1;
            continue;
        }

    }

    log::info!("Parsed {} total sites", total_variants);
    log::info!("Skipped {} sites that were not SNPs", not_snp);
    log::info!("Kept {} di-allelic sites", num_variants2);
    log::info!("Kept {} tri-allelic sites", num_variants3);
    log::info!("Kept {} tetra-allelic sites", num_variants4);

    let gl2 = Array2::from_shape_vec((num_variants2, num_samples), gl2).unwrap();
    let gl3 = Array2::from_shape_vec((num_variants3, num_samples), gl3).unwrap();
    let gl4 = Array2::from_shape_vec((num_variants4, num_samples), gl4).unwrap();

    let af = if parse_af {
        Some(AlleleFrequencies { af2, af3, af4 })
    } else {
        None
    };

    let gl = GenotypeLikelihoods::new(gl2, gl3, gl4);

    let parse_gl = ParseGL { samples, gl, af };

    Ok(parse_gl)
}


fn parse_samples_gl<const G: usize>(
    header: &Header,
    record: &Record,
    gl_vec: &mut Vec<[f64; G]>,
) -> Result<()> {

    let samples = record.samples();
    let gl_series = samples.select("GL").context("No GL data found")?;

    for result in gl_series.iter(header) {
        let value = result?.context("No genotype likelihood for sample found")?;
        if let SeriesValue::Array(gl_array) = value {
            if let SeriesArray::Float(gl_float) = gl_array {
                let mut gl_arr = [0.0f64; G];

                let mut num_gl = 0;
                let mut max_gl = f64::NEG_INFINITY;
                for result_gl in gl_float.iter() {
                    let option_gl = result_gl.context("Error reading genotype likelihood")?;
                    // If a site hasn't been sequenced then the GL data will be missing
                    // e.g. vcfgl produces 0:.,.,.,.,.,.,.,.,.,. when this happens
                    // It behooves us for it to becometh a NaN
                    let gl: f64 = option_gl.unwrap_or(f32::NAN).into();
                    let index = gl_arr.get_mut(num_gl).context("Too many genotype likelihoods!")?;
                    max_gl = max_gl.max(gl);
                    *index = gl;
                    num_gl += 1;
                }

                if num_gl != G {
                    bail!("Not enough genotype likelihoods!")
                }

                // Normalize by the maximum GL to avoid possible underflow
                // (This matches what PL does)
                let prob = std::array::from_fn(|i| 10.0f64.powf(gl_arr[i] - max_gl));

                // If parsing any of the samples fails we will abort, so it's ok
                // to push straight to the output vector.
                gl_vec.push(prob)
            } else {
                bail!("Array {:?} does not contain floats", gl_array);
            }
        } else {
            bail!("Value {:?} is not an array", value);
        }
    }

    Ok(())
}

fn parse_samples_pl<const G: usize>(
    header: &Header,
    record: &Record,
    gl_vec: &mut Vec<[f64; G]>,
) -> Result<()> {

    let samples = record.samples();
    let pl_series = samples.select("PL").context("No PL data found")?;

    for result in pl_series.iter(header) {
        let value = result?.context("No genotype likelihood for sample found")?;
        if let SeriesValue::Array(pl_array) = value {
            if let SeriesArray::Integer(pl_int) = pl_array {
                let mut pl_arr = [0.0f64; G];

                let mut num_pl = 0;
                for result_pl in pl_int.iter() {
                    let option_pl = result_pl.context("Error reading genotype likelihood")?;
                    let pl = option_pl.map(|p| f64::from(p)).unwrap_or(f64::NAN);
                    let index = pl_arr.get_mut(num_pl).context("Too many genotype likelihoods!")?;
                    *index = pl;
                    num_pl += 1;
                }

                if num_pl != G {
                    bail!("Not enough genotype likelihoods!")
                }

                let prob = std::array::from_fn(|i| 10.0f64.powf(-pl_arr[i]/ 10.0));

                // If parsing any of the samples fails we will abort, so it's ok
                // to push straight to the output vector.
                gl_vec.push(prob)
            } else {
                bail!("Array {:?} does not contain floats", pl_array);
            }
        } else {
            bail!("Value {:?} is not an array", value);
        }
    }

    Ok(())
}

// VCF spec says this must be A, C, G, T, or N (case insensitive)
fn count_snp_bases(record: &Record) -> Result<Option<usize>> {
    let ref_bases = record.reference_bases();

    if !matches!(ref_bases, "A" | "C" | "G" | "T" | "a" | "c" | "g" | "t") {
        return Ok(None);
    }

    let mut num_bases = 1;

    for result_alt in record.alternate_bases().iter() {
        let alt = result_alt.context("Error reading alternate bases")?;

        if !matches!(alt, "A" | "C" | "G" | "T" | "a" | "c" | "g" | "t") {
            return Ok(None);
        }

        num_bases += 1;
    }

    Ok(Some(num_bases))
}

fn parse_variant_af<const A: usize>(
    header: &Header,
    record: &Record,
) -> Result<[f64; A]> {
    let info = record.info();
    let af_info = info
        .get(header, "AF")
        .context("No AF info found")?
        .context("Error reading AF info")?
        .context("No AF info found")?;

    if let InfoValue::Array(af_array) = af_info {
        if let InfoArray::Float(af_float) = af_array {
            let mut af_arr = [0.0; A];

            let mut num_af = 1;
            let mut alt_sum = 0.0f64;
            for af in af_float.iter() {
                let af: f64 = af.context("Error reading AF")?.context("No AF data found")?.into();
                let index = af_arr.get_mut(num_af).context("Too many allele frequencies!")?;
                alt_sum += af;
                *index = af;
                num_af += 1;
            }

            if num_af != A {
                bail!("Not enough allele frequencies!")
            }

            let ref_af = 1.0 - alt_sum.clamp(0.0, 1.0);
            af_arr[0] = ref_af;

            Ok(af_arr)

        } else {
            bail!("Array {:?} is does not contain floats", af_array);
        }
    } else {
        bail!("Value {:?} is not an array", af_info);
    }
}

