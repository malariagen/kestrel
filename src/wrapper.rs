use std::num::{NonZeroU64, NonZeroUsize};

use ndarray::{Array1, Array2, ArrayView2, ArrayView3, Axis};
use paralight::threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPool, ThreadPoolBuilder};

use crate::{
    coefficients::{Output, calculate_allele_frequencies}, lanevector::{LaneVector, LaneVectorBuilder}, simd::{Lane, simd::Simd}, util::ResourceTracker, vcf::{AlleleFrequencies, GenotypeLikelihoods},
};

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::Avx2Lane;
#[cfg(target_arch = "x86_64")]
use crate::simd::avx512::Avx512Lane;
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::NeonLane;

pub fn calculate_relatedness_coefficients_gt(
    genotypes: ArrayView3<u8>,
    allele_frequencies: Option<ArrayView2<f64>>,
    threads: Option<NonZeroUsize>,
    iterations: NonZeroU64,
) -> Vec<Vec<Output>> {
    // This works, suprisingly
    let calculated_af;
    let allele_frequencies = match allele_frequencies {
        Some(af) => af,
        None => {
            calculated_af = calculate_allele_frequencies(genotypes);
            calculated_af.view()
        }
    };

    let num_v = genotypes.shape()[0];
    let num_s = genotypes.shape()[1];
    let num_h = genotypes.shape()[2];

    assert!(num_v > 0, "Must have at least one variant");
    assert!(num_s > 0, "Must have at least one sample");
    assert!(num_h == 2, "Must have a ploidy of 2");

    // TODO in theory we could do some checks here that the af are non-negative, sum to one
    // Indexing will work, etc.

    assert!(allele_frequencies.shape()[0] == num_v, "Must have same dimension");
    assert!(allele_frequencies.shape()[1] == 4, "Must have all four alleles");

    let simd = Simd::detect();

    let mut thread_pool = build_thread_pool(threads);

    let outputs = match simd {
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => crate::coefficients::calculate_coefficients_gt::<Avx2Lane>(
            genotypes,
            allele_frequencies,
            &mut thread_pool,
            iterations,
        ),
        #[cfg(target_arch = "x86_64")]
        Simd::Avx512 => crate::coefficients::calculate_coefficients_gt::<Avx512Lane>(
            genotypes,
            allele_frequencies,
            &mut thread_pool,
            iterations,
        ),
        #[cfg(target_arch = "aarch64")]
        Simd::Neon => crate::coefficients::calculate_coefficients_gt::<NeonLane>(
            genotypes,
            allele_frequencies,
            &mut thread_pool,
            iterations,
        ),
        Simd::Scalar => crate::coefficients::calculate_coefficients_gt::<f64>(
            genotypes,
            allele_frequencies,
            &mut thread_pool,
            iterations,
        ),
    };

    outputs
}

pub fn calculate_relatedness_coefficients_gl(
    genotype_likelihoods: GenotypeLikelihoods,
    allele_frequencies: Option<AlleleFrequencies>,
    threads: Option<NonZeroUsize>,
    iterations: NonZeroU64,
    minimum_allele_frequency: Option<f64>,
) -> Vec<Vec<Output>> {

    let tracker = ResourceTracker::new();

    // This is the smallest allele frequency possible given our sample size
    // Anything lower than this is just noise
    let maf = minimum_allele_frequency.unwrap_or(1.0 / (2.0 * genotype_likelihoods.samples as f64));

    assert!(
        0.0 <= maf && maf <= 1.0,
        "Minimum allele frequency must be within 0 to 1"
    );

    let simd = Simd::detect();

    let mut thread_pool = build_thread_pool(threads);

    let calculated_af;
    let allele_frequencies = match allele_frequencies {
        Some(af) => {
            // let (anum_v, num_a) = af.dim();
            // assert!(anum_v == num_v, "Must have same dimension");
            // assert!(num_a == 4, "Must have all four alleles");

            unimplemented!()
            // af.af4
            // af.map_axis(Axis(1), |lane| std::array::from_fn(|i| lane[i]))
        }
        None => {
            calculated_af = match simd {
                #[cfg(target_arch = "x86_64")]
                Simd::Avx512 => crate::allele::calculate_allele_frequencies::<Avx512Lane>(
                    &genotype_likelihoods,
                    &mut thread_pool,
                    iterations,
                    maf,
                ),
                #[cfg(target_arch = "x86_64")]
                Simd::Avx2 => crate::allele::calculate_allele_frequencies::<Avx2Lane>(
                    &genotype_likelihoods,
                    &mut thread_pool,
                    iterations,
                    maf,
                ),
                #[cfg(target_arch = "aarch64")]
                Simd::Neon => crate::allele::calculate_allele_frequencies::<NeonLane>(
                    &genotype_likelihoods,
                    &mut thread_pool,
                    iterations,
                    maf,
                ),
                Simd::Scalar => crate::allele::calculate_allele_frequencies::<f64>(
                    &genotype_likelihoods,
                    &mut thread_pool,
                    iterations,
                    maf,
                ),
            };

            calculated_af
        }
    };

    tracker.checkpoint();

    // TODO in theory we could do some checks here that the af are non-negative, sum to one
    // Indexing will work, etc.

    let outputs = match simd {
        #[cfg(target_arch = "x86_64")]
        Simd::Avx512 => {
            let glaf = split_gl_af(genotype_likelihoods, allele_frequencies);
            crate::coefficients::calculate_relatedness_coefficients_gl::<Avx512Lane>(&glaf, &mut thread_pool, iterations)
        },
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => {
            let glaf = split_gl_af(genotype_likelihoods, allele_frequencies);
            crate::coefficients::calculate_relatedness_coefficients_gl::<Avx2Lane>(&glaf, &mut thread_pool, iterations)
        }
        #[cfg(target_arch = "aarch64")]
        Simd::Neon => {
            let glaf = split_gl_af(genotype_likelihoods, allele_frequencies);
            crate::coefficients::calculate_relatedness_coefficients_gl::<NeonLane>(&glaf, &mut thread_pool, iterations)
        }
        Simd::Scalar => {
            let glaf = split_gl_af(genotype_likelihoods, allele_frequencies);
            crate::coefficients::calculate_relatedness_coefficients_gl::<f64>(&glaf, &mut thread_pool, iterations)
        }
    };

    tracker.checkpoint();

    outputs
}

fn build_thread_pool(threads: Option<NonZeroUsize>) -> ThreadPool {
    let threads = threads.unwrap_or_else(|| std::thread::available_parallelism().unwrap());

    log::info!("Using thread pool with {threads} threads");

    let pool = ThreadPoolBuilder {
        num_threads: ThreadCount::Count(threads),
        range_strategy: RangeStrategy::WorkStealing,
        cpu_pinning: CpuPinningPolicy::No,
    }
    .build();

    pool
}

pub struct GlAf<L: Lane> {
    pub af2: LaneVector<L, 2>,
    pub af3: LaneVector<L, 3>,
    pub af4: LaneVector<L, 4>,
    pub gl2: Array2<[f64; 3]>,
    pub gl3: Array2<[f64; 6]>,
    pub gl4: Array2<[f64; 10]>,
}

// Multi-allelic sites ... never again
// This function takes ownership of the input arrays so that they are dropped at the end of the function
// (They take up a lot of memory and we don't need them anymore)
fn split_gl_af<L: Lane>(genotype_likelihoods: GenotypeLikelihoods, allele_frequencies: AlleleFrequencies) -> GlAf<L> {
    let num_variants = genotype_likelihoods.variants;
    let num_samples = genotype_likelihoods.samples;

    let mut af2 = LaneVector::<L, 2>::empty();
    let mut af3 = LaneVector::<L, 3>::empty();
    let mut af4 = LaneVector::<L, 4>::empty();

    let mut af2b = LaneVectorBuilder::<L, 2>::new(&mut af2);
    let mut af3b = LaneVectorBuilder::<L, 3>::new(&mut af3);
    let mut af4b = LaneVectorBuilder::<L, 4>::new(&mut af4);

    // TODO convert this to lanes here? Will avoid using more memory
    // (Similar to af using a builder)
    let mut gl2 = Vec::<[f64; 3]>::new();
    let mut gl3 = Vec::<[f64; 6]>::new();
    let mut gl4 = Vec::<[f64; 10]>::new();

    let mut nonzero_af_indices = Vec::with_capacity(4);
    let mut af_buf = Vec::with_capacity(4);
    let mut nonzero_gl_indices = Vec::with_capacity(10);
    let mut gl_buf = Vec::with_capacity(10);

    remove_zero_frequencies(genotype_likelihoods.gl2.view(), &allele_frequencies.af2, &mut af2b, &mut af3b, &mut af4b, &mut gl2, &mut gl3, &mut gl4, &mut nonzero_af_indices, &mut af_buf, &mut nonzero_gl_indices, &mut gl_buf);
    remove_zero_frequencies(genotype_likelihoods.gl3.view(), &allele_frequencies.af3, &mut af2b, &mut af3b, &mut af4b, &mut gl2, &mut gl3, &mut gl4, &mut nonzero_af_indices, &mut af_buf, &mut nonzero_gl_indices, &mut gl_buf);
    remove_zero_frequencies(genotype_likelihoods.gl4.view(), &allele_frequencies.af4, &mut af2b, &mut af3b, &mut af4b, &mut gl2, &mut gl3, &mut gl4, &mut nonzero_af_indices, &mut af_buf, &mut nonzero_gl_indices, &mut gl_buf);

    af2b.finish();
    af3b.finish();
    af4b.finish();

    let di = af2.len();
    let tri = af3.len();
    let tetra = af4.len();
    let mono = num_variants - di - tri - tetra;

    log::info!("Kept {di} di-allelic sites");
    log::info!("Kept {tri} tri-allelic sites");
    log::info!("Kept {tetra} tetra-allelic sites");
    log::info!("Discarded {mono} non-segregating sites");

    let gl2 = Array2::from_shape_vec((di, num_samples), gl2).unwrap();
    let gl3 = Array2::from_shape_vec((tri, num_samples), gl3).unwrap();
    let gl4 = Array2::from_shape_vec((tetra, num_samples), gl4).unwrap();

    GlAf {
        af2,
        af3,
        af4,
        gl2,
        gl3,
        gl4,
    }
}

// Some bases may be observed at a site due to read errors (even though they don't exist there)
// So if the allele frequencies are zero then remove them
fn remove_zero_frequencies<L: Lane, const A: usize, const G: usize>(gl: ArrayView2<[f64; G]>, af: &[[f64; A]],
    af2b: &mut LaneVectorBuilder<'_, L, 2>,
    af3b: &mut LaneVectorBuilder<'_, L, 3>,
    af4b: &mut LaneVectorBuilder<'_, L, 4>,
    gl2: &mut Vec<[f64; 3]>,
    gl3: &mut Vec<[f64; 6]>,
    gl4: &mut Vec<[f64; 10]>,
    nonzero_af_indices: &mut Vec<usize>,
    af_buf: &mut Vec<f64>,
    nonzero_gl_indices: &mut Vec<usize>,
    gl_buf: &mut Vec<f64>) {

        gl.outer_iter()
        .zip(af)
        .for_each(|(gl, af)| {
            nonzero_af_indices.clear();
            af_buf.clear();

            for (i, a) in af.iter().enumerate() {
                if *a > 0.0 {
                    nonzero_af_indices.push(i);
                    af_buf.push(*a);
                }
            }

            if let Some(af_arr) = af_buf.as_array::<2>() {
                af2b.push(af_arr);
            } else if let Some(af_arr) = af_buf.as_array::<3>() {
                af3b.push(af_arr);
            } else if let Some(af_arr) = af_buf.as_array::<4>() {
                af4b.push(af_arr);
            }

            nonzero_gl_indices.clear();
            // New array must match VCF order
            for (j, b) in nonzero_af_indices.iter().enumerate() {
                for a in &nonzero_af_indices[..=j] {
                    // Index of (a, b) from VCF spec
                    let gl_index = b * (b + 1) / 2 + a;
                    nonzero_gl_indices.push(gl_index);
                }
            }

            for sample_gls in gl.iter() {
                gl_buf.clear();
                for i in nonzero_gl_indices.iter() {
                    gl_buf.push(sample_gls[*i]);
                }

                if let Some(gl_arr) = gl_buf.as_array::<3>() {
                    gl2.push(*gl_arr);
                } else if let Some(gl_arr) = gl_buf.as_array::<6>() {
                    gl3.push(*gl_arr);
                } else if let Some(gl_arr) = gl_buf.as_array::<10>() {
                    gl4.push(*gl_arr);
                }
            }
        });
}
