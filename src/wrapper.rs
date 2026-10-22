use std::{alloc::handle_alloc_error, num::{NonZeroU64, NonZeroUsize}};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Axis};
use paralight::threads::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPool, ThreadPoolBuilder};

use crate::{
    simd::simd::Simd,
    coefficients::{Output, calculate_allele_frequencies},
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
    genotype_likelihoods: ArrayView3<f64>,
    allele_frequencies: Option<ArrayView2<f64>>,
    threads: Option<NonZeroUsize>,
    iterations: NonZeroU64,
    minimum_allele_frequency: Option<f64>
) -> Vec<Vec<Output>> {

    let (num_v, num_s, num_l) = genotype_likelihoods.dim();

    assert!(num_v > 0, "Must have at least one variant");
    assert!(num_s > 0, "Must have at least one sample");
    assert!(num_l == 10, "Must have exactly 10 GLs per sample");

    let genotype_likelihoods: Array2<[f64; 10]> = genotype_likelihoods.map_axis(Axis(2), |lane| std::array::from_fn(|i| lane[i]) );

    // This is the smallest allele frequency possible given our sample size
    // Anything lower than this is just noise
    let maf = minimum_allele_frequency.unwrap_or(1.0 / (2.0 * num_s as f64));

    assert!(0.0 <= maf && maf <= 1.0, "Minimum allele frequency must be within 0 to 1");

    let simd = Simd::detect();

    let mut thread_pool = build_thread_pool(threads);

    let calculated_af;
    let allele_frequencies = match allele_frequencies {
        Some(af) => {
            let (anum_v, num_a) = af.dim();
            assert!(anum_v == num_v, "Must have same dimension");
            assert!(num_a == 4, "Must have all four alleles");

            af.map_axis(Axis(1), |lane| std::array::from_fn(|i| lane[i]))
        },
        None => {
            calculated_af = match simd {
                #[cfg(target_arch = "x86_64")]
                Simd::Avx512 => crate::allele::calculate_allele_frequencies::<Avx512Lane>(
                    genotype_likelihoods.view(),
                    &mut thread_pool,
                    iterations,
                    maf
                ),
                #[cfg(target_arch = "x86_64")]
                Simd::Avx2 => crate::allele::calculate_allele_frequencies::<Avx2Lane>(
                    genotype_likelihoods.view(),
                    &mut thread_pool,
                    iterations,
                    maf
                ),
                #[cfg(target_arch = "aarch64")]
                Simd::Neon => crate::allele::calculate_allele_frequencies::<NeonLane>(
                    genotype_likelihoods.view(),
                    &mut thread_pool,
                    iterations,
                    maf
                ),
                Simd::Scalar => crate::allele::calculate_allele_frequencies::<f64>(
                    genotype_likelihoods.view(),
                    &mut thread_pool,
                    iterations,
                    maf
                ),
            };

            calculated_af
        }
    };

    // TODO in theory we could do some checks here that the af are non-negative, sum to one
    // Indexing will work, etc.

    let glaf = split_gl_af(genotype_likelihoods.view(), allele_frequencies.view());

    let outputs = match simd {
        #[cfg(target_arch = "x86_64")]
        Simd::Avx512 => crate::coefficients::calculate_relatedness_coefficients_gl::<Avx512Lane>(
            &glaf,
            &mut thread_pool,
            iterations,
        ),
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => crate::coefficients::calculate_relatedness_coefficients_gl::<Avx2Lane>(
            &glaf,
            &mut thread_pool,
            iterations,
        ),
        #[cfg(target_arch = "aarch64")]
        Simd::Neon => crate::coefficients::calculate_relatedness_coefficients_gl::<NeonLane>(
            &glaf,
            &mut thread_pool,
            iterations,
        ),
        Simd::Scalar => crate::coefficients::calculate_relatedness_coefficients_gl::<f64>(
            &glaf,
            &mut thread_pool,
            iterations,
        ),
    };

    outputs
}

fn build_thread_pool(threads: Option<NonZeroUsize>) -> ThreadPool {
    let threads = threads.unwrap_or_else(|| std::thread::available_parallelism().unwrap());
    // let threads: NonZeroUsize = if simd == Simd::Scalar {
    //     num_cpus::get()
    // } else {
    //     num_cpus::get_physical()
    // }.try_into().unwrap();

    log::info!("Using thread pool with {threads} threads");

    let pool = ThreadPoolBuilder {
        num_threads: ThreadCount::Count(threads),
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::No,
    }
    .build();

    pool
}

pub struct GlAf {
    pub af2: Array1<[f64; 2]>,
    pub af3: Array1<[f64; 3]>,
    pub af4: Array1<[f64; 4]>,
    pub gl2: Array2<[f64; 3]>,
    pub gl3: Array2<[f64; 6]>,
    pub gl4: Array2<[f64; 10]>,
}

// Multi-allelic sites ... never again
fn split_gl_af(genotype_likelihoods: ArrayView2<[f64; 10]>, allele_frequencies: ArrayView1<[f64; 4]>) -> GlAf {
    let (num_variants, num_samples) = genotype_likelihoods.dim();

    let mut af2 = Vec::<[f64; 2]>::new();
    let mut af3 = Vec::<[f64; 3]>::new();
    let mut af4 = Vec::<[f64; 4]>::new();

    let mut gl2 = Vec::<[f64; 3]>::new();
    let mut gl3 = Vec::<[f64; 6]>::new();
    let mut gl4 = Vec::<[f64; 10]>::new();

    let mut nonzero_af_indices = Vec::with_capacity(4);
    let mut af_buf = Vec::with_capacity(4);
    let mut nonzero_gl_indices = Vec::with_capacity(10);
    let mut gl_buf = Vec::with_capacity(10);

    genotype_likelihoods.outer_iter().zip(allele_frequencies.iter()).for_each(|(gl, af)| {

        nonzero_af_indices.clear();
        af_buf.clear();

        for (i, a) in af.iter().enumerate() {
            if *a > 0.0 {
                nonzero_af_indices.push(i);
                af_buf.push(*a);
            }
        }

        if let Some(af_arr) = af_buf.as_array::<2>() {
            af2.push(*af_arr);
        }

        if let Some(af_arr) = af_buf.as_array::<3>() {
            af3.push(*af_arr);
        }

        if let Some(af_arr) = af_buf.as_array::<4>() {
            af4.push(*af_arr);
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
            }

            if let Some(gl_arr) = gl_buf.as_array::<6>() {
                gl3.push(*gl_arr);
            }

            if let Some(gl_arr) = gl_buf.as_array::<10>() {
                gl4.push(*gl_arr);
            }
        }
    });

    let di = af2.len();
    let tri = af3.len();
    let tetra = af4.len();
    let mono = num_variants - di - tri - tetra;

    log::info!("Kept {di} di-allelic sites");
    log::info!("Kept {tri} tri-allelic sites");
    log::info!("Kept {tetra} tetra-allelic sites");
    log::info!("Discarded {mono} non-segregating sites");

    let af2 = Array1::from_vec(af2);
    let af3 = Array1::from_vec(af3);
    let af4 = Array1::from_vec(af4);

    let gl2 = Array2::from_shape_vec((di, num_samples), gl2).unwrap();
    let gl3 = Array2::from_shape_vec((tri, num_samples), gl3).unwrap();
    let gl4 = Array2::from_shape_vec((tetra, num_samples), gl4).unwrap();

    GlAf { af2, af3, af4, gl2, gl3, gl4 }
}