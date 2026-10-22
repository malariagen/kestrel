use std::num::{NonZeroU64, NonZeroUsize};

use ndarray::{ArrayView2, ArrayView3};
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
) -> Vec<Output> {
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
) -> Vec<Output> {

    let num_v = genotype_likelihoods.shape()[0];
    let num_s = genotype_likelihoods.shape()[1];
    let num_l = genotype_likelihoods.shape()[2];

    assert!(num_v > 0, "Must have at least one variant");
    assert!(num_s > 0, "Must have at least one sample");
    assert!(num_l == 10, "Must have exactly 10 GLs per sample");

    // This is the smallest allele frequency possible given our sample size
    // Anything lower than this is just noise
    let maf = minimum_allele_frequency.unwrap_or(1.0 / (2.0 * num_s as f64));

    assert!(0.0 <= maf && maf <= 1.0, "Minimum allele frequency must be within 0 to 1");

    let simd = Simd::detect();

    let mut thread_pool = build_thread_pool(threads);

    // This works, suprisingly
    let calculated_af;
    let allele_frequencies = match allele_frequencies {
        Some(af) => af,
        None => {
            calculated_af = match simd {
                #[cfg(target_arch = "x86_64")]
                Simd::Avx512 => crate::allele::calculate_allele_frequencies::<Avx512Lane>(
                    genotype_likelihoods,
                    &mut thread_pool,
                    iterations,
                    maf
                ),
                #[cfg(target_arch = "x86_64")]
                Simd::Avx2 => crate::allele::calculate_allele_frequencies::<Avx2Lane>(
                    genotype_likelihoods,
                    &mut thread_pool,
                    iterations,
                    maf
                ),
                #[cfg(target_arch = "aarch64")]
                Simd::Neon => crate::allele::calculate_allele_frequencies::<NeonLane>(
                    genotype_likelihoods,
                    &mut thread_pool,
                    iterations,
                    maf
                ),
                Simd::Scalar => crate::allele::calculate_allele_frequencies::<f64>(
                    genotype_likelihoods,
                    &mut thread_pool,
                    iterations,
                    maf
                ),
            };

            calculated_af.view()
        }
    };

    // TODO in theory we could do some checks here that the af are non-negative, sum to one
    // Indexing will work, etc.

    assert!(allele_frequencies.shape()[0] == num_v, "Must have same dimension");
    assert!(allele_frequencies.shape()[1] == 4, "Must have all four alleles");

    let outputs = match simd {
        #[cfg(target_arch = "x86_64")]
        Simd::Avx512 => crate::coefficients::calculate_relatedness_coefficients_gl::<Avx512Lane>(
            genotype_likelihoods,
            allele_frequencies,
            &mut thread_pool,
            iterations,
        ),
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => crate::coefficients::calculate_relatedness_coefficients_gl::<Avx2Lane>(
            genotype_likelihoods,
            allele_frequencies,
            &mut thread_pool,
            iterations,
        ),
        #[cfg(target_arch = "aarch64")]
        Simd::Neon => crate::coefficients::calculate_relatedness_coefficients_gl::<NeonLane>(
            genotype_likelihoods,
            allele_frequencies,
            &mut thread_pool,
            iterations,
        ),
        Simd::Scalar => crate::coefficients::calculate_relatedness_coefficients_gl::<f64>(
            genotype_likelihoods,
            allele_frequencies,
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
