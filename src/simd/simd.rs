#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Simd {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

impl Simd {
    // https://kerkour.com/introduction-rust-simd
    // https://shnatsel.github.io/state-of-simd-rust-2026/
    pub fn detect() -> Self {
        // These extensions correspond to x86-64-v4
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") &&
           is_x86_feature_detected!("avx512bw") &&
           is_x86_feature_detected!("avx512cd") &&
           is_x86_feature_detected!("avx512dq") &&
           is_x86_feature_detected!("avx512vl") {
            log::info!("AVX-512 detected, using SIMD intrinsics");
            return Simd::Avx512;
        }

        // And these correspond to the useful parts of x86-64-v3
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") &&
           is_x86_feature_detected!("fma") &&
           is_x86_feature_detected!("bmi1") &&
           is_x86_feature_detected!("bmi2") &&
           is_x86_feature_detected!("lzcnt") {
            log::info!("AVX2 detected, using SIMD intrinsics");
            return Simd::Avx2;
        }

        // We do not support SSE, since that has no FMA instructions
        // (which we need for logs, etc.)

        // These are the useful parts of ARMv8.3-A
        #[cfg(target_arch = "aarch64")]
        if is_aarch64_feature_detected!("neon") &&
            is_aarch64_feature_detected!("lse") &&
            is_aarch64_feature_detected!("rcpc") {
            log::info!("NEON detected, using SIMD intrinsics");
            return Simd::Neon;
        }

        log::warn!("No SIMD support detected. A scalar fallback will be used instead, which may be quite slow...");
        log::warn!("Considering creating an issue in the git repo if you see this, we try to support common hardware.");
        Simd::Scalar
    }
}
