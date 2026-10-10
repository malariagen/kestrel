#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Simd {
    Avx2,
    Avx512,
    Neon,
    Scalar,
}

impl Simd {
    // https://kerkour.com/introduction-rust-simd
    // https://shnatsel.github.io/state-of-simd-rust-2026/
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            log::info!("AVX-512 detected, using SIMD intrinsics");
            return Simd::Avx512;
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            log::info!("AVX2 detected, using SIMD intrinsics");
            return Simd::Avx2;
        }

        #[cfg(target_arch = "aarch64")]
        if is_aarch64_feature_detected!("neon") {
            log::info!("NEON detected, using SIMD intrinsics");
            return Simd::Neon;
        }

        log::warn!("No SIMD support detected. A scalar fallback will be used instead, which may be quite slow...");
        log::warn!("Considering creating an issue in the git repo if you see this, we try to support common hardware.");
        Simd::Scalar
    }
}
