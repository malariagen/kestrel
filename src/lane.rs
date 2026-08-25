use core::arch::x86_64::*;

pub trait Lane {
    const LEN: usize;
}

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct Lane8(pub [f64; 8]);

impl Lane for Lane8 {
    const LEN: usize = 8;
}

impl Lane8 {
    #[inline]
    pub fn zero() -> Self {
        Lane8([0.0; 8])
    }

    #[inline]
    pub fn load(&self) -> __m512d {
        unsafe { _mm512_load_pd(self.0.as_ptr()) }
    }

    #[inline]
    pub fn store(&mut self, reg: __m512d) {
        unsafe { _mm512_store_pd(self.0.as_mut_ptr(), reg) }
    }
}
