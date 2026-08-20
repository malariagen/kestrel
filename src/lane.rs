use core::arch::x86_64::*;

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct Lane8(pub [f64; 8]);

impl Lane8 {
    pub fn zero() -> Self {
        Lane8([0.0; 8])
    }

    pub fn load(&self) -> __m512d {
        unsafe { _mm512_load_pd(self.0.as_ptr()) }
    }

    pub fn store(&mut self, reg: __m512d) {
        unsafe { _mm512_store_pd(self.0.as_mut_ptr(), reg) }
    }
}
