use crate::arith::Lane;

#[repr(align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Lane2(pub [f64; 2]);

#[repr(align(32))]
#[derive(Clone, Copy, Debug)]
pub struct Lane4(pub [f64; 4]);

impl Lane for Lane2 {
    const N: usize = 2;

    #[inline]
    fn get(&self, i: usize) -> f64 {
        self.0[i]
    }

    #[inline]
    fn set(&mut self, i: usize, val: f64) {
        self.0[i] = val;
    }

    #[inline]
    fn zero() -> Self {
        Self([0.0; Self::N])
    }
}

impl Lane for Lane4 {
    const N: usize = 4;

    #[inline]
    fn get(&self, i: usize) -> f64 {
        self.0[i]
    }

    #[inline]
    fn set(&mut self, i: usize, val: f64) {
        self.0[i] = val;
    }

    #[inline]
    fn zero() -> Self {
        Self([0.0; Self::N])
    }
}
