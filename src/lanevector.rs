use crate::arith::lane::{Lane2, Lane4};
use crate::arith::simd::Simd;
use crate::arith::{Lane, Lane8};

pub struct LaneVector<L: Lane, const R: usize> {
    lanes: Vec<[L; R]>,
    rem: Vec<[f64; R]>,
    len: usize,
}

impl<L: Lane, const R: usize> LaneVector<L, R> {
    pub fn new(len: usize) -> Self {
        let lanes = len / L::N;
        let rem = len % L::N;

        let lane_buf = vec![[L::zero(); R]; lanes];
        let rem_buf = vec![[0.0; R]; rem];

        LaneVector {
            lanes: lane_buf,
            rem: rem_buf,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_lanes(&self) -> (&[[L; R]], &[[f64; R]]) {
        (&self.lanes, &self.rem)
    }

    pub fn as_lanes_mut(&mut self) -> (&mut [[L; R]], &mut [[f64; R]]) {
        (&mut self.lanes, &mut self.rem)
    }

    pub fn fill_from_iter(&mut self, mut iter: impl Iterator<Item = [f64; R]>) {
        for block in self.lanes.iter_mut() {
            for i in 0..L::N {
                let row = iter.next().unwrap();
                for r in 0..R {
                    block[r].set(i, row[r]);
                }
            }
        }

        for row in self.rem.iter_mut() {
            *row = iter.next().unwrap();
        }
    }
}

pub enum GenericLaneVector<const R: usize> {
    L8(LaneVector<Lane8, R>),
    L4(LaneVector<Lane4, R>),
    L2(LaneVector<Lane2, R>),
    L1(LaneVector<f64, R>),
}

impl<const R: usize> GenericLaneVector<R> {
    pub fn new(len: usize, simd: Simd) -> Self {
        match simd {
            Simd::Avx512 => Self::L8(LaneVector::new(len)),
            Simd::Avx2 => Self::L4(LaneVector::new(len)),
            Simd::Neon => Self::L2(LaneVector::new(len)),
            Simd::Scalar => Self::L1(LaneVector::new(len)),
        }
    }

    pub fn fill_from_iter(&mut self, iter: impl Iterator<Item = [f64; R]>) {
        match self {
            Self::L8(lv) => lv.fill_from_iter(iter),
            Self::L4(lv) => lv.fill_from_iter(iter),
            Self::L2(lv) => lv.fill_from_iter(iter),
            Self::L1(lv) => lv.fill_from_iter(iter),
        }
    }
}
