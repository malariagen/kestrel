use crate::arith::Lane;

pub struct LaneVector<L: Lane, const R: usize> {
    pub lanes: Vec<[L; R]>,
    pub rem: Vec<[f64; R]>,
    len: usize,
}

impl<L: Lane, const R: usize> LaneVector<L, R> {
    pub fn new(len: usize) -> Self {
        let lanes = len / L::N;
        let rem = len % L::N;

        let lane_buf = vec![[L::default(); R]; lanes];
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

        assert!(iter.next().is_none());
    }
}
