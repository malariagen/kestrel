use crate::arith::Lane;

pub struct LaneBuffer<L : Lane, const R: usize> {
    lanes: Vec<[L; R]>,
    rem: Vec<[f64; R]>,
    len: usize,
}

impl<L: Lane + Copy, const R: usize> LaneBuffer<L, R> {

    pub fn new(len: usize) -> Self {
        let lanes = len / L::N;
        let rem = len % L::N;

        let lane_buf = vec![[L::zero(); R]; lanes];
        let rem_buf = vec![[0.0; R]; rem];

        LaneBuffer { lanes: lane_buf, rem: rem_buf, len }
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
                for c in 0..R {
                    // This is column-major
                    block[c].set(i, row[c]);
                }
            }
        }

        for row in self.rem.iter_mut() {
            *row = iter.next().unwrap();
        }
    }
}
