use crate::arith::Lane;

pub struct LaneBuffer<L : Lane, const R: usize> {
    lanes: Vec<[L; R]>,
    rem: Vec<[f64; R]>,
    len: usize,
}

impl<L: Lane + Copy, const R: usize> LaneBuffer<L, R> {

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_lanes(&self) -> (&[[L; R]], &[[f64; R]]) {
        (&self.lanes, &self.rem)
    }

    pub fn as_lanes_mut(&mut self) -> (&mut [[L; R]], &mut [[f64; R]]) {
        (&mut self.lanes, &mut self.rem)
    }

    pub fn new(len: usize, mut iter: impl Iterator<Item = [f64; R]>) -> Self {
        let lanes = len / L::N;
        let rem = len % L::N;

        let mut lane_buf = Vec::with_capacity(lanes);
        let mut rem_buf = Vec::with_capacity(rem);

        for _ in 0..lanes {
            let mut block = [L::zero(); R];

            for i in 0..L::N {
                let row = iter.next().unwrap();
                for c in 0..R {
                    // This is column-major
                    block[c].set(i, row[c]);
                }
            }

            lane_buf.push(block)
        }

        for _ in 0..rem {
            let row = iter.next().unwrap();
            rem_buf.push(row);
        }

        LaneBuffer { lanes: lane_buf, rem: rem_buf, len }
    }
}
