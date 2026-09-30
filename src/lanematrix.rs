use crate::arith::Lane;

pub struct LaneMatrix<L: Lane, const R: usize, const C: usize> {
    lanes: Vec<[[L; C]; R]>,
    rem: Vec<[[f64; C]; R]>,
    len: usize,
}

impl<L: Lane + Copy, const R: usize, const C: usize> LaneMatrix<L, R, C> {
    pub fn new(len: usize) -> Self {
        let lanes = len / L::N;
        let rem = len % L::N;

        let lane_buf = vec![[[L::zero(); C]; R]; lanes];
        let rem_buf = vec![[[0.0; C]; R]; rem];

        LaneMatrix {
            lanes: lane_buf,
            rem: rem_buf,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_lanes(&self) -> (&[[[L; C]; R]], &[[[f64; C]; R]]) {
        (&self.lanes, &self.rem)
    }

    pub fn as_lanes_mut(&mut self) -> (&mut [[[L; C]; R]], &mut [[[f64; C]; R]]) {
        (&mut self.lanes, &mut self.rem)
    }

    pub fn fill_from_iter(&mut self, mut iter: impl Iterator<Item = [[f64; C]; R]>) {
        for block in self.lanes.iter_mut() {
            for i in 0..L::N {
                let row = iter.next().unwrap();
                for r in 0..R {
                    for c in 0..C {
                        // This is column-major
                        block[r][c].set(i, row[r][c]);
                    }
                }
            }
        }

        for row in self.rem.iter_mut() {
            *row = iter.next().unwrap();
        }
    }
}
