use crate::simd::Lane;

pub type M<T, const R: usize> = [[[T; 9]; R]; R];

pub struct LaneM<L: Lane, const R: usize> {
    pub lanes: Vec<M<L, R>>,
    pub rem: Vec<M<f64, R>>,
}

impl<L: Lane, const R: usize> LaneM<L, R> {
    pub fn new(len: usize) -> Self {
        let lanes = len / L::N;
        let rem = len % L::N;

        let lane_buf = vec![[[[L::default(); 9]; R]; R]; lanes];
        let rem_buf = vec![[[[0.0; 9]; R]; R]; rem];

        LaneM {
            lanes: lane_buf,
            rem: rem_buf,
        }
    }

    // pub fn len(&self) -> usize {
    //     self.len
    // }

    // pub fn as_lanes(&self) -> (&[[[L; C]; R]], &[[[f64; C]; R]]) {
    //     (&self.lanes, &self.rem)
    // }

    // pub fn as_lanes_mut(&mut self) -> (&mut [[[L; C]; R]], &mut [[[f64; C]; R]]) {
    //     (&mut self.lanes, &mut self.rem)
    // }

    pub fn fill_from_iter(&mut self, mut iter: impl Iterator<Item = M<f64, R>>) {
        for block in self.lanes.iter_mut() {
            for i in 0..L::N {
                let m = iter.next().unwrap();
                for a in 0..R {
                    for b in 0..R {
                        for c in 0..9 {
                            block[a][b][c].set(i, m[a][b][c]);
                        }
                    }
                }
            }
        }

        for row in self.rem.iter_mut() {
            *row = iter.next().unwrap();
        }

        assert!(iter.next().is_none());
    }
}
