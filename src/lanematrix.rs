use crate::simd::Lane;

pub type M4<T> = [[[T; 9]; 10]; 10];

pub struct LaneM4<L: Lane> {
    pub lanes: Vec<M4<L>>,
    pub rem: Vec<M4<f64>>,
}

impl<L: Lane> LaneM4<L> {
    pub fn new(len: usize) -> Self {
        let lanes = len / L::N;
        let rem = len % L::N;

        let lane_buf = vec![M4::default(); lanes];
        let rem_buf = vec![M4::default(); rem];

        LaneM4 {
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

    pub fn fill_from_iter(&mut self, mut iter: impl Iterator<Item = M4<f64>>) {
        for block in self.lanes.iter_mut() {
            for i in 0..L::N {
                let m = iter.next().unwrap();
                for a in 0..10 {
                    for b in 0..10 {
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
