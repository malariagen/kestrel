use crate::simd::Lane;

pub struct LaneVector<L: Lane, const R: usize> {
    lanes: Vec<[L; R]>,
    rem: Vec<[f64; R]>,
}

impl<L: Lane, const R: usize> LaneVector<L, R> {
    pub fn new(max_len: usize) -> Self {
        let lanes = max_len / L::N;

        let lane_buf = Vec::with_capacity(lanes);
        let rem_buf = Vec::with_capacity(L::N);

        LaneVector {
            lanes: lane_buf,
            rem: rem_buf,
        }
    }

    pub fn lanes(&self) -> &[[L; R]] {
        &self.lanes
    }

    pub fn rem(&self) -> &[[f64; R]] {
        &self.rem
    }

    pub fn fill_from_slice(&mut self, rows: &[[f64; R]]) {
        self.lanes.clear();
        self.rem.clear();

        let mut curr_row = 0;
        let mut block = [L::default(); R];

        for row in rows {
            if row.iter().any(|f| f.is_nan()) {
                continue;
            }

            for r in 0..R {
                block[r].set(curr_row, row[r]);
            }

            curr_row += 1;

            if curr_row == L::N {
                self.lanes.push(block);
                curr_row = 0;
            }
        }

        // Now we need to copy out the last few rows that don't fit into a whole block
        for i in 0..curr_row {
            let row = std::array::from_fn(|r| block[r].get(i));
            self.rem.push(row);
        }

        assert!(!self.lanes.is_empty() || !self.rem.is_empty(), "Must have at least one row to analyze")
    }
}
