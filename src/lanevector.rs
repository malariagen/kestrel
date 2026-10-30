use crate::simd::Lane;

pub struct LaneVector<L: Lane, const R: usize> {
    lanes: Vec<[L; R]>,
    rem: Vec<[f64; R]>,
}

impl<L: Lane, const R: usize> LaneVector<L, R> {

    pub fn empty() -> Self {
        let lane_buf = Vec::new();
        let rem_buf = Vec::with_capacity(L::N);

        LaneVector {
            lanes: lane_buf,
            rem: rem_buf,
        }
    }

    pub fn new(max_len: usize) -> Self {
        let lanes = max_len / L::N;

        let lane_buf = Vec::with_capacity(lanes);
        let rem_buf = Vec::with_capacity(L::N);

        LaneVector {
            lanes: lane_buf,
            rem: rem_buf,
        }
    }

    pub fn len(&self) -> usize {
        L::N * self.lanes.len() + self.rem.len()
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

pub struct LaneVectorBuilder<'a, L: Lane, const R: usize> {
    // Not a complete fan of this, but it allows usage when you can't own the lv
    lv: &'a mut LaneVector<L, R>,
    block: [L; R],
    curr_row: usize,
}

impl<'a, L: Lane, const R: usize> LaneVectorBuilder<'a, L, R> {

    pub fn new(lv: &'a mut LaneVector<L, R>) -> Self {
        lv.lanes.clear();
        lv.rem.clear();
        Self { lv, block: [L::default(); R], curr_row: 0}
    }

    pub fn push(&mut self, row: &[f64; R]) {
        if row.iter().any(|f| f.is_nan()) {
            return;
        }

        for r in 0..R {
            self.block[r].set(self.curr_row, row[r]);
        }

        self.curr_row += 1;

        if self.curr_row == L::N {
            self.lv.lanes.push(self.block);
            self.curr_row = 0;
        }
    }

    pub fn push_block<M: Lane>(&mut self, block: &[M; R]) {
        for i in 0..M::N {
            let row: [f64; R] = std::array::from_fn(|r| block[r].get(i));
            self.push(&row);
        }
    }

    pub fn finish(self) {
        // Copy out the last few rows that didn't fit into a whole block
        for i in 0..self.curr_row {
            let row = std::array::from_fn(|r| self.block[r].get(i));
            self.lv.rem.push(row);
        }

        assert!(!self.lv.lanes.is_empty() || !self.lv.rem.is_empty(), "Must have at least one row to analyze");
    }

}