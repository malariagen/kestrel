use ndarray::ArrayView3;

use crate::simd::Lane;

// This is really an Array3 with a fixed third dimension,
// And we organize it in this way
// We need to pick which dimension we compress into the lanes
// The variant dimension
// So the format is N x compressed x fixed
// One could also store this as Vec<compressed x fixed>,
// but that's worse for cache locality

// We store the Matrix as two flat buffers instead (row major)
pub struct LaneArray2<L: Lane, const R: usize> {
    pub lanes: Vec<[L; R]>,
    pub rem: Vec<[f64; R]>,
    pub rows: usize,
    pub num_lanes: usize,
    pub num_rem: usize,
}

impl<L: Lane, const R: usize> LaneArray2<L, R> {
    pub fn new(rows: usize, cols: usize) -> Self {
        let num_lanes = cols / L::N;
        let num_rem = cols % L::N;

        let lanes = vec![[L::default(); R]; rows * num_lanes];
        let rem = vec![[0.0; R]; rows * num_rem];

        LaneArray2 {
            lanes,
            rem,
            rows,
            num_lanes,
            num_rem,
        }
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = (&[[L; R]], &[[f64; R]])> {
        let num_lanes = self.num_lanes;
        let num_rem = self.num_rem;

        (0..self.rows).map(move |r| {
            let lane_row = &self.lanes[r * num_lanes..(r + 1) * num_lanes];
            let rem_row = &self.rem[r * num_rem..(r + 1) * num_rem];
            (lane_row, rem_row)
        })
    }
}

pub fn convert_to_lanes<L: Lane>(genotype_likelihoods: ArrayView3<f64>) -> LaneArray2<L, 10> {
    let num_v = genotype_likelihoods.shape()[0];
    let num_s = genotype_likelihoods.shape()[1];

    let mut lane_array = LaneArray2::<L, 10>::new(num_s, num_v);

    let num_lanes = num_v / L::N;
    let num_rem = num_v % L::N;

    for s in 0..num_s {
        for l in 0..num_lanes {
            let v_base = l * L::N;
            let block = &mut lane_array.lanes[s * num_lanes + l];

            for i in 0..L::N {
                let v = v_base + i;
                for r in 0..10 {
                    block[r].set(i, genotype_likelihoods[[v, s, r]]);
                }
            }
        }

        for r in 0..num_rem {
            let v = num_lanes * L::N + r;
            let rem_block = &mut lane_array.rem[s * num_rem + r];

            for i in 0..10 {
                rem_block[i] = genotype_likelihoods[[v, s, i]];
            }
        }
    }

    lane_array
}
