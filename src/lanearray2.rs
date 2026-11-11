use ndarray::ArrayView2;

use crate::simd::Lane;

// This is really an Array3 with a fixed third dimension,
// And we organize it in this way
// We need to pick which dimension we compress into the lanes
// The variant dimension
// So the format is N x compressed x fixed
// One could also store this as Vec<compressed x fixed>,
// but that's worse for cache locality

// We store the Matrix as two flat buffers instead (row major)
pub struct LaneArray2<L: Lane, const G: usize> {
    pub lanes: Vec<[L; G]>,
    pub rem: Vec<[f64; G]>,
    pub num_lanes: usize,
    pub num_rem: usize,
}

impl<L: Lane, const G: usize> LaneArray2<L, G> {
    pub fn new(nums_samples: usize, num_sites: usize) -> Self {
        let num_lanes = num_sites / L::N;
        let num_rem = num_sites % L::N;

        let lanes = vec![[L::default(); G]; nums_samples * num_lanes];
        let rem = vec![[0.0; G]; nums_samples * num_rem];

        LaneArray2 {
            lanes,
            rem,
            num_lanes,
            num_rem,
        }
    }

    pub fn sample(&self, s: usize) -> (&[[L; G]], &[[f64; G]]) {
        let num_lanes = self.num_lanes;
        let num_rem = self.num_rem;

        let lane_row = &self.lanes[s * num_lanes..(s + 1) * num_lanes];
        let rem_row = &self.rem[s * num_rem..(s + 1) * num_rem];
        (lane_row, rem_row)
    }
}

pub fn convert_to_lanes<L: Lane, const G: usize>(genotype_likelihoods: ArrayView2<[f64; G]>) -> LaneArray2<L, G> {
    let (num_v, num_s) = genotype_likelihoods.dim();

    let mut lane_array = LaneArray2::<L, G>::new(num_s, num_v);

    let num_lanes = num_v / L::N;
    let num_rem = num_v % L::N;

    for s in 0..num_s {
        for l in 0..num_lanes {
            let v_base = l * L::N;
            let block = &mut lane_array.lanes[s * num_lanes + l];

            for i in 0..L::N {
                let v = v_base + i;
                for r in 0..G {
                    block[r].set(i, genotype_likelihoods[[v, s]][r]);
                }
            }
        }

        for r in 0..num_rem {
            let v = num_lanes * L::N + r;
            let rem_block = &mut lane_array.rem[s * num_rem + r];

            for i in 0..G {
                rem_block[i] = genotype_likelihoods[[v, s]][i];
            }
        }
    }

    lane_array
}
