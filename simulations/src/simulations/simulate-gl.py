import argparse
import gzip
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from util import ResourceTracker

def simulate(rep_dir, seed_seq, depths, error_rate):

    rng = np.random.default_rng(seed_seq)

    for depth in depths:

        dir_name = f"{rep_dir}/{depth}x"

        Path(dir_name).mkdir(exist_ok=True, parents=True)

        seed = rng.integers(low=1, high=2**31)

        # vcfgl can't handle multiple chroms in one file
        # angsd also segfaults if the file contains sites with empty ALT, so remove them
        vcfgl = ["vcfgl", "-i", f"{rep_dir}/GT.vcf.gz", "--source", "1", "-o", f"{dir_name}/GL", "-O", "z", "-d", str(depth), "--seed", str(seed), "-e", str(error_rate), "-doUnobserved", "0", "--rm-invar-sites", "4", "--rm-empty-sites", "1"]
        print(" ".join(vcfgl))
        subprocess.run(vcfgl, check=True, text=True, capture_output=True)

if __name__ == '__main__':

    random_seed = 0x136f8bf6e0d7d68f03bf7b8c855c7ec4

    p = argparse.ArgumentParser()
    p.add_argument("dir", type=str, help="VCF file directory")
    p.add_argument("-s", "--seed", type=int, default=random_seed, help="random seed")
    p.add_argument("-p", "--threads", type=int, default=-1, help="number of threads to use for simulations")
    p.add_argument("-d", "--depths", type=lambda x: [int(d) for d in x.split(",")], help="list of depths to simulate")
    p.add_argument("-e", "--error-rate", type=float, default=0.001, help="probability of a read error")

    args = p.parse_args()

    threads = args.threads if args.threads > 0 else None

    # https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning
    seed_seq = np.random.SeedSequence(args.seed)

    base_dir = Path(args.dir)

    rep_dirs = [path for path in base_dir.iterdir() if path.is_dir()]

    num_reps = len(rep_dirs)

    with ResourceTracker(), ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(simulate, rep_dir, seed, args.depths, args.error_rate) for (rep_dir, seed) in zip(rep_dirs, seed_seq.spawn(num_reps))
        ]

        for f in futures:
            f.result()


# Q30 phred score is very accurate (standard), maybe Q20 to have some error (0.01)
# in theory we could also sample the errors from a beta distribution centered at the error above with some variance (?)
# (-eq 2 --bv 0.001)

# angsd -vcf-gl AnoGam-2L-GL.vcf.gz -domajorminor 1 -domaf 1 -doGlf 3 -out sim
# zcat sim.mafs.gz | cut -f5 | sed 1d > freqs.txt
# then only keep the first 16 samples and calculate pair-wise for that
# ngsRelate -g sim.glf.gz -n 516 -f freqs.txt -O newres -l 0.0

