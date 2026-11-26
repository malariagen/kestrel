import gzip
import argparse
import subprocess
from pathlib import Path
import shutil

import numpy as np

from util import ResourceTracker

def calculate(rep_dir, seed_seq, threads):

    rng = np.random.default_rng(seed_seq)

    print(f"Calculating {rep_dir}...")

    seed = rng.integers(low=1, high=2**31)
    ngsrelate = ["ngsRelate", "-h", f"{rep_dir}/GT.vcf.gz", "-O", f"{rep_dir}/ngsrelate.tsv", "-T", "GT", "-l", "0.0", "-r", str(seed), "-p", f"{threads}"]
    print(" ".join(ngsrelate))
    with open(f"{rep_dir}/ngsrelate.out", "w") as f:
        subprocess.run(ngsrelate, check=True, text=True, stdout=f, stderr=f)

    kestrel = ["kestrel", "-p", f"{threads}", "-t", "GT", f"{rep_dir}/GT.vcf.gz", f"{rep_dir}/kestrel.tsv"]
    print(" ".join(kestrel))
    with open(f"{rep_dir}/kestrel.out", "w") as f:
        subprocess.run(kestrel, check=True, text=True, stdout=f, stderr=f)

if __name__ == '__main__':

    random_seed = 0x7b6487444a601ababb499d55df9816cd

    p = argparse.ArgumentParser()
    p.add_argument("dir", type=str, help="VCF file directory")
    p.add_argument("-s", "--seed", type=int, default=random_seed, help="random seed")
    p.add_argument("-p", "--threads", type=int, default=-1, help="number of threads to use for simulations")

    args = p.parse_args()

    threads = args.threads if args.threads > 0 else None

    # https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning
    seed_seq = np.random.SeedSequence(args.seed)

    base_dir = Path(args.dir)

    rep_dirs = [path for path in base_dir.iterdir() if path.is_dir()]

    with ResourceTracker():
        for (rep_dir, seq) in zip(rep_dirs, seed_seq.spawn(len(rep_dirs))):
            calculate(rep_dir, seq, threads)

