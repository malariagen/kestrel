import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from Bio import bgzf
import numpy as np

from util import ResourceTracker

def thin_and_subsample_vcf(input_path: str, out_dir: str, select_sites: int, min_dist: int, rep: int, seed):

    print("Subsampling rep", rep)

    rng = np.random.default_rng(seed)

    rep_name = f"{out_dir}/{rep}"
    Path(rep_name).mkdir(exist_ok=True, parents=True)

    last_chrom = None
    last_pos = -1
    n_thinned = 0

    with bgzf.open(input_path, "rb") as infile:
        for line in infile:
            if not line.startswith(b"#"):
                chrom, pos_str, _ = line.split(b"\t", 2)

                pos = int(pos_str)

                if chrom != last_chrom or (pos - last_pos) >= min_dist:
                    n_thinned += 1
                    last_chrom = chrom
                    last_pos = pos

    print(f"Thinned to {n_thinned} sites")

    assert select_sites <= n_thinned

    selected = set(rng.choice(n_thinned, size=select_sites, replace=False).tolist())

    last_chrom = None
    last_pos = -1
    thin_index = 0
    with (
        bgzf.open(input_path, "rb") as infile,
        bgzf.open(rep_name + "/GT.vcf.gz", "wb") as ofile
    ):
        for line in infile:
            if line.startswith(b"#"):
                ofile.write(line)
            else:
                chrom, pos_str, _ = line.split(b"\t", 2)

                pos = int(pos_str)

                if chrom != last_chrom or (pos - last_pos) >= min_dist:
                    if thin_index in selected:
                        ofile.write(line)
                    thin_index += 1
                    last_chrom = chrom
                    last_pos = pos


def subsample_vcf(input_path: str, out_dir: str, n_sites: int, rep: int, seed):

    print("Subsampling rep", rep)

    rng = np.random.default_rng(seed)

    rep_name = f"{out_dir}/{rep}"
    Path(rep_name).mkdir(exist_ok=True, parents=True)

    with gzip.open(input_path, "rb") as infile:
        total_sites = sum(1 for line in infile if not line.startswith(b"#"))

    k = min(n_sites, total_sites)

    selected = set(rng.choice(total_sites, size=k, replace=False).tolist())

    index = 0
    with (
        gzip.open(input_path, "rb") as infile,
        bgzf.open(rep_name + "/GT.vcf.gz", "wb") as ofile
    ):
        for line in infile:
            if line.startswith(b"#"):
                ofile.write(line)
            else:
                if index in selected:
                    ofile.write(line)
                index += 1

if __name__ == "__main__":

    # Only the finest bits from /dev/urandom
    random_seed = 0x2570d7da84fa357bab6aee78a8e3e4b2

    p = argparse.ArgumentParser(
        description="Subsample a VCF file",
    )
    p.add_argument("vcf", type=str, help="input VCF file")
    p.add_argument("sites", type=int, help="number of random sites to select")
    p.add_argument("out_dir", type=str, help="output directory")
    p.add_argument("-d", "--distance", type=int, help="minimum distance between sites")
    p.add_argument("-r", "--reps", type=int, default=1, help="number of subsamples to run")
    p.add_argument("-s", "--seed", type=int, default=random_seed, help="random seed")
    p.add_argument("-t", "--threads", type=int, default=-1, help="number of threads to use for subsampling")

    args = p.parse_args()

    threads = args.threads if args.threads > 0 else None

    # https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning
    seed_seq = np.random.SeedSequence(args.seed)

    with ResourceTracker(), ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(thin_and_subsample_vcf, args.vcf, args.out_dir, args.sites, args.distance, rep, seed) for (rep, seed) in enumerate(seed_seq.spawn(args.reps))
        ]

        # This is needed to propogate exceptions
        for f in futures:
            f.result()
