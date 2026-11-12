import gzip
import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm
import numpy as np

from util import ResourceTracker

def extract_freq(infile, outfile):
    print("Extracting frequencies from", infile)
    with gzip.open(infile, "rt") as fin, open(outfile, "w") as fout:
        header = fin.readline().strip().split()
        em_idx = header.index("knownEM")

        for line in fin:
            fields = line.strip().split()
            if fields:
                fout.write(f"{fields[em_idx]}\n")

def extract_samples(infile, outfile, samples_keep, samples_total):
    BYTES_PER_SAMPLE = 3 * 8  # 3 doubles
    BLOCK_KEEP = samples_keep * BYTES_PER_SAMPLE
    BLOCK_TOTAL = samples_total * BYTES_PER_SAMPLE

    # Calculating all pairwise coeffs for pairs takes very long and is unnecessary
    # We only care about the family samples
    print("Extracting samples of interest from", infile)
    with gzip.open(infile, "rb") as fin, gzip.open(outfile, "wb") as fout:
        while chunk := fin.read(BLOCK_TOTAL):
            fout.write(chunk[:BLOCK_KEEP])

def analyze(rep_dir, seed_seq, threads, samples_keep, samples_total):

    rng = np.random.default_rng(seed_seq)

    depth_dirs = [path for path in rep_dir.iterdir() if path.is_dir()]

    for depth_dir in depth_dirs:

        seed = rng.integers(low=1, high=2**31)

        # https://popgen.dk/angsd/index.php/Allele_Frequencies
        # https://popgen.dk/angsd/index.php/Major_Minor
        # https://popgen.dk/angsd/index.php/Genotype_Likelihoods
        angsd = ["angsd", "-vcf-gl", f"{depth_dir}/GL.vcf.gz", "-P", f"{threads}", "-nInd", f"{samples_total}", "-doMajorMinor", "1", "-doMaf", "1", "-doGlf", "3", "-out", f"{depth_dir}/GL-angsd"]
        print(" ".join(angsd))
        subprocess.run(angsd, check=True, text=True, capture_output=True)

        extract_freq(f"{depth_dir}/GL-angsd.mafs.gz", f"{depth_dir}/angsd-freq.txt")
        extract_samples(f"{depth_dir}/GL-angsd.glf.gz", f"{depth_dir}/GL-angsd-{samples_keep}.glf.gz", samples_keep, samples_total)

        seed = rng.integers(low=1, high=2**31)
        ngsrelate = ["ngsRelate", "-g", f"{depth_dir}/GL-angsd-{samples_keep}.glf.gz", "-n", f"{samples_keep}", "-f", f"{depth_dir}/angsd-freq.txt", "-O", f"{depth_dir}/ngsrelate.tsv", "-l", "0.0", "-r", str(seed), "-p", f"{threads}"]
        print(" ".join(ngsrelate))
        subprocess.run(ngsrelate, check=True, text=True, capture_output=True)

        kestrel = ["kestrel", "-s", f"{samples_keep}", "-p", f"{threads}", f"{depth_dir}/GL.vcf.gz", f"{depth_dir}/kestrel.tsv"]
        print(" ".join(kestrel))
        with open(f"{depth_dir}/kestrel.out", "w") as f:
            subprocess.run(kestrel, check=True, text=True, stdout=f, stderr=f)

if __name__ == '__main__':

    random_seed = 0x136f8bf6e0d7d68f03bf7b8c855c7ec4

    p = argparse.ArgumentParser()
    p.add_argument("dir", type=str, help="VCF file directory")
    p.add_argument("-s", "--seed", type=int, default=random_seed, help="random seed")
    p.add_argument("-p", "--threads", type=int, default=-1, help="number of threads to use for simulations")
    p.add_argument("-k", "--samples-keep", type=int, default=16, help="number of samples to keep for analysis")
    p.add_argument("-t", "--samples-total", type=int, help="total number of samples")

    args = p.parse_args()

    threads = args.threads if args.threads > 0 else None

    # https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning
    seed_seq = np.random.SeedSequence(args.seed)

    base_dir = Path(args.dir)

    rep_dirs = [path for path in base_dir.iterdir() if path.is_dir()]

    with ResourceTracker():
        for (rep_dir, seq) in zip(rep_dirs, seed_seq.spawn(len(rep_dirs))):
            analyze(rep_dir, seq, threads, args.samples_keep, args.samples_total)


# angsd -vcf-gl AnoGam-2L-GL.vcf.gz -domajorminor 1 -domaf 1 -doGlf 3 -out sim
# zcat sim.mafs.gz | cut -f5 | sed 1d > freqs.txt
# then only keep the first 16 samples and calculate pair-wise for that
# ngsRelate -g sim.glf.gz -n 516 -f freqs.txt -O newres -l 0.0

