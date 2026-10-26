import gzip
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def analyze(rep_dir, seed_seq, samples_keep, samples_total):

    rng = np.random.default_rng(seed_seq)

    vcf = f"{rep_dir}/sim"

    if Path(f"{vcf}.vcf").is_file():
        bgzip = ["bgzip", f"{vcf}.vcf"]
        subprocess.run(bgzip, check=True, text=True, capture_output=True)

    for depth in ("1", "5",):
    #for depth in ("1", "5", "10", "20", "30"):
        error_rate = 0.001 # Q30

        dir_name = f"{rep_dir}/{depth}x"

        Path(dir_name).mkdir(exist_ok=True, parents=True)

        seed = rng.integers(low=1, high=2**31)

        # vcfgl can't handle multiple chroms in one file, so we will concat them after
        vcfgl = ["vcfgl", "-i", f"{vcf}.vcf.gz", "--source", "1", "-o", f"{dir_name}/sim-GL", "-O", "z", "-d", str(depth), "--seed", str(seed), "-e", str(error_rate), "-doUnobserved", "3"]

        print(" ".join(vcfgl))

        subprocess.run(vcfgl, check=True, text=True, capture_output=True)

        #bcftools = ["bcftools", "concat"] + [f"{dir_name}/AnoGam-{arm}-GL.vcf.gz" for arm in arms] + ["-O", "z", "-o", f"{dir_name}/AnoGam-GL.vcf.gz"]
        #print(" ".join(bcftools))
        #subprocess.run(bcftools, check=True, text=True, capture_output=True)

        # https://popgen.dk/angsd/index.php/Allele_Frequencies
        # https://popgen.dk/angsd/index.php/Major_Minor
        # https://popgen.dk/angsd/index.php/Genotype_Likelihoods
        angsd = ["angsd", "-vcf-gl", f"{dir_name}/sim-GL.vcf.gz", "-nInd", f"{samples_total}", "-doMajorMinor", "1", "-doMaf", "1", "-doGlf", "3", "-out", f"{dir_name}/sim-GL-angsd"]
        print(" ".join(angsd))
        subprocess.run(angsd, check=True, text=True, capture_output=True)

        extract_freq(f"{dir_name}/sim-GL-angsd.mafs.gz", f"{dir_name}/angsd-freq.txt")
        extract_samples(f"{dir_name}/sim-GL-angsd.glf.gz", f"{dir_name}/sim-GL-angsd-{samples_keep}.glf.gz", samples_keep, samples_total)

        seed = rng.integers(low=1, high=2**31)
        ngsrelate = ["ngsRelate", "-g", f"{dir_name}/sim-GL-angsd-{samples_keep}.glf.gz", "-n", f"{samples_keep}", "-f", f"{dir_name}/angsd-freq.txt", "-O", f"{dir_name}/ngsrelate.tsv", "-l", "0.0", "-r", str(seed), "-p", "1"]
        print(" ".join(ngsrelate))
        subprocess.run(ngsrelate, check=True, text=True, capture_output=True)

        kestrel = ["kestrel", f"{dir_name}/sim-GL.vcf.gz", f"{dir_name}/kestrel.tsv"]
        print(" ".join(kestrel))
        subprocess.run(kestrel, check=True, text=True, capture_output=True)

if __name__ == '__main__':

    samples_keep = 100
    samples_total = 516

    random_seed = 0x136f8bf6e0d7d68f03bf7b8c855c7ec4

    p = argparse.ArgumentParser(
        description="Hello"
    )
    p.add_argument("dir", type=str, help="VCF file directory")
    p.add_argument("-s", "--seed", type=int, default=random_seed, help="random seed")
    p.add_argument("-t", "--threads", type=int, default=-1, help="number of threads to use for simulations")

    args = p.parse_args()

    threads = args.threads if args.threads > 0 else None

    # https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning
    seed_seq = np.random.SeedSequence(args.seed)

    base_dir = Path(args.dir)

    rep_dirs = [path for path in base_dir.iterdir() if path.is_dir()]

    num_reps = len(rep_dirs)

    with ResourceTracker(), ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(analyze, rep_dir, seed, samples_keep, samples_total) for (rep_dir, seed) in zip(rep_dirs, seed_seq.spawn(num_reps))
        ]
        results = [
            f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Analyzing")
        ]


# Q30 phred score is very accurate (standard), maybe Q20 to have some error (0.01)
# in theory we could also sample the errors from a beta distribution centered at the error above with some variance (?)
# (-eq 2 --bv 0.001)

# angsd -vcf-gl AnoGam-2L-GL.vcf.gz -domajorminor 1 -domaf 1 -doGlf 3 -out sim
# zcat sim.mafs.gz | cut -f5 | sed 1d > freqs.txt
# then only keep the first 16 samples and calculate pair-wise for that
# ngsRelate -g sim.glf.gz -n 516 -f freqs.txt -O newres -l 0.0
# then we need to manually go back to the 0-1, 2-3, etc. ugh.

