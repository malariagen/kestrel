import gzip
import subprocess
import numpy as np

Ne = 10000

arms = ("2L", "2R", "3L", "3R")
samples_keep = 16
samples_total = 516

global_seed = 0x136f8bf6e0d7d68f03bf7b8c855c7ec4

def extract_freq(infile, outfile):
    print("Extracting frequencies from", infile)
    with gzip.open(infile, "rt") as fin, open(outfile, "w") as fout:
        header = fin.readline().strip().split()
        em_idx = header.index("knownEM")

        for line in fin:
            fields = line.strip().split()
            if fields:
                fout.write(f"{fields[em_idx]}\n")

def extract_samples(infile, outfile):
    BYTES_PER_SAMPLE = 3 * 8  # 3 doubles
    BLOCK_KEEP = samples_keep * BYTES_PER_SAMPLE
    BLOCK_TOTAL = samples_total * BYTES_PER_SAMPLE

    # Calculating all pairwise coeffs for 516 pairs takes very long and is unnecessary
    # We only care about the 16 family samples
    print("Extracting samples of interest from", infile)
    with gzip.open(infile, "rb") as fin, gzip.open(outfile, "wb") as fout:
        while chunk := fin.read(BLOCK_TOTAL):
            fout.write(chunk[:BLOCK_KEEP])

for arm in arms:
    continue
    # TODO add multiple depths, maybe Q20
    error_rate = 0.001
    depth = 10
    vcf = f"Ne_{Ne}/AnoGam-{arm}"

    # vcfgl can't handle multiple chroms in one file, so we will concat them after
    vcfgl = ["vcfgl", "-i", f"{vcf}.vcf", "--source", "1", "-o", f"{vcf}-GL", "-O", "z", "-d", str(depth), "--seed", "42", "-e", str(error_rate), "-doUnobserved", "3"]

    print(" ".join(vcfgl))

    subprocess.run(vcfgl, check=True, text=True, capture_output=True)

bcftools = ["bcftools", "concat"] + [f"Ne_{Ne}/AnoGam-{arm}-GL.vcf.gz" for arm in arms] + ["-O", "z", "-o", f"Ne_{Ne}/AnoGam-GL.vcf.gz"]
print(" ".join(bcftools))
#subprocess.run(bcftools, check=True, text=True, capture_output=True)

# TODO look more at the args for this, like the p-value and stuff
angsd = ["angsd", "-vcf-gl", f"Ne_{Ne}/AnoGam-GL.vcf.gz", "-nInd", f"{samples_total}", "-doMajorMinor", "1", "-doMaf", "1", "-doGlf", "3", "-out", f"Ne_{Ne}/AnoGam-GL-angsd"]
print(" ".join(angsd))
#subprocess.run(angsd, check=True, text=True, capture_output=True)
raise

extract_freq(f"Ne_{Ne}/AnoGam-GL-angsd.mafs.gz", f"Ne_{Ne}/freq.txt")
extract_samples(f"Ne_{Ne}/AnoGam-GL-angsd.glf.gz", f"Ne_{Ne}/AnoGam-GL-angsd-{samples_keep}.glf.gz")

# TODO seed and threads
ngsrelate = ["ngsRelate", "-g", f"Ne_{Ne}/AnoGam-GL-angsd-{samples_keep}.glf.gz", "-n", f"{samples_keep}", "-f", f"Ne_{Ne}/freq.txt", "-O", f"Ne_{Ne}/ngsrelate.tsv", "-l", "0.0"]
print(" ".join(ngsrelate))
subprocess.run(ngsrelate, check=True, text=True, capture_output=True)

ngsrelate = ["kestrel", ]
print(" ".join(kestrel))
subprocess.run(kestrel, check=True, text=True, capture_output=True)

if __name__ == '__main__':
    pass

# Q30 phred score is very accurate (standard), maybe Q20 to have some error (0.01)
# in theory we could also sample the errors from a beta distribution centered at the error above with some variance (?)
# (-eq 2 --bv 0.001)

# angsd -vcf-gl AnoGam-2L-GL.vcf.gz -domajorminor 1 -domaf 1 -doGlf 3 -out sim
# zcat sim.mafs.gz | cut -f5 | sed 1d > freqs.txt
# then only keep the first 16 samples and calculate pair-wise for that
# ngsRelate -g sim.glf.gz -n 516 -f freqs.txt -O newres -l 0.0
# then we need to manually go back to the 0-1, 2-3, etc. ugh.

# then run kestrel and ngsrelate on all the files, and collect the results (timing? perhaps ignore for now)

