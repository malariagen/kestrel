# Kestrel

Kestrel estimates relatedness coefficients from genomic data. It was started during [GSoC 2026](https://summerofcode.withgoogle.com/programs/2026/projects/ar0jqOKO) and uses an SQP algorithm to calculate the Jacquard coefficients using a maximum likelihood estimator.

## Installation

Install [Rust](https://rust-lang.org/tools/install/) (usually using `rustup` from your package manager). Then clone the repo and run

```
RUSTFLAGS='-C target-cpu=native' cargo build --release
```

The binary will be built in `target/release/kestrel`. Currently the algorithms use AVX-512 intrinsics to accelerate the mathematical computations, and will likely run slower if your computer does not support it.

## Usage

Kestrel can be used as a Python library, but the command line interface is probably the simplest to use. It accepts VCF files with pre-computing AF values.

```
kestrel input.vcf output.txt
```

This will generate a three column file of the kinship of each pair of individuals in the VCF.


## Simulations

`simulations/sims.py` simulates replicate pairs of a given pedigree relationship with `msprime` and writes their genotypes for benchmarking Kestrel's kinship estimates against known ground truth.

Requires `msprime`, `tskit`, `numpy`, and `zarr` (for zarr output) or `pysam` (for VCF output).

```
cd simulations
python sims.py <relationship> <reps> <zarr|vcf> <genome_size> <mu> <recomb> [-o out] [--panel N] [--ne N] [--seed N]
```

For example, to simulate 50 full-sibling pairs over a 30 Mb genome:

```
python sims.py full_sib 50 zarr 30e6 1.29e-8 1.3e-8 -o full_sib.zarr
```

See `pedigrees.py` for available relationships (e.g. `full_sib`, `half_sib`, `first_cousin`, `parent_offspring`, `unrelated`, and inbred variants of each). Each run also writes a `<out>.pairs.tsv` sidecar with the realised and expected kinship for every pair, so output can be scored against Kestrel's estimates. See `pedigree-sims.ipynb` for a worked example.

### Parameters

**Parameter choice:** You can use the excellent [stdpopsim catalogue](https://popsim-consortium.github.io/stdpopsim-docs/stable/index.html) to design a simulation realistic for many common study organisms.

- **`genome_size`**: total sequence length simulated, in bp (accepts scientific notation, e.g. `3e7`). More sequence means more independent recombination events, so the *realised* IBD sharing between a pair converges more tightly around its pedigree-expected value.
- **`mu`**: per-base, per-generation mutation rate. 
- **`recomb`**: per-base, per-generation recombination rate. Together with `genome_size` this sets how many independent chunks of the genome are inherited, which is what drives variance in realised relatedness away from its pedigree expectation.
- **`--ne`**: founder (ancestral) effective population size for the Hudson coalescent phase (default `10000`). Determines background diversity/LD among the founders the pedigree is grafted onto.
- **`--panel`**: number of unrelated reference-panel diploids simulated alongside the focal pairs, used to compute population allele frequencies (default `100`).
- **`--seed`**: random seed (default `1`).

**Why simulate across genome sizes / recombination rates?** Two individuals with the same pedigree relationship (e.g. full siblings, both expected kinship = 0.25) don't inherit *exactly* the same fraction of shared genome every time -- recombination is stochastic, so realised IBD sharing varies around the pedigree expectation, and that variance shrinks as the genome (and the number of independent recombining segments) grows. This is explored in detail in [Visscher et al. 2006, *Assumption-Free Estimation of Heritability from Genome-Wide Identity-by-Descent Sharing*, PLoS Genetics](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020041). Simulating over a range of `genome_size`/`recomb` values lets you characterise how much this realised-vs-expected variance affects Kestrel's estimates, and how much genome coverage is needed before its kinship estimates reliably resolve close relationships.
