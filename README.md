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

