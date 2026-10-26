import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
import msprime
import numpy as np
import pandas as pd
import stdpopsim
import tskit

import ibd
import ped
from util import ResourceTracker

# A separate copy of this will be created for each subprocess (ugh, Python is gross)
pedigree = None
pairs = None
base = None

def init_pedigree(panel_size: int, genome_length: int):
    global pedigree
    global pairs
    global base
    pairs, base, pedigree = ped.create_pedigree(panel_size, genome_length)

# https://tskit.dev/tutorials/tables_and_editing.html#editing-tree-sequences
def thin_tree(ts, min_distance):
    remove = []
    last_kept = None

    for variant in ts.variants():
        if last_kept is None or variant.site.position - last_kept >= min_distance:
            last_kept = variant.site.position
        else:
            remove.append(variant.site.id)

    return ts.delete_sites(remove)


# Based on
# https://tskit.dev/msprime/docs/stable/replication.html#running-in-parallel
def simulate(rep, seed, out_dir, ne, recomb, mu):
    global pedigree
    global pairs
    global base

    rng = np.random.default_rng(seed)

    rep_name = f"{out_dir}/{rep}"
    Path(rep_name).mkdir(exist_ok=True, parents=True)

    # Based on the following
    # https://tskit.dev/msprime/docs/stable/ancestry.html#example-1-simulating-with-a-pedigree

    # First simulate the known pedigree relationships exactly
    ts_ped = msprime.sim_ancestry(
        initial_state=pedigree,
        model=msprime.FixedPedigree(),
        recombination_rate=recomb,
        random_seed=rng.integers(low=1, high=2**31),
    )

    # Next count the number of IBD modes across the arm
    deltas = dict()
    for (rel, pair) in pairs.items():
        deltas[rel] = ibd.count_ibd_modes(ts_ped, pair[0], pair[1])

    # Then simulate to coalescence using two models:
    # DTWF for 20 generations, and then Hudson after.
    # https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1008619
    ts_chrom = msprime.sim_ancestry(
        initial_state=ts_ped,
        model=[
            msprime.DiscreteTimeWrightFisher(duration=20),
            msprime.StandardCoalescent(),
        ],
        population_size=ne,
        recombination_rate=recomb,
        random_seed=rng.integers(low=1, high=2**31),
    )

    # Finally, simulate the mutations
    ts_mut = msprime.sim_mutations(
        ts_chrom,
        model=msprime.JC69(),
        rate=mu,
        random_seed=rng.integers(low=1, high=2**31),
    )

    individuals = []
    individual_names = []
    for (rel, pair) in pairs.items():
        individuals.append(pair[0])
        individuals.append(pair[1])
        individual_names.append(f"{rel}0")
        individual_names.append(f"{rel}1")

    for i, ind in enumerate(base):
        individuals.append(ind)
        individual_names.append(f"ref{i}")

    assert len(individuals) == len(individual_names)

    with open(f"{rep_name}/sim.vcf", "w") as vcf:
        ts_mut.write_vcf(vcf, individuals=individuals, individual_names=individual_names)

    for rel in deltas:
        deltas[rel] /= np.sum(deltas[rel])

    kinship_vec = np.array([1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0])

    columns = np.array([f"delta{i}" for i in range(1, 10)])
    df = pd.DataFrame.from_dict(deltas, orient="index", columns=columns)
    df.index.name = "rel"
    df.reset_index(inplace=True)
    df.insert(1, "kinship", df[columns] @ kinship_vec)
    df.to_csv(f"{rep_name}/sim.tsv", sep="\t", index=False)


if __name__ == "__main__":

    # Only the finest bits from /dev/urandom
    random_seed = 0xcbc8bf613dc84639e312d7bca02a98cc

    p = argparse.ArgumentParser(
        description="Hello"
    )
    p.add_argument("length", type=lambda x: int(float(x)), help="genome length in base pairs")
    p.add_argument("recomb", type=float, help="per-base recombination rate")
    p.add_argument("mu", type=float, help="per-base mutation rate")
    p.add_argument("out_dir", type=str, help="output directory")
    p.add_argument("-p", "--panel", type=int, default=500, help="size of reference panel of unrelated individuals")
    p.add_argument("--ne", type=float, default=1e4, help="effective population size")
    p.add_argument("-r", "--reps", type=int, default=50, help="number of simulations to run")
    p.add_argument("-s", "--seed", type=int, default=random_seed, help="random seed")
    p.add_argument("-t", "--threads", type=int, default=-1, help="number of threads to use for simulations")

    args = p.parse_args()

    threads = args.threads if args.threads > 0 else None

    # https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning
    seed_seq = np.random.SeedSequence(args.seed)

    with ResourceTracker(), ProcessPoolExecutor(max_workers=threads, initializer=init_pedigree, initargs=(args.panel, args.length)) as executor:
        futures = [
            executor.submit(simulate, rep, seed, args.out_dir, args.ne, args.recomb, args.mu) for (rep, seed) in enumerate(seed_seq.spawn(args.reps))
        ]
        results = [
            f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Simulating")
        ]
        # The list is needed to propagate exceptions
        #results = list(tqdm(executor.map(simulate, range(args.reps), seed_seq.spawn(args.reps), repeat(args.out_dir), repeat(args.ne), repeat(args.recomb), repeat(args.mu)), total=args.reps))
