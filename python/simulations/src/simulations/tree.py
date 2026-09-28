import msprime
import pandas as pd
import stdpopsim
import tskit
import numpy as np
from pathlib import Path

import ibd
import ped

from concurrent.futures import ProcessPoolExecutor

# A separate copy of this will be created for each subprocess (ugh, Python is gross)
pedigree = None
pairs = None
base = None

def init_pedigree():
    global pedigree
    global pairs
    global base
    pairs, base, pedigree = ped.create_pedigree(500)

# Based on
# https://tskit.dev/msprime/docs/stable/replication.html#running-in-parallel
def simulate(rep, seed):
    global pedigree
    global pairs
    global base

    rng = np.random.default_rng(seed)

    species = stdpopsim.get_species("AnoGam")

    # Would be great, but too slow
    # Ne = int(species.population_size)
    ne = 10000

    rep_name = f"Ne_{ne}m/{rep}"
    Path(rep_name).mkdir(exist_ok=True, parents=True)

    print("Simulating rep", rep)

    kinship_vec = np.array([1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0])

    deltas = dict()
    for i, arm in enumerate(("2L", "2R", "3L", "3R")):

        contig = species.get_contig(arm)

        # Based on the following
        # https://tskit.dev/msprime/docs/stable/ancestry.html#example-1-simulating-with-a-pedigree
        pedigree.sequence_length = int(contig.recombination_map.position[-1])

        # First simulate the known pedigree relationships exactly
        ts_ped = msprime.sim_ancestry(
            initial_state=pedigree,
            model=msprime.FixedPedigree(),
            recombination_rate=contig.recombination_map,
            random_seed=rng.integers(low=1, high=2**31),
        )

        # Next count the number of IBD modes across the arm
        for (rel, pair) in pairs.items():
            if rel not in deltas:
                deltas[rel] = ibd.count_ibd_modes(ts_ped, pair[0], pair[1])
            else:
                deltas[rel] += ibd.count_ibd_modes(ts_ped, pair[0], pair[1])

        # Then simulate to coalescence using two models:
        # DTWF for 20 generations, and then Hudson after.
        # https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1008619
        ts_chrom = msprime.sim_ancestry(
            initial_state=ts_ped,
            model=[
                msprime.DiscreteTimeWrightFisher(duration=20),
                msprime.StandardCoalescent(),
            ],
            # model="dtwf",
            # model="hudson",
            population_size=ne,
            # demography=msprime.Demography.isolated_model([10000]),
            # demography=model.model,
            recombination_rate=contig.recombination_map,
            random_seed=rng.integers(low=1, high=2**31),
        )

        # Finally, simulate the mutations
        ts_mut = msprime.sim_mutations(
            ts_chrom,
            model=msprime.JC69(),
            rate=contig.mutation_rate * 100,
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
            individual_names.append(f"base{i}")

        assert len(individuals) == len(individual_names)

        with open(f"{rep_name}/AnoGam-{arm}.vcf", "w") as vcf:
            ts_mut.write_vcf(vcf, contig_id=arm, individuals=individuals, individual_names=individual_names)

    for rel in deltas:
        deltas[rel] /= np.sum(deltas[rel])

    columns = np.array([f"ibd{i}" for i in range(1, 10)])
    df = pd.DataFrame.from_dict(deltas, orient="index", columns=columns)
    df.index.name = "rel"
    df.reset_index(inplace=True)
    df.to_csv(f"{rep_name}/AnoGam.tsv", sep="\t", index=False)


if __name__ == "__main__":

    reps = 5

    # https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning
    random_seed = 0xcbc8bf613dc84639e312d7bca02a98cc
    seed_seq = np.random.SeedSequence(random_seed)

    with ProcessPoolExecutor(max_workers=10, initializer=init_pedigree) as executor:
        # The list is needed to propagate exceptions
        results = list(executor.map(simulate, range(reps), seed_seq.spawn(reps)))
