import msprime
import pandas as pd
import stdpopsim
import tskit
import numpy as np
import io
from pathlib import Path

import ibd

species = stdpopsim.get_species("AnoGam")
# model = species.get_demographic_model("GabonAg1000G_1A17")

# 6, 4
def add_inbred_fullsibs(pd):
    # Two grandparents have two full-sib children
    # Those two mate to have two other full-sib children
    gf = pd.add_individual(time=2)
    gm = pd.add_individual(time=2)
    dad = pd.add_individual(time=1, parents=[gf, gm])
    mom = pd.add_individual(time=1, parents=[gf, gm])
    child1 = pd.add_individual(time=0, parents=[dad, mom], is_sample=True)
    child2 = pd.add_individual(time=0, parents=[dad, mom], is_sample=True)
    return (child1, child2)

# 7, 5
def add_inbred_halfsibs(pd):
    # Two grandparents have three full-sib children
    # SibC mates with SibA to have childA
    # SibC mates with SibB to have childB
    gf = pd.add_individual(time=2)
    gm = pd.add_individual(time=2)
    siba = pd.add_individual(time=1, parents=[gf, gm])
    sibb = pd.add_individual(time=1, parents=[gf, gm])
    sibc = pd.add_individual(time=1, parents=[gf, gm])
    childa = pd.add_individual(time=0, parents=[siba, sibc], is_sample=True)
    childb = pd.add_individual(time=0, parents=[sibb, sibc], is_sample=True)
    return (childa, childb)

# 8, 6
def add_inbred_firstcousins(pd):
    # Two grandparents have four full-sib children
    # SibA mates with SibC to have childA
    # SibB mates with SibD to have childB
    gf = pd.add_individual(time=2)
    gm = pd.add_individual(time=2)
    siba = pd.add_individual(time=1, parents=[gf, gm])
    sibb = pd.add_individual(time=1, parents=[gf, gm])
    sibc = pd.add_individual(time=1, parents=[gf, gm])
    sibd = pd.add_individual(time=1, parents=[gf, gm])
    childa = pd.add_individual(time=0, parents=[siba, sibc], is_sample=True)
    childb = pd.add_individual(time=0, parents=[sibb, sibd], is_sample=True)
    return (childa, childb)

# 10, 6
def add_inbred_unrelated(pd):
    gf1 = pd.add_individual(time=2)
    gm1 = pd.add_individual(time=2)
    dad1 = pd.add_individual(time=1, parents=[gf1, gm1])
    mom1 = pd.add_individual(time=1, parents=[gf1, gm1])
    child1 = pd.add_individual(time=0, parents=[dad1, mom1], is_sample=True)

    gf2 = pd.add_individual(time=2)
    gm2 = pd.add_individual(time=2)
    dad2 = pd.add_individual(time=1, parents=[gf2, gm2])
    mom2 = pd.add_individual(time=1, parents=[gf2, gm2])
    child2 = pd.add_individual(time=0, parents=[dad2, mom2], is_sample=True)

    return (child1, child2)

# 4, 2
def add_outbred_fullsibs(pd):
    # Two unrelated parents two mate to have two full-sib children
    dad = pd.add_individual(time=1)
    mom = pd.add_individual(time=1)
    child1 = pd.add_individual(time=0, parents=[dad, mom], is_sample=True)
    child2 = pd.add_individual(time=0, parents=[dad, mom], is_sample=True)
    return (child1, child2)

# 5, 2
def add_outbred_halfsibs(pd):
    # Three unrelated individuals, one mates with the two others
    # SibC mates with SibA to have childA
    # SibC mates with SibB to have childB
    parent1 = pd.add_individual(time=1)
    parent2 = pd.add_individual(time=1)
    parent3 = pd.add_individual(time=1)
    childa = pd.add_individual(time=0, parents=[parent1, parent3], is_sample=True)
    childb = pd.add_individual(time=0, parents=[parent2, parent3], is_sample=True)
    return (childa, childb)

# 10, 6
def add_outbred_firstcousins(pd):
    # Two grandparents have two full-sib children
    # SibA mates with SibC to have childA
    # SibB mates with SibD to have childB
    gf1 = pd.add_individual(time=2)
    gm1 = pd.add_individual(time=2)
    dad1 = pd.add_individual(time=1, parents=[gf1, gm1])
    mom1 = pd.add_individual(time=1, parents=[gf1, gm1])

    gf2 = pd.add_individual(time=2)
    gm2 = pd.add_individual(time=2)
    dad2 = pd.add_individual(time=1, parents=[gf2, gm2])
    mom2 = pd.add_individual(time=1, parents=[gf2, gm2])

    child1 = pd.add_individual(time=0, parents=[dad1, mom2], is_sample=True)
    child2 = pd.add_individual(time=0, parents=[dad2, mom1], is_sample=True)
    return (child1, child2)

# 2, 0
def add_outbred_unrelated(pd):
    a = pd.add_individual(time=0, is_sample=True)
    b = pd.add_individual(time=0, is_sample=True)
    return (a, b)

def create_pedigree(unrelated):
    pd = msprime.PedigreeBuilder()

    ifs = add_inbred_fullsibs(pd)
    ihs = add_inbred_halfsibs(pd)
    ifc = add_inbred_firstcousins(pd)
    iur = add_inbred_unrelated(pd)

    ofs = add_outbred_fullsibs(pd)
    ohs = add_outbred_halfsibs(pd)
    ofc = add_outbred_firstcousins(pd)
    our = add_outbred_unrelated(pd)

    pairs = {
        "ifs": ifs,
        "ihs": ihs,
        "ifc": ifc,
        "iur": iur,
        "ofs": ofs,
        "ohs": ohs,
        "ofc": ofc,
        "our": our
    }

    # Unrelated base population
    base = [pd.add_individual(time=0, is_sample=True) for _ in range(unrelated)]

    return pairs, base, pd.finalise()

def export_multi_vcf(output_path, ts_map, individuals, individual_names):
    with open(output_path, "w") as vcf:
        first_contig = True

        for contig, ts in ts_map.items():
            buf = io.StringIO()
            ts.write_vcf(vcf, contig_id=contig, individuals=individuals, individual_names=individual_names)
            buf.seek(0)

            for line in buf:
                if not first_contig and not line.startswith("#"):
                    vcf.write(line)
                else:
                    vcf.write(line)

            first_contig = False

# Would be great, but too slow
# Ne = int(species.population_size)
Ne = 10000

pairs, base, pedigree = create_pedigree(500)

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

ne_name = f"Ne_{Ne}"
Path(ne_name).mkdir(exist_ok=True)

total_deltas = dict()
ts_map = dict()
for i, arm in enumerate(("2L", "2R", "3L", "3R")):
# for i, arm in enumerate(("2L",)):
    # TODO num_replicates, also run in parallel, better RNG

    print("Simulating", arm)
    contig = species.get_contig(arm)

    # Based on the following
    # https://tskit.dev/msprime/docs/stable/ancestry.html#example-1-simulating-with-a-pedigree
    pedigree.sequence_length = int(contig.recombination_map.position[-1])

    # First simulate the known pedigree relationships exactly
    ts_ped = msprime.sim_ancestry(
        initial_state=pedigree,
        model=msprime.FixedPedigree(),
        recombination_rate=contig.recombination_map,
        random_seed=3*i+1,
    )

    # Next count the number of IBD modes across the arm
    for (rel, pair) in pairs.items():
        deltas = ibd.count_ibd_modes(ts_ped, pair[0], pair[1])
        if rel not in total_deltas:
            total_deltas[rel] = deltas
        else:
            total_deltas[rel] += deltas

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
        # population_size=,
        population_size=Ne,
        # demography=msprime.Demography.isolated_model([10000]),
        # demography=model.model,
        recombination_rate=contig.recombination_map,
        random_seed=3*i+2,
    )

    # Finally, simulate the mutations
    ts_mut = msprime.sim_mutations(
        ts_chrom,
        model=msprime.JC69(),
        rate=contig.mutation_rate,
        random_seed=3*i+3,
    )

    with open(f"{ne_name}/AnoGam-{arm}.vcf", "w") as vcf:
        ts_mut.write_vcf(vcf, contig_id=arm, individuals=individuals, individual_names=individual_names)

for (rel, deltas) in total_deltas.items():
    total_deltas[rel] = deltas / np.sum(deltas)

columns = np.array([f"ibd{i}" for i in range(1, 10)])
df = pd.DataFrame.from_dict(total_deltas, orient="index", columns=columns)
df.index.name = "rel"
df.reset_index(inplace=True)

df.to_csv(f"{ne_name}/AnoGam.tsv", sep="\t", index=False)

    # bgzip, then bcftools concat

    # vcfgl for each arm
    # then thin?
    # then process in ngsrelate and kestrel