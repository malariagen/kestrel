#!/usr/bin/env python
"""sims.py -- simulate replicate pairs of one relationship and write genotypes.

One command, one output file. Given a relationship from the pedigree catalog
(`pedigrees.py`) and a number of replicates, this:

  1. stacks `reps` independent copies of that family PLUS one shared, unrelated
     reference panel into a SINGLE coalescent, so every pair lands on the same
     set of variant sites (this is why the reps can share one file);
  2. runs the two-phase simulation -- pedigree inheritance (the ground-truth
     IBD) then a constant-size Hudson coalescent over the founders (Ne, default
     1e4) plus mutations;
  3. writes the focal individuals' genotypes together with the population allele
     frequencies (estimated from the reference panel, NOT the related focal
     samples) as either a `.zarr` store or a bgzipped, tabix-indexed `.vcf.gz`
     (with an AF INFO field).

`reps=50` therefore means 50 pairs of that relationship (100 individuals) in one
file. A `<out>.pairs.tsv` sidecar records each pair's realised (ground-truth)
kinship so the output can be scored.

Examples
--------
    python sims.py full_sib 50 zarr 30e6 1.29e-8 1.3e-8 -o full_sib_100k
    python sims.py inbred_1_FS 50 vcf 5e7 3.5e-9 1e-8 --panel 200

Positional args (in this order): relationship reps outfmt genome_size mu recomb
"""

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import msprime
import numpy as np
import tskit

import pedigrees

DEFAULT_NE = 10_000


# ---------------------------------------------------------------------------
# 1. Stack `reps` independent families + one shared panel into one pedigree
# ---------------------------------------------------------------------------
def build_stacked(relationship, reps, seq_length, n_panel):
    """Replicate a catalog family `reps` times into one TableCollection.

    Works for any relationship in the catalog (including the inbred variants):
    we read the family's structure -- each individual's time, sample flag and
    parents -- from a single template build, then re-add that structure `reps`
    times into one PedigreeBuilder with fresh ids each time. Finally one shared
    reference panel of `n_panel` unrelated present-day diploids is added.

    Returns (tables, focal_pairs, panel_ids) where focal_pairs is a list of
    `reps` (ind_a, ind_b) id tuples -- the pair from each replicate family.
    """
    template, focal0, _ = pedigrees.get_pedigree(relationship).builder(
        seq_length, n_panel=0)

    n = template.individuals.num_rows
    nodes = template.nodes
    ind_time = {}
    ind_is_sample = {}
    for nid in range(nodes.num_rows):
        iid = nodes.individual[nid]
        if iid == tskit.NULL:
            continue
        ind_time[iid] = float(nodes.time[nid])
        if nodes.flags[nid] & tskit.NODE_IS_SAMPLE:
            ind_is_sample[iid] = True
    # Builders always add parents before children, so id order is a valid
    # topological order: parents[i] only references ids < i, added already.
    parents = [[p for p in template.individuals[i].parents if p != tskit.NULL]
               for i in range(n)]

    pb = msprime.PedigreeBuilder()
    focal_pairs = []
    for _ in range(reps):
        remap = {}
        for i in range(n):
            ps = [remap[p] for p in parents[i]]
            remap[i] = pb.add_individual(
                time=ind_time[i], parents=ps or None,
                is_sample=ind_is_sample.get(i, False))
        focal_pairs.append(tuple(remap[f] for f in focal0))

    panel_ids = [pb.add_individual(time=0, is_sample=True)
                 for _ in range(n_panel)]
    return pb.finalise(sequence_length=seq_length), focal_pairs, panel_ids


# ---------------------------------------------------------------------------
# 2. Two-phase simulation
# ---------------------------------------------------------------------------
def simulate(tables, seq_length, Ne, mu, recomb, seed):
    """Phase 1 (fixed pedigree) then phase 2 (Hudson over founders) + mutations."""
    ts_ped = msprime.sim_ancestry(
        initial_state=tables, model="fixed_pedigree",
        recombination_rate=recomb, random_seed=seed)
    ts_full = msprime.sim_ancestry(
        initial_state=ts_ped, model="hudson",
        demography=msprime.Demography.isolated_model([Ne]),
        recombination_rate=recomb, random_seed=seed + 1)
    ts_mut = msprime.sim_mutations(ts_full, rate=mu, random_seed=seed + 2)
    return ts_ped, ts_mut


def pair_kinships(ts_ped, focal_pairs):
    """Realised ground-truth kinship for each focal pair, from pedigree IBD."""
    L = ts_ped.sequence_length
    founder_time = max(ind.time for ind in ts_ped.individuals())
    focal_ids = [i for pair in focal_pairs for i in pair]
    focal_nodes = [n for i in focal_ids for n in ts_ped.individual(i).nodes]
    totals = {tuple(sorted(p)): 0.0 for p in focal_pairs}
    ibd = ts_ped.ibd_segments(within=focal_nodes, max_time=founder_time,
                              store_pairs=True)
    for (na, nb), segs in ibd.items():
        ia = ts_ped.node(na).individual
        ib = ts_ped.node(nb).individual
        key = tuple(sorted((ia, ib)))
        if ia != ib and key in totals:
            totals[key] += segs.total_span
    return {pair: totals[tuple(sorted(pair))] / (4 * L) for pair in focal_pairs}


# ---------------------------------------------------------------------------
# 3. Extract genotypes + population allele frequencies
# ---------------------------------------------------------------------------
def _individual_columns(ts):
    node_to_col = {n: c for c, n in enumerate(ts.samples())}
    cols = {}
    for n in ts.samples():
        cols.setdefault(ts.node(n).individual, []).append(node_to_col[n])
    return cols


def extract(ts_mut, focal_pairs, panel_ids, restrict_to_panel_sites=True):
    """Return focal genotypes, population allele freqs, positions and alleles.

    Allele frequencies come from the unrelated reference panel (the population
    the samples are drawn from), not the related focal individuals.

    geno    int8    (n_focal, n_sites, 2)   individual-major (coeffs' (I, L, P))
    af      float64 (n_sites, A)            per-site allele freq over the panel
    pos     int64   (n_sites,)              bp positions
    alleles object  (n_sites,)              allele tuple per site (ref first)
    """
    if not panel_ids:
        raise ValueError("A reference panel (n_panel > 0) is required for "
                         "population allele frequencies.")
    focal_ids = [i for pair in focal_pairs for i in pair]
    G = ts_mut.genotype_matrix()                    # (n_sites, n_sample_nodes)
    cols = _individual_columns(ts_mut)
    A = int(G.max()) + 1

    panel_cols = [c for i in panel_ids for c in cols[i]]
    panelG = G[:, panel_cols]
    if restrict_to_panel_sites:
        keep = panelG.max(axis=1) != panelG.min(axis=1)
    else:
        keep = np.ones(G.shape[0], dtype=bool)
    idx = np.flatnonzero(keep)

    panelG = panelG[keep]
    counts = np.stack([(panelG == a).sum(axis=1) for a in range(A)],
                      axis=1).astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    af = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    geno = np.stack([G[np.ix_(idx, cols[i])] for i in focal_ids],
                    axis=0).astype(np.int8)
    all_alleles = [v.alleles for v in ts_mut.variants()]
    pos = ts_mut.tables.sites.position.astype(np.int64)[keep]
    alleles = np.array([all_alleles[k] for k in idx], dtype=object)
    return geno, af, pos, alleles


# ---------------------------------------------------------------------------
# 4. Writers
# ---------------------------------------------------------------------------
def sample_names(relationship, focal_pairs):
    """Two names per pair: <rel>_r<rep>_a / _b."""
    names, pair_idx = [], []
    for r in range(len(focal_pairs)):
        names += [f"{relationship}_r{r}_a", f"{relationship}_r{r}_b"]
        pair_idx += [r, r]
    return names, np.array(pair_idx, dtype=np.int32)


def write_zarr(path, relationship, geno, af, pos, alleles, names, pair_idx,
               attrs):
    """vcf-zarr-style store: call_genotype (variants, samples, 2) + variant_*."""
    import zarr

    L, A = af.shape
    maxlen = max((max((len(a) for a in al), default=1) for al in alleles),
                 default=1)
    allele_arr = np.zeros((L, A), dtype=f"S{maxlen}")
    for i, al in enumerate(alleles):
        for j, a in enumerate(al):
            allele_arr[i, j] = a.encode()

    g = zarr.open_group(str(path), mode="w")
    g.attrs.update(attrs)
    # genotypes as (variants, samples, ploidy) -- the vcf-zarr / sgkit layout
    g.create_dataset("call_genotype", data=geno.transpose(1, 0, 2),
                     chunks=(min(L, 10000), geno.shape[0], 2))
    g.create_dataset("variant_position", data=pos)
    g.create_dataset("variant_allele", data=allele_arr)
    g.create_dataset("variant_allele_frequency", data=af)
    g.create_dataset("sample_id", data=np.array(names, dtype="S"))
    g.create_dataset("sample_pair", data=pair_idx)
    return path


def write_vcf(path, geno, af, pos, alleles, names, contig, seq_length):
    """bgzipped, tabix-indexed VCF v4.2 with an AF INFO field (per-ALT freq).

    `path` is the final `.vcf.gz`. We write the plain text to a sibling `.vcf`,
    bgzip it, index it (`.vcf.gz.tbi`), and drop the plain intermediate. Sites
    come out sorted by position (msprime orders them) on a single contig, which
    is what tabix needs.
    """
    import pysam

    path = Path(path)
    plain = path.with_suffix("")  # foo.vcf.gz -> foo.vcf
    n_focal, L, _ = geno.shape
    with open(plain, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##contig=<ID={contig},length={int(seq_length)}>\n")
        fh.write('##INFO=<ID=AF,Number=A,Type=Float,Description='
                 '"Population allele frequency (reference panel), per ALT">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join(names) + "\n")
        for l in range(L):
            al = alleles[l]
            ref, alt = al[0], al[1:]
            n_alt = len(alt)
            af_alt = ",".join(f"{af[l, a]:.6g}" for a in range(1, n_alt + 1))
            gts = "\t".join(
                f"{int(geno[i, l, 0])}|{int(geno[i, l, 1])}"
                if geno[i, l, 0] >= 0 and geno[i, l, 1] >= 0 else ".|."
                for i in range(n_focal))
            fh.write(f"{contig}\t{int(pos[l])}\t.\t{ref}\t{','.join(alt)}\t"
                     f".\t.\tAF={af_alt}\tGT\t{gts}\n")

    pysam.tabix_compress(str(plain), str(path), force=True)
    plain.unlink()
    pysam.tabix_index(str(path), preset="vcf", force=True)
    return path


def write_pairs_tsv(path, relationship, focal_pairs, names, kin):
    """Sidecar: per-pair realised kinship + expected kinship for the relationship."""
    expected = pedigrees.get_pedigree(relationship).expected_kinship
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["rep", "relationship", "sample_a", "sample_b",
                    "realised_kinship", "expected_kinship"])
        for r, pair in enumerate(focal_pairs):
            w.writerow([r, relationship, names[2 * r], names[2 * r + 1],
                        f"{kin[pair]:.6f}", expected])
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(relationship, reps, outfmt, genome_size, mu, recomb,
        panel=100, Ne=DEFAULT_NE, seed=1, out=None,
        restrict_to_panel_sites=True):
    default_ext = "vcf.gz" if outfmt == "vcf" else "zarr"
    outpath = Path(out) if out else Path(f"sims/{relationship}_n{reps}.{default_ext}")
    if outfmt == "vcf":
        # always emit a bgzipped .vcf.gz regardless of how -o was spelled
        name = outpath.name
        if name.endswith(".vcf.gz") or name.endswith(".gz"):
            pass
        elif name.endswith(".vcf"):
            outpath = outpath.with_name(name + ".gz")
        else:
            outpath = outpath.with_name(name + ".vcf.gz")

    tables, focal_pairs, panel_ids = build_stacked(
        relationship, reps, genome_size, panel)
    ts_ped, ts_mut = simulate(tables, genome_size, Ne, mu, recomb, seed)
    geno, af, pos, alleles = extract(
        ts_mut, focal_pairs, panel_ids,
        restrict_to_panel_sites=restrict_to_panel_sites)
    kin = pair_kinships(ts_ped, focal_pairs)
    names, pair_idx = sample_names(relationship, focal_pairs)

    attrs = {"relationship": relationship, "reps": reps,
             "genome_size": float(genome_size), "mutation_rate": mu,
             "recombination_rate": recomb, "Ne": Ne, "seed": seed,
             "n_sites": int(len(pos)), "n_focal": int(geno.shape[0])}

    if outfmt == "zarr":
        write_zarr(outpath, relationship, geno, af, pos, alleles, names,
                   pair_idx, attrs)
    elif outfmt == "vcf":
        write_vcf(outpath, geno, af, pos, alleles, names, contig="1",
                  seq_length=genome_size)
    else:
        raise ValueError(f"unknown outfmt {outfmt!r}")

    # sidecar sits next to the output; strip any of .vcf.gz/.vcf/.zarr/.gz
    base = outpath.name
    for ext in (".vcf.gz", ".vcf", ".zarr", ".gz"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    tsv = write_pairs_tsv(outpath.with_name(base + ".pairs.tsv"), relationship,
                          focal_pairs, names, kin)
    extra = "  (+ .tbi)" if outfmt == "vcf" else ""
    print(f"wrote {outpath}{extra}  ({geno.shape[0]} samples, {len(pos):,} sites)")
    print(f"wrote {tsv}")
    return outpath


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("relationship", choices=pedigrees.list_pedigrees(),
                   help="pedigree id from the catalog")
    p.add_argument("reps", type=int,
                   help="number of replicate pairs (e.g. 50 -> 50 pairs)")
    p.add_argument("outfmt", choices=["zarr", "vcf"], help="output format")
    p.add_argument("genome_size", type=lambda x: int(float(x)),
                   help="sequence length in bp (accepts 3e7)")
    p.add_argument("mu", type=float, help="per-base mutation rate")
    p.add_argument("recomb", type=float, help="per-base recombination rate")
    p.add_argument("--panel", type=int, default=100,
                   help="reference-panel diploids for allele freqs (default 100)")
    p.add_argument("--ne", type=float, default=DEFAULT_NE,
                   help="founder population size (default 10000)")
    p.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    p.add_argument("-o", "--out", default=None,
                   help="output file to match outfmt (vcf -> bgzipped .vcf.gz "
                        "+ .tbi; zarr -> .zarr store). Default "
                        "<relationship>_n<reps>.<ext>. The .pairs.tsv sidecar "
                        "is written alongside it.")
    p.add_argument("--no-restrict", action="store_true",
                   help="keep all mutated sites, not only those segregating in "
                        "the panel")
    a = p.parse_args(argv)
    run(a.relationship, a.reps, a.outfmt, a.genome_size, a.mu, a.recomb,
        panel=a.panel, Ne=a.ne, seed=a.seed, out=a.out,
        restrict_to_panel_sites=not a.no_restrict)


if __name__ == "__main__":
    main()
