"""Catalog of pedigree structures for the kinship-benchmarking simulations.

A *pedigree* is a family tree: it says who the parents of each individual are.
Each entry in this catalog is a small, standard relationship type (full
siblings, half siblings, first cousins, parent-offspring, unrelated) together
with its metadata and a builder function that constructs it.

This is deliberately separate from sim.py (the engine) so the set of supported
relationships can grow independently. The design echoes stdpopsim's own catalog: 
look an entry up by id, read its metadata, get the thing that builds it.

A builder takes `(seq_length, n_panel=0)` and returns
`(tables, focal_ids, panel_ids)`:
    tables    : msprime TableCollection ready for sim_ancestry
    focal_ids : individual ids whose relatedness we care about (the samples)
    panel_ids : individual ids of an optional unrelated reference panel, used
                later for allele-frequency estimation (n_panel=0 -> []).

Conventions match msprime: `time` is in generations and runs *backwards*
(0 = present-day samples, larger = further into the past). Individuals with no
parents are founders.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import msprime


def _add_reference_panel(pb, n_panel) -> List[int]:
    """Add `n_panel` unrelated present-day diploids and return their ids.

    Each is a founder (no parents) sampled at time 0, so its two genomes drop
    straight into the phase-2 coalescent: an unrelated draw from the same
    population as the pedigree founders. Used only to estimate population allele
    frequencies (which must NOT come from the related focal individuals). They
    are excluded from the IBD/ground-truth computation.
    """
    return [pb.add_individual(time=0, is_sample=True) for _ in range(n_panel)]


def _add_inbred_individual(pb, time, is_sample=False) -> int:
    """Add an individual whose *own parents* are in a parent-offspring
    relationship, and return its id.

    This is how inbreeding is injected into the `inbred_*` pedigrees: we make
    ONE parent (or focal individual) inbred rather than relating the two parents
    of a tested pair. Three fresh founder-lineage ancestors are added above it:

        gp        -- founder                       (time + 2)
        gp_mate   -- founder, gp's partner         (time + 2)
        mid       -- child(gp, gp_mate), gp's kid  (time + 1)

    The returned individual = child(gp, mid): `gp` mates with its own offspring
    `mid`. Its two parents (gp, mid) therefore have a parent-offspring level of
    relatedness (kinship 1/4), giving the returned individual an inbreeding
    coefficient F = 1/4. The lineage is self-contained (all-new founders), so it
    adds inbreeding without creating relatedness to anyone else in the pedigree.
    """
    gp = pb.add_individual(time=time + 2)
    gp_mate = pb.add_individual(time=time + 2)
    mid = pb.add_individual(time=time + 1, parents=[gp, gp_mate])
    return pb.add_individual(time=time, parents=[gp, mid], is_sample=is_sample)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def full_siblings(seq_length, n_panel=0):
    """Two founders -> two full siblings."""
    pb = msprime.PedigreeBuilder()
    dad = pb.add_individual(time=1)
    mum = pb.add_individual(time=1)
    s1 = pb.add_individual(time=0, parents=[dad, mum], is_sample=True)
    s2 = pb.add_individual(time=0, parents=[dad, mum], is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [s1, s2], panel


def half_siblings(seq_length, n_panel=0):
    """Three founders -> two half siblings sharing one parent."""
    pb = msprime.PedigreeBuilder()
    shared = pb.add_individual(time=1)
    other1 = pb.add_individual(time=1)
    other2 = pb.add_individual(time=1)
    s1 = pb.add_individual(time=0, parents=[shared, other1], is_sample=True)
    s2 = pb.add_individual(time=0, parents=[shared, other2], is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [s1, s2], panel


def first_cousins(seq_length, n_panel=0):
    """Two grandparents -> two parent-siblings -> two first cousins."""
    pb = msprime.PedigreeBuilder()
    gf = pb.add_individual(time=2)
    gm = pb.add_individual(time=2)
    par1 = pb.add_individual(time=1, parents=[gf, gm])
    par2 = pb.add_individual(time=1, parents=[gf, gm])
    oth1 = pb.add_individual(time=1)  # unrelated partners
    oth2 = pb.add_individual(time=1)
    s1 = pb.add_individual(time=0, parents=[par1, oth1], is_sample=True)
    s2 = pb.add_individual(time=0, parents=[par2, oth2], is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [s1, s2], panel


def parent_offspring(seq_length, n_panel=0):
    """One parent + one unrelated partner -> one offspring; sample parent + child."""
    pb = msprime.PedigreeBuilder()
    parent = pb.add_individual(time=1, is_sample=True)
    other = pb.add_individual(time=1)  # unrelated other parent
    child = pb.add_individual(time=0, parents=[parent, other], is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [parent, child], panel


def unrelated(seq_length, n_panel=0):
    """Two independent founders with no shared ancestry (expected kinship 0)."""
    pb = msprime.PedigreeBuilder()
    s1 = pb.add_individual(time=0, is_sample=True)
    s2 = pb.add_individual(time=0, is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [s1, s2], panel


# ---------------------------------------------------------------------------
# Inbred variants
#
# Each mirrors a base pedigree but with ONE parent made inbred via a
# parent-offspring mating in its own parents (F = 1/4, see
# _add_inbred_individual). The tested focal pair is unchanged in *type*; the
# inbreeding is injected on one lineage. For most types the two tested samples
# stay outbred (their own parents are unrelated) and only carry an inbred
# parent; for parent_offspring and unrelated the inbred individual is itself
# one of the tested pair (there is no separate parent generation to place it in).
# ---------------------------------------------------------------------------
def inbred_full_siblings(seq_length, n_panel=0):
    """Full siblings whose mother is inbred (F=1/4); both share that mother."""
    pb = msprime.PedigreeBuilder()
    dad = pb.add_individual(time=1)
    mum = _add_inbred_individual(pb, time=1)  # inbred parent, F=1/4
    s1 = pb.add_individual(time=0, parents=[dad, mum], is_sample=True)
    s2 = pb.add_individual(time=0, parents=[dad, mum], is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [s1, s2], panel


def inbred_half_siblings(seq_length, n_panel=0):
    """Half siblings; the non-shared parent of one sib is inbred (F=1/4)."""
    pb = msprime.PedigreeBuilder()
    shared = pb.add_individual(time=1)
    other1 = _add_inbred_individual(pb, time=1)  # s1's non-shared parent, inbred
    other2 = pb.add_individual(time=1)
    s1 = pb.add_individual(time=0, parents=[shared, other1], is_sample=True)
    s2 = pb.add_individual(time=0, parents=[shared, other2], is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [s1, s2], panel


def inbred_first_cousins(seq_length, n_panel=0):
    """First cousins; the non-cousin-side parent of one cousin is inbred (F=1/4)."""
    pb = msprime.PedigreeBuilder()
    gf = pb.add_individual(time=2)
    gm = pb.add_individual(time=2)
    par1 = pb.add_individual(time=1, parents=[gf, gm])
    par2 = pb.add_individual(time=1, parents=[gf, gm])
    oth1 = _add_inbred_individual(pb, time=1)  # s1's other parent, inbred
    oth2 = pb.add_individual(time=1)
    s1 = pb.add_individual(time=0, parents=[par1, oth1], is_sample=True)
    s2 = pb.add_individual(time=0, parents=[par2, oth2], is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [s1, s2], panel


def inbred_parent_offspring(seq_length, n_panel=0):
    """A parent and its offspring, where the parent is inbred (F=1/4)."""
    pb = msprime.PedigreeBuilder()
    parent = _add_inbred_individual(pb, time=1, is_sample=True)  # inbred parent
    other = pb.add_individual(time=1)  # unrelated other parent
    child = pb.add_individual(time=0, parents=[parent, other], is_sample=True)
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [parent, child], panel


def inbred_unrelated(seq_length, n_panel=0):
    """Two unrelated individuals, one of which is inbred (F=1/4)."""
    pb = msprime.PedigreeBuilder()
    s1 = _add_inbred_individual(pb, time=0, is_sample=True)  # inbred
    s2 = pb.add_individual(time=0, is_sample=True)  # unrelated, outbred
    panel = _add_reference_panel(pb, n_panel)
    return pb.finalise(sequence_length=seq_length), [s1, s2], panel


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pedigree:
    """A catalog entry: a relationship type and how to build it."""

    id: str
    description: str
    expected_kinship: float
    builder: Callable[..., Tuple]


CATALOG = {
    p.id: p
    for p in [
        Pedigree("full_sib", "Two full siblings (shared mother and father).",
                 0.25, full_siblings),
        Pedigree("half_sib", "Two half siblings (one shared parent).",
                 0.125, half_siblings),
        Pedigree("first_cousin", "Two first cousins (shared grandparents).",
                 0.0625, first_cousins),
        Pedigree("parent_offspring", "A parent and its offspring.",
                 0.25, parent_offspring),
        Pedigree("unrelated", "Two unrelated individuals.",
                 0.0, unrelated),
        # Inbred variants: one parent made inbred via a parent-offspring mating
        # (F=1/4). Pair kinship shifts only where the inbred lineage lies on the
        # shared path (full sib) or is itself a tested sample (parent-offspring).
        Pedigree("inbred_1_FS",
                 "Full siblings with an inbred mother (F=1/4); sibs outbred.",
                 0.28125, inbred_full_siblings),
        Pedigree("inbred_1_HS",
                 "Half siblings; one sib's non-shared parent is inbred (F=1/4).",
                 0.125, inbred_half_siblings),
        Pedigree("inbred_1_FC",
                 "First cousins; one cousin's non-cousin-side parent is inbred (F=1/4).",
                 0.0625, inbred_first_cousins),
        Pedigree("inbred_1_PO",
                 "A parent (inbred, F=1/4) and its outbred offspring.",
                 0.3125, inbred_parent_offspring),
        Pedigree("inbred_1_UN",
                 "Two unrelated individuals, one inbred (F=1/4).",
                 0.0, inbred_unrelated),
    ]
}


def get_pedigree(pedigree_id: str) -> Pedigree:
    """Look up a catalog entry by id, with a helpful error if it's missing."""
    try:
        return CATALOG[pedigree_id]
    except KeyError:
        raise KeyError(
            f"unknown pedigree {pedigree_id!r}; "
            f"available: {', '.join(CATALOG)}"
        ) from None


def list_pedigrees() -> List[str]:
    """Return the available pedigree ids."""
    return list(CATALOG)


if __name__ == "__main__":
    for p in CATALOG.values():
        print(f"  {p.id:18} kinship~{p.expected_kinship:<7} {p.description}")
