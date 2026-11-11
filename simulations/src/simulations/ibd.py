import numpy as np
import tskit

def generate_ibd_lookup_table() -> np.ndarray:
    # For 4 items, there are 4 choose 2 = 6 different equality combinations
    # between items. Each combination has 2 outcomes, so there are 2^6 = 64
    # possible outcomes. However, only 15 of the possibilities define valid
    # equivalence relations.
    table = np.zeros((2, 2, 2, 2, 2, 2), dtype=np.int8)

    # 1
    # all equal
    # ii-ii
    table[1, 1, 1, 1, 1, 1] = 1

    # 2
    # i == j, k == l
    # ii-kk
    table[1, 0, 0, 0, 0, 1] = 2

    # 3
    # i == j, i == k, j == k
    # ii-il
    table[1, 1, 0, 1, 0, 0] = 3
    # i == j, i == l, j == l
    # ii-ki
    table[1, 0, 1, 0, 1, 0] = 3

    # 4
    # i == j
    # ii-kl
    table[1, 0, 0, 0, 0, 0] = 4

    # 5
    # i == k, i == l, k == l
    # ij-ii
    table[0, 1, 1, 0, 0, 1] = 5
    # j == k, j == l, k == l
    # ij-jj
    table[0, 0, 0, 1, 1, 1] = 5

    # 6
    # k == l
    # ij-kk
    table[0, 0, 0, 0, 0, 1] = 6

    # 7
    # i == k, j == l
    # ij-ij
    table[0, 1, 0, 0, 1, 0] = 7
    # i == l, j == k
    # ij-ji
    table[0, 0, 1, 1, 0, 0] = 7

    # 8
    # i == k, ij-il
    table[0, 1, 0, 0, 0, 0] = 8
    # i == l, ij-ki
    table[0, 0, 1, 0, 0, 0] = 8
    # j == k, ij-jl
    table[0, 0, 0, 1, 0, 0] = 8
    # j == l, ij-kj
    table[0, 0, 0, 0, 1, 0] = 8

    # 9
    # none equal, ij-kl
    table[0, 0, 0, 0, 0, 0] = 9

    return table


ibd_table = generate_ibd_lookup_table()


def calc_ibd_mode(tree, a1, a2, b1, b2) -> int:
    c1 = int(tree.mrca(a1, a2) != tskit.NULL)
    c2 = int(tree.mrca(a1, b1) != tskit.NULL)
    c3 = int(tree.mrca(a1, b2) != tskit.NULL)
    c4 = int(tree.mrca(a2, b1) != tskit.NULL)
    c5 = int(tree.mrca(a2, b2) != tskit.NULL)
    c6 = int(tree.mrca(b1, b2) != tskit.NULL)

    a = ibd_table[c1, c2, c3, c4, c5, c6]
    assert a != 0
    return a


def count_ibd_modes(ts_ped, a, b):
    # These are the two chromosomes in each individual
    a1, a2 = ts_ped.individual(a).nodes
    b1, b2 = ts_ped.individual(b).nodes

    delta = np.zeros(9, dtype=np.float64)

    for tree in ts_ped.trees():
        mode = calc_ibd_mode(tree, a1, a2, b1, b2)
        delta[mode - 1] += tree.span

    return delta