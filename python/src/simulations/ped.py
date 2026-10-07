import msprime

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
