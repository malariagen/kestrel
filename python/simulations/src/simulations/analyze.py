import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

ne = 10000

dir_name = f"Ne_{ne}m2/0"

irels = ["ifs", "ihs", "ifc", "iur"]
orels = ["ofs", "ohs", "ofc", "our"]
rels = irels + orels

def parse_true(infile):
    df = pd.read_csv(infile, sep='\t')

    jac = dict()
    for rel in rels:
        match = (df['rel'] == rel)
        cols = [f'ibd{i}' for i in range(1, 10)]
        jac[rel] = df.loc[match, cols].to_numpy().flatten()

    return jac

def parse_ngsrelate(infile):
    df = pd.read_csv(infile, sep='\t')

    indices = { "ifs": (0, 1), "ihs": (2, 3), "ifc": (4, 5), "iur": (6, 7), "ofs": (8, 9), "ohs": (10, 11), "ofc": (12, 13), "our": (14, 15) }

    jac = dict()
    for (rel, pair) in indices.items():
        match = (df['a'] == pair[0]) & (df['b'] == pair[1])
        cols = [f'J{i}' for i in range(1, 10)]
        jac[rel] = df.loc[match, cols].to_numpy().flatten()

    return jac

def parse_kestrel(infile):
    df = pd.read_csv(infile, sep='\t')

    jac = dict()
    for rel in rels:
        match = (df['sample1'] == f"{rel}0") & (df['sample2'] == f"{rel}1")
        cols = [f'delta{i}' for i in range(1, 10)]
        jac[rel] = df.loc[match, cols].to_numpy().flatten()

    return jac

def frm(p, q):
    return 2*np.arccos(np.sum(np.sqrt(p*q)))

def tvd(p, q):
    return np.sum(np.abs(p - q)) / 2.0

def kinship(x):
    vec = np.array([1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0])
    return np.dot(vec, x)
    
tjac = parse_true(f"{dir_name}/AnoGam.tsv")
njac = parse_ngsrelate(f"{dir_name}/ngsrelate.tsv")
kjac = parse_kestrel(f"{dir_name}/kestrel.tsv")

tkin = np.array([kinship(tjac[rel]) for rel in rels])
nkin = np.array([kinship(njac[rel]) for rel in rels])
kkin = np.array([kinship(kjac[rel]) for rel in rels])

nkl = [entropy(tjac[rel], njac[rel]) for rel in rels]
kkl = [entropy(tjac[rel], kjac[rel]) for rel in rels]

njs = [jensenshannon(tjac[rel], njac[rel]) for rel in rels]
kjs = [jensenshannon(tjac[rel], kjac[rel]) for rel in rels]

nfrm = [frm(tjac[rel], njac[rel]) for rel in rels]
kfrm = [frm(tjac[rel], kjac[rel]) for rel in rels]

ntvd = [tvd(tjac[rel], njac[rel]) for rel in rels]
ktvd = [tvd(tjac[rel], kjac[rel]) for rel in rels]

# KL divergence
# Jensen shannon distance - sqrt(JSD), metric
# TVD - metric. Simple to understand.
# Hellinger distance - chord length between points on sphere, squared HD is lower bound of KL
# Bhattacharyaa Coefficient = dot product between two vectors
# Fisher Rao Metric - path length between points on the sphere

print("KL")
print(np.mean(nkl))
print(np.mean(kkl))

print("JSD")
print(np.mean(njs))
print(np.mean(kjs))

print("FRM")
print(np.mean(nfrm))
print(np.mean(kfrm))

print("TVD")
print(np.mean(ntvd))
print(np.mean(ktvd))

print(np.mean(np.abs(tkin - nkin)))
print(np.mean(np.abs(tkin - kkin)))

nerr = np.array([np.abs(tjac[rel] - njac[rel]) for rel in rels])
kerr = np.array([np.abs(tjac[rel] - kjac[rel]) for rel in rels])

print(np.mean(nerr, axis=0))
print(np.mean(kerr, axis=0))
