import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

irels = ["ifs", "ihs", "ifc", "iur"]
orels = ["ofs", "ohs", "ofc", "our"]
rels = irels + orels

def parse_true(infile):
    df = pd.read_csv(infile, sep='\t')

    jac = dict()
    for rel in rels:
        match = (df['rel'] == rel)
        cols = [f'jacquard{i}' for i in range(1, 10)]
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

    iters = dict()
    for (rel, pair) in indices.items():
        match = (df['a'] == pair[0]) & (df['b'] == pair[1])
        iters[rel] = float(df.loc[match, 'nIter'].item())

    return jac, iters

def parse_kestrel(infile):
    df = pd.read_csv(infile, sep='\t')

    jac = dict()
    for rel in rels:
        match = (df['sample1'] == f"{rel}0") & (df['sample2'] == f"{rel}1")
        cols = [f'jacquard{i}' for i in range(1, 10)]
        jac[rel] = df.loc[match, cols].to_numpy().flatten()

    iters = dict()
    for rel in rels:
        match = (df['sample1'] == f"{rel}0") & (df['sample2'] == f"{rel}1")
        iters[rel] = float(df.loc[match, "iterations"].item())

    return jac, iters

def process_reps(base_dir):
    records = []
    base_path = Path(base_dir)
    rep_dirs = [path for path in base_path.iterdir() if path.is_dir()]

    for rep_dir in rep_dirs:
        tjac = parse_true(f"{rep_dir}/ibd.tsv")
        tkin = {rel: kinship(tjac[rel]) for rel in rels}

        depth_dirs = [path for path in rep_dir.iterdir() if path.is_dir()]
        for depth_dir in depth_dirs:
            depth = int(depth_dir.name.rstrip("x"))

            njac, ngs_iter = parse_ngsrelate(f"{depth_dir}/ngsrelate.tsv")
            kjac, kes_iter = parse_kestrel(f"{depth_dir}/kestrel.tsv")

            nkin = np.array([kinship(njac[rel]) for rel in rels])
            kkin = np.array([kinship(kjac[rel]) for rel in rels])

            for rel in rels:
                records.append({
                    "rep": rep_dir.name,
                    "depth": depth,
                    "rel": rel,
                    "tool": "true",
                    "iterations": 0,
                    "kinship": tkin[rel],
                    "jac_tvd": 0.0,
                    "kin_abs_err": 0.0
                } | {f"jac{i+1}": tjac[rel][i] for i in range(9)}
                               )

                ngs_kin = kinship(njac[rel])
                records.append({
                    "rep": rep_dir.name,
                    "depth": depth,
                    "rel": rel,
                    "tool": "ngsrelate",
                    "iterations": ngs_iter[rel],
                    "kinship": ngs_kin,
                    "jac_tvd": tvd(tjac[rel], njac[rel]),
                    "kin_abs_err":  np.abs(tkin[rel] - ngs_kin)
                } | {f"jac{i+1}": njac[rel][i] for i in range(9)}
                               )

                kestrel_kin = kinship(kjac[rel])
                records.append({
                    "rep": rep_dir.name,
                    "depth": depth,
                    "rel": rel,
                    "tool": "kestrel",
                    "iterations": kes_iter[rel],
                    "kinship": kestrel_kin,
                    "jac_tvd": tvd(tjac[rel], kjac[rel]),
                    "kin_abs_err":  np.abs(tkin[rel] - kestrel_kin)
                } | {f"jac{i+1}": kjac[rel][i] for i in range(9)}
                               )

    return pd.DataFrame(records)

def print_tables(df, tool):
    print(tool)

    df_tool = df[df["tool"] == tool]

    tvd_table = df_tool.pivot_table(
        index="rel",
        columns="depth",
        values="jac_tvd",
        aggfunc="mean"
    ).reindex(rels)

    kin_table = df_tool.pivot_table(
        index="rel",
        columns="depth",
        values="kin_abs_err",
        aggfunc="mean"
    ).reindex(rels)

    iter_table = df_tool.pivot_table(
        index="rel",
        columns="depth",
        values="iterations",
        aggfunc="mean"
    ).reindex(rels)

    tvd_table = tvd_table.reindex(sorted(tvd_table.columns), axis=1)
    kin_table = kin_table.reindex(sorted(kin_table.columns), axis=1)
    iter_table = iter_table.reindex(sorted(iter_table.columns), axis=1)

    print("Jacquard TVD")
    print(tvd_table.round(4))

    print("Kinship Err")
    print(kin_table.round(4))

    print("Average iterations")
    print(iter_table.round(4))

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument("dir", type=str, help="VCF file directory")
    args = p.parse_args()

    df = process_reps(args.dir)
    print_tables(df, "ngsrelate")
    print_tables(df, "kestrel")

    df = df.sort_values(by="depth")
    df["depth_str"] = df["depth"].astype(str) + "x"

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Color palette including true, kestrel, and ngsrelate
    cb_palette = {
        "true": "#E69F00",
        "kestrel": "#56B4E9",
        "ngsrelate": "#009E73"
    }

    rel_order = ["ifs", "ihs", "ifc", "iur", "ofs", "ohs", "ofc", "our"]

    g = sns.catplot(
        data=df,
        x="depth_str",
        y="kinship",
        hue="tool",
        hue_order=["true", "kestrel", "ngsrelate"],
        palette=cb_palette,
        col="rel",
        col_order=rel_order,
        col_wrap=4,
        kind="strip",
        #inner="quartile",
        #cut=0,
        #height=3.5,
        #aspect=1.2
)

    g.set_axis_labels("Coverage", "Kinship")
    g.set_titles(col_template="Relationship: {col_name}")
    g.fig.suptitle("Kinship Coefficient Distributions Across Coverage Depths", y=1.03, fontsize=16, fontweight="bold")

    plt.savefig(f"kinship_distribution_{args.dir}.png", bbox_inches="tight")

    sns.set_theme(style="whitegrid", font_scale=1.1)

    g = sns.relplot(
        data=df,
        x="depth",
        y="kin_abs_err",
        hue="tool",
        style="tool",            # Adds distinct line styles and markers per tool
        markers={"kestrel": "o", "ngsrelate": "s"},  # Circle vs Square
        dashes={"kestrel": "", "ngsrelate": (2, 2)},  # Solid vs Dashed line
        hue_order=["kestrel", "ngsrelate"],
        palette=cb_palette,
        col="rel",
        col_order=rel_order,
        col_wrap=4,
        kind="line",
        linewidth=2.5,
        markersize=7,
        errorbar="sd",
        height=3.5,
        aspect=1.2
    )

    g.set_axis_labels("Coverage Depth", "Kinship Absolute Error")
    g.set_titles(col_template="Relationship: {col_name}")
    g.fig.suptitle("Kinship Coefficient Error Comparison (Mean ± SD)", y=1.03, fontsize=16, fontweight="bold")

    plt.savefig(f"kinship_error_{args.dir}.png", bbox_inches="tight")

