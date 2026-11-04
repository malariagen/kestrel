import argparse
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

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

    return jac

def parse_kestrel(infile):
    df = pd.read_csv(infile, sep='\t')

    jac = dict()
    for rel in rels:
        match = (df['sample1'] == f"{rel}0") & (df['sample2'] == f"{rel}1")
        cols = [f'jacquard{i}' for i in range(1, 10)]
        jac[rel] = df.loc[match, cols].to_numpy().flatten()

    return jac

def frm(p, q):
    return 2*np.arccos(np.sum(np.sqrt(p*q)))

def tvd(p, q):
    return np.sum(np.abs(p - q)) / 2.0

def kinship(x):
    vec = np.array([1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.25, 0.0])
    return np.dot(vec, x)

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

            njac = parse_ngsrelate(f"{depth_dir}/ngsrelate.tsv")
            kjac = parse_kestrel(f"{depth_dir}/kestrel.tsv")

            nkin = np.array([kinship(njac[rel]) for rel in rels])
            kkin = np.array([kinship(kjac[rel]) for rel in rels])

            for rel in rels:

                ngs_kin = kinship(njac[rel])
                records.append({
                    "rep": rep_dir.name,
                    "depth": depth,
                    "rel": rel,
                    "tool": "ngsrelate",
                    "jac_tvd": tvd(tjac[rel], njac[rel]),
                    "kin_abs_err":  np.abs(tkin[rel] - ngs_kin)
                })

                kestrel_kin = kinship(kjac[rel])
                records.append({
                    "rep": rep_dir.name,
                    "depth": depth,
                    "rel": rel,
                    "tool": "kestrel",
                    "jac_tvd": tvd(tjac[rel], kjac[rel]),
                    "kin_abs_err":  np.abs(tkin[rel] - kestrel_kin)
                })

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

    tvd_table = tvd_table.reindex(sorted(tvd_table.columns), axis=1)
    kin_table = kin_table.reindex(sorted(kin_table.columns), axis=1)

    print("Jacquard TVD")
    print(tvd_table.round(4))

    print("Kinship Err")
    print(kin_table.round(4))

# KL divergence
# Jensen shannon distance - sqrt(JSD), metric
# TVD - metric. Simple to understand.
# Hellinger distance - chord length between points on sphere, squared HD is lower bound of KL
# Bhattacharyaa Coefficient = dot product between two vectors
# Fisher Rao Metric - path length between points on the sphere

#nerr = np.array([np.abs(tjac[rel] - njac[rel]) for rel in rels])
#kerr = np.array([np.abs(tjac[rel] - kjac[rel]) for rel in rels])

#print(np.mean(nerr, axis=0))
#print(np.mean(kerr, axis=0))

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument("dir", type=str, help="VCF file directory")
    args = p.parse_args()

    df = process_reps(args.dir)
    print_tables(df, "ngsrelate")
    print_tables(df, "kestrel")


    sns.set_theme(style="whitegrid", font_scale=1.1)

# Okabe-Ito Color-Blind Friendly Palette
    cb_palette = {
        "kestrel": "#0072B2",    # High-contrast Blue
        "ngsrelate": "#D55E00"   # High-contrast Vermilion (Red-Orange)
    }

    rel_order = ["ifs", "ihs", "ifc", "iur", "ofs", "ohs", "ofc", "our"]

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

    g.set_axis_labels("Read Depth", "Kinship Absolute Error")
    g.set_titles(col_template="Relationship: {col_name}")
    g.fig.suptitle("Kinship Coefficient Error Comparison (Mean ± SD)", y=1.03, fontsize=16, fontweight="bold")

    plt.savefig("kinship_error_colorblind.pdf", bbox_inches="tight")

    summary_df = (
        df.groupby(["rel", "depth", "tool"])["kin_abs_err"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    rel_order = ["ifs", "ihs", "ifc", "iur", "ofs", "ohs", "ofc", "our"]

# Okabe-Ito colorblind palette + distinct markers/styles
    styles = {
        "kestrel": {"color": "#0072B2", "marker": "o", "linestyle": "-"},    # Blue, circle, solid
        "ngsrelate": {"color": "#D55E00", "marker": "s", "linestyle": "--"}  # Vermilion, square, dashed
    }

    for i, rel in enumerate(rel_order):
        ax = axes[i]
        rel_data = summary_df[summary_df["rel"] == rel]
        
        for tool in ["kestrel", "ngsrelate"]:
            tool_data = rel_data[rel_data["tool"] == tool].sort_values("depth")
            ax.plot(
                tool_data["depth"],
                tool_data["kin_abs_err"],
                label=tool,
                linewidth=2.5,
                markersize=6,
                **styles[tool]
            )
        
        ax.set_title(rel.upper(), fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.supxlabel("Read Depth", fontsize=14, y=0.02)
    fig.supylabel("Mean Kinship Absolute Error", fontsize=14, x=0.01)
    fig.suptitle("Kinship Coefficient Error: Kestrel vs NGSrelate", fontsize=16, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.98), frameon=True)

    plt.tight_layout(rect=[0.02, 0.04, 1, 0.95])
    plt.savefig("kinship_error_colorblind_mpl.pdf")
