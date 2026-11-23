import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

def parse_kestrel_runtime(path):

    seen = False
    times = []

    with open(path) as infile:
        for line in infile:
            if "Wall Time" in line:
                if not seen:
                    seen = True
                else:
                    words = line.strip().split()
                    word = words[4]
                    assert word.endswith("s,")
                    time = float(word[:-2])
                    times.append(time)

    assert len(times) == 1

    return times[0]

def parse_ngsrelate_runtime(path):

    with open(path) as infile:
        for line in infile:
            if "walltime" in line:
                words = line.strip().split()
                t1 = float(words[5])
                t2 = float(words[9])
                return t1 - t2

def process_reps(base_dir):
    records = []
    base_path = Path(base_dir)
    rep_dirs = [path for path in base_path.iterdir() if path.is_dir()]

    gt_dfs = []
    gl_dfs = []
    records = []

    for rep_dir in rep_dirs:
        kes_gt = pd.read_csv(f"{rep_dir}/kestrel.tsv", sep='\t')
        # Remove self-analysis since ngsrelate doesn't do that
        kes_gt = kes_gt[kes_gt['sample1'] != kes_gt['sample2']]

        kes_gt = kes_gt.assign(repetition=rep_dir.name, tool='kestrel')
        gt_dfs.append(kes_gt)

        depth_dirs = [path for path in rep_dir.iterdir() if path.is_dir()]
        for depth_dir in depth_dirs:
            depth = int(depth_dir.name.rstrip("x"))

            cols = ["sample1", "sample2", "kinship"] + [f"jacquard{i}" for i in range(1, 10)] + ["iterations"]

            ngs_df = pd.read_csv(f"{depth_dir}/ngsrelate.tsv", sep='\t')
            ngs_df = ngs_df.rename(columns={"theta": "kinship", "nIter": "iterations", "a": "sample1", "b": "sample2"} | {f"J{i}": f"jacquard{i}" for i in range(1, 10)})

            kes_df = pd.read_csv(f"{depth_dir}/kestrel.tsv", sep='\t')
            # Remove self-analysis since ngsrelate doesn't do that
            kes_df = kes_df[kes_df['sample1'] != kes_df['sample2']]

            assert len(ngs_df) == len(kes_df)

            # Overwrite the ngsrelate names with the kestrel ones, since those match the VCF
            ngs_df[["sample1", "sample2"]] = kes_df[["sample1", "sample2"]].to_numpy()

            ngs_sub = ngs_df[cols].assign(tool="ngsrelate")
            kes_sub = kes_df[cols].assign(tool="kestrel")

            combined = pd.concat([ngs_sub, kes_sub], ignore_index=True).assign(repetition=rep_dir.name, depth=depth)
            gl_dfs.append(combined)

            ngs_time = parse_ngsrelate_runtime(f"{depth_dir}/ngsrelate.out")
            kes_time = parse_kestrel_runtime(f"{depth_dir}/kestrel.out")

            records.append({
                "repetition": rep_dir.name,
                "depth": depth,
                "tool": "kestrel",
                "time": kes_time
            })

            records.append({
                "repetition": rep_dir.name,
                "depth": depth,
                "tool": "ngsrelate",
                "time": ngs_time
            })

    return pd.concat(gt_dfs, ignore_index=True), pd.concat(gl_dfs, ignore_index=True), pd.DataFrame(records)

def plot_kinship_graph(gt_df, min_kin):

    rel_df = gt_df[gt_df["kinship"] >= min_kin].drop_duplicates(subset=["sample1", "sample2"])

    print(len(rel_df))

    graph = nx.from_pandas_edgelist(rel_df, source="sample1", target="sample2", edge_attr="kinship")

    nodes_to_keep = [node for comp in nx.connected_components(graph) if len(comp) >= 3 for node in comp]
    graph = graph.subgraph(nodes_to_keep).copy()
    #pos = nx.nx_agraph.pygraphviz_layout(graph, prog="neato", args="-Goverlap=false")
    pos = nx.nx_agraph.pygraphviz_layout(graph, prog="neato", args="-Gpackmode=node -Gsep=+5")

    fig, ax = plt.subplots(figsize=(10, 8))

# Scale line thickness by kinship (e.g., 0.25 vs 0.50)
    edge_widths = [graph[u][v]["kinship"] * 6 for u, v in graph.edges()]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=100,
        node_color="#56B4E9",
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color="#555555",
        alpha=0.8,
        ax=ax,
    )
    #nx.draw_networkx_labels(
        #graph,
        #pos,
        #font_size=8,
        #font_weight="bold",
        #ax=ax,
    #)

    #edge_labels = {(u, v): f"{d['kinship']:.2f}" for u, v, d in graph.edges(data=True)}
    #nx.draw_networkx_edge_labels(graph, pos, font_size=7, ax=ax)

    ax.set_title(f"Hard-called GT Relatedness Network (Kinship ≥ {min_kin})")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"kinship_network_{min_kin}.png", dpi=300, bbox_inches="tight")

def print_kinship_err(gt_df, gl_df, min_kin):
# 1. Filter GT pairs with kinship >= 0.2
    gt_rel = (
        gt_df[gt_df["kinship"] >= min_kin][["repetition", "sample1", "sample2", "kinship"]]
        .rename(columns={"kinship": "kinship_gt"})
    )

    # 2. Keep only those pairs across all depths and tools in df
    rel_eval = gl_df.merge(gt_rel, on=["repetition", "sample1", "sample2"], how="inner")

    # 3. Compute signed error (bias) and absolute error relative to GT
    rel_eval["error"] = rel_eval["kinship"] - rel_eval["kinship_gt"]
    rel_eval["abs_error"] = rel_eval["error"].abs()

    # 4. Summarize across depths and pivot tools side-by-side
    error_summary = (
        rel_eval.groupby(["depth", "tool"], observed=True)
        .agg(
            MAE=("abs_error", "mean"),
            Bias=("error", "mean"),
            RMSE=("error", lambda s: np.sqrt(np.mean(s**2))),
        )
        .reset_index()
    )

    error_table = (
        error_summary.pivot(index="depth", columns="tool", values=["MAE", "Bias", "RMSE"])
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1)
    )
    error_table.index = [f"{d}x" for d in error_table.index]
    error_table.index.name = "Depth"

    print(error_table.round(4))


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument("dir", type=str, help="VCF file directory")
    args = p.parse_args()

    gt_df, df, df_time = process_reps(args.dir)

    df = df.sort_values(by="depth")
    df["depth_str"] = df["depth"].astype(str) + "x"

    time_summary = (
        df_time.groupby(["tool"], observed=True)["time"]
        #df_time.groupby(["depth", "tool"], observed=True)["time"]
        .agg(["min", "mean", "median", "max"])
        .reset_index()
    )

    print(time_summary)

    plot_kinship_graph(gt_df, 0.2)
    print_kinship_err(gt_df, df, 0.2)

    sns.set_theme(style="whitegrid")

    cb_palette = {
        "kestrel": "#56B4E9",    # High-contrast Blue
        "ngsrelate": "#009E73"   # High-contrast Vermilion
    }

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="depth_str",
        y="jacquard7",
        hue="tool",
        #split=True,
        #inner="quart",          # draws median and IQR lines inside each half
        #gap=0.1,                # adds a small visual gap between the two halves
        #cut=0,                  # prevents KDE tails from extending below 0
        #density_norm="width",   # keeps violin widths uniform across depths
        palette=cb_palette,
        ax=ax,
    )

    ax.set_xlabel("Sequencing Depth")
    ax.set_ylabel("Estimated J8")
    ax.set_title("Kinship Estimates by Depth: ngsRelate vs. Kestrel")
    ax.legend(title="Tool", frameon=True)

    plt.tight_layout()
    plt.savefig("j7_distribution_real.png", dpi=300, bbox_inches="tight")

    summary_by_tool = (
        #df.groupby(["depth", "tool"], observed=True)["iterations"]
        df.groupby(["tool"], observed=True)["iterations"]
        .agg(["min", "mean", "median", "max"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.stripplot(
        data=df,
        x="depth_str",
        y="iterations",
        hue="tool",
        dodge=True,             # places ngsrelate and kestrel side-by-side at each depth
        jitter=0.25,            # spreads overlapping points horizontally
        alpha=0.25,             # transparency reveals point density at the 10,003 ceiling
        size=3.5,
        palette=cb_palette,
        ax=ax,
    )

# Log scale with standard integer tick labels
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.set_ylim(1, 22000)

# Mark the ngsRelate 10,000-iteration ceiling
    ax.axhline(10000, color="#B22222", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(
        x=df["depth_str"].nunique() - 0.52,
        y=11500,
        s="ngsRelate cap (10,000)",
        color="#B22222",
        fontsize=9,
        ha="right",
    )

# Keep legend dots fully opaque even when plot points are translucent
    leg = ax.legend(title="Tool", frameon=True, loc="upper left")
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    ax.set_xlabel("Sequencing Depth")
    ax.set_ylabel("Iterations (log scale)")
    ax.set_title("Convergence Speed by Depth: ngsRelate vs. Kestrel")

    plt.tight_layout()
    plt.savefig("iterations_real.png", dpi=300)

    print(summary_by_tool)

    fig, ax = plt.subplots(figsize=(9, 5))
# Set log scale BEFORE calling violinplot so the density curve is computed in log space
    ax.set_yscale("log")

    sns.violinplot(
        data=df,
        x="depth_str",
        y="iterations",
        hue="tool",
        split=True,
        inner="quart",          # draws median and IQR lines inside each half
        gap=0.1,
        cut=0,                  # clamps violins at exact min/max (won't bleed past 10,003)
        density_norm="width",   # keeps violin widths comparable across depths
        palette=cb_palette,
        ax=ax,
    )

# Format y-axis ticks as readable integers (1, 10, 100, 1,000, 10,000)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.set_ylim(1, 22000)

# Mark the ngsRelate 10,000-iteration ceiling
    ax.axhline(10000, color="#B22222", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(
        x=df["depth_str"].nunique() - 0.52,
        y=11500,
        s="ngsRelate cap (10,000)",
        color="#B22222",
        fontsize=9,
        ha="right",
    )

    ax.set_xlabel("Sequencing Depth")
    ax.set_ylabel("Iterations (log scale)")
    ax.set_title("Convergence Speed by Depth: ngsRelate vs. Kestrel")
    ax.legend(title="Tool", frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig("iterations-violin.png", dpi=300)

    #stuffff

    ref_30x = (
        df[df["depth"] == 30][["repetition", "tool", "sample1", "sample2", "kinship"]]
        .rename(columns={"kinship": "kinship_30x"})
    )

    gt_rel_pairs = gt_df[gt_df["kinship"] >= 0.2][["repetition", "sample1", "sample2"]].assign(is_related=True)

    self_eval = (
        df[df["depth"] < 30]
        .merge(ref_30x, on=["repetition", "tool", "sample1", "sample2"])
        .merge(gt_rel_pairs, on=["repetition", "sample1", "sample2"], how="left")
    )
    self_eval["is_related"] = self_eval["is_related"].fillna(False)
    self_eval["error"] = self_eval["kinship"] - self_eval["kinship_30x"]
    self_eval["abs_error"] = self_eval["error"].abs()

    for label, subset in [("All pairs", self_eval), ("Related pairs (GT >= 0.2)", self_eval[self_eval["is_related"]])]:
        tbl = (
            subset.groupby(["depth", "tool"], observed=True)
            .agg(
                Bias=("error", "mean"),
                MAE=("abs_error", "mean"),
                RMSE=("error", lambda s: np.sqrt(np.mean(s**2))),
            )
            .unstack("tool")
            .swaplevel(0, 1, axis=1)
            .sort_index(axis=1)
        )
        tbl.index = [f"{d}x" for d in tbl.index]
        print(f"\n--- {label} (vs. 30x) ---")
        print(tbl.round(4))

    k20 = df[(df["tool"] == "ngsrelate") & (df["depth"] == 20)]
    k30 = df[(df["tool"] == "ngsrelate") & (df["depth"] == 30)]
    gt_sub = gt_df[gt_df["kinship"] >= 0.2][["repetition", "sample1", "sample2", "kinship"]].rename(columns={"kinship": "gt"})

    check = (
        k20.merge(k30, on=["repetition", "sample1", "sample2"], suffixes=("_20x", "_30x"))
        .merge(gt_sub, on=["repetition", "sample1", "sample2"])
    )
    check["diff_30_20"] = check["kinship_30x"] - check["kinship_20x"]

    print("20x vs 30x related RMSE:", np.sqrt(np.mean(check["diff_30_20"] ** 2)).round(4))
    print(
        check.reindex(check["diff_30_20"].abs().sort_values(ascending=False).index)
        [["repetition", "sample1", "sample2", "gt", "kinship_20x", "kinship_30x", "diff_30_20", "iterations_20x", "iterations_30x"]]
        .head(10)
    )

    target_samples = {"VBS19141-5563STDY7800225", "VBS19133-5563STDY7800217"}
    j_cols = ["kinship"] + [f"jacquard{i}" for i in range(1, 10)]

    sub = df[
        (df["repetition"] == "0")
        & (df["depth"].isin([20, 30]))
        & (df["sample1"].isin(target_samples))
        & (df["sample2"].isin({
            "VBS19143-5563STDY7800227",
            "VBS19155-5563STDY7800241",
            "VBS19145-5563STDY7800229",
        }))
    ]

    print(sub.set_index(["sample1", "sample2", "tool", "depth"])[j_cols].round(3))

    # pairwise concordance
    piv = df.pivot(
        index=["repetition", "depth", "depth_str", "sample1", "sample2"],
        columns="tool",
        values="kinship",
    ).reset_index()

    depths = sorted(df["depth"].unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(3.5 * len(depths), 3.5), sharex=True, sharey=True)

    for ax, d in zip(axes, depths):
        sub = piv[piv["depth"] == d]
        ax.plot([0, 0.55], [0, 0.55], ls="--", color="gray", lw=1, zorder=1)
        for thresh in [0.125, 0.25]:
            ax.axhline(thresh, ls=":", color="#BBBBBB", lw=0.8, zorder=0)
            ax.axvline(thresh, ls=":", color="#BBBBBB", lw=0.8, zorder=0)
        ax.scatter(sub["ngsrelate"], sub["kestrel"], alpha=0.45, s=16, color="#0072B2", edgecolors="none")
        ax.set_title(f"{d}x")
        ax.set_xlabel("ngsRelate kinship")
        ax.set_xlim(-0.02, 0.55)
        ax.set_ylim(-0.02, 0.55)

    axes[0].set_ylabel("Kestrel kinship")
    fig.suptitle("Pairwise Kinship Concordance Across Sequencing Depths", y=1.02)
    plt.tight_layout()
    plt.savefig("concordance_by_depth.png", dpi=300, bbox_inches="tight")

    # split
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: Unrelated background (< 0.05) — checks low-depth false-positive inflation
    sns.boxplot(
        data=df[df["kinship"] < 0.05],
        x="depth_str",
        y="kinship",
        hue="tool",
        palette=cb_palette,
        fliersize=1.5,
        ax=axes[0],
    )
    axes[0].set_title("Unrelated Background (Kinship < 0.05)")
    axes[0].set_xlabel("Sequencing Depth")
    axes[0].set_ylabel("Estimated Kinship")

# Right: Putative relatives (>= 0.05) — mirrored KDEs across depth
    sns.violinplot(
        data=df[df["kinship"] >= 0.05],
        x="depth_str",
        y="kinship",
        hue="tool",
        split=True,
        inner="quart",
        cut=0,
        density_norm="width",
        palette=cb_palette,
        ax=axes[1],
    )
    for thresh in [0.125, 0.25]:
        axes[1].axhline(thresh, ls="--", color="#666666", lw=0.9)
    axes[1].set_title("Putative Relatives (Kinship ≥ 0.05)")
    axes[1].set_xlabel("Sequencing Depth")
    axes[1].set_ylabel("Estimated Kinship")

    plt.tight_layout()
    plt.savefig("kinship_distributions_split.png", dpi=300, bbox_inches="tight")

    # depth
    rep0 = sorted(df["repetition"].unique())[0]
    rel_pairs = gt_df[(gt_df["repetition"] == rep0) & (gt_df["kinship"] >= 0.1)][["sample1", "sample2"]]

    traj_df = df[df["repetition"] == rep0].merge(rel_pairs, on=["sample1", "sample2"])
    traj_df["pair_id"] = traj_df["sample1"] + "_" + traj_df["sample2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, tool in zip(axes, ["kestrel", "ngsrelate"]):
        sub = traj_df[traj_df["tool"] == tool]
        sns.lineplot(
            data=sub,
            x="depth_str",
            y="kinship",
            units="pair_id",
            estimator=None,
            color=cb_palette[tool],
            alpha=0.45,
            lw=1.3,
            marker="o",
            markersize=4,
            ax=ax,
        )
        for thresh in [0.125, 0.25]:
            ax.axhline(thresh, ls="--", color="#666666", lw=0.9)
        ax.set_title(f"{tool} Pair Trajectories Across Depth")
        ax.set_xlabel("Sequencing Depth")
        ax.set_ylabel("Estimated Kinship")

    plt.tight_layout()
    plt.savefig("kinship_trajectories.png", dpi=300, bbox_inches="tight")
