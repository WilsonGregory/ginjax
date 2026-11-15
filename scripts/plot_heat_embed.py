"""
Paper-style "error vs flops" plots (cf. Fig. 2a / Fig. 4a of arXiv:2609.38916) combining
    - results_D{D}.npy written by scripts/heat_equation.py (Baseline, Nepo[, Copy, Zeros]) and/or
    - results_*.json written by scripts/heat_embed.py (baseline, nepo, embed; files are merged).

Same encoding as anyd_helpers.plot_results: median over trials (line, 'x' markers), interquartile
range (band), one dot per trial, log y-axis, x = mean training flops over trials.

The x-axis is in true GFLOPs (forward-pass flops x samples x epochs / 1e9). anyd_helpers divides by
1e6 and labels the axis "Gigaflops", so the paper's tick values are 1000x these.

usage (from the repo root):
    python -m scripts.plot_heat_embed --anyd RUN/results/results_D2.npy \
        --embed OUT/results/results_1to2_*.json --title "Heat equation, 1D to 2D" --out PREFIX
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import anyd_helpers  # noqa: F401  (needed to unpickle the ModelLabel keys)

# Baseline and Nepo keep the paper's colors (matplotlib "b", "g"); Embed is a magenta that stays
# distinguishable from both under protan/deutan/tritan simulation (OKLab dE >= 13).
STYLE = {
    "Baseline": ("#0000ff", "solid"),
    "Nepo": ("#008000", "dotted"),
    "Embed": ("#d55181", "dashed"),
    "Copy": ("#e66100", "dashdot"),
    "Zeros": ("#9b30b0", (0, (3, 1, 1, 1, 1, 1))),
}
ORDER = list(STYLE)
JSON_NAMES = {"baseline": "Baseline", "nepo": "Nepo", "embed": "Embed", "embedrand": "Embed"}


def load(
    anyd_path: str | None,
    embed_paths: list[str],
    include: set[str],
    lr_map: dict | None = None,
    zero_shot: bool = False,
):
    """
    returns {method: {n: [(rel, smse, flops), ...over trials]}}, {method: [pretrain flops]}, and
    whether embed pre-training is on the x-axis. Every method's pre-training is put on the x-axis.
    Old-format heat_embed.py files (per-trial "by_n") store Embed's fine-tuning flops only, so its
    pre-training is added here, unless zero_shot. Record-format files ("records") already include
    it, and need --lr-map when they contain a learning-rate sweep.
    """
    data: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    pretrain: dict[str, list[float]] = defaultdict(list)
    pretrain_on_axis = True
    seen_from_anyd = set()
    if anyd_path:
        res = np.load(anyd_path, allow_pickle=True).item()
        for label, arr in res.items():
            name = label.display_name()
            if name not in include:
                continue
            seen_from_anyd.add(name)
            for row in np.asarray(arr):
                data[name][int(label.tune_L)].append((row[0], row[1], row[3]))

    seen_trials = defaultdict(set)
    seen_runs = set()
    for path in embed_paths:
        res = json.load(open(path))
        for r in res.get("records", []):
            name = JSON_NAMES[r["method"]]
            if r["kind"] != "tune" or name not in include or name in seen_from_anyd:
                continue
            if lr_map:
                if r["method"] in lr_map["pretrain"] and r["lr_pre"] != lr_map["pretrain"][r["method"]]:
                    continue
                if r["n"] != 0 and r["lr"] != lr_map["tune"][r["method"]][str(r["n"])]:
                    continue
            key = (name, r["trial"], r["n"])
            if key in seen_runs:
                raise ValueError(f"{key} appears twice (a learning-rate sweep?): pass --lr-map")
            seen_runs.add(key)
            data[name][r["n"]].append((r["rel"], r["smse"], r["pretrain_flops"] + r["tune_flops"]))
            if r["n"] == 0:
                pretrain[name].append(r["pretrain_flops"])
        if "records" in res:
            continue
        for trial, methods in res.get("trials", {}).items():
            for method, m in methods.items():
                name = JSON_NAMES[method]
                if name not in include:
                    continue
                if name in seen_from_anyd:
                    print(f"note: {name} from {path} ignored, already loaded from {anyd_path}")
                    continue
                if trial in seen_trials[name]:
                    raise ValueError(f"trial {trial} of {name} appears in two result files")
                seen_trials[name].add(trial)
                pretrain[name].append(m["pretrain_flops"])
                # pre-training flops not already counted in by_n (Nepo's are, Embed's are not)
                uncounted = m["pretrain_flops"] - m["by_n"]["0"]["flops"]
                shift = 0.0
                if uncounted > 0:
                    if zero_shot:
                        pretrain_on_axis = False
                    else:
                        shift = uncounted
                for n, e in m["by_n"].items():
                    data[name][int(n)].append((e["rel"], e["smse"], e["flops"] + shift))
    return data, pretrain, pretrain_on_axis


def plot(data, metric_idx: int, ylabel: str, title: str, saveloc: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))  # same size/fonts as anyd_helpers.plot_results
    for name in [m for m in ORDER if m in data]:
        ns = sorted(data[name])
        vals = [np.array(data[name][n]) for n in ns]
        x = np.array([v[:, 2].mean() for v in vals]) / 1e9
        y = [v[:, metric_idx] for v in vals]
        color, ls = STYLE[name]
        ax.plot(x, [np.median(v) for v in y], marker="x", linestyle=ls, color=color, label=name,
                linewidth=2, markersize=8)
        ax.fill_between(x, [np.quantile(v, 0.25) for v in y], [np.quantile(v, 0.75) for v in y],
                        color=color, alpha=0.2, linewidth=0)
        for xi, v in zip(x, y):
            ax.scatter(np.full(len(v), xi), v, marker=".", color=color, s=30, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Gigaflops", fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_title(title, fontsize=24)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)
    print(f"saved {saveloc}")


def table(data, pretrain, pretrain_on_axis: bool) -> str:
    names = [m for m in ORDER if m in data]
    ns = sorted({n for m in names for n in data[m]})
    lines = []
    for idx, label, fmt in [(0, "median relative error (%)", "{:.1f}%"), (1, "median l2 (SMSE)", "{:.1e}"),
                            (2, "mean training GFLOPs on the x-axis", "{:.3g}")]:
        lines += [f"\n{label}", "| method | " + " | ".join(f"n={n}" for n in ns) + " | trials |",
                  "|---|" + "---|" * (len(ns) + 1)]
        for m in names:
            cells = []
            for n in ns:
                v = np.array(data[m].get(n, [(np.nan,) * 3]))[:, idx]
                cells.append(fmt.format(np.mean(v) / 1e9 if idx == 2 else np.median(v)))
            k = len(data[m][ns[0]]) if ns[0] in data[m] else 0
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {k} |")
    for m, p in pretrain.items():
        if m == "Embed":
            where = "included on the x-axis" if pretrain_on_axis else "not on the x-axis"
            lines.append(f"\nEmbed pre-training ({where}): {np.mean(p) / 1e9:.3g} GFLOPs per trial")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anyd", type=str, default=None, help="results_D{D}.npy from heat_equation.py")
    p.add_argument("--embed", type=str, nargs="*", default=[], help="results_*.json from heat_embed.py")
    p.add_argument("--include", type=lambda s: set(s.split(",")), default="Baseline,Nepo,Embed")
    p.add_argument("--lr-map", type=str, default=None, help="JSON from heat_select_lr.py")
    p.add_argument("--zero-shot", action="store_true",
                   help="old-format files: leave Embed pre-training off the x-axis")
    p.add_argument("--title", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="output prefix")
    args = p.parse_args()

    lr_map = json.load(open(args.lr_map)) if args.lr_map else None
    data, pretrain, pretrain_on_axis = load(args.anyd, args.embed, args.include, lr_map, args.zero_shot)
    plot(data, 0, "Relative error (%)", args.title, f"{args.out}_rel_vs_gflops.png")
    plot(data, 1, "L2 error", args.title, f"{args.out}_l2_vs_gflops.png")
    t = table(data, pretrain, pretrain_on_axis)
    print(t)
    with open(f"{args.out}_table.md", "w") as f:
        f.write(f"# {args.title}\n{t}\n")


if __name__ == "__main__":
    main()
