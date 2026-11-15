"""Compare a results_D2.npy from scripts/heat_equation.py against arXiv:2609.38916 Tables 1/4/5/6.

usage (from the repo root): python -m scripts.heat_compare_paper /path/to/results_D2.npy
"""
import sys

import numpy as np
import jax.numpy as jnp

from . import anyd_helpers  # noqa: F401  (needed to unpickle the ModelLabel keys)

NS = [0, 1, 4, 8, 32, 128]
# Tables 1 & 4 (median relative error %, 5 trials)
PAPER_REL = {
    "Baseline": [105.3, 21.0, 20.8, 19.8, 3.3, 0.5],
    "Copy": [12797.3, 131.6, 138.5, 152.5, 52.7, 15.0],
    "Zeros": [138.3, 8.1, 8.1, 8.4, 3.1, 1.0],
    "Nepo": [5.7, 2.2, 2.2, 2.1, 0.7, 0.5],
}
# Table 5 (median l2 / SMSE)
PAPER_L2 = {
    "Baseline": [4.4e-02, 1.7e-03, 1.7e-03, 1.5e-03, 4.5e-05, 1.0e-06],
    "Copy": [6.5e02, 6.9e-02, 7.6e-02, 9.3e-02, 1.1e-02, 9.0e-04],
    "Zeros": [7.6e-02, 2.6e-04, 2.9e-04, 2.8e-04, 3.8e-05, 4.8e-06],
    "Nepo": [1.3e-04, 2.0e-05, 1.9e-05, 1.9e-05, 2.2e-06, 1.0e-06],
}
# Table 6 (value as printed in the paper; = forward flops * n * epochs / 1e6)
PAPER_FLOPS = {
    "Baseline": [0, 8.732e04, 3.493e05, 6.986e05, 2.794e06, 1.117e07],
    "Nepo": [3.594e05, 4.467e05, 7.087e05, 1.058e06, 3.153e06, 1.153e07],
}

res = jnp.load(sys.argv[1], allow_pickle=True).item()
ours = {}  # name -> {n: (n_trials, 4) array [rel, smse, time, flops]}
for label, arr in res.items():
    ours.setdefault(label.display_name(), {})[label.tune_L] = np.asarray(arr)

def fmt(x, sci):
    return "   --   " if x is None else (f"{x:8.1e}" if sci else f"{x:8.1f}")

for title, idx, paper, sci in [
    ("Relative error % (Tables 1/4), median over trials", 0, PAPER_REL, False),
    ("l2 / SMSE (Table 5), median over trials", 1, PAPER_L2, True),
    ("Flops / 1e6 (Table 6), mean over trials", 3, PAPER_FLOPS, True),
]:
    print(f"\n== {title} ==")
    print(f"{'model':<9}{'':<6}" + "".join(f"{'n=' + str(n):>9}" for n in NS))
    for name in ["Baseline", "Nepo", "Zeros", "Copy"]:
        if name not in ours:
            continue
        row = []
        for n in NS:
            a = ours[name].get(n)
            if a is None:
                row.append(None)
            elif idx == 3:
                row.append(float(np.mean(a[:, idx])) / 1e6)
            else:
                row.append(float(np.median(a[:, idx])))
        trials = {n: len(ours[name][n]) for n in ours[name]}
        print(f"{name:<9}{'ours':<6}" + "".join(f" {fmt(x, sci)}" for x in row) + f"   trials={trials}")
        if name in paper:
            print(f"{'':<9}{'paper':<6}" + "".join(f" {fmt(x, sci)}" for x in paper[name]))

# per-trial spread for the n=0 column, since medians of 5 can hide a lot
print("\n== n=0 per-trial relative error % ==")
for name in ["Baseline", "Nepo", "Zeros", "Copy"]:
    if name in ours and 0 in ours[name]:
        print(f"{name:<9}", np.round(np.sort(ours[name][0][:, 0]), 1))
