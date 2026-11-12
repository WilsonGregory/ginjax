"""
Pick learning rates from a scripts/heat_embed.py sweep, on validation error only (as in the paper:
"Learning rates were chosen ... based on performance on a validation set").

    pretrain: for embed/embedrand, the lr_pre with the lowest median embedded-validation error.
    tune:     for every method and n > 0, the fine-tuning lr with the lowest median validation error
              (for embed methods, among runs that start from the selected pre-trained model).

usage (from the repo root):
    python -m scripts.heat_select_lr SWEEP.json [MORE.json ...] --out lr_map.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def best_lr(med: dict) -> float:
    """lr with the lowest median validation error; diverged (NaN) runs count as worst"""
    return min(med, key=lambda lr: med[lr] if np.isfinite(med[lr]) else np.inf)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    records = [r for f in args.results for r in json.load(open(f))["records"]]

    pre = defaultdict(lambda: defaultdict(list))  # method -> lr_pre -> [val_rel]
    for r in records:
        if r["kind"] == "pretrain":
            pre[r["method"]][r["lr_pre"]].append(r["val_rel"])
    chosen_pre = {}
    for method, by_lr in pre.items():
        med = {lr: float(np.median(v)) for lr, v in by_lr.items()}
        chosen_pre[method] = best_lr(med)
        print(f"{method} pre-training, median embedded-val rel. error (%): "
              + "  ".join(f"{lr:g}: {m:.3g}" for lr, m in sorted(med.items()))
              + f"  -> {chosen_pre[method]:g}")

    tune = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # method -> n -> lr -> [val]
    for r in records:
        if r["kind"] != "tune" or r["n"] == 0:
            continue
        if r["method"] in chosen_pre and r["lr_pre"] != chosen_pre[r["method"]]:
            continue
        tune[r["method"]][r["n"]][r["lr"]].append(r["val_rel"])
    chosen_tune = {}
    for method, by_n in tune.items():
        chosen_tune[method] = {}
        print(f"\n{method} fine-tuning, median val rel. error (%) by lr:")
        for n in sorted(by_n):
            med = {lr: float(np.median(v)) for lr, v in by_n[n].items()}
            best = best_lr(med)
            chosen_tune[method][str(n)] = best
            print(f"   n={n:<4}" + "  ".join(f"{lr:g}: {m:8.3g}" for lr, m in sorted(med.items()))
                  + f"   -> {best:g}")

    out = {"pretrain": chosen_pre, "tune": chosen_tune, "sources": args.results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
