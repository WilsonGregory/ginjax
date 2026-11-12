"""
Heat equation, (D-1) -> D: compare ways of getting a D-dimensional model.

    baseline:  D-dimensional UNet trained only on the D-dimensional fine-tuning data.
    nepo:      (D-1)-dimensional UNet pre-trained on (D-1)-dimensional data, converted with the Nepo
               warm-start (geom.Rescaling.NEPO), then fine-tuned on D-dimensional data.
    embed:     the (D-1)-dimensional pre-training pairs lifted to D dimensions by making each pair
               constant along one axis (once per axis, D copies of the set); the D-dimensional UNet
               is pre-trained on them, then fine-tuned on D-dimensional data.
    embedrand: (D-1)-dimensional initial conditions on an --embed-N grid, each lifted to D
               dimensions along one random lattice direction n in {-1,0,1}^D (constant along n),
               with the exact D-dimensional heat solution as target; the D-dimensional UNet is
               pre-trained on them (cheap for a small --embed-N), then fine-tuned on D-dim data.

Training, conversion, flop counting, and evaluation reuse scripts/anyd_helpers.py (AdamW,
warmup-cosine schedule, NRMSE train loss, final-epoch model). There is no per-epoch validation pass:
it never affects the returned model, so each trained model is instead evaluated once on the
validation set (for learning-rate selection) and on the test set.

Learning rates: without --lr-map every fine-tuning run is repeated for each value in --lrs and the
embed pre-training for each value in --lr-pretrain (the best on the embedded validation set is
fine-tuned), i.e. a sweep. scripts/heat_select_lr.py turns a sweep into an --lr-map file
{"pretrain": {method: lr}, "tune": {method: {n: lr}}} for the remaining trials.

Data files have the same names and format as scripts/heat_equation.py, so its data dir can be reused.
New data are generated with an FFT implementation of the same torus heat-kernel convolution.
Results are appended to RESULTS_DIR/results_{lo}to{hi}_{tag}.json as a list of records.

Example (2D -> 3D sweep on trial 0, then the remaining trials with the selected learning rates):
    python -m scripts.heat_embed --low-D 2 --methods baseline,nepo,embedrand --embed-N 16 \
        --trials 0 --tag sweep --data DATA --model-dir OUT/models --results-dir OUT/results
    python -m scripts.heat_select_lr OUT/results/results_2to3_sweep.json --out OUT/lr_map.json
    python -m scripts.heat_embed ... --trials 1,2,3,4 --lr-map OUT/lr_map.json --tag final
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import time
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models
from . import anyd_helpers

MAX_TEMP = math.sqrt(3)  # uniform on [-sqrt(3), sqrt(3)] has unit variance
T = 1  # time step; an int so file names match heat_equation.py ("_t1_")
EVAL_LOSSES = [geom.Losses.L2_REL, geom.Losses.SMSE]
PRETRAINED = ("nepo", "embed", "embedrand")


def heat_step_fft(D: int, x0: np.ndarray, t: float, k: float) -> np.ndarray:
    """
    Same operator as heat_equation.heat_step on the torus: circular convolution with the heat kernel
    exp(-|d|^2 / 4kt) / (4 pi k t)^(D/2), where d is the per-axis minimal-image offset. Computed in
    float64 with FFTs instead of the O(N^(2D)) direct sum.

    args:
        D: dimension
        x0: initial temperature, shape (batch,) + (N,)*D
        t: time step
        k: diffusion coefficient

    returns:
        the temperature after time t, same shape as x0, float32
    """
    N = x0.shape[1]
    assert x0.shape[1:] == (N,) * D
    m = np.arange(N)
    m2 = np.minimum(m, N - m).astype(np.float64) ** 2
    d2 = sum(np.meshgrid(*([m2] * D), indexing="ij"))
    kernel = np.exp(-d2 / (4 * k * t)) / (4 * np.pi * k * t) ** (D / 2)
    axes = tuple(range(1, D + 1))
    x0_hat = np.fft.rfftn(np.asarray(x0, dtype=np.float64), axes=axes)
    out = np.fft.irfftn(x0_hat * np.fft.rfftn(kernel)[None], s=(N,) * D, axes=axes)
    return out.astype(np.float32)


def data_path(data_dir: pathlib.Path, split: str, D: int, N: int, n: int, k: float) -> pathlib.Path:
    # identical naming to heat_equation.get_data_d (is_torus=True, no anisotropy)
    return data_dir / split / f"D{D}_N{N}_istorus1_n{n}_k{k}_t{T}_maxtemp{MAX_TEMP}.npy"


def get_arrays(
    data_dir: pathlib.Path, split: str, D: int, N: int, n: int, k: float, key: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    """Load (x0, xt) of shape (n,) + (N,)*D, generating and caching them if the file is missing."""
    if n == 0:
        return np.zeros((0,) + (N,) * D, np.float32), np.zeros((0,) + (N,) * D, np.float32)

    path = data_path(data_dir, split, D, N, n, k)
    if path.is_file():
        dataset = np.load(path, allow_pickle=True).item()
        return np.asarray(dataset["x0"]), np.asarray(dataset["xt"])

    print(f"Creating data {path}...", flush=True)
    start = time.time()
    _, subkey = random.split(key)  # same sampling call as heat_equation.get_data_d
    x0 = random.uniform(subkey, shape=(n,) + (N,) * D, minval=-MAX_TEMP, maxval=MAX_TEMP)
    x0 = np.asarray(x0)
    xt = heat_step_fft(D, x0, T, k)
    gen_time = time.time() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npy")  # atomic write
    np.save(tmp, {"x0": x0, "xt": xt, "generation_time": gen_time, "generator": "fft"})
    os.replace(tmp, path)
    print(f"Finished in {gen_time:.2f} seconds.", flush=True)
    return x0, xt


def embed_constant(x: np.ndarray, D: int) -> np.ndarray:
    """
    Lift (n,) + (N,)*(D-1) fields to (D*n,) + (N,)*D by making them constant along one axis, once
    for each axis: constant along the last axis first, then the next-to-last, ..., then the first.
    E.g. D=3: U(i,j,l) = u(i,j), then u(i,l), then u(j,l).
    """
    n, N = x.shape[0], x.shape[1]
    lifted = [
        np.broadcast_to(np.expand_dims(x, axis=1 + axis), (n,) + (N,) * D)
        for axis in reversed(range(D))
    ]
    return np.ascontiguousarray(np.concatenate(lifted, axis=0))


def lattice_directions(D: int) -> list[np.ndarray]:
    """Nonzero vectors of {-1,0,1}^D up to sign (first nonzero entry +1): 4 for D=2, 13 for D=3."""
    dirs = []
    for v in itertools.product((-1, 0, 1), repeat=D):
        nz = np.flatnonzero(v)
        if len(nz) and v[nz[0]] == 1:
            dirs.append(np.array(v))
    return dirs


def lift_matrix(n: np.ndarray) -> np.ndarray:
    """
    Integer (D-1, D) matrix A with A n = 0 such that X -> A X mod N maps Z_N^D onto Z_N^(D-1), so
    U(X) = u(A X mod N) is a periodic grid field that is constant along n. Rows n_p e_q - n_q e_p for
    q != p, where p is the first nonzero entry of n (n_p = +-1). For n = e_D this is U(X) = u(X[:-1]).
    """
    D = len(n)
    p = int(np.flatnonzero(n)[0])
    rows = []
    for q in range(D):
        if q != p:
            row = np.zeros(D, dtype=int)
            row[q] += n[p]
            row[p] -= n[q]
            rows.append(row)
    return np.stack(rows)


def lift_along(x: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Lift one (N,)*(D-1) field to (N,)*D, constant along the lattice direction n."""
    N, D = x.shape[0], len(n)
    grid = np.indices((N,) * D).reshape(D, -1)
    coords = (lift_matrix(n) @ grid) % N
    return x[tuple(coords)].reshape((N,) * D)


def lift_random(
    x0: np.ndarray, D: int, k: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lift each (D-1)-dim initial condition along an independently, uniformly chosen lattice direction
    and solve the D-dim heat equation exactly (FFT) for the target. For axis directions the target
    is the lifted (D-1)-dim solution; for diagonal ones it equals the lifted solution of a (D-1)-dim
    anisotropic heat equation (diffusion tensor A A^T), i.e. still a (D-1)-dim computation.

    returns:
        (X0, XT) of shape (n,) + (N,)*D, and the direction index of each sample
    """
    dirs = lattice_directions(D)
    idx = rng.integers(len(dirs), size=len(x0))
    X0 = np.stack([lift_along(x, dirs[i]) for x, i in zip(x0, idx)])
    return X0, heat_step_fft(D, X0, T, k), idx


def make_dataloader(
    x0: np.ndarray, xt: np.ndarray, D: int, batch_size: int, shuffle: bool
) -> DataLoader[ml.MultiImageDataset]:
    # identical dataloader construction to heat_equation.get_data
    x0_img = geom.MultiImage({(0, 0): jnp.asarray(x0)[:, None]}, D, True)
    xt_img = geom.MultiImage({(0, 0): jnp.asarray(xt)[:, None]}, D, True)
    dataset = ml.MultiImageDataset(x0_img, xt_img)
    n = len(x0)
    sampler = RandomSampler(dataset) if (shuffle and n > 0) else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=BatchSampler(sampler, batch_size, drop_last=n > batch_size),
        collate_fn=lambda x: x[0],
    )


def float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-D", type=int, required=True, help="pre-training dimension (1 or 2)")
    parser.add_argument("--data", type=str, required=True, help="data dir (train/val/test subdirs)")
    parser.add_argument("--model-dir", type=str, default=None, help="save/load trained models here")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None, help="results file tag (default: trials)")
    parser.add_argument("--methods", type=lambda s: s.split(","), default="baseline,nepo,embedrand")
    parser.add_argument("--trials", type=lambda s: [int(x) for x in s.split(",")], default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-N", type=int, default=64)
    parser.add_argument("--diffusion-coef", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=128, help="(D-1)-dim pre-training points")
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument(
        "--n-tune-range", type=lambda s: [int(x) for x in s.split(",")], default="0,1,4,8,32,128"
    )
    parser.add_argument("--embed-N", type=int, default=16, help="grid side for embedrand pre-training")
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("--lr-low-pretrain", type=float, default=None, help="nepo (D-1)-dim pretrain")
    parser.add_argument("--lr-pretrain", type=float_list, default="1e-5,5e-5,1e-4,5e-4,1e-3,5e-3",
                        help="embed/embedrand pre-training learning rates (sweep)")
    parser.add_argument("--lrs", type=float_list, default="1e-5,5e-5,1e-4,5e-4",
                        help="fine-tuning learning rates for every method (sweep)")
    parser.add_argument("--lr-map", type=str, default=None, help="JSON from heat_select_lr.py")
    parser.add_argument("--data-only", action="store_true", help="generate/check data, then exit")
    parser.add_argument("-v", "--verbose", type=int, default=1)
    args = parser.parse_args()
    if args.lr_low_pretrain is None:  # Table 3: 1D pre-train 1e-3; 2D models at n=128 use 5e-4
        args.lr_low_pretrain = 1e-3 if args.low_D == 1 else 5e-4
    if args.tag is None:
        args.tag = "trials" + "-".join(str(t) for t in args.trials)
    return args


def main() -> None:
    args = handle_args()
    lo, hi = args.low_D, args.low_D + 1
    N, k, eN = args.N, args.diffusion_coef, args.embed_N
    data_dir = pathlib.Path(args.data)
    model_dir = pathlib.Path(args.model_dir) if args.model_dir else None
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
    lr_map = json.loads(pathlib.Path(args.lr_map).read_text()) if args.lr_map else None
    print(f"heat_embed {lo}D -> {hi}D  methods={args.methods}  trials={args.trials}", flush=True)
    print(vars(args), flush=True)

    # ---- data: generated up front with fixed keys, so separate processes share identical data
    data_key = random.fold_in(random.PRNGKey(args.seed), 1_000_003)
    k_lo_train, k_lo_val, k_hi_val, k_hi_test, k_tune = random.split(data_key, 5)
    k_er_train, k_er_val = random.split(random.fold_in(data_key, eN), 2)
    lo_train = get_arrays(data_dir, "train", lo, N, args.n_train, k, k_lo_train)
    lo_val = get_arrays(data_dir, "val", lo, N, args.n_val, k, k_lo_val)
    hi_val = get_arrays(data_dir, "val", hi, N, args.n_val, k, k_hi_val)
    hi_test = get_arrays(data_dir, "test", hi, N, args.n_test, k, k_hi_test)
    hi_tune = {
        n: get_arrays(data_dir, "train", hi, N, n, k, random.fold_in(k_tune, n))
        for n in args.n_tune_range
    }
    pretrain_sets = {}  # method -> (train (x0, xt), embedded val (x0, xt), info)
    if "embed" in args.methods:
        pretrain_sets["embed"] = (
            (embed_constant(lo_train[0], hi), embed_constant(lo_train[1], hi)),
            (embed_constant(lo_val[0], hi)[: args.n_val], embed_constant(lo_val[1], hi)[: args.n_val]),
            {"N": N, "copies": hi},
        )
    if "embedrand" in args.methods:
        # (D-1)-dim initial conditions on the eN grid (same files/keys as the N grid when eN == N)
        er_train = get_arrays(data_dir, "train", lo, eN, args.n_train, k, k_lo_train if eN == N else k_er_train)
        er_val = get_arrays(data_dir, "val", lo, eN, args.n_val, k, k_lo_val if eN == N else k_er_val)
        rng = np.random.default_rng([args.seed, eN, 7919])  # same lifts in every process/trial
        X0, XT, train_dirs = lift_random(er_train[0], hi, k, rng)
        V0, VT, _ = lift_random(er_val[0], hi, k, rng)
        dirs = lattice_directions(hi)
        counts = {str(tuple(int(c) for c in dirs[i])): int((train_dirs == i).sum()) for i in range(len(dirs))}
        print(f"embedrand: {len(X0)} lifts to {X0.shape[1:]}, directions {counts}", flush=True)
        pretrain_sets["embedrand"] = ((X0, XT), (V0, VT), {"N": eN, "directions": counts})
    for name, (x0, xt) in [("pretrain", lo_train), ("test", hi_test)]:
        print(f"{name}: {x0.shape}  x0 std {x0.std():.3e}  xt std {xt.std():.3e}", flush=True)
    if args.data_only:
        return

    B = args.batch
    lo_train_dl = make_dataloader(*lo_train, lo, B, shuffle=True)
    hi_val_dl = make_dataloader(*hi_val, hi, B, shuffle=False)
    hi_test_dl = make_dataloader(*hi_test, hi, B, shuffle=False)
    pretrain_dls = {
        m: (
            make_dataloader(tr[0], tr[1], hi, B, shuffle=True),
            make_dataloader(va[0], va[1], hi, B, shuffle=False),
            info,
        )
        for m, (tr, va, info) in pretrain_sets.items()
    }

    # ---- models, training, and evaluation as in heat_equation.py / anyd_helpers.py
    free_filters, upsample_filters = anyd_helpers.generate_filters((lo, hi), [0])

    def make_unet(D: int, key: jax.Array) -> models.UNet:
        sig = geom.MultiImage({(0, 0): jnp.zeros((1, 1) + (N,) * D)}, D, True).get_signature()
        return models.UNet(
            D,
            sig,
            sig,
            depth=48,
            activation_f=jax.nn.gelu,
            conv_filters=free_filters[D],
            upsample_filters=upsample_filters[D],
            key=key,
        )

    def train(label, model, train_dl, lr):
        # no per-epoch validation pass: it is only logged and never changes the returned model
        return anyd_helpers.train_model(
            (train_dl, None), label, model, lr, geom.Losses.NRMSE, B, args.epochs, model_dir,
            False, None, verbose=args.verbose,
        )

    def evaluate(model, dl) -> tuple[float, float]:
        rel, smse = ml.map_loss_in_batches_dl(ml.Mapper(EVAL_LOSSES, eps=1e-9), model, dl)
        return float(rel), float(smse)

    results_file = results_dir / f"results_{lo}to{hi}_{args.tag}.json"
    results: dict[str, Any] = (
        json.loads(results_file.read_text()) if results_file.is_file() else {"records": []}
    )
    results["meta"] = {
        **vars(args),
        "pretrain_sets": {m: info for m, (_, _, info) in pretrain_sets.items()},
        "flops_note": "forward-pass flops x samples x epochs (anyd_helpers.train_model); nepo "
        "pretrain_flops include the conversion (param count), as in anyd_helpers.convert_and_tune",
    }
    done = {
        (r["kind"], r["trial"], r["method"], r.get("lr_pre"), r.get("lr"), r.get("n"))
        for r in results["records"]
    }

    def save(record: dict) -> None:
        results["records"].append(record)
        results_file.write_text(json.dumps(results, indent=1))

    ModelLabel = anyd_helpers.ModelLabel
    for trial in args.trials:
        k_base, k_nepo, k_embed, k_conv = random.split(random.fold_in(random.PRNGKey(args.seed), trial), 4)

        # method -> (initial hi-D model, pretrain flops, pretrain lr, pretrain_D, pretrain_L)
        starts = {}
        if "baseline" in args.methods:
            starts["baseline"] = (make_unet(hi, k_base), 0, None, None, None)
        if "nepo" in args.methods:
            label = ModelLabel("heat_nepo", trial, pretrain_D=lo, pretrain_L=args.n_train)
            lo_model, _, pre_flops = train(
                label, make_unet(lo, k_nepo), lo_train_dl, args.lr_low_pretrain
            )
            converted = lo_model.convertD(
                free_filters[hi], geom.Rescaling.NEPO, k_conv, upsample_filters=upsample_filters[hi]
            )
            flops = pre_flops + models.count_params(lo_model)  # same accounting as convert_and_tune
            starts["nepo"] = (converted, flops, args.lr_low_pretrain, lo, args.n_train)
        for method in ("embed", "embedrand"):
            if method not in args.methods:
                continue
            train_dl, val_dl, info = pretrain_dls[method]
            L = len(train_dl.dataset)
            lrs_pre = [lr_map["pretrain"][method]] if lr_map else args.lr_pretrain
            best = None
            for lr_pre in lrs_pre:
                name = f"heat_{method}{info['N'] if method == 'embedrand' else ''}_lr{lr_pre:g}"
                label = ModelLabel(name, trial, pretrain_D=lo, pretrain_L=L)
                model, _, pre_flops = train(label, make_unet(hi, k_embed), train_dl, lr_pre)
                val_rel, val_smse = evaluate(model, val_dl)  # embedded validation set
                print(f"Pretrain {label}: embedded val {val_smse:.3e} ({val_rel:.3f}%)", flush=True)
                if ("pretrain", trial, method, lr_pre, None, None) not in done:
                    save({"kind": "pretrain", "trial": trial, "method": method, "lr_pre": lr_pre,
                          "val_rel": val_rel, "val_smse": val_smse, "pretrain_flops": pre_flops})
                if best is None or val_rel < best[0]:
                    best = (val_rel, model, pre_flops, lr_pre)
            assert best is not None, f"{method}: no pre-training learning rate given"
            _, model, pre_flops, lr_pre = best
            print(f"{method}: fine-tuning the lr_pre={lr_pre:g} pre-trained model", flush=True)
            starts[method] = (model, pre_flops, lr_pre, lo, L)

        for n in args.n_tune_range:
            tune_dl = make_dataloader(*hi_tune[n], hi, B, shuffle=True)
            for method, (start_model, pre_flops, lr_pre, pD, pL) in starts.items():
                if n == 0:
                    lrs = [None]
                elif lr_map:
                    lrs = [lr_map["tune"][method][str(n)]]
                else:
                    lrs = args.lrs
                for lr in lrs:
                    rec_pre = lr_pre if method in PRETRAINED else None
                    if ("tune", trial, method, rec_pre, lr, n) in done:
                        continue
                    if n == 0:
                        tuned, tune_time, tune_flops = start_model, 0.0, 0
                    else:
                        pre_str = f"_pre{lr_pre:g}" if method in ("embed", "embedrand") else ""
                        label = ModelLabel(f"heat_{method}{pre_str}_lr{lr:g}", trial, pD, pL, None, hi, n)
                        tuned, tune_time, tune_flops = train(label, start_model, tune_dl, lr)
                    val_rel, val_smse = evaluate(tuned, hi_val_dl)
                    rel, smse = evaluate(tuned, hi_test_dl)
                    save({"kind": "tune", "trial": trial, "method": method, "lr_pre": rec_pre,
                          "lr": lr, "n": n, "rel": rel, "smse": smse, "val_rel": val_rel,
                          "val_smse": val_smse, "tune_time": tune_time, "tune_flops": tune_flops,
                          "pretrain_flops": pre_flops})
                    print(f"Eval {method} trial{trial} n={n} lr={lr}: test {smse:.3e} ({rel:.3f}%)"
                          f"  val ({val_rel:.3f}%)", flush=True)

    print(f"Done. {results_file}", flush=True)


if __name__ == "__main__":
    main()
