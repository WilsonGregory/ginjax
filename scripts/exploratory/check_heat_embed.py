"""
Checks for scripts/heat_embed.py:
  1) heat_step_fft matches the original heat_equation.heat_step (torus heat-kernel sum) in 1D/2D/3D;
  2) optionally, heat_step_fft reproduces data files written by heat_equation.py (--data DIR);
  3) lifting a (D-1)-dim pair to D dims by making it constant along an axis gives an exact D-dim
     heat solution, and embed_constant orders the copies as constant along axis D, D-1, ..., 1;
  4) random lattice lifts (embedrand): for every direction n in {-1,0,1}^D the lift is periodic,
     constant along n, uses every value of the (D-1)-dim field, and the D-dim heat solution of the
     lift equals the lift of the (D-1)-dim heat solution with diffusion tensor A A^T.

usage (from the repo root): python -m scripts.exploratory.check_heat_embed [--data RUN/data]
"""

import argparse
import ast
import itertools
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from scripts.heat_embed import (
    embed_constant,
    heat_step_fft,
    lattice_directions,
    lift_along,
    lift_matrix,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def anisotropic_heat_fft(u0: np.ndarray, K: np.ndarray, t: float = 1.0) -> np.ndarray:
    """
    Heat equation u_t = div(K grad u) on the torus: convolve with the periodized anisotropic heat
    kernel (sum over neighboring periodic images, which matter on small grids when K > 1).
    """
    N, d = u0.shape[0], u0.ndim
    m = np.arange(N)
    m = np.where(m <= N // 2, m, m - N)  # signed minimal-image offsets
    a = np.stack(np.meshgrid(*([m] * d), indexing="ij"), axis=-1).astype(np.float64)
    Kinv = np.linalg.inv(K)
    kernel = np.zeros(a.shape[:-1])
    for shift in itertools.product((-1, 0, 1), repeat=d):
        b = a + N * np.array(shift)
        kernel += np.exp(-np.einsum("...i,ij,...j->...", b, Kinv, b) / (4 * t))
    kernel /= (4 * np.pi * t) ** (d / 2) * np.sqrt(np.linalg.det(K))
    return np.real(np.fft.ifftn(np.fft.fftn(u0) * np.fft.fftn(kernel)))


def load_original_heat_step():
    # heat_equation.py runs its experiment on import, so pull out heat_step verbatim
    src = (REPO / "scripts" / "heat_equation.py").read_text()
    fn = next(
        n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "heat_step"
    )
    ns = {"jax": jax, "jnp": jnp}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "heat_step", "exec"), ns)
    return ns["heat_step"]


def rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="data dir written by heat_equation.py")
    p.add_argument("--tol", type=float, default=1e-6, help="max relative l2 difference")
    args = p.parse_args()
    heat_step = load_original_heat_step()
    key = random.PRNGKey(0)
    ok = True

    def check(name, value):
        nonlocal ok
        ok &= value < args.tol
        print(f"   {name}: {value:.2e} {'ok' if value < args.tol else 'FAIL'}")

    print("1) heat_step_fft vs original heat_step (float32 rounding is ~1e-7)")
    for D, N, b in [(1, 64, 8), (2, 16, 4), (2, 64, 2), (3, 16, 2)]:
        key, sub = random.split(key)
        x0 = random.uniform(sub, (b,) + (N,) * D, minval=-np.sqrt(3), maxval=np.sqrt(3))
        check(f"D={D} N={N}", rel(heat_step_fft(D, np.asarray(x0), 1, 1.0), heat_step(D, x0, 1, 1.0, True)))

    if args.data:
        print("2) heat_step_fft vs heat_equation.py data files")
        for f in sorted(pathlib.Path(args.data).glob("*/D*.npy")):
            d = np.load(f, allow_pickle=True).item()
            check(f"{f.parent.name}/{f.name[:20]}", rel(heat_step_fft(d["x0"].ndim - 1, d["x0"], 1, 1.0), d["xt"]))

    print("3) lifted (D-1)-dim pairs (target from the original heat_step) solved in D dims")
    for D in (2, 3):
        key, sub = random.split(key)
        x0 = random.uniform(sub, (2,) + (64,) * (D - 1), minval=-np.sqrt(3), maxval=np.sqrt(3))
        xt = heat_step(D - 1, x0, 1, 1.0, True)
        X0, XT = embed_constant(np.asarray(x0), D), embed_constant(np.asarray(xt), D)
        check(f"{D - 1}D -> {D}D, shapes {x0.shape} -> {X0.shape}", rel(heat_step_fft(D, X0, 1, 1.0), XT))
    i, j, l = 5, 17, 40
    order = (
        np.allclose(X0[0, i, j, l], x0[0, i, j])
        and np.allclose(X0[2, i, j, l], x0[0, i, l])
        and np.allclose(X0[4, i, j, l], x0[0, j, l])
    )
    ok &= order
    print(f"   copy order u(i,j), u(i,l), u(j,l): {'ok' if order else 'FAIL'}")

    print("4) random lattice lifts, N=16: D-dim solve of the lift vs lift of the anisotropic (D-1)-dim solve")
    print("   (agreement is limited by Poisson-summation error ~exp(-4 pi^2 t / |n|^2), < 1e-5)")
    N = 16
    for D in (2, 3):
        for n in lattice_directions(D):
            A = lift_matrix(n)
            key, sub = random.split(key)
            u0 = np.asarray(random.uniform(sub, (N,) * (D - 1), minval=-np.sqrt(3), maxval=np.sqrt(3)))
            U0 = lift_along(u0, n)
            structure = (
                not np.any(A @ n)
                and np.array_equal(U0, np.roll(U0, shift=tuple(-n), axis=tuple(range(D))))
                and len(np.unique(U0)) == len(np.unique(u0))
            )
            err = rel(heat_step_fft(D, U0[None], 1, 1.0)[0], lift_along(anisotropic_heat_fft(u0, A @ A.T), n))
            ok &= structure and err < 1e-5
            print(f"   D={D} n={str(tuple(int(c) for c in n)):<12} K=AA^T diag {np.diag(A @ A.T)}: "
                  f"constant along n / onto: {'ok' if structure else 'FAIL'}   solve: {err:.1e} "
                  f"{'ok' if err < 1e-5 else 'FAIL'}")

    print("ALL OK" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
