import jax.numpy as jnp
import numpy as np
import sympy as sp
from jaxtyping import Array, Float

import ginjax.geometric as geom


def pi_n(
    arr: Float[Array, "... tensor_highD"], lowD: int, k: int
) -> Float[Array, "... tensor_lowD"]:
    """
    Perform the tensor projection from highD to lowD on a k tensor.
    """
    assert k > 0
    return arr[*((slice(None),) * (arr.ndim - k)), *((slice(0, lowD),) * k)]


sp.init_printing(use_unicode=True)

full_D_range = [2, 3]

free_filters_dict = {}

conv_filters_1d = geom.get_invariant_filters(
    [3], [0], [0], 1, geom.make_all_operators(1), scale=geom.FilterScaling.ONE
)
conv_filters_2d = geom.get_invariant_filters(
    [3], [0, 1, 2], [0], 2, geom.make_all_operators(2), scale=geom.FilterScaling.ONE
)
conv_filters_3d = geom.get_invariant_filters(
    [3], [0, 1, 2], [0], 3, geom.make_all_operators(3), scale=geom.FilterScaling.ONE
)

print("1d scalar filters:", conv_filters_1d[((), 0)].shape)
print("2d scalar filters:", conv_filters_2d[((), 0)].shape)  # (n_filters,spatial)
summed_2d = jnp.sum(conv_filters_2d[((), 0)], axis=-1)  # (n_filters,spatial_1d)
combined = jnp.concat([summed_2d.T, -conv_filters_1d[((), 0)].T], axis=1)

print(combined.shape)
print(combined)

sp_combined = sp.Matrix(combined)
sp.pprint(sp_combined)
rref_combined = sp_combined.rref()[0]
# result is columns of alpha_prime, then columns of alpha
sp.pprint(rref_combined)

lowD = 2
highD = 3
for (k, p), filters_2d in conv_filters_2d.items():
    print(f"(k,p): ({len(k)},{p})")
    filters_3d = conv_filters_3d[k, p]
    summed_3d = jnp.sum(filters_3d, axis=highD)  # axis is dimension
    if len(k) > 0:
        # do the pi projection on each tensor
        summed_3d = pi_n(summed_3d, lowD, len(k))
        print(summed_3d.shape)

    # shape (spatial_lowD*tensor_lowD,n_filters_highD+n_filters_lowD)
    combined = jnp.concat(
        [summed_3d.reshape(len(summed_3d), -1).T, -filters_2d.reshape((len(filters_2d), -1)).T],
        axis=1,
    )

    # TODO: I might want to convert some of the floats in combined to rationals
    sp_combined = sp.Matrix(combined)
    rref_combined = sp_combined.rref()[0]
    # result is columns of alpha_prime, then columns of alpha
    sp.pprint(rref_combined)


b0, b1, b2, b3, b4, b5, b6, b7 = sp.symbols("b0 b1 b2 b3 b4 b5 b6 b7")
a0, a1, a2, a3, a4 = sp.symbols("a0 a1 a2 a3 a4")

rref_combined_short = rref_combined[:5, :]
vars_col = sp.Matrix([b0, b1, b2, b3, b4, b5, b6, b7, a0, a1, a2, a3, a4])
eqs = rref_combined_short @ (-1 * vars_col)
sp.pprint(sp.Array(eqs))
rhs_eqs = sp.Array(eqs).reshape(5) + sp.Array([b0, b1, b2, b4, b6])
rhs_eqs = sp.Array([rhs_eqs[0], rhs_eqs[1], rhs_eqs[2], b3, rhs_eqs[3], b5, rhs_eqs[4], b7])
sp.pprint(rhs_eqs)


# do the zero pad embeddings varphi.
zeros_pad = jnp.zeros((5, 3, 3, 2, 2))
# (n_filters,spatial_3d,tensor_2d)
filters_2d3d = jnp.stack([zeros_pad, conv_filters_2d[((False, False), 0)], zeros_pad], axis=3)
print(filters_2d3d.shape)
sp_filters_2d3d = sp.Array(np.array(filters_2d3d))
filters_2d3d_var = sp.Array(
    [
        (a0 * sp_filters_2d3d[0]).tolist(),
        (a1 * sp_filters_2d3d[1]).tolist(),
        (a2 * sp_filters_2d3d[2]).tolist(),
        (a3 * sp_filters_2d3d[3]).tolist(),
        (a4 * sp_filters_2d3d[4]).tolist(),
    ],
)
print(filters_2d3d_var.shape)

filters_3d = pi_n(conv_filters_3d[((False, False), 0)], lowD, 2)
sp_filters_3d = sp.Array(np.array(filters_3d))
filters_3d_var = sp.Array(
    [(bi * filters_3d_i).tolist() for bi, filters_3d_i in zip(rhs_eqs, sp_filters_3d)]
)
print(filters_3d_var.shape)

filters_2d3d_var = (
    filters_2d3d_var[0]
    + filters_2d3d_var[1]
    + filters_2d3d_var[2]
    + filters_2d3d_var[3]
    + filters_2d3d_var[4]
)

print(filters_2d3d_var.shape)
sp.pprint(filters_2d3d_var)

filters_3d_var = (
    filters_3d_var[0]
    + filters_3d_var[1]
    + filters_3d_var[2]
    + filters_3d_var[3]
    + filters_3d_var[4]
    + filters_3d_var[5]
    + filters_3d_var[6]
    + filters_3d_var[7]
)
print(filters_3d_var.shape)
sp.pprint(filters_3d_var)

diff = filters_2d3d_var - filters_3d_var
print("diff.shape", diff.shape)
total = sum(sp.Matrix(diff.reshape(108, 1)).applyfunc(lambda x: x**2))
print("total:")
print(total)

print("\nexpanded:")
reduced_total = sp.expand(total) / 6
print(reduced_total)

d_db3 = sp.diff(reduced_total, b3) / 8
d_db5 = sp.diff(reduced_total, b5) / 16
d_db7 = sp.diff(reduced_total, b7) / 6
print("d/db3", d_db3)
print("d/db5", d_db5)
print("d/db7", d_db7)
