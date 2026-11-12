import jax.numpy as jnp
import math
import numpy as np
import sympy as sp
from jaxtyping import Array, Float

import ginjax.geometric as geom


def pi_n(
    arr: Float[Array, "... tensor_highD"], lowD: int, total_k: int, shrink_k: int | None = None
) -> Float[Array, "... tensor_lowD"]:
    """
    Perform the tensor projection from highD to lowD on a k tensor.
    """
    shrink_k = total_k if shrink_k is None else shrink_k  # default shrink them all
    if shrink_k == 0:
        return arr

    # last total_k - shrink_k indices are not shrunk, so we don't need to specify
    return arr[*((slice(None),) * (arr.ndim - total_k) + (slice(0, lowD),) * shrink_k)]


def tilde_phi_n(
    arr: Float[Array, "... tensor_lowD"],
    k: int,
    extraD: int = 1,
) -> Float[Array, "... tensor_highD"]:
    """
    Zero pad the k tensors at the end of arr.

    It feels like there should be a better way of doing this, but no dice.
    """
    if k == 0:
        return arr

    batch_axes = arr.ndim - k
    for axis in range(k):
        added_shape = list(arr.shape)
        added_shape[batch_axes + axis] = extraD
        arr = jnp.concat([arr, jnp.zeros(tuple(added_shape))], axis=batch_axes + axis)

    return arr


def varphi_spatial(
    lowD: int, highD: int, low_filters: Float[Array, "n_low_filters spatial tensor"]
) -> Float[Array, "n_low_filters high_spatial low_tensor"]:
    """
    Pad a low_filter in the spatial dimensions only. For example, (5,3,3,2,2) -> (5,3,3,3,2,2)
    """
    N = low_filters.shape[1]
    assert low_filters.shape[1 : 1 + lowD] == (N,) * lowD
    assert N % 2 == 1, f"varphi_spatial: Spatial dims must be odd, got N={N}"
    assert highD == lowD + 1
    N_half = (N - 1) // 2

    spatial_pad = jnp.zeros(
        low_filters.shape[: 1 + lowD] + (N_half,) + low_filters.shape[1 + lowD :]
    )
    low_filters = low_filters.reshape(
        low_filters.shape[: 1 + lowD] + (1,) + low_filters.shape[1 + lowD :]
    )
    return jnp.concat([spatial_pad, low_filters, spatial_pad], axis=1 + lowD)


def compute_constraints(
    lowD: int,
    highD: int,
    k_p: tuple[tuple[bool, ...], int],
    low_filters: Float[Array, "n_low_filters spatial tensor"],
    high_filters: Float[Array, "n_high_filters spatial tensor"],
    input_k: int,
) -> None:
    assert highD == lowD + 1
    k, p = k_p
    assert input_k <= len(k)
    n_low_ff = len(low_filters)
    n_high_ff = len(high_filters)

    print(f"Converting (k,p)=({len(k)},{p}) {lowD}D to {highD}D with input_k={input_k}.")
    print(f"There are {n_low_ff} {lowD}D filters and {n_high_ff} {highD}D filters.")

    # sum the high filters spatially
    summed_high_filters = jnp.sum(high_filters, axis=highD)

    # embed the low dimensional tensors with zero pad embedding
    embedded_low_filters = tilde_phi_n(low_filters, len(k))

    assert summed_high_filters.shape[1:] == embedded_low_filters.shape[1:]

    summed_high_filters = pi_n(summed_high_filters, lowD, input_k)
    embedded_low_filters = pi_n(embedded_low_filters, lowD, input_k)

    assert summed_high_filters.shape[1:] == embedded_low_filters.shape[1:]

    # shape (spatial_lowD*tensor_lowD,n_filters_highD+n_filters_lowD)
    combined = jnp.concat(
        [
            summed_high_filters.reshape(len(summed_high_filters), -1).T,
            -embedded_low_filters.reshape((len(embedded_low_filters), -1)).T,
        ],
        axis=1,
    )

    # TODO: I might want to convert some of the floats in combined to rationals
    # put the constraints into a matrix
    sp_combined = sp.Matrix(combined)

    rref_combined, rref_pivots = sp_combined.rref()
    rref_combined_short = rref_combined[: len(rref_pivots), :]
    print(
        "The rref of the constraints, each column is a variable, rows are components but they are "
        "in rref here. The first "
        f"{n_high_ff} columns are the {highD}D variables, and the last "
        f"{n_low_ff} columns are -1 times the {lowD}D variables."
        "Each row with a leading 1 is a fixed "
        "parameter, and the columns without associated rows are free parametes."
    )
    sp.pprint(rref_combined_short)
    print(f"The {len(rref_pivots)} pivot columns: {rref_pivots}")
    if max(rref_pivots) > n_high_ff:
        print(
            "WARNING: The constraints are not satisfied by the choice of higher dimensional "
            "filters. Exiting\n"
        )
        return

    print("\nNow we optimize the free params.")
    high_vars = sp.symbols(f"b:{n_high_ff}")
    # b0, b1, b2, b3, b4, b5, b6, b7 = sp.symbols("b0 b1 b2 b3 b4 b5 b6 b7")
    low_vars = sp.symbols(f"a:{n_low_ff}")
    # a0, a1, a2, a3, a4 = sp.symbols("a0 a1 a2 a3 a4")

    vars_col = sp.Matrix(high_vars + low_vars)
    eqs = rref_combined_short @ (-1 * vars_col)

    # cancel out the pivots to get the constraint equations equal to the pivots
    rhs_eqs = sp.Array(eqs).reshape(len(eqs)) + sp.Array([high_vars[i] for i in rref_pivots])

    # TODO: need to get the row,
    rhs_eqs = sp.Array(
        [
            rhs_eqs[tuple(rref_pivots).index(i)] if i in rref_pivots else high_vars[i]
            for i in range(n_high_ff)
        ]
    )
    free_high_vars = list(
        filter(None, [False if i in rref_pivots else rhs_eqs[i] for i in range(n_high_ff)])
    )
    print("Here are the constraint equations, one for each b_i:")
    for high_var, rhs_eq in zip(high_vars, rhs_eqs):
        sp.pprint(sp.Eq(high_var, rhs_eq, evaluate=False))

    # sp.pprint(sp.Array(rhs_eqs))
    print("Note that the free vars are:")
    sp.pprint(free_high_vars)

    # do the zero pad embeddings varphi. # (n_filters,spatial_3d,tensor_2d)
    varphi_low_filters = varphi_spatial(lowD, highD, low_filters)
    sp_varphi_low_filters = sp.Array(np.array(varphi_low_filters))
    varphi_low_filters = sp.Array(
        [(ai * ff).tolist() for ai, ff in zip(low_vars, sp_varphi_low_filters)]
    )

    pi_n_high_filters = pi_n(high_filters, lowD, len(k))
    sp_pi_n_high_filters = sp.Array(np.array(pi_n_high_filters))
    sp_pi_n_high_filters = sp.Array(
        [(bi * ff).tolist() for bi, ff in zip(rhs_eqs, sp_pi_n_high_filters)]
    )

    assert varphi_low_filters.shape[1:] == sp_pi_n_high_filters.shape[1:]

    # throws an error but this works
    summed_varphi_low_filters = varphi_low_filters[0]
    assert isinstance(summed_varphi_low_filters, sp.ImmutableDenseNDimArray)
    for i in range(1, varphi_low_filters.shape[0]):
        summed_varphi_low_filters += varphi_low_filters[i]

    summed_pi_n_high_filters = sp_pi_n_high_filters[0]
    assert isinstance(summed_pi_n_high_filters, sp.ImmutableDenseNDimArray)
    for i in range(1, sp_pi_n_high_filters.shape[0]):
        summed_pi_n_high_filters += sp_pi_n_high_filters[i]

    assert (
        summed_varphi_low_filters.shape == summed_pi_n_high_filters.shape
    ), f"{summed_varphi_low_filters.shape} != {summed_pi_n_high_filters.shape}"

    diff = summed_varphi_low_filters - summed_pi_n_high_filters
    total = sum(sp.Matrix(diff.reshape(math.prod(diff.shape), 1)).applyfunc(lambda x: x**2))
    print("Here is the squared filter difference quantity added into a single equation:")
    reduced_total = sp.expand(total)
    sp.pprint(reduced_total)

    for free_high_var in free_high_vars:
        print(
            f"The derivative w.r.t. {sp.pretty(free_high_var)} is: "
            f"{sp.pretty(sp.diff(reduced_total, free_high_var))}"
        )

    print(
        "To finish, set these equal to 0 and solve for the free variable. "
        "This may require solving a system of equations.\n"
    )


# MAIN
sp.init_printing(use_unicode=True)

conv_filters_1d = geom.get_invariant_filters(
    [3], [0], [0], 1, geom.make_all_operators(1), scale=geom.FilterScaling.ONE
)
conv_filters_2d = geom.get_invariant_filters(
    [3], [0, 1, 2], [0], 2, geom.make_all_operators(2), scale=geom.FilterScaling.ONE
)
conv_filters_3d = geom.get_invariant_filters(
    [3], [0, 1, 2], [0], 3, geom.make_all_operators(3), scale=geom.FilterScaling.ONE
)

# 1D -> 2D
for k_p, ff_low in conv_filters_1d.items():
    compute_constraints(1, 2, k_p, ff_low, conv_filters_2d[k_p], 0)

# 2D -> 3D
for k_p, ff_low in conv_filters_2d.items():
    k = len(k_p[0])
    for input_k in range(k + 1):
        compute_constraints(2, 3, k_p, ff_low, conv_filters_3d[k_p], input_k)
