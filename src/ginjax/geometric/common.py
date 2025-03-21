from __future__ import annotations
from typing import Optional, Sequence

import itertools as it
import numpy as np

import jax.numpy as jnp
import jax.lax
import jax

from ginjax.geometric.constants import TINY
from ginjax.geometric.geometric_image import GeometricImage, GeometricFilter
from ginjax.geometric.multi_image import MultiImage
from ginjax.geometric.functional_geometric_image import times_group_element

# ------------------------------------------------------------------------------
# PART 1: Make and test a complete group


def permutation_matrix_from_sequence(seq: Sequence[int]) -> np.ndarray:
    """
    Give a sequence tuple, return the permutation matrix for that sequence

    args:
        seq: the sequence

    returns:
        the permutation matrix of that sequence
    """
    D = len(seq)
    permutation_matrix = []
    for num in seq:
        row = [0] * D
        row[num] = 1
        permutation_matrix.append(row)
    return np.array(permutation_matrix)


def make_all_operators(D: int) -> list[np.ndarray]:
    """
    Construct all operators of dimension D that are rotations of 90 degrees, or reflections, or a combination of the
    two. This is equivalent to all the permutation matrices where each entry can either be +1 or -1

    args:
        D: dimension of the operator

    returns:
        the operators as a list of arrays
    """

    # permutation matrices, one for each permutation of length D
    permutation_matrices = [
        permutation_matrix_from_sequence(seq) for seq in it.permutations(range(D))
    ]
    # possible entries, e.g. for D=2: (1,1), (-1,1), (1,-1), (-1,-1)
    possible_entries = [np.diag(prod) for prod in it.product([1, -1], repeat=D)]

    # combine all the permutation matrices with the possible entries, then flatten to a single array of operators
    return list(
        it.chain(
            *list(
                map(
                    lambda matrix: [matrix @ prod for prod in possible_entries],
                    permutation_matrices,
                )
            )
        )
    )


# ------------------------------------------------------------------------------
# PART 2: Use group averaging to find unique invariant filters.

basis_cache = {}


def get_basis(key: str, shape: tuple[int, ...]) -> jax.Array:
    """
    Return a basis for the given shape. Bases are cached so we only have to calculate them once. The
    result will be a jnp.array of shape (len, shape) where len is the shape all multiplied together.

    args:
        key: basis cache key for this basis, will be combined with the shape
        shape: the shape of the basis

    returns:
        the basis
    """
    actual_key = key + ":" + str(shape)
    if actual_key not in basis_cache:
        size = np.multiply.reduce(shape)
        basis_cache[actual_key] = jnp.eye(size).reshape((size,) + shape)

    return basis_cache[actual_key]


def get_unique_invariant_filters(
    M: int,
    k: int,
    parity: int,
    D: int,
    operators: Sequence[np.ndarray],
    scale: str = "normalize",
) -> list[GeometricFilter]:
    """
    Use group averaging to generate all the unique invariant filters

    args:
        M: filter side length
        k: tensor order
        parity:  0 or 1, 0 is for normal tensors, 1 for pseudo-tensors
        D: image dimension
        operators: array of operators of a group
        scale: option for scaling the values of the filters, 'normalize' (default) to make amplitudes of each
            tensor +/- 1. 'one' to set them all to 1.

    returns:
        the unique invariant filters
    """
    assert scale == "normalize" or scale == "one"

    # make the seed filters
    shape = (M,) * D + (D,) * k

    basis = get_basis("image", shape)  # (N**D * D**k, (N,)*D, (D,)*k)
    # not a true vmap because we can't vmap over the operators, but equivalent (if slower)
    vmap_times_group = lambda ff, precision: jnp.stack(
        [times_group_element(D, ff, parity, gg, precision) for gg in operators]
    )
    # vmap over the elements of the basis
    group_average = jax.vmap(
        lambda ff: jnp.sum(vmap_times_group(ff, jax.lax.Precision.HIGH), axis=0)
    )
    filter_matrix = group_average(basis).reshape(len(basis), -1)

    # remove rows of all zeros
    filter_matrix = filter_matrix[jnp.sum(jnp.abs(filter_matrix), axis=1) != 0.0]
    # get the leading signs of each row
    leading_signs = jnp.sign(
        filter_matrix[(jnp.arange(len(filter_matrix)), jnp.argmax(filter_matrix != 0, axis=1))]
    )
    # set the leading signs to positive
    filter_matrix = filter_matrix * leading_signs[:, None]
    # jax unique has issues (https://github.com/jax-ml/jax/issues/17370), do it with numpy
    amps = jnp.array(np.unique(np.array(filter_matrix), axis=0))

    # set the amps to generally positive
    signs = jnp.sign(jnp.sum(amps, axis=1, keepdims=True))
    signs = jnp.where(
        signs == 0, jnp.ones(signs.shape), signs
    )  # if signs is 0, just want to multiply by 1
    amps = amps * signs

    # scale the largest value to 1
    amps /= jnp.max(jnp.abs(amps), axis=1, keepdims=True)

    # order them
    filters = [GeometricFilter(aa.reshape(shape), parity, D) for aa in amps]
    if scale == "normalize":
        filters = [ff.normalize() for ff in filters]

    norms = [ff.bigness() for ff in filters]
    I = np.argsort(norms)
    filters = [filters[i] for i in I]

    # now do k-dependent rectification:
    filters = [ff.rectify() for ff in filters]

    return filters


def get_invariant_filters_dict(
    Ms: Sequence[int],
    ks: Sequence[int],
    parities: Sequence[int],
    D: int,
    operators: Sequence[np.ndarray],
    scale: str = "normalize",
) -> tuple[dict[tuple[int, int, int, int], list[GeometricFilter]], dict[tuple[int, int], int]]:
    """
    Use group averaging to generate all the unique invariant filters for the ranges of Ms, ks, and
    parities. Returns the filters as dictionary along with a dictionary of the number of filters of
    each type.

    args:
        Ms: filter side lengths
        ks: tensor orders
        parities:  0 or 1, 0 is for normal tensors, 1 for pseudo-tensors
        D: image dimension
        operators: array of operators of a group
        scale: option for scaling the values of the filters, 'normalize' (default) to make
            amplitudes of each tensor +/- 1. 'one' to set them all to 1.

    returns:
        allfilters: a dictionary of filters of the specified D, M, k, and parity
        maxn: a dictionary that tracks the longest number of filters per key, for a particular D,M combo.
    """
    assert scale == "normalize" or scale == "one"

    allfilters = {}
    maxn = {}
    for M in Ms:  # filter side length
        maxn[(D, M)] = 0
        for k in ks:  # tensor order
            for parity in parities:  # parity
                key = (D, M, k, parity)
                allfilters[key] = get_unique_invariant_filters(M, k, parity, D, operators, scale)
                n = len(allfilters[key])
                if n > maxn[(D, M)]:
                    maxn[(D, M)] = n

    if allfilters == {}:
        print(
            f"WARNING get_invariant_filters_dict(Ms={Ms}, ks={ks}, parities={parities}, D={D}): No invariant filters."
        )

    return allfilters, maxn


def get_invariant_filters_list(
    Ms: Sequence[int],
    ks: Sequence[int],
    parities: Sequence[int],
    D: int,
    operators: Sequence[np.ndarray],
    scale: str = "normalize",
) -> list[GeometricFilter]:
    """
    Use group averaging to generate all the unique invariant filters for the ranges of Ms, ks, and
    parities. Returns the filters as a single list.

    args:
        Ms: filter side lengths
        ks: tensor orders
        parities:  0 or 1, 0 is for normal tensors, 1 for pseudo-tensors
        D: image dimension
        operators: array of operators of a group
        scale: option for scaling the values of the filters, 'normalize' (default) to make
            amplitudes of each tensor +/- 1. 'one' to set them all to 1.

    returns:
        a list of filters of the specified D, M, k, and parity
    """
    allfilters, _ = get_invariant_filters_dict(Ms, ks, parities, D, operators, scale)
    return list(it.chain(*list(allfilters.values())))  # list of GeometricFilters


def get_invariant_filters(
    Ms: Sequence[int],
    ks: Sequence[int],
    parities: Sequence[int],
    D: int,
    operators: Sequence[np.ndarray],
    scale: str = "normalize",
) -> MultiImage:
    """
    Use group averaging to generate all the unique invariant filters for the ranges of Ms, ks, and
    parities. Returns the filters as a single list.

    args:
        Ms: filter side lengths
        ks: tensor orders
        parities:  0 or 1, 0 is for normal tensors, 1 for pseudo-tensors
        D: image dimension
        operators: array of operators of a group
        scale: option for scaling the values of the filters, 'normalize' (default) to make
            amplitudes of each tensor +/- 1. 'one' to set them all to 1.

    returns:
        the filter of the specified D, M, k, and parity as a MultiImage
    """
    allfilters_list = get_invariant_filters_list(Ms, ks, parities, D, operators, scale)
    return MultiImage.from_images(allfilters_list)


def tensor_name(k: int, parity: int) -> str:
    """
    Return the given tensor name for the specified tensor order and parity.

    args:
        k: tensor order
        parity: tensor parity, either 0 or 1

    returns:
        a string of the tensor name
    """
    nn = "tensor"
    if k == 0:
        nn = "scalar"
    if k == 1:
        nn = "vector"
    if parity % 2 == 1 and k < 2:
        nn = "pseudo" + nn
    if k > 1:
        if parity == 0:
            nn = r"${}_{}-$".format(k, "{(+)}") + nn
        else:
            nn = r"${}_{}-$".format(k, "{(-)}") + nn

    return nn
