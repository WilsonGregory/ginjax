from __future__ import annotations
from typing import Optional, Sequence

import itertools as it
import numpy as np

import jax.numpy as jnp
import jax.lax
import jax

from ginjax.geometric.constants import TINY, LeviCivitaSymbol
from ginjax.geometric.geometric_image import GeometricImage, GeometricFilter
from ginjax.geometric.multi_image import MultiImage
from ginjax.geometric.functional_geometric_image import times_group_element, times_D8_element

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


def make_D8_group(D: int) -> list[np.ndarray]:
    """
    Construct D_8, the Dihedral group with 16 elements, aka rotations of 45 degrees and flips.
    In D=2 this is the symmetries of an octagon.
    """
    if D == 1:
        return make_C2_group(D)
    elif D == 2:
        ggs = []
        for i in range(8):
            theta = 2 * jnp.pi * i / 8
            ggs.append(
                np.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
            )

        for i in range(8):
            theta = 2 * jnp.pi * i / 8
            ggs.append(
                np.array([[jnp.cos(theta), jnp.sin(theta)], [jnp.sin(theta), -jnp.cos(theta)]])
            )

        return ggs
    else:
        raise NotImplementedError


def make_C2_group(D: int) -> list[np.ndarray]:
    """
    Construct the group C2 x C2 x ... x C2, D times. On a D-dimensional space this is the group
    which flips each axis.

    args:
        D: the dimension of the space

    returns:
        the operators as a list of numpy arrays
    """
    return [np.diag(prod) for prod in it.product([1, -1], repeat=D)]


# ------------------------------------------------------------------------------
# PART 2: Use group averaging to find unique invariant filters.

basis_cache = {}


def get_k2_irrep_basis(M: int, k: int, D: int):
    shape = (M,) * D + (D,) * k
    actual_key = "k2_irrep_basis:" + str(shape)
    if actual_key not in basis_cache:
        if D == 1:
            basis = jnp.eye(M**D).reshape((M**D,) + (M,) * D + (D,) * k)
        elif D == 2:
            # this specific basis is for the irreducibles of O(2), and is nicer for visualizing
            # the basis elements of that one
            levi_civita = LeviCivitaSymbol.get(D)
            basis = []
            for i in range(M):
                for j in range(M):
                    # kronecker delta coefficient (trace)
                    elem = np.zeros((M,) * D + (D,) * k)
                    elem[i, j] = np.eye(D)
                    basis.append(elem)

                    # levi civita coefficient (antisymmetric matrix)
                    elem = np.zeros((M,) * D + (D,) * k)
                    elem[i, j] = levi_civita
                    basis.append(elem)

                    # symmetric traceless matrix, diagonal
                    elem = np.zeros((M,) * D + (D,) * k)
                    elem[i, j, 0, 0] = 1
                    elem[i, j, 1, 1] = -1
                    basis.append(elem)

                    # symmetric traceless matrix, off-diagonal
                    elem = np.zeros((M,) * D + (D,) * k)
                    elem[i, j, 0, 1] = 1
                    elem[i, j, 1, 0] = 1
                    basis.append(elem)

            basis = jnp.stack(basis)
        elif D == 3:
            basis = []
            for i in range(M):
                for j in range(M):
                    for l in range(M):
                        # kronecker delta coefficient (trace)
                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l] = np.eye(D)
                        basis.append(elem)

                        # levi civita coefficient (antisymmetric matrix)
                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l, 0, 1] = 1
                        elem[i, j, l, 1, 0] = -1
                        basis.append(elem)

                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l, 0, 2] = 1
                        elem[i, j, l, 2, 0] = -1
                        basis.append(elem)

                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l, 1, 2] = 1
                        elem[i, j, l, 2, 1] = -1
                        basis.append(elem)

                        # symmetric traceless matrix, diagonal
                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l, 0, 0] = 1
                        elem[i, j, l, 2, 2] = -1
                        basis.append(elem)

                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l, 1, 1] = 1
                        elem[i, j, l, 2, 2] = -1
                        basis.append(elem)

                        # symmetric traceless matrix, off-diagonal
                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l, 0, 1] = 1
                        elem[i, j, l, 1, 0] = 1
                        basis.append(elem)

                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l, 0, 2] = 1
                        elem[i, j, l, 2, 0] = 1
                        basis.append(elem)

                        elem = np.zeros((M,) * D + (D,) * k)
                        elem[i, j, l, 1, 2] = 1
                        elem[i, j, l, 2, 1] = 1
                        basis.append(elem)

            basis = jnp.stack(basis)
        else:
            raise NotImplementedError(f"k2_irrep_basis only implemented for D=2,3, but got D={D}")

        basis_cache[actual_key] = basis

    return basis_cache[actual_key]


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
    exclude_corners: bool = False,
    k2_irreps_basis: bool = True,
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
        exclude_corners: if true, only keep filters that are copies/rotations of D=1 filters. This
            ensures that D=100 has the same number of filters as D=1. Defaults to False.
        k2_irreps_basis: for D=2, k=2 filters, use the irreps basis. Defaults to True.

    returns:
        the unique invariant filters
    """
    assert scale in {"normalize", "one", "gaussian", "zero_sum"}

    # make the seed filters
    shape = (M,) * D + (D,) * k

    # (N**D * D**k, (N,)*D, (D,)*k)
    basis = (
        get_k2_irrep_basis(M, k, D) if (k == 2 and k2_irreps_basis) else get_basis("image", shape)
    )

    # not a true vmap because we can't vmap over the operators, but equivalent (if slower)
    # covariant axes should maybe be true? For G = O(D), they are equivalent.
    vmap_times_group = lambda ff: jnp.stack(
        [
            times_group_element(D, ff, parity, gg, (False,) * k, jax.lax.Precision.HIGHEST)
            # times_D8_element(D, ff, parity, gg, (False,) * k, jax.lax.Precision.HIGHEST)
            for gg in operators
        ]
    )
    # vmap over the elements of the basis
    group_average = jax.vmap(lambda ff: jnp.sum(vmap_times_group(ff), axis=0))
    filter_matrix = group_average(basis).reshape(len(basis), -1)

    # remove rows of all zeros
    filter_matrix = filter_matrix[
        ~jnp.isclose(jnp.sum(jnp.abs(filter_matrix), axis=1), 0.0, rtol=TINY, atol=TINY)
    ]
    # Scale filters so that they all add up to 1
    filter_matrix /= jnp.sum(jnp.abs(filter_matrix), axis=1, keepdims=True)
    # D4 operators are only +/- 1, but D8 are fractions so tiny values distinct from 0 are there
    filter_matrix = jnp.round(filter_matrix, 5)
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
    filters = sorted([GeometricFilter(aa.reshape(shape), parity, D) for aa in amps])

    # now do k-dependent rectification:
    filters = [ff.rectify() for ff in filters]

    if D > 1 and exclude_corners:
        assert (M % 2) == 1  # currently can only handle odd filters for cornerless
        m = (M - 1) // 2
        meshgrid_dims = (jnp.arange(-m, m + 1),) * D
        idxs = jnp.stack(jnp.meshgrid(*meshgrid_dims, indexing="ij"), axis=-1).reshape((-1, D))
        corner_idxs = jnp.sum(idxs != 0, axis=1) > 1  # central idxs have at most 1 nonzero

        cornerless_filters = []
        for ff in filters:
            if jnp.allclose(ff.data.reshape((M**D, D**k))[corner_idxs, :], 0):
                cornerless_filters.append(ff)

        filters = cornerless_filters

    if len(filters) > 0:
        if scale == "one":
            filters = [ff * float(1 / jnp.max(jnp.abs(ff.data))) for ff in filters]
        if scale == "normalize":
            filters = [ff.normalize() for ff in filters]
        elif scale == "gaussian":
            filters = [ff.normalize() for ff in filters]  # first set the norms to 1
            # scale according to the rbf kernel, or like a multivariate gaussian
            meshgrid_dims = tuple(jnp.arange(M1) for M1 in filters[0].image_shape())
            idxs = jnp.stack(jnp.meshgrid(*meshgrid_dims, indexing="ij"), axis=-1).reshape((-1, D))
            idxs -= jnp.array(filters[0].image_shape()) / 2
            dist_scaling = jnp.exp(-0.25 * (jnp.linalg.norm(idxs, axis=1) ** 2))
            # I should maybe account for the fact that in k=2, some pixels have multiple filters
            nonempty_pixels = jnp.any(
                jnp.stack(
                    [
                        jnp.any(~jnp.isclose(ff.data.reshape((M**D, D**k)), 0), axis=1)
                        for ff in filters
                    ],
                    axis=1,
                ),
                axis=1,
            ).astype(int)
            gaussian_sum = jnp.sum(dist_scaling * nonempty_pixels)
            normalized_dist_scaling = GeometricFilter(
                dist_scaling.reshape((M,) * D) / gaussian_sum, 0, D
            )
            filters = [ff * normalized_dist_scaling for ff in filters]
        elif scale == "zero_sum":
            filters = [ff.normalize() for ff in filters]  # first set the norms to 1

            assert k < 2, f"zero_sum only currently valid for k equals 0 or 1, but got {k}"

            # George's recommended scaling. Sum of the filters has to equal 0.

            # vector filters already add up to 0
            if k == 0:  # scalar
                center_ff = filters[0] * -1

                # (M**D,offcenter_filters)
                nonempty_pixels = jnp.stack(
                    [
                        jnp.any(~jnp.isclose(ff.data.reshape((M**D, D**k)), 0), axis=1)
                        for ff in filters[1:]
                    ],
                    axis=1,
                )

                pixel_sum = int(jnp.sum(nonempty_pixels))
                offcenter_filters = [ff * (1 / pixel_sum) for ff in filters[1:]]

                filters = [center_ff] + offcenter_filters

            filter_sum = jnp.sum(
                jnp.stack([ff.data.reshape((M**D,) + (D,) * ff.k) for ff in filters], axis=0),
                axis=(0, 1),
            )
            assert jnp.allclose(filter_sum, 0, rtol=TINY, atol=TINY)

    return filters


def get_invariant_filters_dict(
    Ms: Sequence[int],
    ks: Sequence[int],
    parities: Sequence[int],
    D: int,
    operators: Sequence[np.ndarray],
    scale: str = "normalize",
    exclude_corners: bool = False,
    k2_irreps_basis: bool = True,
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
        exclude_corners: if true, only keep filters that are copies/rotations of D=1 filters. This
            ensures that D=100 has the same number of filters as D=1. Defaults to False.
        k2_irreps_basis: for D=2, k=2 filters, use the irreps basis. Defaults to True.

    returns:
        allfilters: a dictionary of filters of the specified D, M, k, and parity
        maxn: a dictionary that tracks the longest number of filters per key, for a particular D,M combo.
    """
    assert scale in {"normalize", "one", "gaussian", "zero_sum"}

    allfilters = {}
    maxn = {}
    for M in Ms:  # filter side length
        maxn[(D, M)] = 0
        for k in ks:  # tensor order
            for parity in parities:  # parity
                key = (D, M, k, parity)
                allfilters[key] = get_unique_invariant_filters(
                    M, k, parity, D, operators, scale, exclude_corners, k2_irreps_basis
                )
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
    exclude_corners: bool = False,
    k2_irreps_basis: bool = True,
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
        exclude_corners: if true, only keep filters that are copies/rotations of D=1 filters. This
            ensures that D=100 has the same number of filters as D=1. Defaults to False.
        k2_irreps_basis: for D=2, k=2 filters, use the irreps basis. Defaults to True.

    returns:
        a list of filters of the specified D, M, k, and parity
    """
    allfilters, _ = get_invariant_filters_dict(
        Ms, ks, parities, D, operators, scale, exclude_corners, k2_irreps_basis
    )
    return list(it.chain(*list(allfilters.values())))  # list of GeometricFilters


def get_invariant_filters(
    Ms: Sequence[int],
    ks: Sequence[int],
    parities: Sequence[int],
    D: int,
    operators: Sequence[np.ndarray],
    scale: str = "normalize",
    exclude_corners: bool = False,
    k2_irreps_basis: bool = True,
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
        exclude_corners: if true, only keep filters that are copies/rotations of D=1 filters. This
            ensures that D=100 has the same number of filters as D=1. Defaults to False.
        k2_irreps_basis: for D=2, k=2 filters, use the irreps basis. Defaults to True.

    returns:
        the filter of the specified D, M, k, and parity as a MultiImage
    """
    allfilters_list = get_invariant_filters_list(
        Ms, ks, parities, D, operators, scale, exclude_corners, k2_irreps_basis
    )
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
