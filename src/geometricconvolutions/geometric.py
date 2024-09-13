"""
# Core code for GeometricConvolutions

## License:
Copyright 2022 David W. Hogg and contributors.
The code in GeometricConvolutions is licensed under the open-source MIT License.
See the file `LICENSE` for more details.

## Authors:
- David W. Hogg (NYU)
- Wilson Gregory (JHU)

## To-do items:
- Fix the norm() operations so they are makeable with index summations! Duh. sqrt(a_hij a_hij / d^(k-2)) maybe??
- Fix sizing of multi-filter plots.
- Need to implement bin-down and bin-up operators.
"""

from __future__ import annotations
from typing import Union, Sequence, Optional, Generator, Any, Callable
from typing_extensions import Self

import itertools as it
import numpy as np  # removing this
from functools import partial, reduce
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.lax
import jax.nn
import jax
from jax import jit, vmap
from jax.tree_util import register_pytree_node_class

import geometricconvolutions.utils as utils

TINY = 1.0e-5
LETTERS = "abcdefghijklmnopqrstuvwxyxABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ------------------------------------------------------------------------------
# PART 1: Make and test a complete group


def permutation_matrix_from_sequence(seq: Sequence[int]) -> np.ndarray:
    """
    Give a sequence tuple, return the permutation matrix for that sequence
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
        D (int): dimension of the operator
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
# PART 2: Define the Kronecker Delta and Levi Civita symbols to be used in Levi Civita contractions


class KroneckerDeltaSymbol:
    # we only want to create each dimension of levi civita symbol once, so we cache them in this dictionary
    symbol_dict = {}

    @classmethod
    def get(cls, D: int, k: int) -> np.ndarray:
        """
        Get the Levi Civita symbol for dimension D from the cache, or creating it on a cache miss
        args:
            D (int): dimension of the Kronecker symbol
            k (int): order of the Kronecker Delta symbol
        """
        assert D > 1
        assert k > 1
        if (D, k) not in cls.symbol_dict:
            arr = np.zeros((k * (D,)), dtype=int)
            for i in range(D):
                arr[(i,) * k] = 1
            cls.symbol_dict[(D, k)] = arr

        return cls.symbol_dict[(D, k)]

    @classmethod
    def get_image(cls, N: int, D: int, k: int) -> GeometricImage:
        return GeometricImage(
            jnp.stack([cls.get(D, k) for _ in range(N**D)]).reshape(((N,) * D + (D,) * k)),
            0,
            D,
        )


def permutation_parity(pi: Sequence[int]) -> int:
    """
    Code taken from Sympy Permutations: https://github.com/sympy/sympy/blob/26f7bdbe3f860e7b4492e102edec2d6b429b5aaf/sympy/combinatorics/permutations.py#L114
    Slightly modified to return 1 for even permutations, -1 for odd permutations, and 0 for repeated digits
    Permutations of length n must consist of numbers {0, 1, ..., n-1}
    """
    if len(np.unique(pi)) != len(pi):
        return 0

    n = len(pi)
    a = [0] * n
    c = 0
    for j in range(n):
        if a[j] == 0:
            c += 1
            a[j] = 1
            i = j
            while pi[i] != j:
                i = pi[i]
                a[i] = 1

    # code originally returned 1 for odd permutations (we want -1) and 0 for even permutations (we want 1)
    return -2 * ((n - c) % 2) + 1


class LeviCivitaSymbol:

    # we only want to create each dimension of levi civita symbol once, so we cache them in this dictionary
    symbol_dict = {}

    @classmethod
    def get(cls, D: int) -> jnp.ndarray:
        """
        Get the Levi Civita symbol for dimension D from the cache, or creating it on a cache miss
        args:
            D (int): dimension of the Levi Civita symbol
        """
        assert D > 1
        if D not in cls.symbol_dict:
            arr = np.zeros((D * (D,)), dtype=int)
            for index in it.product(range(D), repeat=D):
                arr[index] = permutation_parity(index)
            cls.symbol_dict[D] = jnp.array(arr)

        return cls.symbol_dict[D]


# ------------------------------------------------------------------------------
# PART 3: Use group averaging to find unique invariant filters.

basis_cache = {}


def get_basis(key: str, shape: tuple[int]) -> jnp.ndarray:
    """
    Return a basis for the given shape. Bases are cached so we only have to calculate them once. The
    result will be a jnp.array of shape (len, shape) where len is the shape all multiplied together.
    args:
        key (string): basis cache key for this basis, will be combined with the shape
        shape (tuple of ints): the shape of the basis
    """
    actual_key = key + ":" + str(shape)
    if actual_key not in basis_cache:
        size = np.multiply.reduce(shape)
        basis_cache[actual_key] = jnp.eye(size).reshape((size,) + shape)

    return basis_cache[actual_key]


def get_operators_on_coeffs(
    D: int, operators: Sequence[np.ndarray], library: jnp.ndarray
) -> jnp.ndarray:
    """
    Given the operators of a group and a library of vector image basis functions, find the action of
    the group on the coefficients of the library of basis functions.
    args:
        D (int): dimension of the space
        operators (list of jnp.array): list of group operators, each is shape (D, D)
        library (jnp.array): basis functions, shape (N**D, num_coeffs)
    returns: list of jnp.array, each is shape (num_coeffs*D, num_coeffs*D)
    """
    num_coeffs = library.shape[1]
    library_N = int(jnp.power(library.shape[0], 1.0 / D))  # N is the 1/D th root.
    library_pinv = jnp.linalg.pinv(
        library
    )  # library is (N**D, num_coeffs), pinv is (num_coeffs, N**D)

    def action(basis_element, gg):
        vec_img = library @ basis_element.reshape((num_coeffs, D))  # (N**D, D)
        vec_img = vec_img.reshape((library_N,) * D + (D,))

        rotated_img = times_group_element(D, vec_img, 0, gg, jax.lax.Precision.HIGHEST).reshape(
            (library_N**D, D)
        )
        rotated_coeffs = library_pinv @ rotated_img  # (num_coeffs, D)
        # this numerical method is a little messy, but in actuality the rotation matrix on the
        # coefficients will all be either 1s or -1s. Thus we aggressively round them off.
        return (jnp.round(rotated_coeffs, decimals=2) + 0.0).reshape(-1)

    vmap_action = vmap(action, in_axes=(0, None))
    # the output vector of action is a column of the operator, so we take the transpose
    return [vmap_action(jnp.eye(num_coeffs * D), gg).transpose() for gg in operators]


def get_operators_on_layer(operators: Sequence[np.ndarray], layer: Layer) -> jnp.ndarray:
    """
    Given the operators of a group and a Layer, find the action of the group on the vectorized version
    of the layer. That is, the output `gg_out` is such that:
    gg_out @ layer.to_vector() == layer.times_group_element(gg)
    args:
        operators (list of jnp.arrays): group operators, each is shape (D, D)
        layer (Layer): can be a layer of any shape, just cannot be a BatchLayer
    returns: list of jnp.array, each is shape (num_coeffs*D, num_coeffs*D)
    """
    basis_len = layer.size()
    layer_basis = vmap(lambda e: layer.__class__.from_vector(e, layer))(jnp.eye(basis_len))

    vmap_times_gg = vmap(
        lambda gg, e: e.times_group_element(gg, jax.lax.Precision.HIGHEST).to_vector(),
        in_axes=(None, 0),
    )

    # the output of the vmap is a column of the operator, so we take the transpose
    return [vmap_times_gg(gg, layer_basis).transpose() for gg in operators]


def get_equivariant_map_to_coeffs(
    layer: Layer, operators: Sequence[np.ndarray], library: jnp.ndarray
) -> jnp.ndarray:
    """
    Find the linear maps from a layer of the specified shape to the coefficients of a basis of
    vector images given by library that is equivariant to the operators of some group.
    args:
        layer (Layer): a layer of any shape, but must be a Layer and not a BatchLayer
        operators (list): list of operators given by make_all_operators, each is (D,D)
        library (jnp.array): library of vector image basis, shape (N**D, num_coeffs)
    """
    # First, we construct basis of layer elements
    num_coeffs = library.shape[1]
    basis_len = layer.size()

    # Get the representation of the group operators on the specified layer
    operators_on_layer = jnp.stack(get_operators_on_layer(operators, layer))

    # Get the representation of the group on the coefficients of the library
    operators_on_coeffs = jnp.stack(get_operators_on_coeffs(layer.D, operators, library))

    # Now get a basis of the linear maps, and do the group averaging
    lin_map_basis = get_basis("equivariant_linear_map", (num_coeffs * layer.D, basis_len))
    conjugate = vmap(
        vmap(
            lambda gg_coeffs, gg_layer, basis_elem: gg_coeffs.T @ basis_elem @ gg_layer,
            in_axes=(0, 0, None),
        ),
        in_axes=(None, None, 0),
    )

    # before sum, (N**D * num_coeffs * D, group_size, num_coeffs*D, basis_len)
    maps_matrix = jnp.sum(conjugate(operators_on_coeffs, operators_on_layer, lin_map_basis), axis=1)
    maps_matrix = maps_matrix.reshape((len(maps_matrix), -1))

    # do the SVD
    _, s, v = jnp.linalg.svd(maps_matrix)
    sbig = s > TINY
    if not jnp.any(sbig):
        return []

    return v[sbig].reshape((len(v[sbig]), num_coeffs * layer.D, basis_len))


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
        M (int): filter side length
        k (int): tensor order
        parity (int):  0 or 1, 0 is for normal tensors, 1 for pseudo-tensors
        D (int): image dimension
        operators (jnp-array): array of operators of a group
        scale (string): option for scaling the values of the filters, 'normalize' (default) to make amplitudes of each
        tensor +/- 1. 'one' to set them all to 1.
    """
    assert scale == "normalize" or scale == "one"

    # make the seed filters
    shape = (M,) * D + (D,) * k
    operators = jnp.stack(operators)

    basis = get_basis("image", shape)  # (N**D * D**k, (N,)*D, (D,)*k)
    # not a true vmap because we can't vmap over the operators, but equivalent (if slower)
    vmap_times_group = lambda ff, precision: jnp.stack(
        [times_group_element(D, ff, parity, gg, precision) for gg in operators]
    )
    # vmap over the elements of the basis
    group_average = vmap(lambda ff: jnp.sum(vmap_times_group(ff, jax.lax.Precision.HIGH), axis=0))
    filter_matrix = group_average(basis).reshape(len(basis), -1)

    # do the SVD
    _, s, v = np.linalg.svd(filter_matrix)
    sbig = s > TINY
    if not np.any(sbig):
        return []

    # normalize the amplitudes so they max out at +/- 1.
    amps = v[sbig] / jnp.max(jnp.abs(v[sbig]), axis=1, keepdims=True)
    # make sure the amps are positive, generally
    signs = jnp.sign(jnp.sum(amps, axis=1, keepdims=True))
    signs = jnp.where(
        signs == 0, jnp.ones(signs.shape), signs
    )  # if signs is 0, just want to multiply by 1
    amps *= signs
    # make sure that the zeros are zeros.
    amps = jnp.round(amps, decimals=5) + 0.0

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


def get_invariant_filters(
    Ms: Sequence[int],
    ks: Sequence[int],
    parities: Sequence[int],
    D: int,
    operators: Sequence[np.ndarray],
    scale: str = "normalize",
    return_type: str = "layer",
    return_maxn: bool = False,
):
    """
    Use group averaging to generate all the unique invariant filters for the ranges of Ms, ks, and parities. By default
    it returns the filters in a dictionary with the key (D,M,k,parity), but flattens to a list if return_list=True
    args:
        Ms (iterable of int): filter side lengths
        ks (iterable of int): tensor orders
        parities (iterable of int):  0 or 1, 0 is for normal tensors, 1 for pseudo-tensors
        D (int): image dimension
        operators (jnp-array): array of operators of a group
        scale (string): option for scaling the values of the filters, 'normalize' (default) to make amplitudes of each
        tensor +/- 1. 'one' to set them all to 1.
        return_type (string): returns the filters as the dict, a list, or a Layer, defaults to layer
        return_maxn (bool): defaults to False, if true returns the length of the max list for each D, M
    returns:
        allfilters: a dictionary of filters of the specified D, M, k, and parity. If return_list=True, this is a list
        maxn: a dictionary that tracks the longest number of filters per key, for a particular D,M combo. Not returned
            if return_list=True
    """
    assert scale == "normalize" or scale == "one"
    assert return_type in {"dict", "list", "layer"}

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

    allfilters_list = list(it.chain(*list(allfilters.values())))
    if return_type == "list":
        allfilters = allfilters_list
    elif return_type == "layer":
        allfilters = Layer.from_images(allfilters_list)
    # else, allfilters is the default structure

    if return_maxn:
        return allfilters, maxn
    else:
        return allfilters


def get_invariant_image(
    N: int,
    D: int,
    k: int,
    parity: int = 0,
    is_torus: bool = True,
    data_only: bool = True,
) -> Union[jnp.ndarray, GeometricImage]:
    """
    Get the G_{N,D} invariant image
    """
    # is this assertion true for odd parity?
    assert (k % 2) == 0, "get_invariant_image: There only exists even tensor order invariant images"
    if parity != 0:
        raise Exception("get_invariant_image: Odd parity currently not implemented")

    images = [
        GeometricImage.fill(N, parity, D, KroneckerDeltaSymbol.get(D, 2), is_torus)
        for _ in range(k // 2)
    ]
    image = reduce(lambda a, b: a * b, images, GeometricImage.fill(N, parity, D, 1, is_torus))

    return image.data if data_only else image


def get_contraction_map(D: int, k: int, indices: tuple[tuple[int]]) -> jnp.ndarray:
    """
    Get the linear map of contracting a tensor. Since contractions of geometric images happen pixel wise,
    we only need this map to apply to every pixel (tensor), saving space over finding the entire map.
    args:
        D (int): dimension of the tensor
        k (int): order of the tensor
        indices (tuple of tuple of int pairs): the indices of one multicontraction
    """
    basis = get_basis("tensor", (D,) * k)

    out = vmap(multicontract, in_axes=(0, None))(basis, indices).reshape((len(basis), -1))
    return jnp.transpose(out)


# ------------------------------------------------------------------------------
# PART 4: Functional Programming GeometricImages
# This section contains pure functions of geometric images that allows easier use of JAX fundamentals
# such as vmaps, loops, jit, and so on. All functions in this section take in images as their jnp.array data
# only, and return them as that as well.


def parse_shape(shape: tuple[int], D: int) -> tuple[tuple[int], int]:
    """
    Given a geometric image shape and dimension D, return the sidelength tuple and tensor order k.
    args:
        shape (shape tuple): the shape of the data of a single geoemtric image
        D (int): dimension of the image
    """
    assert isinstance(shape, tuple), f"parse_shape: Shape must be a tuple, but it is {type(shape)}"
    assert len(shape) >= D, f"parse_shape: Shape {shape} is shorter than D={D}"
    return shape[:D], len(shape) - D


def hash(D: int, image: jnp.ndarray, indices: jnp.ndarray) -> tuple[jnp.ndarray]:
    """
    Deals with torus by modding (with `np.remainder()`).
    args:
        D (int): dimension of hte image
        image (jnp.array): image data
        indices (jnp.array): array of indices, shape (num_idx, D) to apply the remainder to
    """
    spatial_dims = jnp.array(parse_shape(image.shape, D)[0]).reshape((1, D))
    return tuple(jnp.remainder(indices, spatial_dims).transpose().astype(int))


def get_torus_expanded(
    D: int,
    image: jnp.ndarray,
    is_torus: tuple[bool],
    filter_spatial_dims: tuple[int],
    rhs_dilation: tuple[int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    For a particular filter, expand the image so that we no longer have to do convolutions on the torus, we are
    just doing convolutions on the expanded image and will get the same result. Return a new GeometricImage
    args:
        D (int): dimension of the image
        image (jnp.array): image data
        is_torus (tuple of bool): d-length tuple of bools specifying which spatial dimensions are toroidal
        filter_spatial_dims (tuple of ints): the spatial dimensions of the filter
        rhs_dilation (tuple of int): dilation to apply to each filter dimension D
    """
    spatial_dims, img_k = parse_shape(image.shape, D)
    # assert all the filter side lengths are odd
    assert reduce(lambda carry, M: carry and (M % 2 == 1), filter_spatial_dims, True)

    # for each torus dimension, calculate the torus padding
    padding_f = lambda M, dilation, torus: (((M - 1) // 2) * dilation) if torus else 0
    zipped_dims = zip(filter_spatial_dims, rhs_dilation, is_torus)
    torus_padding = tuple(padding_f(M, dilation, torus) for M, dilation, torus in zipped_dims)

    # calculate indices for torus padding, then use hash to select the appropriate pixels
    ranges = list(jnp.arange(-pad, N + pad) for N, pad in zip(spatial_dims, torus_padding))
    mesh = jnp.stack(jnp.meshgrid(*ranges, indexing="ij"), axis=-1)
    new_spatial_dims = mesh.shape[:D]
    indices = mesh.reshape((np.multiply.reduce(new_spatial_dims), D))
    expanded_image = image[hash(D, image, indices)].reshape(new_spatial_dims + (D,) * img_k)

    # zero_pad where we don't torus pad
    zero_padding = get_same_padding(
        filter_spatial_dims,
        rhs_dilation,
        tuple(not torus for torus in is_torus),
    )

    return expanded_image, zero_padding


def get_same_padding(
    filter_spatial_dims: tuple[int],
    rhs_dilation: tuple[int],
    pad_dims: Optional[tuple[bool]] = None,
) -> tuple[tuple[int]]:
    """
    Calculate the padding for each dimension D necessary for 'SAME' padding, including rhs_dilation.
    args:
        filter_spatial_dims (tuple of ints): filter spatial dimensions, length D tuple
        rhs_dilation (tuple of ints): rhs (filter) dilation, length D tuple
        pad_dims (tuple of bool): which dimensions to pad, defaults to None which is all dimensions
    """
    pad_dims = (True,) * len(filter_spatial_dims) if pad_dims is None else pad_dims
    padding_f = lambda M, dilation, pad: ((((M - 1) // 2) * dilation,) * 2 if pad else (0, 0))
    zipped_dims = zip(filter_spatial_dims, rhs_dilation, pad_dims)
    return tuple(padding_f(M, dilation, pad) for M, dilation, pad in zipped_dims)


def pre_tensor_product_expand(
    D: int,
    image_a: jnp.ndarray,
    image_b: jnp.ndarray,
    a_offset: int = 0,
    b_offset: int = 0,
    dtype: Optional[jnp.dtype] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rather than take a tensor product of two tensors, we can first take a tensor product of each with a tensor of
    ones with the shape of the other. Then we have two matching shapes, and we can then do whatever operations.
    args:
        D (int): dimension of the image
        image_a (GeometricImage like): one geometric image whose tensors we will later be doing tensor products on
        image_b (GeometricImage like): other geometric image
        dtype (dtype): if present, cast both outputs to dtype, defaults to None
    """
    _, img_a_k = parse_shape(image_a.shape[a_offset:], D)
    _, img_b_k = parse_shape(image_b.shape[b_offset:], D)

    if img_b_k > 0:
        image_a_expanded = jnp.tensordot(
            image_a,
            jnp.ones((D,) * img_b_k),
            axes=0,
        )
    else:
        image_a_expanded = image_a

    if img_a_k > 0:
        break1 = img_a_k + b_offset + D  # after outer product, end of image_b N^D axes
        # we want to expand the ones in the middle (D^ki), so add them on the front, then move to middle

        # (b_offset,b_spatial,b_tensor) -> (a_tensor,b_offset,b_spatial,b_tensor)
        image_b_expanded = jnp.tensordot(jnp.ones((D,) * img_a_k), image_b, axes=0)

        # (a_tensor,b_offset,b_spatial,b_tensor) -> (b_offset,b_spatial,a_tensor,b_tensor)
        idxs = (
            tuple(range(img_a_k, break1))
            + tuple(range(img_a_k))
            + tuple(range(break1, break1 + img_b_k))
        )
        image_b_expanded = image_b_expanded.transpose(idxs)
    else:
        image_b_expanded = image_b

    if dtype is not None:
        image_a_expanded = image_a_expanded.astype(dtype)
        image_b_expanded = image_b_expanded.astype(dtype)

    return image_a_expanded, image_b_expanded


def conv_contract_image_expand(D: int, image: jnp.ndarray, filter_k: int) -> jnp.ndarray:
    """
    For conv_contract, we will be immediately performing a contraction, so we don't need to fully expand
    each tensor, just the k image to the k+k' conv filter.
    args:
        D (int): dimension of the space
        image (jnp.array): image data, shape (in_c,spatial,tensor)
        filter_k (int): the filter tensor order
    """
    _, img_k = parse_shape(image.shape[2:], D)
    k_prime = filter_k - img_k  # not to be confused with Coach Prime
    assert k_prime >= 0

    return jnp.tensordot(image, jnp.ones((D,) * k_prime), axes=0)


def mul(
    D: int,
    image_a: jnp.ndarray,
    image_b: jnp.ndarray,
    a_offset: int = 0,
    b_offset: int = 0,
) -> jnp.ndarray:
    """
    Multiplication operator between two images, implemented as a tensor product of the pixels.
    args:
        D (int): dimension of the images
        image_a (jnp.array): image data
        image_b (jnp.array): image data
        a_offset (int): number of axes before the spatial axes (batch, channels, etc.), default 0
        b_offset (int): number of axes before the spatial axes (batch, channels, etc.), default 0
    """
    image_a_data, image_b_data = pre_tensor_product_expand(D, image_a, image_b, a_offset, b_offset)
    return image_a_data * image_b_data  # now that shapes match, do elementwise multiplication


@partial(jit, static_argnums=[0, 3, 4, 5, 6, 7, 8])
def convolve(
    D: int,
    image: jnp.ndarray,
    filter_image: jnp.ndarray,
    is_torus: Union[tuple[bool], bool],
    stride: Optional[tuple[int]] = None,
    padding: Optional[tuple[int]] = None,
    lhs_dilation: Optional[tuple[int]] = None,
    rhs_dilation: Optional[tuple[int]] = None,
    tensor_expand: bool = True,
) -> jnp.ndarray:
    """
    Here is how this function works:
    1. Expand the geom_image to its torus shape, i.e. add filter.m cells all around the perimeter of the image
    2. Do the tensor product (with 1s) to each image.k, filter.k so that they are both image.k + filter.k tensors.
    That is if image.k=2, filter.k=1, do (D,D) => (D,D) x (D,) and (D,) => (D,D) x (D,) with tensors of 1s
    3. Now we shape the inputs to work with jax.lax.conv_general_dilated
    4. Put image in NHWC (batch, height, width, channel). Thus we vectorize the tensor
    5. Put filter in HWIO (height, width, input, output). Input is 1, output is the vectorized tensor
    6. Plug all that stuff in to conv_general_dilated, and feature_group_count is the length of the vectorized
    tensor, and it is basically saying that each part of the vectorized tensor is treated separately in the filter.
    It must be the case that channel = input * feature_group_count
    See: https://jax.readthedocs.io/en/latest/notebooks/convolutions.html#id1 and
    https://www.tensorflow.org/xla/operation_semantics#conv_convolution

    args:
        D (int): dimension of the images
        image (jnp.array): image data, shape (batch,in_c,spatial,tensor)
        filter_image (jnp.array): the convolution filter, shape (out_c,in_c,spatial,tensor)
        is_torus (bool or tuple of bool): what dimensions of the image are toroidal
        stride (tuple of ints): convolution stride, defaults to (1,)*self.D
        padding (either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs):
            defaults to 'TORUS' if image.is_torus, else 'SAME'
        lhs_dilation (tuple of ints): amount of dilation to apply to image in each dimension D, also transposed conv
        rhs_dilation (tuple of ints): amount of dilation to apply to filter in each dimension D, defaults to 1
        tensor_expand (bool): expand the tensor of image and filter to do tensor convolution, defaults to True.
            If there is something more complicated going on (e.g. conv_contract), you can skip this step.
    returns: (jnp.array) convolved_image, shape (batch,out_c,spatial,tensor)
    """
    assert (D == 2) or (D == 3)
    assert (isinstance(is_torus, tuple) and len(is_torus) == D) or isinstance(is_torus, bool), (
        "geom::convolve" f" is_torus must be bool or tuple of bools, but got {is_torus}"
    )
    assert image.shape[1] == filter_image.shape[1], (
        f"Second axis (in_channels) for image and filter_image "
        f"must equal, but got image {image.shape} and filter {filter_image.shape}"
    )
    dtype = "float32"

    if isinstance(is_torus, bool):
        is_torus = (is_torus,) * D

    filter_spatial_dims, _ = parse_shape(filter_image.shape[2:], D)
    out_c, in_c = filter_image.shape[
        :2
    ]  # in_c and out_c are in_channels and out_channels respectively
    batch = len(image)

    assert not (
        reduce(lambda carry, N: carry or (N % 2 == 0), filter_spatial_dims, False)
        and (padding == "TORUS" or padding == "SAME" or padding is None)
    ), f"convolve: Filters with even sidelengths {filter_spatial_dims} require literal padding, not {padding}"

    if rhs_dilation is None:
        rhs_dilation = (1,) * D

    if stride is None:
        stride = (1,) * D

    if padding is None:  # if unspecified, infer from is_torus
        padding = "TORUS" if len(list(filter(lambda x: x, is_torus))) else "SAME"

    if (lhs_dilation is not None) and isinstance(padding, str):
        print(
            "WARNING convolve: lhs_dilation (transposed convolution) should specify padding exactly, "
            "see https://arxiv.org/pdf/1603.07285.pdf for the appropriate cases."
        )

    if padding == "TORUS":
        in_axes, out_axes = (None, 0, None, None, None), (0, None)
        vmap_torus_expand = vmap(vmap(get_torus_expanded, in_axes, out_axes), in_axes, out_axes)
        image, padding_literal = vmap_torus_expand(
            D, image, is_torus, filter_spatial_dims, rhs_dilation
        )
    elif padding == "VALID":
        padding_literal = ((0, 0),) * D
    elif padding == "SAME":
        padding_literal = get_same_padding(filter_spatial_dims, rhs_dilation)
    else:
        padding_literal = padding

    if tensor_expand:
        img_expanded, filter_expanded = pre_tensor_product_expand(
            D,
            image,
            filter_image,
            a_offset=2,
            b_offset=2,
            dtype=dtype,
        )
    else:
        img_expanded, filter_expanded = image, filter_image

    _, output_k = parse_shape(filter_expanded.shape[2:], D)
    channel_length = D**output_k
    image_spatial_dims, _ = parse_shape(img_expanded.shape[2:], D)

    # convert the image to NHWC (or NHWDC), treating all the pixel values as channels
    # (batch,in_c,spatial,tensor) -> (batch,spatial,tensor,in_c)
    img_formatted = jnp.moveaxis(img_expanded, 1, -1)
    # (batch,spatial,tensor,in_c) -> (batch,spatial,tensor*in_c)
    img_formatted = img_formatted.reshape((batch,) + image_spatial_dims + (channel_length * in_c,))

    # convert filter to HWIO (or HWDIO)
    # (out_c,in_c,spatial,tensor) -> (spatial,in_c,tensor,out_c)
    transpose_idxs = (
        tuple(range(2, 2 + D)) + (1,) + tuple(range(2 + D, filter_expanded.ndim)) + (0,)
    )
    filter_formatted = filter_expanded.transpose(transpose_idxs)
    # (spatial,in_c,out_c,tensor) -> (spatial,in_c,tensor*out_c)
    filter_formatted = filter_formatted.reshape(
        filter_spatial_dims + (in_c, out_c * channel_length)
    )

    convolved_array = jax.lax.conv_general_dilated(
        img_formatted,  # lhs
        filter_formatted,  # rhs
        stride,
        padding_literal,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=(("NHWC", "HWIO", "NHWC") if D == 2 else ("NHWDC", "HWDIO", "NHWDC")),
        feature_group_count=channel_length,  # each tensor component is treated separately
    )
    out_shape = convolved_array.shape[:-1] + (D,) * output_k + (out_c,)
    return jnp.moveaxis(convolved_array.reshape(out_shape), -1, 1)  # move out_c to 2nd axis


@partial(jit, static_argnums=[0, 3, 4, 5, 6, 7])
def convolve_contract(
    D: int,
    image: jnp.ndarray,
    filter_image: jnp.ndarray,
    is_torus: Union[bool, tuple[bool]],
    stride: Optional[tuple[int]] = None,
    padding: Optional[tuple[int]] = None,
    lhs_dilation: Optional[tuple[int]] = None,
    rhs_dilation: Optional[tuple[int]] = None,
) -> jnp.ndarray:
    """
    Given an input k image and a k+k' filter, take the tensor convolution that contract k times with one index
    each from the image and filter. This implementation is slightly more efficient then doing the convolution
    and contraction separately by avoiding constructing the k+k+k' intermediate tensor. See convolve for a
    full description of the convolution including the args.
    args:
        image (jnp.array): image data, shape (batch,in_c,spatial,tensor)
        filter_image (jnp.array): filter data, shape (out_c,in_c,spatial,tensor)
    returns: (jnp.array) output of shape (batch,out_c,spatial,tensor)
    """
    dtype = "float32"
    _, img_k = parse_shape(image.shape[2:], D)
    _, filter_k = parse_shape(filter_image.shape[2:], D)
    img_expanded = conv_contract_image_expand(D, image, filter_k).astype(dtype)
    convolved_img = convolve(
        D,
        img_expanded,
        filter_image,
        is_torus,
        stride,
        padding,
        lhs_dilation,
        rhs_dilation,
        tensor_expand=False,
    )
    # then sum along first img_k tensor axes, this is the contraction
    return jnp.sum(convolved_img, axis=range(2 + D, 2 + D + img_k))


def get_contraction_indices(
    initial_k: int,
    final_k: int,
    swappable_idxs: tuple[tuple[tuple[int]]] = (),
) -> list[tuple[tuple[int]]]:
    """
    Get all possible unique indices for multicontraction. Returns a list of indices. The indices are a tuple of tuples
    where each of the inner tuples are pairs of indices. For example, if initial_k=5, final_k = 4, one element of the
    list that is returned will be ((0,1), (2,3)), another will be ((1,4), (0,2)), etc.

    Note that contracting (0,1) is the same as contracting (1,0). Also, contracting ((0,1),(2,3)) is the same as
    contracting ((2,3),(0,1)). In both of those cases, they won't be returned. There is also the optional
    argument swappable_idxs to specify indices that can be swapped without changing the contraction. Suppose
    we have A * c1 where c1 is a k=2, parity=0 invariant conv_filter. In that case, we can contract on either of
    its indices and it won't change the result because transposing the axes is a group operation.
    args:
        initial_k (int): the starting number of indices that we have
        final_k (int): the final number of indices that we want to end up with
        swappable_idxs (tuple of tuple pairs of ints): Indices that can swapped w/o changing the contraction
    """
    assert ((initial_k + final_k) % 2) == 0
    assert initial_k >= final_k
    assert final_k >= 0

    tuple_pairs = it.combinations(it.combinations(range(initial_k), 2), (initial_k - final_k) // 2)
    rows = np.array([np.array(pair).reshape((initial_k - final_k,)) for pair in tuple_pairs])
    unique_rows = np.array([True if len(np.unique(row)) == len(row) else False for row in rows])
    unique_pairs = rows[unique_rows]  # remove rows which have an index multiple times

    # replace every element of the second term of the swappable pair with the first term
    for a, b in swappable_idxs:
        unique_pairs[np.where(np.isin(unique_pairs, b))] = a

    # convert back to lists
    sorted_tuples = [
        sorted(sorted([x, y]) for x, y in zip(row[0::2], row[1::2])) for row in unique_pairs
    ]
    sorted_rows = np.array(
        [np.array(pair).reshape((initial_k - final_k,)) for pair in sorted_tuples]
    )
    unique_sorted_rows = np.unique(sorted_rows, axis=0)  # after sorting remove redundant rows

    # restore by elements of the swappable pairs to being in the sequences
    for pair in swappable_idxs:
        for row in unique_sorted_rows:
            locs = np.isin(row, pair)
            if len(np.where(locs)[0]) > 0:
                row[np.max(np.where(locs))] = pair[1]
                row[np.min(np.where(locs))] = pair[
                    0
                ]  # if there is only 1, it will get set to pair 0

    return [tuple((x, y) for x, y in zip(idxs[0::2], idxs[1::2])) for idxs in unique_sorted_rows]


@partial(jit, static_argnums=[1, 2])
def multicontract(data: jnp.ndarray, indices: tuple[tuple[int]], idx_shift: int = 0) -> jnp.ndarray:
    """
    Perform the Kronecker Delta contraction on the data. Must have at least 2 dimensions, and because we implement with
    einsum, must have at most 52 dimensions. Indices a tuple of pairs of indices, also tuples.
    args:
        data (np.array-like): data to perform the contraction on
        indices (tuple of tuples of ints): index pairs to perform the contractions on
        idx_shift (int): indices are the tensor indices, so if data has spatial indices or channel/batch
            indices in the beginning we shift over by idx_shift
    """
    dimensions = len(data.shape)
    assert dimensions + len(indices) < 52
    assert dimensions >= 2
    # all indices must be unique, indices must be greater than 0 and less than dimensions

    einstr = list(LETTERS[:dimensions])
    for i, (idx1, idx2) in enumerate(indices):
        einstr[idx1 + idx_shift] = einstr[idx2 + idx_shift] = LETTERS[-(i + 1)]

    return jnp.einsum("".join(einstr), data)


def apply_contraction_map(
    D: int, image_data: jnp.ndarray, contract_map: jnp.ndarray, final_k: int
) -> jnp.ndarray:
    """
    Contract the image_data using the contraction map.
    """
    spatial_dims, k = parse_shape(image_data.shape, D)
    spatial_size = np.multiply.reduce(spatial_dims)
    vmap_mult = vmap(lambda image, map: map @ image, in_axes=(0, None))
    return vmap_mult(image_data.reshape((spatial_size, (D**k))), contract_map).reshape(
        spatial_dims + (D,) * final_k
    )


@jit
def linear_combination(images: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    A method takes a list of parameters, a list of geometric images and returns the linear combination.
    args:
        images (jnp.array): block of image data where the first axis is the image
        params (jnp.array): scalar multipliers of the images
    """
    return jnp.sum(vmap(lambda image, param: image * param)(images, params), axis=0)


def get_rotated_keys(D: int, data: jnp.ndarray, gg: np.ndarray) -> np.ndarray:
    """
    Get the rotated keys of data when it will be rotated by gg. Note that we rotate the key vector indices
    by the inverse of gg per the definition (this is done by key_array @ gg, rather than gg @ key_array).
    When the spatial_dims are not square, this gets a little tricky.
    The gg needs to be a concrete (numpy) array, not a traced jax array.
    args:
        D (int): dimension of image
        data (jnp.array): data array to be rotated
        gg (jnp array-like): group operation
    """
    spatial_dims, _ = parse_shape(data.shape, D)
    rotated_spatial_dims = tuple(np.abs(gg @ np.array(spatial_dims)))

    # When spatial_dims is nonsquare, we have to subtract one version, then add the rotated version.
    centering_coords = (np.array(rotated_spatial_dims).reshape((1, D)) - 1) / 2
    rotated_centering_coords = np.abs(gg @ centering_coords.reshape((D, 1))).reshape((1, D))
    # rotated keys will need to have the rotated_spatial_dims numbers
    key_array = np.array([key for key in it.product(*list(range(N) for N in rotated_spatial_dims))])
    shifted_key_array = key_array - centering_coords
    return np.rint((shifted_key_array @ gg) + rotated_centering_coords).astype(int)


def times_group_element(
    D: int,
    data: jnp.ndarray,
    parity: int,
    gg: np.ndarray,
    precision: Optional[jax.lax.Precision] = None,
) -> jnp.ndarray:
    """
    Apply a group element of SO(2) or SO(3) to the geometric image. First apply the action to the location of the
    pixels, then apply the action to the pixels themselves.
    args:
        D (int): dimension of the data
        data (jnp.array): data block of image data to rotate
        parity (int): parity of the data, 0 for even parity, 1 for odd parity
        gg (group operation matrix): a DxD matrix that rotates the tensor. Note that you cannot vmap
            by this argument because it needs to deal with concrete values
        precision (jax.lax.Precision): eisnum precision, normally uses lower precision, use
            jax.lax.Precision.HIGH for testing equality in unit tests
    """
    spatial_dims, k = parse_shape(data.shape, D)
    sign, _ = jnp.linalg.slogdet(gg)
    parity_flip = sign**parity  # if parity=1, the flip operators don't flip the tensors

    rotated_spatial_dims = tuple(np.abs(gg @ np.array(spatial_dims)))
    rotated_keys = get_rotated_keys(D, data, gg)
    rotated_pixels = data[hash(D, data, rotated_keys)].reshape(rotated_spatial_dims + (D,) * k)

    if k == 0:
        newdata = 1.0 * rotated_pixels * parity_flip
    else:
        # applying the rotation to tensors is essentially multiplying each index, which we can think of as a
        # vector, by the group action. The image pixels have already been rotated.
        einstr = LETTERS[: len(data.shape)] + ","
        einstr += ",".join([LETTERS[i + 13] + LETTERS[i + D] for i in range(k)])
        tensor_inputs = (rotated_pixels,) + k * (gg,)
        newdata = jnp.einsum(einstr, *tensor_inputs, precision=precision) * (parity_flip)

    return newdata


def tensor_times_gg(
    tensor: jnp.ndarray,
    parity: int,
    gg: np.ndarray,
    precision: Optional[jax.lax.Precision] = None,
) -> jnp.ndarray:
    """
    Apply a group element of SO(2) or SO(3) to a single tensor.
    args:
        D (int): dimension of the data
        tensor (jnp.array): data of the tensor
        parity (int): parity of the data, 0 for even parity, 1 for odd parity
        gg (group operation matrix): a DxD matrix that rotates the tensor. Note that you cannot vmap
            by this argument because it needs to deal with concrete values
        precision (jax.lax.Precision): eisnum precision, normally uses lower precision, use
            jax.lax.Precision.HIGH for testing equality in unit tests
    """
    k = len(tensor.shape)
    sign, _ = jnp.linalg.slogdet(gg)
    parity_flip = sign**parity  # if parity=1, the flip operators don't flip the tensors

    if k == 0:
        newdata = 1.0 * tensor * parity_flip
    else:
        # applying the rotation to tensors is essentially multiplying each index, which we can think of as a
        # vector, by the group action. The image pixels have already been rotated.
        einstr = LETTERS[:k] + ","
        einstr += ",".join([LETTERS[i + 13] + LETTERS[i] for i in range(k)])
        tensor_inputs = (tensor,) + k * (gg,)
        newdata = jnp.einsum(einstr, *tensor_inputs, precision=precision) * (parity_flip)

    return newdata


def norm(idx_shift: int, data: jnp.ndarray, keepdims: bool = False) -> jnp.ndarray:
    """
    Perform the frobenius norm on each pixel tensor, returning a scalar image
    args:
        idx_shift (int): the number of leading axes before the tensor, should be D for spatial plus
            the batch and spatial axes if they
        data (jnp.array): image data, shape (N,)*D + (D,)*k
        keepdims (bool): passed to jnp.linalg.norm, defaults to False
    """
    assert (
        idx_shift <= data.ndim
    ), f"norm: idx shift must be at most ndim, but {idx_shift} > {data.ndim}"
    if data.ndim == idx_shift:  # in this case, reshape creates an axis, so we need to collapse it
        keepdims = False

    return jnp.linalg.norm(
        data.reshape(data.shape[:idx_shift] + (-1,)), axis=idx_shift, keepdims=keepdims
    )


@partial(jax.jit, static_argnums=[0, 2, 3])
def max_pool(
    D: int,
    image_data: jnp.ndarray,
    patch_len: int,
    use_norm: bool = True,
    comparator_image: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Perform a max pooling operation where the length of the side of each patch is patch_len. Max is
    determined by the value of comparator_image if present, then the norm of image_data if use_norm
    is true, then finally the image_data otherwise.
    args:
        D (int): the dimension of the space
        image_data (jnp.array): the image data, currently shape (N,)*D + (D,)*k
        patch_len (int): the side length of the patches, must evenly divide all spatial dims
        use_norm (bool): if true, use the norm (over the tensor) of the image as the comparator image
        comparator_image (jnp.array): scalar image whose argmax is used to determine what value to use.
    """
    spatial_dims, k = parse_shape(image_data.shape, D)
    assert (comparator_image is not None) or use_norm or (k == 0)

    # TODO: use the batch dimension of dilated_patches correctly
    patches = jax.lax.conv_general_dilated_patches(
        image_data.reshape((1,) + spatial_dims + (-1,)).astype("float32"),  # NHWDC
        filter_shape=(patch_len,) * D,  # filter_shape
        window_strides=(patch_len,) * D,
        padding=((0, 0),) * D,  # padding
        dimension_numbers=(("NHWDC", "OIHWD", "NCHWD") if D == 3 else ("NHWC", "OIHW", "NCHW")),
    )[
        0
    ]  # no batch. Out shape (batch,channels,spatial)

    new_spatial_dims = patches.shape[1:]
    patches = patches.reshape((D**k, patch_len**D, -1))  # (tensor,patch,num_patches)

    if comparator_image is not None:
        assert comparator_image.shape == spatial_dims
        comparator_patches = jax.lax.conv_general_dilated_patches(
            comparator_image.reshape((1,) + spatial_dims + (-1,)).astype("float32"),  # NHWDC
            filter_shape=(patch_len,) * D,  # filter_shape
            window_strides=(patch_len,) * D,
            padding=((0, 0),) * D,  # padding
            dimension_numbers=(("NHWDC", "OIHWD", "NCHWD") if D == 3 else ("NHWC", "OIHW", "NCHW")),
        )[0]
        comparator_patches = comparator_patches.reshape((patch_len**D, -1))
    elif use_norm:
        comparator_patches = jnp.linalg.norm(patches, axis=0)  # (patch,num_patches)
    else:
        assert len(patches) == 1  # can only use image as your comparator if its a scalar image
        comparator_patches = patches[0]

    idxs = jnp.argmax(comparator_patches, axis=0)  # (num_patches,)
    vmap_max = vmap(lambda patch, idx: patch[:, idx], in_axes=(2, 0))
    return vmap_max(patches, idxs).reshape(new_spatial_dims + (D,) * k)


@partial(jit, static_argnums=[0, 2])
def average_pool(D: int, image_data: jnp.ndarray, patch_len: int) -> jnp.ndarray:
    """
    Perform a average pooling operation where the length of the side of each patch is patch_len. This is
    equivalent to doing a convolution where each element of the filter is 1 over the number of pixels in the
    filter, the stride length is patch_len, and the padding is 'VALID'.
    args:
        D (int): dimension of data
        image_data (jnp.array): image data, shape (h,w,tensor) or (h,w,d,tensor)
        patch_len (int): the side length of the patches, must evenly divide the sidelength
    """
    spatial_dims, _ = parse_shape(image_data.shape, D)
    assert reduce(lambda carry, N: carry and (N % patch_len == 0), spatial_dims, True)
    # convolve expects (out_c,in_c,h,w)
    filter_data = (1 / (patch_len**D)) * jnp.ones((1, 1) + (patch_len,) * D)

    # reshape to (1,h,w,tensor) because convolve expects (c,h,w,tensor)
    return convolve(
        D,
        image_data[None, None],
        filter_data,
        False,
        stride=(patch_len,) * D,
        padding="VALID",
    )[0, 0]


# ------------------------------------------------------------------------------
# PART 5: Define geometric (k-tensor, torus) images.


def tensor_name(k: int, parity: int) -> str:
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


@register_pytree_node_class
class GeometricImage:

    # Constructors

    @classmethod
    def zeros(
        cls,
        N: int,
        k: int,
        parity: int,
        D: int,
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        """
        Class method zeros to construct a geometric image of zeros
        args:
            N (int or tuple of ints): length of all sides if an int, otherwise a tuple of the side lengths
            k (int): the order of the tensor in each pixel, i.e. 0 (scalar), 1 (vector), 2 (matrix), etc.
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        spatial_dims = N if isinstance(N, tuple) else (N,) * D
        assert len(spatial_dims) == D
        return cls(jnp.zeros(spatial_dims + (D,) * k), parity, D, is_torus)

    @classmethod
    def fill(
        cls,
        N: int,
        parity: int,
        D: int,
        fill: Union[jnp.ndarray, int, float],
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        """
        Class method fill constructor to construct a geometric image every pixel as fill
        args:
            N (int or tuple of ints): length of all sides if an int, otherwise a tuple of the side lengths
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            fill (jnp.ndarray or number): tensor to fill the image with
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        spatial_dims = N if isinstance(N, tuple) else (N,) * D
        assert len(spatial_dims) == D

        k = (
            len(fill.shape)
            if (isinstance(fill, jnp.ndarray) or isinstance(fill, np.ndarray))
            else 0
        )
        data = jnp.stack([fill for _ in range(np.multiply.reduce(spatial_dims))]).reshape(
            spatial_dims + (D,) * k
        )
        return cls(data, parity, D, is_torus)

    def __init__(
        self: Self,
        data: jnp.ndarray,
        parity: int,
        D: int,
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        """
        Construct the GeometricImage. It will be (N^D x D^k), so if N=100, D=2, k=1, then it's (100 x 100 x 2)
        args:
            data (array-like):
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool or tuple of bools): whether the datablock is a torus, used for convolutions.
                Takes either a tuple of bools of length D specifying whether each dimension is toroidal,
                or simply True or False which sets all dimensions to that value. Defaults to True.
        """
        self.D = D
        self.spatial_dims, self.k = parse_shape(data.shape, D)
        assert data.shape[D:] == self.k * (
            self.D,
        ), "GeometricImage: each pixel must be D cross D, k times"
        self.parity = parity % 2

        assert (isinstance(is_torus, tuple) and (len(is_torus) == D)) or isinstance(is_torus, bool)
        if isinstance(is_torus, bool):
            is_torus = (is_torus,) * D

        self.is_torus = is_torus

        self.data = jnp.copy(
            data
        )  # TODO: don't need to copy if data is already an immutable jnp array

    def copy(self: Self) -> Self:
        return self.__class__(self.data, self.parity, self.D, self.is_torus)

    # Getters, setters, basic info

    def hash(self: Self, indices: jnp.ndarray) -> jnp.ndarray:
        """
        Deals with torus by modding (with `np.remainder()`).
        args:
            indices (tuple of ints): indices to apply the remainder to
        """
        return hash(self.D, self.data, indices)

    def __getitem__(self: Self, key) -> jnp.ndarray:
        """
        Accessor for data values. Now you can do image[key] where k are indices or array slices and it will just work
        Note that JAX does not throw errors for indexing out of bounds
        args:
            key (index): JAX/numpy indexer, i.e. "0", "0,1,3", "4:, 2:3, 0" etc.
        """
        return self.data[key]

    def __setitem__(self: Self, key, val) -> Self:
        """
        Jax arrays are immutable, so this reconstructs the data object with copying, and is potentially slow
        """
        self.data = self.data.at[key].set(val)
        return self

    def shape(self: Self) -> tuple[int]:
        """
        Return the full shape of the data block
        """
        return self.data.shape

    def image_shape(self: Self, plus_Ns: Optional[tuple[int]] = None) -> tuple[int]:
        """
        Return the shape of the data block that is not the ktensor shape, but what comes before that.
        args:
            plus_Ns (tuple of ints): d-length tuple, N to add to each spatial dim
        """
        plus_Ns = (0,) * self.D if (plus_Ns is None) else plus_Ns
        return tuple(N + plus_N for N, plus_N in zip(self.spatial_dims, plus_Ns))

    def pixel_shape(self: Self) -> tuple[int]:
        """
        Return the shape of the data block that is the ktensor, aka the pixel of the image.
        """
        return self.k * (self.D,)

    def pixel_size(self: Self) -> int:
        """
        Get the size of the pixel shape, i.e. (D,D,D) = D**3
        """
        return self.D**self.k

    def __str__(self: Self) -> str:
        return "<{} object in D={} with spatial_dims={}, k={}, parity={}, is_torus={}>".format(
            self.__class__,
            self.D,
            self.spatial_dims,
            self.k,
            self.parity,
            self.is_torus,
        )

    def keys(self: Self) -> Sequence[Sequence[int]]:
        """
        Iterate over the keys of GeometricImage
        """
        return it.product(*list(range(N) for N in self.spatial_dims))

    def key_array(self: Self) -> jnp.ndarray:
        # equivalent to the old pixels function
        return jnp.array([key for key in self.keys()], dtype=int)

    def pixels(self: Self) -> Generator[jnp.ndarray]:
        """
        Iterate over the pixels of GeometricImage.
        """
        for key in self.keys():
            yield self[key]

    def items(self: Self) -> Generator[tuple[Any, jnp.ndarray]]:
        """
        Iterate over the key, pixel pairs of GeometricImage.
        """
        for key in self.keys():
            yield (key, self[key])

    # Binary Operators, Complicated functions

    def __eq__(self: Self, other: Self) -> bool:
        """
        Equality operator, must have same shape, parity, and data within the TINY=1e-5 tolerance.
        """
        return (
            self.D == other.D
            and self.spatial_dims == other.spatial_dims
            and self.k == other.k
            and self.parity == other.parity
            and self.is_torus == other.is_torus
            and self.data.shape == other.data.shape
            and jnp.allclose(self.data, other.data, rtol=TINY, atol=TINY)
        )

    def __add__(self: Self, other: Self) -> Self:
        """
        Addition operator for GeometricImages. Both must be the same size and parity. Returns a new GeometricImage.
        args:
            other (GeometricImage): other image to add the the first one
        """
        assert self.D == other.D
        assert self.spatial_dims == other.spatial_dims
        assert self.k == other.k
        assert self.parity == other.parity
        assert self.is_torus == other.is_torus
        assert self.data.shape == other.data.shape
        return self.__class__(self.data + other.data, self.parity, self.D, self.is_torus)

    def __sub__(self: Self, other: Self) -> Self:
        """
        Subtraction operator for GeometricImages. Both must be the same size and parity. Returns a new GeometricImage.
        args:
            other (GeometricImage): other image to add the the first one
        """
        assert self.D == other.D
        assert self.spatial_dims == other.spatial_dims
        assert self.k == other.k
        assert self.parity == other.parity
        assert self.is_torus == other.is_torus
        assert self.data.shape == other.data.shape
        return self.__class__(self.data - other.data, self.parity, self.D, self.is_torus)

    def __mul__(self: Self, other: Union[Self, float, int]) -> Self:
        """
        If other is a scalar, do scalar multiplication of the data. If it is another GeometricImage, do the tensor
        product at each pixel. Return the result as a new GeometricImage.
        args:
            other (GeometricImage or number): scalar or image to multiply by
        """
        if isinstance(other, GeometricImage):
            assert self.D == other.D
            assert self.spatial_dims == other.spatial_dims
            assert self.is_torus == other.is_torus
            return self.__class__(
                mul(self.D, self.data, other.data),
                self.parity + other.parity,
                self.D,
                self.is_torus,
            )
        else:  # its an integer or a float, or something that can we can multiply a Jax array by (like a DeviceArray)
            return self.__class__(self.data * other, self.parity, self.D, self.is_torus)

    def __rmul__(self: Self, other: Union[Self, float, int]) -> Self:
        """
        If other is a scalar, multiply the data by the scalar. This is necessary for doing scalar * image, and it
        should only be called in that case.
        """
        return self * other

    def transpose(self: Self, axes_permutation: Sequence[int]) -> Self:
        """
        Transposes the axes of the tensor, keeping the image axes in the front the same
        args:
            axes_permutation (iterable of indices): new axes order
        """
        idx_shift = len(self.image_shape())
        new_indices = tuple(
            tuple(range(idx_shift)) + tuple(axis + idx_shift for axis in axes_permutation)
        )
        return self.__class__(
            jnp.transpose(self.data, new_indices), self.parity, self.D, self.is_torus
        )

    @partial(jit, static_argnums=[2, 3, 4, 5])
    def convolve_with(
        self: Self,
        filter_image: Self,
        stride: Optional[tuple[int]] = None,
        padding: Optional[tuple[int]] = None,
        lhs_dilation: Optional[tuple[int]] = None,
        rhs_dilation: Optional[tuple[int]] = None,
    ) -> Self:
        """
        See convolve for a description of this function.
        """
        convolved_array = convolve(
            self.D,
            self.data[None, None],  # add batch, in_channels axes
            filter_image.data[None, None],  # add out_channels, in_channels axes
            self.is_torus,
            stride,
            padding,
            lhs_dilation,
            rhs_dilation,
        )
        return self.__class__(
            convolved_array[0, 0],  # ignore batch, out_channels axes
            self.parity + filter_image.parity,
            self.D,
            self.is_torus,
        )

    @partial(jit, static_argnums=[1, 2])
    def max_pool(self: Self, patch_len: int, use_norm: bool = True) -> Self:
        """
         Perform a max pooling operation where the length of the side of each patch is patch_len. Max is determined
        by the norm of the pixel when use_norm is True. Note that for scalars, this will be the absolute value of
        the pixel. If you want to use the max instead, set use_norm to False (requires scalar images).
        args:
            patch_len (int): the side length of the patches, must evenly divide all spatial dims
        """
        return self.__class__(
            max_pool(self.D, self.data, patch_len, use_norm),
            self.parity,
            self.D,
            self.is_torus,
        )

    @partial(jit, static_argnums=1)
    def average_pool(self: Self, patch_len: int) -> Self:
        """
        Perform a average pooling operation where the length of the side of each patch is patch_len. This is
        equivalent to doing a convolution where each element of the filter is 1 over the number of pixels in the
        filter, the stride length is patch_len, and the padding is 'VALID'.
        args:
            patch_len (int): the side length of the patches, must evenly divide self.N
        """
        return self.__class__(
            average_pool(self.D, self.data, patch_len),
            self.parity,
            self.D,
            self.is_torus,
        )

    @partial(jit, static_argnums=1)
    def unpool(self: Self, patch_len: int) -> Self:
        """
        Each pixel turns into a (patch_len,)*self.D patch of that pixel. Also called "Nearest Neighbor" unpooling
        args:
            patch_len (int): side length of the patch of our unpooled images
        """
        grow_filter = GeometricImage(jnp.ones((patch_len,) * self.D), 0, self.D)
        return self.convolve_with(
            grow_filter,
            padding=((patch_len - 1,) * 2,) * self.D,
            lhs_dilation=(patch_len,) * self.D,
        )

    def times_scalar(self: Self, scalar: float) -> Self:
        """
        Scale the data by a scalar, returning a new GeometricImage object. Alias of the multiplication operator.
        args:
            scalar (number): number to scale everything by
        """
        return self * scalar

    @jit
    def norm(self: Self) -> Self:
        """
        Calculate the norm pixel-wise. This becomes a 0 parity image.
        returns: scalar image
        """
        return self.__class__(norm(self.D, self.data), 0, self.D, self.is_torus)

    def normalize(self: Self) -> Self:
        """
        Normalize so that the max norm of each pixel is 1, and all other tensors are scaled appropriately
        """
        max_norm = jnp.max(self.norm().data)
        if max_norm > TINY:
            return self.times_scalar(1.0 / max_norm)
        else:
            return self.times_scalar(1.0)

    def activation_function(self: Self, function: Callable[[jnp.ndarray], jnp.ndarray]) -> Self:
        assert (
            self.k == 0
        ), "Activation functions only implemented for k=0 tensors due to equivariance"
        return self.__class__(function(self.data), self.parity, self.D, self.is_torus)

    @partial(jit, static_argnums=[1, 2])
    def contract(self: Self, i: int, j: int) -> Self:
        """
        Use einsum to perform a kronecker contraction on two dimensions of the tensor
        args:
            i (int): first index of tensor
            j (int): second index of tensor
        """
        assert self.k >= 2
        idx_shift = len(self.image_shape())
        return self.__class__(
            multicontract(self.data, ((i, j),), idx_shift),
            self.parity,
            self.D,
            self.is_torus,
        )

    @partial(jit, static_argnums=1)
    def multicontract(self: Self, indices: tuple[tuple[int]]) -> Self:
        """
        Use einsum to perform a kronecker contraction on two dimensions of the tensor
        args:
            indices (tuple of tuples of ints): indices to contract
        """
        assert self.k >= 2
        idx_shift = len(self.image_shape())
        return self.__class__(
            multicontract(self.data, indices, idx_shift),
            self.parity,
            self.D,
            self.is_torus,
        )

    def levi_civita_contract(self: Self, indices: tuple[tuple[int]]) -> Self:
        """
        Perform the Levi-Civita contraction. Outer product with the Levi-Civita Symbol, then perform D-1 contractions.
        Resulting image has k= self.k - self.D + 2
        args:
            indices (int, or tuple, or list): indices of tensor to perform contractions on
        """
        assert self.k >= (
            self.D - 1
        )  # so we have enough indices to work on since we perform D-1 contractions
        if self.D == 2 and not (isinstance(indices, tuple) or isinstance(indices, list)):
            indices = (indices,)
        assert len(indices) == self.D - 1

        levi_civita = LeviCivitaSymbol.get(self.D)
        outer = jnp.tensordot(self.data, levi_civita, axes=0)

        # make contraction index pairs with one of specified indices, and index (in order) from the levi_civita symbol
        idx_shift = len(self.image_shape())
        zipped_indices = tuple(
            (i + idx_shift, j + idx_shift)
            for i, j in zip(indices, range(self.k, self.k + len(indices)))
        )
        return self.__class__(
            multicontract(outer, zipped_indices), self.parity + 1, self.D, self.is_torus
        )

    def get_rotated_keys(self: Self, gg: np.ndarray) -> np.ndarray:
        """
        Slightly messier than with GeometricFilter because self.N-1 / 2 might not be an integer, but should work
        args:
            gg (jnp array-like): group operation
        """
        return get_rotated_keys(self.D, self.data, gg)

    def times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision] = None
    ) -> Self:
        """
        Apply a group element of SO(2) or SO(3) to the geometric image. First apply the action to the location of the
        pixels, then apply the action to the pixels themselves.
        args:
            gg (group operation matrix): a DxD matrix that rotates the tensor
            precision (jax.lax.Precision): precision level for einsum, for equality tests use Precision.HIGH
        """
        assert self.k < 14
        assert gg.shape == (self.D, self.D)

        return self.__class__(
            times_group_element(self.D, self.data, self.parity, gg, precision=precision),
            self.parity,
            self.D,
            self.is_torus,
        )

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        title: str = "",
        boxes: bool = False,
        fill: bool = True,
        symbols: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = False,
        vector_scaling: float = 0.5,
    ) -> None:
        # plot functions should fail gracefully
        if self.D != 2 and self.D != 3:
            print(
                f"GeometricImage::plot: Can only plot dimension 2 or 3 images, but got D={self.D}"
            )
            return
        if self.k > 2:
            print(
                f"GeometricImage::plot: Can only plot tensor order 0,1, or 2 images, but got k={self.k}"
            )
            return
        if self.k == 2 and self.D == 3:
            print(f"GeometricImage::plot: Cannot plot D=3, k=2 geometric images.")
            return

        ax = utils.setup_plot() if ax is None else ax

        # This was breaking earlier with jax arrays, not sure why. I really don't want plotting to break,
        # so I am will swap to numpy arrays just in case.
        xs, ys, *zs = np.array(self.key_array()).T
        if self.D == 3:
            xs = xs + utils.XOFF * zs
            ys = ys + utils.YOFF * zs

        pixels = np.array(list(self.pixels()))

        if self.k == 0:
            vmin = np.min(pixels) if vmin is None else vmin
            vmax = np.max(pixels) if vmax is None else vmax
            utils.plot_scalars(
                ax,
                self.spatial_dims,
                xs,
                ys,
                pixels,
                boxes=boxes,
                fill=fill,
                symbols=symbols,
                vmin=vmin,
                vmax=vmax,
                colorbar=colorbar,
            )
        elif self.k == 1:
            vmin = 0.0 if vmin is None else vmin
            vmax = 2.0 if vmax is None else vmax
            utils.plot_vectors(
                ax,
                xs,
                ys,
                pixels,
                boxes=boxes,
                fill=fill,
                vmin=vmin,
                vmax=vmax,
                scaling=vector_scaling,
            )
        else:  # self.k == 2
            utils.plot_tensors(ax, xs, ys, pixels, boxes=boxes)

        utils.finish_plot(ax, title, xs, ys, self.D)

    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[jnp.ndarray], dict[str, Union[int, Union[bool, tuple[bool]]]]]:
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly. Children and aux_data
        must contain all the variables that are passed in __init__()
        """
        children = (self.data,)  # arrays / dynamic values
        aux_data = {
            "D": self.D,
            "parity": self.parity,
            "is_torus": self.is_torus,
        }  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly.
        """
        return cls(*children, **aux_data)


# ------------------------------------------------------------------------------
# PART 3: Define a geometric (k-tensor) filter.


@register_pytree_node_class
class GeometricFilter(GeometricImage):

    def __init__(
        self: Self,
        data: jnp.ndarray,
        parity: int,
        D: int,
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        super(GeometricFilter, self).__init__(data, parity, D, is_torus)
        assert (
            self.spatial_dims == (self.spatial_dims[0],) * self.D
        ), "GeometricFilter: Filters must be square."  # I could remove  this requirement in the future

    @classmethod
    def from_image(cls, geometric_image: GeometricImage) -> Self:
        """
        Constructor that copies a GeometricImage and returns a GeometricFilter
        """
        return cls(
            geometric_image.data,
            geometric_image.parity,
            geometric_image.D,
            geometric_image.is_torus,
        )

    def __str__(self: Self) -> str:
        return "<geometric filter object in D={} with spatial_dims={}, k={}, parity={}, and is_torus={}>".format(
            self.D, self.spatial_dims, self.k, self.parity, self.is_torus
        )

    def bigness(self: Self) -> float:
        """
        Gives an idea of size for a filter, sparser filters are smaller while less sparse filters are larger
        """
        norms = self.norm().data
        numerator = 0.0
        for key in self.key_array():
            numerator += jnp.linalg.norm(key * norms[tuple(key)], ord=2)

        denominator = jnp.sum(norms)
        return numerator / denominator

    def rectify(self: Self) -> Self:
        """
        Filters form an equivalence class up to multiplication by a scalar, so if its negative we want to flip the sign
        """
        if self.k == 0:
            if jnp.sum(self.data) < 0:
                return self.times_scalar(-1)
        elif self.k == 1:
            if self.parity % 2 == 0:
                if np.sum([np.dot(np.array(key), self[key]) for key in self.keys()]) < 0:
                    return self.times_scalar(-1)
            elif self.D == 2:
                if np.sum([np.cross(np.array(key), self[key]) for key in self.keys()]) < 0:
                    return self.times_scalar(-1)
        return self

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        title: str = "",
        boxes: bool = True,
        fill: bool = True,
        symbols: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = False,
        vector_scaling: float = 0.33,
    ) -> None:
        if self.k == 0:
            vmin = -3.0 if vmin is None else vmin
            vmax = 3.0 if vmax is None else vmax
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 3.0 if vmax is None else vmax

        super(GeometricFilter, self).plot(
            ax, title, boxes, fill, symbols, vmin, vmax, colorbar, vector_scaling
        )


@register_pytree_node_class
class Layer:

    # Constructors

    def __init__(
        self: Self, data: jnp.ndarray, D: int, is_torus: Union[bool, tuple[bool]] = True
    ) -> Self:
        """
        Construct a layer
        args:
            data (dictionary of jnp.array): dictionary by k of jnp.array
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        self.D = D
        assert (isinstance(is_torus, tuple) and (len(is_torus) == D)) or isinstance(is_torus, bool)
        if isinstance(is_torus, bool):
            is_torus = (is_torus,) * D

        self.is_torus = is_torus
        # copy dict, but image_block is immutable jnp array
        self.data = {key: image_block for key, image_block in data.items()}

    def copy(self: Self) -> Self:
        return self.__class__(self.data, self.D, self.is_torus)

    def empty(self: Self) -> Self:
        return self.__class__({}, self.D, self.is_torus)

    @classmethod
    def from_images(cls, images: list[GeometricImage]) -> Self:
        # We assume that all images have the same D and is_torus
        if len(images) == 0:
            return None

        out_layer = cls({}, images[0].D, images[0].is_torus)
        for image in images:
            out_layer.append(image.k, image.parity, image.data.reshape((1,) + image.data.shape))

        return out_layer

    @classmethod
    def from_vector(cls, vector: jnp.ndarray, layer: Self) -> Self:
        """
        Convert a vector to a layer, using the shape and parity of the provided layer.
        args:
            vector (jnp.array): a 1-D array of values
            layer (Layer): a layer providing the parity and shape for the resulting new layer
        """
        idx = 0
        out_layer = layer.empty()
        for (k, parity), img in layer.items():
            out_layer.append(k, parity, vector[idx : (idx + img.size)].reshape(img.shape))
            idx += img.size

        return out_layer

    def __str__(self: Self) -> str:
        layer_repr = f"{self.__class__} D: {self.D}, is_torus: {self.is_torus}\n"
        for k, image_block in self.items():
            layer_repr += f"\t{k}: {image_block.shape}\n"

        return layer_repr

    def size(self: Self) -> int:
        return reduce(lambda size, img: size + img.size, self.values(), 0)

    def get_spatial_dims(self: Self) -> tuple[int]:
        """
        Get the spatial dimensions. Use this function with caution, if the layer is being vmapped, it will
        return the incorrect spatial dims.
        """
        if len(self.values()) == 0:
            return None

        return next(iter(self.values())).shape[1 : 1 + self.D]

    # Functions that map directly to calling the function on data

    def keys(self: Self) -> Generator[tuple[int, int]]:
        return self.data.keys()

    def values(self: Self) -> Generator[jnp.ndarray]:
        return self.data.values()

    def items(self: Self) -> Generator[tuple[tuple[int, int], jnp.ndarray]]:
        return self.data.items()

    def __getitem__(self: Self, idx: tuple[int, int]) -> jnp.ndarray:
        return self.data[idx]

    def __setitem__(self: Self, idx: tuple[int, int], val: jnp.ndarray) -> jnp.ndarray:
        self.data[idx] = val
        return self.data[idx]

    def __contains__(self: Self, idx: tuple[int, int]) -> bool:
        return idx in self.data

    def __eq__(self: Self, other: Self, rtol: float = TINY, atol: float = TINY) -> bool:
        if (
            (self.D != other.D)
            or (self.is_torus != other.is_torus)
            or (self.keys() != other.keys())
        ):
            return False

        for key in self.keys():
            if not jnp.allclose(self[key], other[key], rtol, atol):
                return False

        return True

    # Other functions

    def append(self: Self, k: int, parity: int, image_block: jnp.ndarray, axis: int = 0) -> Self:
        """
        Append an image block at (k,parity). It will be concatenated along axis=0, so channel for Layer
        and vmapped BatchLayer, and batch for normal BatchLayer
        """
        parity = parity % 2
        # will this work for BatchLayer?
        if (
            k > 0
        ):  # very light shape checking, other problematic cases should be caught in concatenate
            assert image_block.shape[-k:] == (self.D,) * k

        if (k, parity) in self:
            self[(k, parity)] = jnp.concatenate((self[(k, parity)], image_block), axis=axis)
        else:
            self[(k, parity)] = image_block

        return self

    def __add__(self: Self, other: Self) -> Self:
        """
        Addition operator for Layers, must have the same types of layers, adds them together
        """
        assert type(self) == type(
            other
        ), f"{self.__class__}::__add__: Types of layers being added must match, had {type(self)} and {type(other)}"
        assert (
            self.D == other.D
        ), f"{self.__class__}::__add__: Dimension of layers must match, had {self.D} and {other.D}"
        assert (
            self.is_torus == other.is_torus
        ), f"{self.__class__}::__add__: is_torus of layers must match, had {self.is_torus} and {other.is_torus}"
        assert (
            self.keys() == other.keys()
        ), f"{self.__class__}::__add__: Must have same types of images, had {self.keys()} and {other.keys()}"

        return self.__class__.from_vector(self.to_vector() + other.to_vector(), self)

    def __mul__(self: Self, other: Union[Self, float]) -> Self:
        """
        Multiplication operator for a layer and a scalar
        """
        assert not isinstance(
            other, Layer
        ), f"Layer multiplication is only implemented for numbers, got {type(other)}."

        return self.__class__.from_vector(self.to_vector() * other, self)

    def __truediv__(self: Self, other: float) -> Self:
        """
        True division (a/b) for a layer and a scalar.
        """
        return self * (1.0 / other)

    def concat(self: Self, other: Self, axis: int = 0) -> Self:
        """
        Concatenate the layers along a specified axis.
        args:
            other (Layer): a layer with the same dimension and qualities as this one
            axis (int): the axis along with the concatenate the other layer
        """
        assert type(self) == type(
            other
        ), f"{self.__class__}::concat: Types of layers being added must match, had {type(self)} and {type(other)}"
        assert (
            self.D == other.D
        ), f"{self.__class__}::concat: Dimension of layers must match, had {self.D} and {other.D}"
        assert (
            self.is_torus == other.is_torus
        ), f"{self.__class__}::concat: is_torus of layers must match, had {self.is_torus} and {other.is_torus}"

        new_layer = self.copy()
        for (k, parity), image_block in other.items():
            new_layer.append(k, parity, image_block, axis)

        return new_layer

    def to_images(self: Self) -> list[GeometricImage]:
        # Should only be used in Layer of vmapped BatchLayer
        images = []
        for image_block in self.values():
            for image in image_block:
                images.append(
                    GeometricImage(image, 0, self.D, self.is_torus)
                )  # for now, assume 0 parity

        return images

    def to_vector(self: Self) -> jnp.ndarray:
        """
        Vectorize a layer in the natural way
        """
        return reduce(
            lambda x, y: jnp.concatenate([x, y.reshape(-1)]),
            self.values(),
            jnp.zeros(0),
        )

    def to_scalar_layer(self: Self) -> Self:
        """
        Convert layer to a layer where all the channels and components are in the scalar
        """
        # convert to channels of a scalar layer
        out_layer = self.empty()
        for (k, _), image in self.items():
            transpose_idxs = (
                (0,) + tuple(range(1 + self.D, 1 + self.D + k)) + tuple(range(1, 1 + self.D))
            )
            out_layer.append(
                0,
                0,
                image.transpose(transpose_idxs).reshape((-1,) + image.shape[1 : 1 + self.D]),
            )

        return out_layer

    def from_scalar_layer(self: Self, layout: dict[tuple[int, int], int]) -> Self:
        """
        Convert a scalar layer back to a layer with the specified layout
        args:
            layout (dict): dictionary of keys (k,parity) and values num_channels for the output layer
        """
        assert list(self.keys()) == [(0, 0)]
        spatial_dims = self[(0, 0)].shape[1:]

        out_layer = self.empty()
        idx = 0
        for (k, parity), num_channels in layout.items():
            length = num_channels * (self.D**k)
            # reshape, it is (num_channels*(D**k), spatial_dims) -> (num_channels, (D,)*k, spatial_dims)
            reshaped_data = self[(0, 0)][idx : idx + length].reshape(
                (num_channels,) + (self.D,) * k + spatial_dims
            )
            # tranpose (num_channels, (D,)*k, spatial_dims) -> (num_channels, spatial_dims, (D,)*k)
            transposed_data = reshaped_data.transpose(
                (0,) + tuple(range(1 + k, 1 + k + self.D)) + tuple(range(1, 1 + k))
            )
            out_layer.append(k, parity, transposed_data)
            idx += length

        return out_layer

    def times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision] = None
    ) -> Self:
        """
        Apply a group element of O(2) or O(3) to the layer. First apply the action to the location of the
        pixels, then apply the action to the pixels themselves.
        args:
            gg (group operation matrix): a DxD matrix that rotates the tensor
            precision (jax.lax.Precision): precision level for einsum, for equality tests use Precision.HIGH
        """
        vmap_rotate = vmap(times_group_element, in_axes=(None, 0, None, None, None))
        out_layer = self.empty()
        for (k, parity), image_block in self.items():
            out_layer.append(k, parity, vmap_rotate(self.D, image_block, parity, gg, precision))

        return out_layer

    def norm(self: Self) -> Self:
        out_layer = self.empty()
        for image_block in self.values():
            out_layer.append(0, 0, norm(self.D + 1, image_block))  # norm is even parity

        return out_layer

    def get_component(
        self: Self,
        component: int,
        future_steps: int = 1,
        as_layer: bool = True,
    ) -> Union[Self, jnp.ndarray]:
        """
        Given a layer with data with shape (channels*future_steps,spatial,tensor), combine all
        fields into a single block of data (future_steps,spatial,channels*tensor) then pick the
        ith channel in the last axis, where i = component. For example, if the layer has density (scalar),
        pressure (scalar), and velocity (vector) then i=0 -> density, i=1 -> pressure, i=2 -> velocity 1,
        and i=3 -> velocity 2. This assumes D=2.
        args:
            component (int): which component to select
            future_steps (int): the number of future timesteps of this layer, defaults to 1
            as_layer (bool): if true, return as a new layer with a scalar feature, otherwise just the data
        """
        # explicitly call Layer's version, even if calling from vmapped BatchLayer
        spatial_dims = Layer.get_spatial_dims(self)

        data = None
        for (k, _), img in self.items():
            exp_data = img.reshape(
                (-1, future_steps) + spatial_dims + (self.D,) * k
            )  # (c,time,spatial,tensor)
            exp_data = jnp.moveaxis(exp_data, 0, 1 + self.D)  # (time,spatial,c,tensor)
            exp_data = exp_data.reshape(
                (future_steps,) + spatial_dims + (-1,)
            )  # (time,spatial,c*tensor)

            data = exp_data if data is None else jnp.concatenate([data, exp_data], axis=-1)

        component_data = data[..., component].reshape((future_steps,) + spatial_dims + (-1,))
        component_data = jnp.moveaxis(component_data, -1, 0).reshape((-1,) + spatial_dims)
        if as_layer:
            return self.__class__({(0, 0): component_data}, self.D, self.is_torus)
        else:
            return component_data

    def device_replicate(self: Self, sharding: jax.sharding.PositionalSharding) -> Self:
        """
        Put the BatchLayer on particular devices according to the sharding and num_devices
        args:
            sharding (jax sharding): jax positional sharding to be reshaped
            num_devices (int): number of gpus to split the batches over
        """
        return self.__class__(
            jax.device_put(self.data, sharding.replicate()), self.D, self.is_torus
        )

    # JAX helpers
    def tree_flatten(self):
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly. Children
        and aux_data must contain all the variables that are passed in __init__()
        """
        children = (self.data,)  # arrays / dynamic values
        aux_data = {
            "D": self.D,
            "is_torus": self.is_torus,
        }  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly.
        """
        return cls(*children, **aux_data)


@register_pytree_node_class
class BatchLayer(Layer):
    # I may want to only have Layer, and find out a better way of tracking this

    # Constructors

    def __init__(
        self: Self, data: jnp.ndarray, D: int, is_torus: Union[bool, tuple[bool]] = True
    ) -> Self:
        """
        Construct a layer
        args:
            data (dictionary of jnp.array): dictionary by k of jnp.array
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        super(BatchLayer, self).__init__(data, D, is_torus)

        self.L = None
        for image_block in data.values():  # if empty, this won't get set
            if isinstance(image_block, jnp.ndarray):
                self.L = len(image_block)  # shape (batch, channels, (N,)*D, (D,)*k)
            break

    @classmethod
    def from_images(cls, images: list[GeometricImage]) -> Self:
        # We assume that all images have the same D and is_torus
        if len(images) == 0:
            return None

        out_layer = cls({}, images[0].D, images[0].is_torus)
        for image in images:
            out_layer.append(image.k, image.parity, image.data.reshape((1, 1) + image.data.shape))

        batch_image_block = list(out_layer.values())[0]
        out_layer.L = batch_image_block.shape[0]

        return out_layer

    def get_spatial_dims(self: Self) -> tuple[int]:
        """
        Get the spatial dims. Use this function with caution, if the BatchLayer is being vmapped, then
        it will give you the incorrect spatial dims.
        """
        if len(self.values()) == 0:
            return None

        return next(iter(self.values())).shape[2 : 2 + self.D]

    def get_L(self: Self) -> int:
        """
        Get the batch size. This will return the wrong value if the batch is vmapped.
        """
        if len(self.values()) == 0:
            return 0

        return len(next(iter(self.values())))

    def get_subset(self: Self, idxs: jnp.ndarray) -> Self:
        """
        Select a subset of the batch, picking the indices idxs
        args:
            idxs (jnp.array): array of indices to select the subset
        """
        assert isinstance(idxs, jnp.ndarray), "BatchLayer::get_subset arg idxs must be a jax array"
        assert len(
            idxs.shape
        ), "BatchLayer::get_subset arg idxs must be a jax array, e.g. jnp.array([0])"
        return self.__class__(
            {k: image_block[idxs] for k, image_block in self.items()},
            self.D,
            self.is_torus,
        )

    def get_one(self: Self, idx: int = 0) -> Self:
        return self.get_subset(jnp.array([idx]))

    def get_one_layer(self: Self, idx: int = 0) -> Layer:
        """
        Similar to get_one, but instead get a single index of a BatchLayer as a Layer
        args:
            idx (int): the index along the batch to get as a Layer.
        """
        return Layer(
            {k: image_block[idx] for k, image_block in self.items()},
            self.D,
            self.is_torus,
        )

    def times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision] = None
    ) -> Self:
        return self._times_group_element(gg, precision)

    @partial(jax.vmap, in_axes=(0, None, None))
    def _times_group_element(self: Self, gg: np.ndarray, precision: jax.lax.Precision) -> Self:
        return super(BatchLayer, self).times_group_element(gg, precision)

    @jax.vmap
    def to_vector(self: Self) -> jnp.ndarray:
        return super(BatchLayer, self).to_vector()

    @jax.vmap
    def to_scalar_layer(self: Self) -> Self:
        return super(BatchLayer, self).to_scalar_layer()

    @partial(jax.vmap, in_axes=(0, None))
    def from_scalar_layer(self: Self, layout: Self) -> Self:
        return super(BatchLayer, self).from_scalar_layer(layout)

    @classmethod
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def from_vector(cls, vector: jnp.ndarray, layer: Self) -> Self:
        return super().from_vector(vector, layer)

    @jax.vmap
    def norm(self: Self) -> Self:
        return super(BatchLayer, self).norm()

    def get_component(
        self: Self,
        component: int,
        future_steps: int = 1,
        as_layer: bool = True,
    ) -> Union[Self, jnp.ndarray]:
        return self._get_component(component, future_steps, as_layer)

    @partial(jax.vmap, in_axes=(0, None, None, None))
    def _get_component(
        self: Self, component: int, future_steps: int, as_layer: bool
    ) -> Union[Self, jnp.ndarray]:
        return super(BatchLayer, self).get_component(component, future_steps, as_layer)

    def device_put(self: Self, sharding: jax.sharding.PositionalSharding, num_devices: int) -> Self:
        """
        Put the BatchLayer on particular devices according to the sharding and num_devices
        args:
            sharding (jax sharding): jax positional sharding to be reshaped
            num_devices (int): number of gpus to split the batches over
        """
        assert (
            self.get_L() % num_devices
        ) == 0  # number of batches must device evenly into number of devices

        new_data = {}
        for key, image_block in self.items():
            sharding_shape = (num_devices,) + (1,) * len(image_block.shape[1:])
            new_data[key] = jax.device_put(image_block, sharding.reshape(sharding_shape))

        return self.__class__(new_data, self.D, self.is_torus)

    def reshape_pmap(self: Self, devices) -> Self:
        """
        Reshape the batch to allow pmap to work. E.g., if shape is (batch,1,N,N) and num_devices=2, then
        reshape to (2,batch/2,1,N,N)
        args:
            devices (list): list of gpus or cpu that we are using
        """
        assert self.get_L() % len(devices) == 0, (
            f"BatchLayer::reshape_pmap: length of devices must evenly "
            f"divide the total batch size, but got batch_size: {self.get_L()}, devices: {devices}"
        )

        num_devices = len(devices)

        out_layer = self.empty()
        for (k, parity), image in self.items():
            out_layer.append(
                k,
                parity,
                image.reshape((num_devices, self.get_L() // num_devices) + image.shape[1:]),
            )

        return out_layer
