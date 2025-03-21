# ------------------------------------------------------------------------------
# Functional Programming GeometricImages
# This section contains pure functions of geometric images that allows easier use of JAX fundamentals
# such as vmaps, loops, jit, and so on. All functions in this section take in images as their jnp.array data
# only, and return them as that as well.

import itertools as it
import functools
import numpy as np
from typing_extensions import Optional, Union

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx

from ginet.geometric.constants import LETTERS


def parse_shape(shape: tuple[int, ...], D: int) -> tuple[tuple[int, ...], int]:
    """
    Given a geometric image shape and dimension D, return the sidelength tuple and tensor order k.

    args:
        shape: the shape of the data of a single geometric image
        D: dimension of the image

    returns:
        tuple of spatial dimensions, tensor order
    """
    assert isinstance(shape, tuple), f"parse_shape: Shape must be a tuple, but it is {type(shape)}"
    assert len(shape) >= D, f"parse_shape: Shape {shape} is shorter than D={D}"
    return shape[:D], len(shape) - D


def hash(D: int, image: jax.Array, indices: ArrayLike) -> tuple[jax.Array]:
    """
    Converts an array of indices to their pixels on the torus by modding the indices with the
    spatial dimensions.

    args:
        D: dimension of the image
        image: image data
        indices: array of indices, shape (num_idx, D) to apply the remainder to

    returns:
        the pixel indices as a d-tuple of jax arrays
    """
    spatial_dims = jnp.array(parse_shape(image.shape, D)[0]).reshape((1, D))
    return tuple(jnp.remainder(indices, spatial_dims).transpose().astype(int))


def get_torus_expanded(
    image: jax.Array,
    is_torus: tuple[bool, ...],
    filter_spatial_dims: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
) -> tuple[jax.Array, tuple[tuple[int, int], ...]]:
    """
    For a particular filter, expand the image so that we no longer have to do convolutions on the torus, we are
    just doing convolutions on the expanded image and will get the same result.

    args:
        image: image data, (batch,spatial,channels)
        is_torus: d-length tuple of bools specifying which spatial dimensions are toroidal
        filter_spatial_dims: d-length tuple of the spatial dimensions of the filter
        rhs_dilation: dilation to apply to each filter dimension D

    Returns:
        The new expanded torus, and the appropriate padding_literal to use in convolve
    """
    # assert all the filter side lengths are odd
    assert functools.reduce(lambda carry, M: carry and (M % 2 == 1), filter_spatial_dims, True)

    # for each torus dimension, calculate the torus padding
    padding_f = lambda M, dilation, torus: ((((M - 1) // 2) * dilation),) * 2 if torus else (0, 0)
    zipped_dims = zip(filter_spatial_dims, rhs_dilation, is_torus)
    torus_padding = tuple(padding_f(M, dilation, torus) for M, dilation, torus in zipped_dims)

    # calculate indices for torus padding, then use hash to select the appropriate pixels
    expanded_image = jnp.pad(image, ((0, 0),) + torus_padding + ((0, 0),), mode="wrap")

    # zero_pad where we don't torus pad
    zero_padding = get_same_padding(
        filter_spatial_dims,
        rhs_dilation,
        tuple(not torus for torus in is_torus),
    )

    return expanded_image, zero_padding


def get_same_padding(
    filter_spatial_dims: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    pad_dims: Optional[tuple[bool, ...]] = None,
) -> tuple[tuple[int, int], ...]:
    """
    Calculate the padding for each dimension D necessary for 'SAME' padding, including rhs_dilation.

    args:
        filter_spatial_dims: filter spatial dimensions, length D tuple
        rhs_dilation: rhs (filter) dilation, length D tuple
        pad_dims: d-tuple of dimensions to pad, default (None) is all dimensions

    returns:
        d-tuple of pairs of amount of pixels to pad
    """
    pad_dims = (True,) * len(filter_spatial_dims) if pad_dims is None else pad_dims

    def padding_f(M: int, dilation: int, pad: int) -> tuple[int, int]:
        if pad:
            return (((M - 1) // 2) * dilation, ((M - 1) // 2) * dilation)
        else:
            return (0, 0)

    zipped_dims = zip(filter_spatial_dims, rhs_dilation, pad_dims)
    return tuple(padding_f(M, dilation, pad) for M, dilation, pad in zipped_dims)


def pre_tensor_product_expand(
    D: int,
    image_a: jax.Array,
    image_b: jax.Array,
    a_offset: int = 0,
    b_offset: int = 0,
    dtype: Optional[jnp.dtype] = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Rather than take a tensor product of two tensors, we can first take a tensor product of each with a tensor of
    ones with the shape of the other. Then we have two matching shapes, and we can then do whatever operations.

    args:
        D: dimension of the image
        image_a: one geometric image whose tensors we will later be doing tensor products on
        image_b: other geometric image
        a_offset: number of axes of image_a prior to the spatial dims
        b_offset: number of axes of image_b prior to the spatial dims
        dtype: if present, cast both outputs to dtype

    returns:
        tuple of the expanded images
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


def conv_contract_image_expand(D: int, image: jax.Array, filter_k: int) -> jax.Array:
    """
    For conv_contract, we will be immediately performing a contraction, so we don't need to fully expand
    each tensor, just the k image to the k+k' conv filter.

    args:
        D: dimension of the space
        image: image data, shape (in_c,spatial,tensor)
        filter_k: the filter tensor order

    returns:
        the expanded image data
    """
    _, img_k = parse_shape(image.shape[2:], D)
    k_prime = filter_k - img_k  # not to be confused with Coach Prime
    assert k_prime >= 0

    return jnp.tensordot(image, jnp.ones((D,) * k_prime), axes=0)


def mul(
    D: int,
    image_a: jax.Array,
    image_b: jax.Array,
    a_offset: int = 0,
    b_offset: int = 0,
) -> jax.Array:
    """
    Multiplication operator between two images, implemented as a tensor product of the pixels.

    args:
        D: dimension of the images
        image_a: image data
        image_b: image data
        a_offset: number of axes before the spatial axes (batch, channels, etc.)
        b_offset: number of axes before the spatial axes (batch, channels, etc.)

    returns:
        the multiplied images
    """
    image_a_data, image_b_data = pre_tensor_product_expand(D, image_a, image_b, a_offset, b_offset)
    return image_a_data * image_b_data  # now that shapes match, do elementwise multiplication


@eqx.filter_jit
def convolve(
    D: int,
    image: jax.Array,
    filter_image: jax.Array,
    is_torus: Union[tuple[bool, ...], bool],
    stride: Union[int, tuple[int, ...]] = 1,
    padding: Optional[Union[str, int, tuple[tuple[int, int], ...]]] = None,
    lhs_dilation: Optional[tuple[int, ...]] = None,
    rhs_dilation: Union[int, tuple[int, ...]] = 1,
    tensor_expand: bool = True,
) -> jax.Array:
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

    It must be the case that channel = input * feature_group_count.
    See: https://jax.readthedocs.io/en/latest/notebooks/convolutions.html#id1 and
    https://www.tensorflow.org/xla/operation_semantics#conv_convolution

    args:
        D: dimension of the images
        image: image data, shape (batch,in_c,spatial,tensor)
        filter_image: the convolution filter, shape (out_c,in_c,spatial,tensor)
        is_torus: what dimensions of the image are toroidal
        stride: convolution stride, defaults to (1,)*self.D
        padding: either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs,
            defaults to 'TORUS' if image.is_torus, else 'SAME'
        lhs_dilation: amount of dilation to apply to image in each dimension D, also transposed conv
        rhs_dilation: amount of dilation to apply to filter in each dimension D, defaults to 1
        tensor_expand: expand the tensor of image and filter to do tensor convolution, defaults to True.
            If there is something more complicated going on (e.g. conv_contract), you can skip this step.

    returns:
        convolved_image, shape (batch,out_c,spatial,tensor)
    """
    assert (D == 2) or (D == 3)
    assert image.shape[1] == filter_image.shape[1], (
        f"Second axis (in_channels) for image and filter_image "
        f"must equal, but got image {image.shape} and filter {filter_image.shape}"
    )

    filter_spatial_dims, _ = parse_shape(filter_image.shape[2:], D)
    out_c, in_c = filter_image.shape[:2]
    batch = len(image)

    if tensor_expand:
        img_expanded, filter_expanded = pre_tensor_product_expand(
            D, image, filter_image, a_offset=2, b_offset=2, dtype=jnp.float32
        )
    else:
        img_expanded, filter_expanded = image, filter_image

    _, output_k = parse_shape(filter_expanded.shape[2:], D)
    image_spatial_dims, input_k = parse_shape(img_expanded.shape[2:], D)
    channel_length = D**input_k

    # convert the image to NHWC (or NHWDC), treating all the pixel values as channels
    # (batch,in_c,spatial,in_tensor) -> (batch,spatial,in_tensor,in_c)
    img_formatted = jnp.moveaxis(img_expanded, 1, -1)
    # (batch,spatial,in_tensor,in_c) -> (batch,spatial,in_tensor*in_c)
    img_formatted = img_formatted.reshape((batch,) + image_spatial_dims + (channel_length * in_c,))

    # convert filter to HWIO (or HWDIO)
    # (out_c,in_c,spatial,out_tensor) -> (spatial,in_c,out_tensor,out_c)
    filter_formatted = jnp.moveaxis(jnp.moveaxis(filter_expanded, 0, -1), 0, D)
    # (spatial,in_c,out_tensor,out_c) -> (spatial,in_c,out_tensor*out_c)
    filter_formatted = filter_formatted.reshape(
        filter_spatial_dims + (in_c, channel_length * out_c)
    )

    # (batch,spatial,out_tensor*out_c)
    convolved_array = convolve_ravel(
        D, img_formatted, filter_formatted, is_torus, stride, padding, lhs_dilation, rhs_dilation
    )
    out_shape = convolved_array.shape[:-1] + (D,) * output_k + (out_c,)
    return jnp.moveaxis(convolved_array.reshape(out_shape), -1, 1)  # move out_c to 2nd axis


@eqx.filter_jit
def convolve_ravel(
    D: int,
    image: jax.Array,
    filter_image: jax.Array,
    is_torus: Union[tuple[bool, ...], bool],
    stride: Union[int, tuple[int, ...]] = 1,
    padding: Optional[Union[str, int, tuple[tuple[int, int], ...]]] = None,
    lhs_dilation: Optional[tuple[int, ...]] = None,
    rhs_dilation: Union[int, tuple[int, ...]] = 1,
) -> jax.Array:
    """
    Raveled verson of convolution. Assumes the channels are all lined up correctly for the tensor
    convolution. This assumes that the feature_group_count is image in_c // filter in_c.

    See [convolve](functional_geometric_image.md#ginet.geometric.functional_geometric_image.convolve) for a full
    description of this function.

    args:
        D: dimension of the images
        image: image data, shape (batch,spatial,tensor*in_c)
        filter_image: the convolution filter, shape (spatial,in_c,tensor*out_c)
        is_torus: what dimensions of the image are toroidal
        stride: convolution stride, defaults to (1,)*self.D
        padding: either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs,
            defaults to 'TORUS' if image.is_torus, else 'SAME'
        lhs_dilation: amount of dilation to apply to image in each dimension D, also transposed conv
        rhs_dilation: amount of dilation to apply to filter in each dimension D, defaults to 1

    returns:
        convolved_image, shape (batch,spatial,tensor*out_c)
    """
    assert (D == 2) or (D == 3)
    assert (isinstance(is_torus, tuple) and len(is_torus) == D) or isinstance(is_torus, bool), (
        "geom::convolve" f" is_torus must be bool or tuple of bools, but got {is_torus}"
    )

    if isinstance(is_torus, bool):
        is_torus = (is_torus,) * D

    filter_spatial_dims, _ = parse_shape(filter_image.shape, D)

    assert not (
        functools.reduce(lambda carry, N: carry or (N % 2 == 0), filter_spatial_dims, False)
        and (padding == "TORUS" or padding == "SAME" or padding is None)
    ), f"convolve: Filters with even sidelengths {filter_spatial_dims} require literal padding, not {padding}"

    if not isinstance(rhs_dilation, tuple):
        rhs_dilation = (rhs_dilation,) * D

    if not isinstance(stride, tuple):
        stride = (stride,) * D

    if padding is None:  # if unspecified, infer from is_torus
        padding = "TORUS" if len(list(filter(lambda x: x, is_torus))) else "SAME"

    if (lhs_dilation is not None) and isinstance(padding, str):
        print(
            "WARNING convolve: lhs_dilation (transposed convolution) should specify padding exactly, "
            "see https://arxiv.org/pdf/1603.07285.pdf for the appropriate cases."
        )

    if padding == "TORUS":
        image, padding_literal = get_torus_expanded(
            image, is_torus, filter_spatial_dims, rhs_dilation
        )
    elif padding == "VALID":
        padding_literal = ((0, 0),) * D
    elif padding == "SAME":
        padding_literal = get_same_padding(filter_spatial_dims, rhs_dilation)
    elif isinstance(padding, int):
        padding_literal = ((padding, padding),) * D
    else:
        padding_literal = padding

    assert (image.shape[-1] // filter_image.shape[-2]) == (image.shape[-1] / filter_image.shape[-2])
    channel_length = image.shape[-1] // filter_image.shape[-2]

    # (batch,spatial,out_tensor*out_c)
    convolved_array = jax.lax.conv_general_dilated(
        image,  # lhs
        filter_image,  # rhs
        stride,
        padding_literal,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=(("NHWC", "HWIO", "NHWC") if D == 2 else ("NHWDC", "HWDIO", "NHWDC")),
        feature_group_count=channel_length,  # each tensor component is treated separately
    )
    return convolved_array


@eqx.filter_jit
def convolve_contract(
    D: int,
    image: jax.Array,
    filter_image: jax.Array,
    is_torus: Union[bool, tuple[bool, ...]],
    stride: Union[int, tuple[int, ...]] = 1,
    padding: Optional[Union[str, int, tuple[tuple[int, int], ...]]] = None,
    lhs_dilation: Optional[tuple[int, ...]] = None,
    rhs_dilation: Union[int, tuple[int, ...]] = 1,
) -> jax.Array:
    """
    Given an input k image and a k+k' filter, take the tensor convolution that contract k times with one index
    each from the image and filter. This implementation is slightly more efficient then doing the convolution
    and contraction separately by avoiding constructing the k+k+k' intermediate tensor. See
    [convolve](functional_geometric_image.md#ginet.geometric.functional_geometric_image.convolve) for a full
    description of the convolution.

    args:
        D: dimension of the images
        image: image data, shape (batch,in_c,spatial,tensor)
        filter_image: the convolution filter, shape (out_c,in_c,spatial,tensor)
        is_torus: what dimensions of the image are toroidal
        stride: convolution stride, defaults to (1,)*self.D
        padding: either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs,
            defaults to 'TORUS' if image.is_torus, else 'SAME'
        lhs_dilation: amount of dilation to apply to image in each dimension D, also transposed conv
        rhs_dilation: amount of dilation to apply to filter in each dimension D, defaults to 1

    returns:
        convolved_image, shape (batch,out_c,spatial,tensor)
    """
    _, img_k = parse_shape(image.shape[2:], D)
    _, filter_k = parse_shape(filter_image.shape[2:], D)
    img_expanded = conv_contract_image_expand(D, image, filter_k).astype("float32")
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
    swappable_idxs: tuple[tuple[int, int], ...] = (),
) -> list[tuple[tuple[int, int], ...]]:
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
        initial_k: the starting number of indices that we have
        final_k: the final number of indices that we want to end up with
        swappable_idxs: Indices that can swapped w/o changing the contraction

    returns:
        all the possible contraction indices
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


@functools.partial(jax.jit, static_argnums=[1, 2])
def multicontract(
    data: jax.Array, indices: tuple[tuple[int, int], ...], idx_shift: int = 0
) -> jax.Array:
    """
    Perform the Kronecker Delta contraction on the data. Must have at least 2 dimensions, and because we implement with
    einsum, must have at most 52 dimensions. Indices a tuple of pairs of indices, also tuples.

    args:
        data: data to perform the contraction on
        indices: index pairs to perform the contractions on
        idx_shift: indices are the tensor indices, so if data has spatial indices or channel/batch
            indices in the beginning we shift over by idx_shift

    returns:
        the contracted data
    """
    dimensions = len(data.shape)
    assert dimensions + len(indices) < 52
    assert dimensions >= 2
    # all indices must be unique, indices must be greater than 0 and less than dimensions

    einstr = list(LETTERS[:dimensions])
    for i, (idx1, idx2) in enumerate(indices):
        einstr[idx1 + idx_shift] = einstr[idx2 + idx_shift] = LETTERS[-(i + 1)]

    return jnp.einsum("".join(einstr), data)


def get_rotated_keys(D: int, data: jax.Array, gg: np.ndarray) -> np.ndarray:
    """
    Get the rotated keys of data when it will be rotated by gg. Note that we rotate the key vector indices
    by the inverse of gg per the definition (this is done by key_array @ gg, rather than gg @ key_array).
    When the spatial_dims are not square, this gets a little tricky.
    The gg needs to be a concrete (numpy) array, not a traced jax array.

    args:
        D: dimension of image
        data: data array to be rotated
        gg: group operation

    returns:
        the rotated keys
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
    data: jax.Array,
    parity: int,
    gg: np.ndarray,
    precision: Optional[jax.lax.Precision] = None,
) -> jax.Array:
    """
    Apply a group element of SO(2) or SO(3) to the geometric image. First apply the action to the location of the
    pixels, then apply the action to the pixels themselves.

    args:
        D: dimension of the data
        data: data block of image data to rotate, shape (spatial,tensor)
        parity: parity of the data, 0 for even parity, 1 for odd parity
        gg: a DxD matrix that rotates the tensor. Note that you cannot vmap
            by this argument because it needs to deal with concrete values
        precision: eisnum precision, normally uses lower precision, use
            jax.lax.Precision.HIGH for testing equality in unit tests

    returns:
        the rotated image data
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
    tensor: jax.Array,
    parity: int,
    gg: np.ndarray,
    precision: Optional[jax.lax.Precision] = None,
) -> jax.Array:
    """
    Apply a group element of SO(2) or SO(3) to a single tensor.

    args:
        tensor: data of the tensor
        parity: parity of the data, 0 for even parity, 1 for odd parity
        gg: a DxD matrix that rotates the tensor. Note that you cannot vmap
            by this argument because it needs to deal with concrete values
        precision: eisnum precision, normally uses lower precision, use
            jax.lax.Precision.HIGH for testing equality in unit tests

    returns:
        rotated tensor data
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


def norm(idx_shift: int, data: jax.Array, keepdims: bool = False) -> jax.Array:
    """
    Perform the frobenius norm on each pixel tensor, returning a scalar image

    args:
        idx_shift: the number of leading axes before the tensor, should be D for spatial plus
            the batch and spatial axes if they
        data: image data, shape (spatial,tensor)
        keepdims: passed to jnp.linalg.norm

    returns:
        the data of a scalar image after performing the norm
    """
    assert (
        idx_shift <= data.ndim
    ), f"norm: idx shift must be at most ndim, but {idx_shift} > {data.ndim}"
    if data.ndim == idx_shift:  # in this case, reshape creates an axis, so we need to collapse it
        keepdims = False

    normed_data = jnp.linalg.norm(data.reshape(data.shape[:idx_shift] + (-1,)), axis=idx_shift)
    if keepdims:
        extra_axes = data.ndim - normed_data.ndim
        return normed_data.reshape(normed_data.shape + (1,) * extra_axes)
    else:
        return normed_data


@functools.partial(jax.jit, static_argnums=[0, 2, 3])
def max_pool(
    D: int,
    image_data: jax.Array,
    patch_len: int,
    use_norm: bool = True,
    comparator_image: Optional[jax.Array] = None,
) -> jax.Array:
    """
    Perform a max pooling operation where the length of the side of each patch is patch_len. Max is
    determined by the value of comparator_image if present, then the norm of image_data if use_norm
    is true, then finally the image_data otherwise.

    args:
        D: the dimension of the space
        image_data: the image data, shape (spatial,tensor)
        patch_len: the side length of the patches, must evenly divide all spatial dims
        use_norm: if true, use the norm (over the tensor) of the image as the comparator image
        comparator_image: scalar image whose argmax is used to determine what value to use.

    returns:
        the image data that has been max pooled, shape (spatial,tensor)
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
    vmap_max = jax.vmap(lambda patch, idx: patch[:, idx], in_axes=(2, 0))
    return vmap_max(patches, idxs).reshape(new_spatial_dims + (D,) * k)


@functools.partial(jax.jit, static_argnums=[0, 2])
def average_pool(D: int, image_data: jax.Array, patch_len: int) -> jax.Array:
    """
    Perform a average pooling operation where the length of the side of each patch is patch_len. This is
    equivalent to doing a convolution where each element of the filter is 1 over the number of pixels in the
    filter, the stride length is patch_len, and the padding is 'VALID'.

    args:
        D: dimension of data
        image_data: image data, shape (spatial,tensor)
        patch_len: the side length of the patches, must evenly divide the sidelength

    returns:
        the image data after being averaged pooled, shape (spatial, tensor)
    """
    spatial_dims, _ = parse_shape(image_data.shape, D)
    assert functools.reduce(lambda carry, N: carry and (N % patch_len == 0), spatial_dims, True)
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
