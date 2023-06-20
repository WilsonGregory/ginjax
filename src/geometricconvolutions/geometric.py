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

import itertools as it
import numpy as np #removing this
import jax.numpy as jnp
import jax.lax
import jax.nn
from jax import jit, vmap
from jax.tree_util import register_pytree_node_class
from functools import partial, reduce

TINY = 1.e-5
LETTERS = 'abcdefghijklmnopqrstuvwxyxABCDEFGHIJKLMNOPQRSTUVWXYZ'

# ------------------------------------------------------------------------------
# PART 1: Make and test a complete group

def permutation_matrix_from_sequence(seq):
    """
    Give a sequence tuple, return the permutation matrix for that sequence
    """
    D = len(seq)
    permutation_matrix = []
    for num in seq:
        row = [0]*D
        row[num] = 1
        permutation_matrix.append(row)
    return jnp.array(permutation_matrix)

def make_all_operators(D):
    """
    Construct all operators of dimension D that are rotations of 90 degrees, or reflections, or a combination of the
    two. This is equivalent to all the permutation matrices where each entry can either be +1 or -1
    args:
        D (int): dimension of the operator
    """

    # permutation matrices, one for each permutation of length D
    permutation_matrices = [permutation_matrix_from_sequence(seq) for seq in it.permutations(range(D))]
    # possible entries, e.g. for D=2: (1,1), (-1,1), (1,-1), (-1,-1)
    possible_entries = [np.diag(prod) for prod in it.product([1,-1], repeat=D)]

    #combine all the permutation matrices with the possible entries, then flatten to a single array of operators
    return list(it.chain(*list(map(lambda matrix: [matrix @ prod for prod in possible_entries], permutation_matrices))))


# ------------------------------------------------------------------------------
# PART 2: Define the Kronecker Delta and Levi Civita symbols to be used in Levi Civita contractions

class KroneckerDeltaSymbol:
    #we only want to create each dimension of levi civita symbol once, so we cache them in this dictionary
    symbol_dict = {}

    @classmethod
    def get(cls, D, k):
        """
        Get the Levi Civita symbol for dimension D from the cache, or creating it on a cache miss
        args:
            D (int): dimension of the Kronecker symbol
            k (int): order of the Kronecker Delta symbol
        """
        assert D > 1
        assert k > 1
        if (D,k) not in cls.symbol_dict:
            arr = np.zeros((k * (D,)), dtype=int)
            for i in range(D):
                arr[(i,)*k] = 1
            cls.symbol_dict[(D,k)] = arr

        return cls.symbol_dict[(D,k)]

    @classmethod
    def get_image(cls, N, D, k):
        return GeometricImage(jnp.stack([cls.get(D,k) for _ in range(N**D)]).reshape(((N,)*D + (D,)*k)), 0, D)

def permutation_parity(pi):
    """
    Code taken from Sympy Permutations: https://github.com/sympy/sympy/blob/26f7bdbe3f860e7b4492e102edec2d6b429b5aaf/sympy/combinatorics/permutations.py#L114
    Slightly modified to return 1 for even permutations, -1 for odd permutations, and 0 for repeated digits
    Permutations of length n must consist of numbers {0, 1, ..., n-1}
    """
    if (len(np.unique(pi)) != len(pi)):
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
    return -2*((n - c) % 2)+1

class LeviCivitaSymbol:

    #we only want to create each dimension of levi civita symbol once, so we cache them in this dictionary
    symbol_dict = {}

    @classmethod
    def get(cls, D):
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

def get_unique_invariant_filters(M, k, parity, D, operators, scale='normalize'):
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
    assert scale == 'normalize' or scale == 'one'

    # make the seed filters
    shape = (M,)*D + (D,)*k
    size = np.multiply.reduce(shape)
    allfilters = []

    for i in range(size):
        data = [0]*size
        data[i] = 1
        allfilters.append(GeometricFilter(jnp.array(data).reshape(shape), parity, D))

    # do the group averaging
    bigshape = (len(allfilters), size)
    filter_matrix = np.zeros(bigshape)
    for i, f1 in enumerate(allfilters):
        ff = GeometricFilter.zeros(M, k, parity, D)
        for gg in operators:
            ff = ff + f1.times_group_element(gg)
        filter_matrix[i] = ff.data.flatten()

    # do the SVD
    u, s, v = np.linalg.svd(filter_matrix)
    sbig = s > TINY
    if not np.any(sbig):
        return []

    # normalize the amplitudes so they max out at +/- 1.
    amps = v[sbig] / np.max(np.abs(v[sbig]), axis=1)[:, None]
    # make sure the amps are positive, generally
    for i in range(len(amps)):
        if np.sum(amps[i]) < 0:
            amps[i] *= -1
    # make sure that the zeros are zeros.
    amps[np.abs(amps) < TINY] = 0.

    # order them
    filters = [GeometricFilter(aa.reshape(shape), parity, D) for aa in amps]
    if (scale == 'normalize'):
        filters = [ff.normalize() for ff in filters]

    norms = [ff.bigness() for ff in filters]
    I = np.argsort(norms)
    filters = [filters[i] for i in I]

    # now do k-dependent rectification:
    filters = [ff.rectify() for ff in filters]

    return filters

def get_invariant_filters(Ms, ks, parities, D, operators, scale='normalize', return_type='layer', return_maxn=False):
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
    assert scale == 'normalize' or scale == 'one'
    assert return_type in { 'dict', 'list', 'layer' }

    allfilters = {}
    maxn = {}
    for M in Ms: #filter side length
        maxn[(D, M)] = 0
        for k in ks: #tensor order
            for parity in parities: #parity
                key = (D, M, k, parity)
                allfilters[key] = get_unique_invariant_filters(M, k, parity, D, operators, scale)
                n = len(allfilters[key])
                if n > maxn[(D, M)]:
                    maxn[(D, M)] = n

    allfilters_list = list(it.chain(*list(allfilters.values())))
    if return_type == 'list':
        allfilters = allfilters_list
    elif return_type == 'layer':
        allfilters = Layer.from_images(allfilters_list)
    # else, allfilters is the default structure

    if return_maxn:
        return allfilters, maxn
    else:
        return allfilters


# ------------------------------------------------------------------------------
# PART 4: Functional Programming GeometricImages
# This section contains pure functions of geometric images that allows easier use of JAX fundamentals 
# such as vmaps, loops, jit, and so on. All functions in this section take in images as their jnp.array data
# only, and return them as that as well.

def parse_shape(shape, D):
    """
    Given a geometric image shape and dimension D, return the sidelength N and tensor order k.
    args:
        shape (shape tuple): the shape of the data of a single geoemtric image
        D (int): dimension of the image
    """
    return shape[0], len(shape) - D

def hash(D, image, indices):
    """
    Deals with torus by modding (with `np.remainder()`).
    args:
        D (int): dimension of hte image
        image (jnp.array): data of the image
        indices (tuple of ints): indices to apply the remainder to
    """
    img_N, _ = parse_shape(image.shape, D)
    return tuple(jnp.remainder(indices, img_N).transpose().astype(int))

def get_torus_expanded(D, image, filter_image, dilation):
        """
        For a particular filter, expand the image so that we no longer have to do convolutions on the torus, we are
        just doing convolutions on the expanded image and will get the same result. Return a new GeometricImage
        args:
            D (int): dimension of the image
            image (jnp.array): image data
            filter_image (jnp.array): filter data, how much is expanded depends on filter_image.m
            dilation (int): dilation to apply to each filter dimension D
        """
        img_N, img_k = parse_shape(image.shape, D)
        filter_N, _ = parse_shape(filter_image.shape, D)
        assert filter_N % 2 == 1

        padding = ((filter_N - 1) // 2) * dilation
        indices = jnp.array(list(it.product(range(img_N + 2 * padding), repeat=D)), dtype=int) - padding
        return image[hash(D, image, indices)].reshape((img_N + 2 * padding,)*D + (D,)*img_k)

def pre_tensor_product_expand(D, image_a, image_b):
    """
    Rather than take a tensor product of two tensors, we can first take a tensor product of each with a tensor of
    ones with the shape of the other. Then we have two matching shapes, and we can then do whatever operations.
    args:
        D (int): dimension of the image
        image_a (GeometricImage like): one geometric image whose tensors we will later be doing tensor products on
        image_b (GeometricImage like): other geometric image
    """
    _, img_a_k = parse_shape(image_a.shape, D)
    _, img_b_k = parse_shape(image_b.shape, D)

    if (img_b_k > 0):
        image_a_expanded = jnp.tensordot(
            image_a,
            jnp.ones((D,)*img_b_k),
            axes=0,
        )
    else:
        image_a_expanded = image_a

    if (img_a_k > 0):
        break1 = img_a_k + D #after outer product, end of image_b N^D axes
        #after outer product: [D^ki, N^D, D^kf], convert to [N^D, D^ki, D^kf]
        # we are trying to expand the ones in the middle (D^ki), so we add them on the front, then move to middle
        image_b_expanded = jnp.transpose(
            jnp.tensordot(jnp.ones((D,)*img_a_k), image_b, axes=0),
            list(
                tuple(range(img_a_k, break1)) + tuple(range(img_a_k)) + tuple(range(break1, break1 + img_b_k))
            ),
        )
    else:
        image_b_expanded = image_b

    return image_a_expanded, image_b_expanded

def mul(D, image_a, image_b):
    """
    Multiplication operator between two images, implemented as a tensor product of the pixels.
    args:
        D (int): dimension of the images
        image_a (jnp.array): image data
        image_b (jnp.array): image data
    """
    image_a_data, image_b_data = pre_tensor_product_expand(D, image_a, image_b)
    return image_a_data * image_b_data #now that shapes match, do elementwise multiplication

@partial(jit, static_argnums=[0,3,4,5,6,7])
def convolve(
    D,
    image,
    filter_image, 
    is_torus,
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None, 
):
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
        image (jnp.array): image data
        filter_image (jnp.array): the convolution filter
        is_torus (bool): whether the images data is on the torus or not
        stride (tuple of ints): convolution stride, defaults to (1,)*self.D
        padding (either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs): 
            defaults to 'TORUS' if image.is_torus, else 'SAME'
        lhs_dilation (tuple of ints): amount of dilation to apply to image in each dimension D, also transposed conv
        rhs_dilation (tuple of ints): amount of dilation to apply to filter in each dimension D, defaults to 1
    """
    assert (D == 2) or (D == 3)
    dtype= 'float32'

    _, img_k = parse_shape(image.shape, D)
    filter_N, filter_k = parse_shape(filter_image.shape, D)

    output_k = img_k + filter_k

    if rhs_dilation is None:
        rhs_dilation = (1,)*D

    if stride is None:
        stride = (1,)*D

    if padding is None: #if unspecified, infer from is_torus
        padding = 'TORUS' if is_torus else 'SAME'

    if padding == 'TORUS':
        image = get_torus_expanded(D, image, filter_image, rhs_dilation[0])
        padding_literal = ((0,0),)*D
    else:
        if padding == 'VALID':
            padding_literal = ((0,0),)*D
        elif padding == 'SAME':
            filter_m = (filter_N - 1) // 2
            padding_literal = ((filter_m * dilation,) * 2 for dilation in rhs_dilation)
        else:
            padding_literal = padding

    img_expanded, filter_expanded = pre_tensor_product_expand(D, image, filter_image)
    img_expanded = img_expanded.astype(dtype)
    filter_expanded = filter_expanded.astype(dtype)

    channel_length = D**output_k

    # convert the image to NHWC (or NHWDC), treating all the pixel values as channels
    # batching is handled by using vmap on this function
    img_formatted = img_expanded.reshape((1,) + tuple(img_expanded.shape[:D]) + (channel_length,))

    # convert filter to HWIO (or HWDIO)
    filter_formatted = filter_expanded.reshape((filter_N,)*D + (1,channel_length))

    convolved_array = jax.lax.conv_general_dilated(
        img_formatted, #lhs
        filter_formatted, #rhs
        stride,
        padding_literal,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=(('NHWC','HWIO','NHWC') if D == 2 else ('NHWDC','HWDIO','NHWDC')),
        feature_group_count=channel_length, #each tensor component is treated separately
    )
    return convolved_array.reshape(convolved_array.shape[1:-1] + (D,)*output_k)

@partial(jit, static_argnums=[0,3,4,5,6,7])
def depth_convolve(
    D,
    image,
    filter_image, 
    is_torus,
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None, 
):
    """
    See convolve for a full description. This function performs depth convolutions by applying a vmap
    over regular convolutions to the image and filter_image arguments, then taking the sum of the result.
    Possibly would be faster to do this inside convolve.
    args:
        image (jnp.array): array of shape (depth, (N,)*D, (D,)*img_k)
        filter_image (jnp.array): array of shape (depth, (N,)*D, (D,)*filter_k)
    """
    assert image.shape[0] == filter_image.shape[0]
    vmap_convolve = vmap(convolve, in_axes=(None, 0, 0, None, None, None, None, None))
    res = vmap_convolve(D, image, filter_image, is_torus, stride, padding, lhs_dilation, rhs_dilation)
    return jnp.sum(res, axis=0)

def get_contraction_indices(initial_k, final_k, swappable_idxs=()):
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

    tuple_pairs = it.combinations(it.combinations(range(initial_k),2),(initial_k - final_k) // 2)
    rows = np.array([np.array(pair).reshape((initial_k - final_k,)) for pair in tuple_pairs])
    unique_rows = np.array([True if len(np.unique(row)) == len(row) else False for row in rows])
    unique_pairs = rows[unique_rows] #remove rows which have an index multiple times

    # replace every element of the second term of the swappable pair with the first term
    for a,b in swappable_idxs:
        unique_pairs[np.where(np.isin(unique_pairs, b))] = a

    # convert back to lists
    sorted_tuples = [sorted(sorted([x,y]) for x,y in zip(row[0::2], row[1::2])) for row in unique_pairs]
    sorted_rows = np.array([np.array(pair).reshape((initial_k - final_k,)) for pair in sorted_tuples])
    unique_sorted_rows = np.unique(sorted_rows, axis=0) #after sorting remove redundant rows

    # restore by elements of the swappable pairs to being in the sequences
    for pair in swappable_idxs:
        for row in unique_sorted_rows:
            locs = np.isin(row, pair)
            if len(np.where(locs)[0]) > 0:
                row[np.max(np.where(locs))] = pair[1]
                row[np.min(np.where(locs))] = pair[0] #if there is only 1, it will get set to pair 0

    return [tuple((x,y) for x,y in zip(idxs[0::2], idxs[1::2])) for idxs in unique_sorted_rows]

@partial(jit, static_argnums=1)
def multicontract(data, indices):
    """
    Perform the Kronecker Delta contraction on the data. Must have at least 2 dimensions, and because we implement with
    einsum, must have at most 52 dimensions. Indices a tuple of pairs of indices, also tuples.
    args:
        data (np.array-like): data to perform the contraction on
        indices (tuple of tuples of ints): index pairs to perform the contractions on
    """
    dimensions = len(data.shape)
    assert dimensions + len(indices) < 52
    assert dimensions >= 2
    #all indices must be unique, indices must be greater than 0 and less than dimensions

    einstr = list(LETTERS[:dimensions])
    for i, (idx1, idx2) in enumerate(indices):
        einstr[idx1] = einstr[idx2] = LETTERS[-(i+1)]
    
    return jnp.einsum(''.join(einstr), data)

@jit
def linear_combination(images, params):
    """
    A method takes a list of parameters, a list of geometric images and returns the linear combination.
    args:
        images (jnp.array): block of image data where the first axis is the image
        params (jnp.array): scalar multipliers of the images
    """
    return jnp.sum(vmap(lambda image, param: image * param)(images, params), axis=0)

# ------------------------------------------------------------------------------
# PART 5: Define geometric (k-tensor, torus) images.

def tensor_name(k, parity):
    nn = "tensor"
    if k == 0:
        nn = "scalar"
    if k == 1:
        nn = "vector"
    if parity % 2 == 1 and k < 2:
        nn = "pseudo" + nn
    if k > 1:
        if parity == 0:
            nn = r'${}_{}-$'.format(k, '{(+)}') + nn
        else:
            nn = r'${}_{}-$'.format(k, '{(-)}') + nn

    return nn

@register_pytree_node_class
class GeometricImage:

    # Constructors

    @classmethod
    def zeros(cls, N, k, parity, D, is_torus=True):
        """
        Class method zeros to construct a geometric image of zeros
        args:
            N (int): length of a side of an image, currently all images must be square N^D pixels
            k (int): the order of the tensor in each pixel, i.e. 0 (scalar), 1 (vector), 2 (matrix), etc.
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        shape = D * (N, ) + k * (D, )
        return cls(jnp.zeros(shape), parity, D, is_torus)

    @classmethod
    def fill(cls, N, parity, D, fill, is_torus=True):
        """
        Class method fill constructor to construct a geometric image every pixel as fill
        args:
            N (int): length of a side of an image, currently all images must be square N^D pixels
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            fill (jnp.ndarray or number): tensor to fill the image with
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        k = len(fill.shape) if isinstance(fill, jnp.ndarray) else 0
        data = jnp.stack([fill for _ in range(N ** D)]).reshape((N,)*D + (D,)*k)
        return cls(data, parity, D, is_torus)

    def __init__(self, data, parity, D, is_torus=True):
        """
        Construct the GeometricImage. It will be (N^D x D^k), so if N=100, D=2, k=1, then it's (100 x 100 x 2)
        args:
            data (array-like):
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        self.D = D
        self.N = len(data)
        self.k = len(data.shape) - D
        assert data.shape[:D] == self.D * (self.N, ), \
        "GeometricImage: data must be square."
        assert data.shape[D:] == self.k * (self.D, ), \
        "GeometricImage: each pixel must be D cross D, k times"
        self.parity = parity % 2
        self.is_torus = is_torus
        self.data = jnp.copy(data) #TODO: don't need to copy if data is already an immutable jnp array

    def copy(self):
        return self.__class__(self.data, self.parity, self.D, self.is_torus)

    # Getters, setters, basic info

    def hash(self, indices):
        """
        Deals with torus by modding (with `np.remainder()`).
        args:
            indices (tuple of ints): indices to apply the remainder to
        """
        return hash(self.D, self.data, indices)

    def __getitem__(self, key):
        """
        Accessor for data values. Now you can do image[key] where k are indices or array slices and it will just work
        Note that JAX does not throw errors for indexing out of bounds
        args:
            key (index): JAX/numpy indexer, i.e. "0", "0,1,3", "4:, 2:3, 0" etc.
        """
        return self.data[key]

    def __setitem__(self, key, val):
        """
        Jax arrays are immutable, so this reconstructs the data object with copying, and is potentially slow
        """
        self.data = self.data.at[key].set(val)
        return self

    def shape(self):
        """
        Return the full shape of the data block
        """
        return self.data.shape

    def image_shape(self, plus_N=0):
        """
        Return the shape of the data block that is not the ktensor shape, but what comes before that. For regular
        GeometricImages, this is shape of the literal image. For BatchGeometricImage it prepends the batch size L.
        args:
            plus_N (int): numer to add to self.N, useful when growing/shrinking the image
        """
        return self.D*(self.N + plus_N,)

    def pixel_shape(self):
        """
        Return the shape of the data block that is the ktensor, aka the pixel of the image.
        """
        return self.k*(self.D,)

    def pixel_size(self):
        """
        Get the size of the pixel shape, i.e. (D,D,D) = D**3
        """
        return self.D ** self.k

    def __str__(self):
        return "<{} object in D={} with N={}, k={}, parity={}, is_torus={}>".format(
            self.__class__, self.D, self.N, self.k, self.parity, self.is_torus)

    def keys(self):
        """
        Iterate over the keys of GeometricImage
        """
        return it.product(range(self.N), repeat=self.D)

    def key_array(self):
        # equivalent to the old pixels function
        return jnp.array([key for key in self.keys()], dtype=int)

    def pixels(self):
        """
        Iterate over the pixels of GeometricImage.
        """
        for key in self.keys():
            yield self[key]

    def items(self):
        """
        Iterate over the key, pixel pairs of GeometricImage.
        """
        for key in self.keys():
            yield (key, self[key])

    # Binary Operators, Complicated functions

    def __eq__(self, other):
        """
        Equality operator, must have same shape, parity, and data within the TINY=1e-5 tolerance.
        """
        return (
            self.D == other.D and
            self.N == other.N and
            self.k == other.k and
            self.parity == other.parity and
            self.is_torus == other.is_torus and
            self.data.shape == other.data.shape and
            jnp.allclose(self.data, other.data, rtol=TINY, atol=TINY)
        )

    def __add__(self, other):
        """
        Addition operator for GeometricImages. Both must be the same size and parity. Returns a new GeometricImage.
        args:
            other (GeometricImage): other image to add the the first one
        """
        assert self.D == other.D
        assert self.N == other.N
        assert self.k == other.k
        assert self.parity == other.parity
        assert self.is_torus == other.is_torus
        assert self.data.shape == other.data.shape
        return self.__class__(self.data + other.data, self.parity, self.D, self.is_torus)

    def __sub__(self, other):
        """
        Subtraction operator for GeometricImages. Both must be the same size and parity. Returns a new GeometricImage.
        args:
            other (GeometricImage): other image to add the the first one
        """
        assert self.D == other.D
        assert self.N == other.N
        assert self.k == other.k
        assert self.parity == other.parity
        assert self.is_torus == other.is_torus
        assert self.data.shape == other.data.shape
        return self.__class__(self.data - other.data, self.parity, self.D, self.is_torus)

    def __mul__(self, other):
        """
        If other is a scalar, do scalar multiplication of the data. If it is another GeometricImage, do the tensor
        product at each pixel. Return the result as a new GeometricImage.
        args:
            other (GeometricImage or number): scalar or image to multiply by
        """
        if (isinstance(other, GeometricImage)):
            assert self.D == other.D
            assert self.N == other.N
            assert self.is_torus == other.is_torus
            return self.__class__(
                mul(self.D, self.data, other.data), 
                self.parity + other.parity, 
                self.D, 
                self.is_torus,
            )
        else: #its an integer or a float, or something that can we can multiply a Jax array by (like a DeviceArray)
            return self.__class__(self.data * other, self.parity, self.D, self.is_torus)

    def __rmul__(self, other):
        """
        If other is a scalar, multiply the data by the scalar. This is necessary for doing scalar * image, and it
        should only be called in that case.
        """
        return self * other

    def transpose(self, axes_permutation):
        """
        Transposes the axes of the tensor, keeping the image axes in the front the same
        args:
            axes_permutation (iterable of indices): new axes order
        """
        idx_shift = len(self.image_shape())
        new_indices = tuple(tuple(range(idx_shift)) + tuple(axis + idx_shift for axis in axes_permutation))
        return self.__class__(jnp.transpose(self.data, new_indices), self.parity, self.D, self.is_torus)

    @partial(jit, static_argnums=[2,3,4,5])
    def convolve_with(
        self, 
        filter_image, 
        stride=None, 
        padding=None,
        lhs_dilation=None, 
        rhs_dilation=None, 
    ):
        """
        See convolve for a description of this function.
        """
        convolved_array = convolve(
            self.D, 
            self.data, 
            filter_image.data, 
            self.is_torus,
            stride,
            padding,
            lhs_dilation,
            rhs_dilation,
        )
        return self.__class__(
            convolved_array, 
            self.parity + filter_image.parity, 
            self.D, 
            self.is_torus,
        )

    @partial(jit, static_argnums=1)
    def max_pool(self, patch_len):
        """
        Perform a max pooling operation where the length of the side of each patch is patch_len. Max is determined by
        the norm of the pixel. Note that for scalars, this will be the absolute value of the pixel.
        args:
            patch_len (int): the side length of the patches, must evenly divide self.N
        """
        assert (self.N % patch_len) == 0
        plus_N = -1*(self.N - int(self.N / patch_len))
        norm_data = self.norm()

        idxs = jnp.array(list(it.product(range(patch_len), repeat=self.D)))
        max_idxs = []
        for base in it.product(range(0, self.N, patch_len), repeat=self.D):
            block_idxs = jnp.array(base) + idxs
            max_hash_idx = jnp.argmax(norm_data[self.hash(block_idxs)])
            max_idxs.append(block_idxs[max_hash_idx])

        max_data = self[self.hash(jnp.array(max_idxs))].reshape(self.image_shape(plus_N) + self.pixel_shape())
        return self.__class__(max_data, self.parity, self.D, self.is_torus)

    @partial(jit, static_argnums=1)
    def average_pool(self, patch_len):
        """
        Perform a average pooling operation where the length of the side of each patch is patch_len. This is 
        equivalent to doing a convolution where each element of the filter is 1 over the number of pixels in the 
        filter, the stride length is patch_len, and the padding is 'VALID'.
        args:
            patch_len (int): the side length of the patches, must evenly divide self.N
        """
        assert (self.N % patch_len) == 0

        avg_filter = GeometricImage((1/(patch_len ** self.D)) * jnp.ones((patch_len,)*self.D), 0, self.D)
        return self.convolve_with(avg_filter, stride=(patch_len,) * self.D, padding='VALID')

    @partial(jit, static_argnums=1)
    def unpool(self, patch_len):
        """
        Each pixel turns into a (patch_len,)*self.D patch of that pixel. Also called "Nearest Neighbor" unpooling
        args:
            patch_len (int): side length of the patch of our unpooled images
        """
        grow_filter = GeometricImage(jnp.ones((patch_len,)*self.D), 0, self.D)
        return self.convolve_with(grow_filter, padding=((patch_len-1,)*2,)*self.D, lhs_dilation=(patch_len,)*self.D)

    def times_scalar(self, scalar):
        """
        Scale the data by a scalar, returning a new GeometricImage object. Alias of the multiplication operator.
        args:
            scalar (number): number to scale everything by
        """
        return self * scalar

    @jit
    def norm(self):
        """
        Calculate the norm pixel-wise
        """
        #not sure if there is a jax way to avoid unrolling this loop
        #construct a norm function that takes the norm of each pixel, so it maps over all the image_shape axes
        vmap_norm = reduce(lambda f,_: vmap(f), range(len(self.image_shape())), jnp.linalg.norm)

        # it is possible that the parity of the new image is always 0
        return self.__class__(vmap_norm(self.data), self.parity, self.D, self.is_torus)

    def normalize(self):
        """
        Normalize so that the max norm of each pixel is 1, and all other tensors are scaled appropriately
        """
        max_norm = jnp.max(self.norm().data)
        if max_norm > TINY:
            return self.times_scalar(1. / max_norm)
        else:
            return self.times_scalar(1.)

    def activation_function(self, function):
        assert self.k == 0, "Activation functions only implemented for k=0 tensors due to equivariance"
        return self.__class__(function(self.data), self.parity, self.D, self.is_torus)

    @partial(jit, static_argnums=[1,2])
    def contract(self, i, j):
        """
        Use einsum to perform a kronecker contraction on two dimensions of the tensor
        args:
            i (int): first index of tensor
            j (int): second index of tensor
        """
        assert self.k >= 2
        idx_shift = len(self.image_shape())
        return self.__class__(
            multicontract(self.data, ((i + idx_shift, j + idx_shift),)),
            self.parity,
            self.D,
            self.is_torus,
        )

    @partial(jit, static_argnums=1)
    def multicontract(self, indices):
        """
        Use einsum to perform a kronecker contraction on two dimensions of the tensor
        args:
            indices (tuple of tuples of ints): indices to contract
        """
        assert self.k >= 2
        idx_shift = len(self.image_shape())
        shifted_indices = tuple((i + idx_shift, j + idx_shift) for i,j in indices)
        return self.__class__(multicontract(self.data, shifted_indices), self.parity, self.D, self.is_torus)

    def levi_civita_contract(self, indices):
        """
        Perform the Levi-Civita contraction. Outer product with the Levi-Civita Symbol, then perform D-1 contractions.
        Resulting image has k= self.k - self.D + 2
        args:
            indices (int, or tuple, or list): indices of tensor to perform contractions on
        """
        assert self.k >= (self.D - 1) # so we have enough indices to work on since we perform D-1 contractions
        if self.D == 2 and not (isinstance(indices, tuple) or isinstance(indices, list)):
            indices = (indices,)
        assert len(indices) == self.D - 1

        levi_civita = LeviCivitaSymbol.get(self.D)
        outer = jnp.tensordot(self.data, levi_civita, axes=0)

        #make contraction index pairs with one of specified indices, and index (in order) from the levi_civita symbol
        idx_shift = len(self.image_shape())
        zipped_indices = tuple((i+idx_shift,j+idx_shift) for i,j in zip(indices, range(self.k, self.k + len(indices))))
        return self.__class__(multicontract(outer, zipped_indices), self.parity + 1, self.D, self.is_torus)

    def anticontract(self, additional_k):
        """
        Expand the ktensor so that the new order is self.k + additional_k. Then elemenet-wise multiply it by a
        special symbol such that no matter what contractions you perform to return it to the orignal k, it equals the
        original image. This only works under certain conditions, given by the asserts.
        args:
            additional_k (int): how many dimensions we are adding
        """
        # Currently these are the only cases it works with and has been tested for.
        assert self.k == 0 or self.k == 1
        assert additional_k == 2 or additional_k == 4
        assert self.D == 2

        expanded_data = jnp.tensordot(self.data, jnp.ones((self.D,)*additional_k), axes=0) #stretch the data

        if self.k == 0: # 1 in the [0,0,...,0] position, zeros everywhere else
            kron_delta = jnp.array(list((1,) + (0,)*(self.D**additional_k - 1))).reshape(((self.D,)*additional_k))
        elif self.k == 1:
            kron_delta = KroneckerDeltaSymbol.get(self.D, additional_k + self.k)

        kron_delta = jnp.tensordot(jnp.ones(self.image_shape()), kron_delta, axes=0)

        assert expanded_data.shape == kron_delta.shape

        return self.__class__(expanded_data * kron_delta, self.parity, self.D, self.is_torus)

    def get_rotated_keys(self, gg):
        """
        Slightly messier than with GeometricFilter because self.N-1 / 2 might not be an integer, but should work
        args:
            gg (jnp array-like): group operation
        """
        key_array = self.key_array() - ((self.N-1) / 2)
        return jnp.rint((key_array @ gg) + (self.N-1) / 2).astype(int)

    def times_group_element(self, gg):
        """
        Apply a group element of SO(2) or SO(3) to the geometric image. First apply the action to the location of the
        pixels, then apply the action to the pixels themselves.
        args:
            gg (group operation matrix): a DxD matrix that rotates the tensor
        """
        assert self.k < 14
        assert gg.shape == (self.D, self.D)
        sign, logdet = np.linalg.slogdet(gg)
        assert logdet == 0. #determinant is +- 1, so abs(log(det)) should be 0
        parity_flip = sign ** self.parity #if parity=1, the flip operators don't flip the tensors

        rotated_keys = self.get_rotated_keys(gg)
        rotated_pixels = self[self.hash(rotated_keys)].reshape(self.shape())

        if self.k == 0:
            newdata = 1. * rotated_pixels * parity_flip
        else:
            # applying the rotation to tensors is essentially multiplying each index, which we can think of as a
            # vector, by the group action. The image pixels have already been rotated.
            einstr = LETTERS[:len(self.shape())] + ','
            einstr += ",".join([LETTERS[i+13] + LETTERS[i + len(self.image_shape())] for i in range(self.k)])
            tensor_inputs = (rotated_pixels, ) + self.k * (gg, )
            newdata = np.einsum(einstr, *tensor_inputs) * (parity_flip)

        return self.__class__(newdata, self.parity, self.D, self.is_torus)

    def tree_flatten(self):
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly. Children and aux_data
        must contain all the variables that are passed in __init__()
        """
        children = (self.data,)  # arrays / dynamic values
        aux_data = {
            'D': self.D,
            'parity': self.parity,
            'is_torus': self.is_torus,
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

    def __init__(self, data, parity, D, is_torus=True):
        super(GeometricFilter, self).__init__(data, parity, D, is_torus)
        self.m = (self.N - 1) // 2
        assert self.N == 2 * self.m + 1, \
        "GeometricFilter: N needs to be odd."

    @classmethod
    def from_image(cls, geometric_image):
        """
        Constructor that copies a GeometricImage and returns a GeometricFilter
        """
        return cls(geometric_image.data, geometric_image.parity, geometric_image.D, geometric_image.is_torus)

    def __str__(self):
        return "<geometric filter object in D={} with N={} (m={}), k={}, parity={}, and is_torus={}>".format(
            self.D, self.N, self.m, self.k, self.parity, self.is_torus)

    def bigness(self):
        """
        Gives an idea of size for a filter, sparser filters are smaller while less sparse filters are larger
        """
        norms = self.norm().data
        numerator = 0.
        for key in self.key_array():
            numerator += jnp.linalg.norm(key * norms[tuple(key)], ord=2)

        denominator = jnp.sum(norms)
        return numerator / denominator

    def key_array(self, centered=False):
        # equivalent to the old pixels function
        if centered:
            return jnp.array([key for key in self.keys()], dtype=int) - self.m
        else:
            return jnp.array([key for key in self.keys()], dtype=int)

    def keys(self, centered=False):
        """
        Enumerate over all the keys in the geometric filter. Use centered=True when using the keys as adjustments
        args:
            centered (bool): if true, the keys range from -m to m rather than 0 to N. Defaults to false.
        """
        for key in super().keys():
            if (centered):
                #subtract m from all the elements of key
                yield tuple([a+b for a,b in zip(key, len(key) * (-self.m,))])
            else:
                yield key

    def items(self, centered=False):
        """
        Enumerate over all the key, pixel pairs in the geometric filter. Use centered=True when using the keys as
        adjustments
        args:
            centered (bool): if true, the keys range from -m to m rather than 0 to N. Defaults to false.
        """
        for key in self.keys(): #dont pass centered along because we need the un-centered keys to access the vals
            value = self[key]
            if (centered):
                #subtract m from all the elements of key
                yield (tuple([a+b for a,b in zip(key, len(key) * (-self.m,))]), value)
            else:
                yield (key, value)

    def get_rotated_keys(self, gg):
        key_array = self.key_array(centered=True)
        return (key_array @ gg) + self.m

    def rectify(self):
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

@register_pytree_node_class
class BatchGeometricImage(GeometricImage):
    """
    A GeometricImage class where the data is actually L geometric images
    """

    # Constructors

    @classmethod
    def fill(cls, N, parity, D, fill, L, is_torus=True):
        """
        Class method fill constructor to construct a geometric image every pixel as fill
        args:
            N (int): length of a side of an image, currently all images must be square N^D pixels
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            fill (jnp.ndarray or number): tensor to fill the image with
        """
        k = len(fill.shape) if isinstance(fill, jnp.ndarray) else 0
        data = jnp.stack([fill for _ in range(L * (N ** D))]).reshape((L,) + (N,)*D + (D,)*k)
        return cls(data, parity, D, is_torus)

    def __init__(self, data, parity, D, is_torus=True):
        """
        Construct the GeometricImage. It will be (L x N^D x D^k), so if L=16, N=100, D=2, k=1, then it's
        (16 x 100 x 100 x 2)
        args:
            data (array-like): the data
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
        """
        super(BatchGeometricImage, self).__init__(data[0], parity, D, is_torus)
        self.L = data.shape[0]
        self.data = data

    @classmethod
    def from_images(cls, images, indices=None):
        """
        Class method to construct a BatchGeometricImage from a list of GeometricImages. All the images should have the
        same parity, D, k, and is_torus.
        args:
            images (list of GeometricImages): images that we are making into a batch image
        """
        if indices is None:
            indices = range(len(images))

        data = jnp.stack([images[idx].data for idx in indices])
        return cls(data, images[0].parity, images[0].D, images[0].is_torus)

    def to_images(self):
        """
        Convert a batch image to a list of the individual images. Generally, doing this, then applying an operation,
        then converting back to the batch will be less efficient than operating on the batch.
        """
        return [GeometricImage(image_data, self.parity, self.D, self.is_torus) for image_data in self.data]

    def image_shape(self, plus_N=0):
        """
        Return the shape of the data block that is not the ktensor shape, but what comes before that. For regular
        GeometricImages, this is shape of the literal image. For BatchGeometricImage it prepends the batch size L.
        args:
            plus_N (int): number to add to self.N, used when shape will be growing/shrinking.
        """
        return (self.L,) + super(BatchGeometricImage, self).image_shape(plus_N)

    def __getitem__(self, key):
        """
        Accessor for data values. For the BatchGeometricImage, we prepend ':' to the key automatically so that you
        are selecting starting with the pixels on ALL images in the batch. If you want to select without this prepend,
        do self.data[key] instead of self[key].
        args:
            key (index): JAX/numpy indexer, i.e. "0", "0,1,3", "4:, 2:3, 0" etc.
        """
        return self.data[(slice(None),)+key]

    def __str__(self):
        return "<{} object with L={} images in D={} with N={}, k={}, parity={}, and is_torus={}>".format(
            self.__class__, self.L, self.D, self.N, self.k, self.parity, self.is_torus)

    # Binary Operators, Complicated functions

    def __eq__(self, other):
        """
        Equality operator, must have same L, shape, parity, and data within the TINY=1e-5 tolerance.
        """
        return self.L == other.L and super(BatchGeometricImage, self).__eq__(other)

    def __mul__(self, other):
        """
        Multiplication operator for BatchGeometricImages. Both must have the same batch size.
        args:
            other (GeometricImage or number): scalar or image to multiply by
        """
        if (isinstance(other, GeometricImage)):
            assert self.D == other.D
            assert self.N == other.N
            assert self.is_torus == other.is_torus
            assert self.L == other.L
            return self.__class__(
                vmap(mul, in_axes=(None, 0, 0))(self.D, self.data, other.data), 
                self.parity + other.parity, 
                self.D, 
                self.is_torus,
            )
        else: #its an integer or a float, or something that can we can multiply a Jax array by (like a DeviceArray)
            return self.__class__(self.data * other, self.parity, self.D, self.is_torus)

    def max_pool(self, patch_len):
        """
        Perform a max pooling operation where the length of the side of each patch is patch_len. Max is determined by
        the norm of the pixel. Note that for scalars, this will be the absolute value of the pixel.
        args:
            patch_len (int): the side length of the patches, must evenly divide self.N
        """
        # there has to be a better way of doing this
        return BatchGeometricImage.from_images([image.max_pool(patch_len) for image in self.to_images()]) 

    # TODO!!
    def normalize(self):
        raise Exception('BatchGeometricImage::normalize is not implemented')

    @partial(jit, static_argnums=[2,3,4,5])
    def convolve_with(
        self, 
        filter_image, 
        stride=None, 
        padding=None,
        lhs_dilation=None, 
        rhs_dilation=None, 
    ):
        """
        See batch_convolve for a description of this function.
        """
        batch_convolve = vmap(convolve, in_axes=(None, 0, None, None, None, None, None, None))
        convolved_array = batch_convolve(
            self.D, 
            self.data, 
            filter_image.data, 
            self.is_torus,
            stride,
            padding,
            lhs_dilation,
            rhs_dilation,
        )
        return self.__class__(
            convolved_array, 
            self.parity + filter_image.parity, 
            self.D, 
            self.is_torus,
        )

@register_pytree_node_class
class Layer:

    # Constructors

    def __init__(self, data, D, is_torus=True):
        """
        Construct a layer
        args:
            data (dictionary of jnp.array): dictionary by k of jnp.array
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        self.D = D
        self.is_torus = is_torus
        #copy dict, but image_block is immutable jnp array
        self.data = { k: image_block for k, image_block in data.items() } 

        self.N = None
        for image_block in data.values(): #if empty, this won't get set
            if isinstance(image_block, jnp.ndarray):
                self.N = image_block.shape[1] #shape (channels, (N,)*D, (D,)*k)
            break

    def copy(self):
        return self.__class__(self.data, self.D, self.is_torus)

    def empty(self):
        return self.__class__({}, self.D, self.is_torus)
    
    @classmethod
    def from_images(cls, images):
        # We assume that all images have the same D and is_torus
        if len(images) == 0:
            return None 
        
        out_layer = cls({}, images[0].D, images[0].is_torus)
        for image in images:
            out_layer.append(image.k, image.data.reshape((1,) + image.data.shape))

        return out_layer

    # Functions that map directly to calling the function on data

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx] = val
        return self.data[idx]
    
    def __contains__(self, idx):
        return idx in self.data
    
    # Other functions

    def append(self, k, image_block):
        """
        Append an image block at k=k. It will be concatenated along axis=0, so channel for Layer
        and vmapped BatchLayer, and batch for normal BatchLayer
        """
        # will this work for BatchLayer?
        if k > 0: #very light shape checking, other problematic cases should be caught in concatenate
            assert image_block.shape[-k:] == (self.D,)*k

        if (k in self):
            self[k] = jnp.concatenate((self[k], image_block))
        else:
            self[k] = image_block

        if self.N is None:
            self.N = image_block.shape[1]

        return self

    def __add__(self, other):
        """
        Addition operator for Layers, merges them together
        """
        assert type(self) == type(other), \
            f'{self.__class__}::__add__: Types of layers being added must match, had {type(self)} and {type(other)}'
        assert self.D == other.D, \
            f'{self.__class__}::__add__: Dimension of layers must match, had {self.D} and {other.D}'
        assert self.is_torus == other.is_torus, \
            f'{self.__class__}::__add__: is_torus of layers must match, had {self.is_torus} and {other.is_torus}'

        new_layer = self.copy()
        for k, image_block in other.items():
            new_layer.append(k, image_block)

        return new_layer
    
    def to_images(self):
        # Should only be used in Layer of vmapped BatchLayer
        images = []
        for image_block in self.values():
            for image in image_block:
                images.append(GeometricImage(image, 0, self.D, self.is_torus)) # for now, assume 0 parity

        return images

    #JAX helpers
    def tree_flatten(self):
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly. Children 
        and aux_data must contain all the variables that are passed in __init__()
        """
        children = (self.data,)  # arrays / dynamic values
        aux_data = {
            'D': self.D,
            'is_torus': self.is_torus,
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

    def __init__(self, data, D, is_torus=True):
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
        for image_block in data.values(): #if empty, this won't get set
            if isinstance(image_block, jnp.ndarray):
                self.L = len(image_block) #shape (batch, channels, (N,)*D, (D,)*k)
                self.N = image_block.shape[2]
            break

    @classmethod
    def from_images(cls, images):
        # We assume that all images have the same D and is_torus
        if len(images) == 0:
            return None 
        
        out_layer = cls({}, images[0].D, images[0].is_torus)
        for image in images:
            out_layer.append(image.k, image.data.reshape((1,1) + image.data.shape))

        batch_image_block = list(out_layer.values())[0]
        out_layer.L = batch_image_block.shape[0]
        out_layer.N = batch_image_block.shape[2]

        return out_layer

    def get_subset(self, idxs):
        """
        Select a subset of the batch, picking the indices idxs
        args:
            idxs (jnp.array): array of indices to select the subset
        """
        assert isinstance(idxs, jnp.ndarray)
        return self.__class__(
            { k: image_block[idxs] for k, image_block in self.items() },
            self.D,
            self.is_torus,
        )
