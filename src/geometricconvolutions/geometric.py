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
- Move over to jax.
- Create tests for group operations on k-tensor images.
- Fix sizing of multi-filter plots.
- Switch over to jax so this is useful for ML people.
- Need to implement index permutation operation.
- Need to build tests for the contractions.
- Need to implement bin-down and bin-up operators.
"""

import itertools as it
import numpy as np #removing this
import jax.numpy as jnp
import jax.lax
import pylab as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import cmastro

TINY = 1.e-5

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
# PART 1.5: Define the Levi Civita symbol to be used in Levi Civita contractions

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
# PART 2: Define a k-tensor.

class Ktensor:

    def name(k, parity):
        nn = "tensor"
        if k == 0:
            nn = "scalar"
        if k == 1:
            nn = "vector"
        if parity % 2 == 1 and k < 2:
            nn = "pseudo" + nn
        if k > 1:
            nn = "${}$-${}$-".format(k, parity) + nn
        return nn

    def __init__(self, data, parity, D):
        self.levi_civita = None
        self.D = D
        assert self.D > 1, \
        "Ktensor: geometry makes no sense if D<2."
        self.parity = parity % 2
        if len(jnp.atleast_1d(data)) == 1:
            self.data = data
            self.k = 0
        else:
            self.data = jnp.array(data)
            self.k = len(data.shape)
            assert jnp.all(jnp.array(data.shape) == self.D), \
            "Ktensor: shape must be (D, D, D, ...), but instead it's {}".format(data.shape)

    def __getitem__(self, key):
        return self.data[key]

    def __add__(self, other):
        assert self.k == other.k, \
        "Ktensor: can't add objects of different k"
        assert self.parity == other.parity, \
        "Ktensor: can't add objects of different parity"
        assert self.D == other.D, \
        "Ktensor: can't add objects of different dimension D"
        return Ktensor(self.data + other.data, self.parity, self.D)

    def __mul__(self, other):
        if self.k == 0 or other.k == 0:
            return Ktensor(self.data * other.data,
                           self.parity + other.parity, self.D)
        return Ktensor(jnp.tensordot(self.data, other.data, axes=0),
                       self.parity + other.parity, self.D)

    def __str__(self):
        return "<k-tensor object in D={} with k={} and parity={}>".format(
            self.D, self.k, self.parity)

    def norm(self):
        if self.k == 0:
            return np.abs(self.data)
        return np.linalg.norm(self.data)

    def times_scalar(self, scalar):
        return Ktensor(scalar * self.data, self.parity, self.D)

    def times_group_element(self, gg):
        """
        Multiply Ktensor by group element, performing necessary adjustments if its a pseudo-tensor
        args:
            gg (jnp array-like): group element
        """
        assert self.k < 14
        assert gg.shape == (self.D, self.D)
        sign, logdet = np.linalg.slogdet(gg)
        assert logdet == 0. #determinant is +- 1, so abs(log(det)) should be 0
        if self.k == 0:
            newdata = 1. * self.data * (sign ** self.parity)
        else:
            firstletters  = "abcdefghijklm"
            secondletters = "nopqrstuvwxyz"
            einstr = "".join([firstletters[i] for i in range(self.k)]) +"," + \
            ",".join([secondletters[i] + firstletters[i] for i in range(self.k)])
            foo = (self.data, ) + self.k * (gg, )
            newdata = np.einsum(einstr, *foo) * (sign ** self.parity)
        return Ktensor(newdata, self.parity, self.D)

    def contract(self, i, j):
        """
        Use einsum to perform a kronecker contraction on two dimensions
        args:
            i (int): first index
            j (int): second index
        """
        assert self.k < 27
        assert self.k >= 2
        assert i != j
        assert i < self.k
        assert j < self.k
        if (j < i): #order of indices does not matter for kronecker contraction
            tmp = i
            i = j
            j = tmp

        letters  = "bcdefghijklmnopqrstuvwxyz"
        einstr = letters[:i] + "a" + letters[i:j-1] + "a" + letters[j-1:self.k-2]
        return Ktensor(np.einsum(einstr, self.data), self.parity, self.D)

    def levi_civita_contract(self, indices):
        """
        Possibly naive implementation of levi_civita contraction
        args:
            indices (array-like): can be list or tuple, indices in order that you want them contracted
        """
        assert (self.k+1) >= self.D # so we have enough indices to work on
        if self.D == 2 and not (isinstance(indices, tuple) or isinstance(indices, list)):
            indices = (indices,)
        assert len(indices) == self.D - 1

        levi_civita = LeviCivitaSymbol.get(self.D)
        outer = Ktensor(jnp.tensordot(self.data, levi_civita, axes=0), self.parity + 1, self.D)

        indices_removed = 0
        while len(indices) > 0:
            idx, *indices = indices
            outer = outer.contract(idx, self.k - indices_removed)
            indices = [x if x < idx else x-1 for x in indices]
            #^ decrement indices larger than the one we just contracted, leave smaller ones alone
            indices_removed += 1

        return outer

# ------------------------------------------------------------------------------
# PART 4: Use group averaging to find unique invariant filters.

def get_unique_invariant_filters(M, k, parity, D, operators):
    """
    Use group averaging to generate all the unique invariant filters
    args:
        M (int): filter side length
        k (int): tensor order
        parity (int):  0 or 1, 0 is for normal tensors, 1 for pseudo-tensors
        D (int): image dimension
        operators (jnp-array): array of operators of a group
    """

    # make the seed filters
    tmp = GeometricFilter.zeros(M, k, parity, D)
    keys, shape = tmp.keys(), tmp.data.shape
    allfilters = []
    if k == 0:
        for kk in keys:
            thisfilter = GeometricFilter.zeros(M, k, parity, D)
            thisfilter[kk] = 1
            allfilters.append(thisfilter)
    else:
        for kk in keys:
            thisfilter = GeometricFilter.zeros(M, k, parity, D)
            for indices in it.product(range(D), repeat=k):
                thisfilter[kk] = thisfilter[kk].at[indices].set(1) #is this even right?
                allfilters.append(thisfilter)

    # do the group averaging
    bigshape = (len(allfilters), ) + thisfilter.data.flatten().shape
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
    filters = [GeometricFilter(aa.reshape(shape), parity, D).normalize() for aa in amps]
    norms = [ff.bigness() for ff in filters]
    I = np.argsort(norms)
    filters = [filters[i] for i in I]

    # now do k-dependent rectification:
    filters = [ff.rectify() for ff in filters]

    return filters

# ------------------------------------------------------------------------------
# PART 5: Define geometric (k-tensor, torus) images.

class GeometricImage:

    # Constructors

    @classmethod
    def zeros(cls, N, k, parity, D):
        """
        Class method zeros to construct a geometric image of zeros
        args:
            N (int): length of a side of an image, currently all images must be square N^D pixels
            k (int): the order of the tensor in each pixel, i.e. 0 (scalar), 1 (vector), 2 (matrix), etc.
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
        """
        shape = D * (N, ) + k * (D, )
        return cls(jnp.zeros(shape), parity, D)

    def __init__(self, data, parity, D):
        """
        Construct the GeometricImage. It will be (N^D x D^k), so if N=100, D=2, k=1, then it's (100 x 100 x 2)
        args:
            data (array-like):
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
        """
        self.D = D
        self.N = len(data)
        self.k = len(data.shape) - D
        assert data.shape[:D] == self.D * (self.N, ), \
        "GeometricImage: data must be square."
        assert data.shape[D:] == self.k * (self.D, ), \
        "GeometricImage: each pixel must be D cross D, k times"
        self.parity = parity % 2
        self.data = jnp.copy(data) #TODO: don't need to copy if data is already an immutable jnp array

    def copy(self):
        return self.__class__(self.data, self.parity, self.D)

    # Getters, setters, basic info

    def hash(self, indices):
        """
        Deals with torus by modding (with `np.remainder()`).
        args:
            indices (tuple of ints): indices to apply the remainder to
        """
        return tuple(jnp.remainder(indices, self.N).transpose().astype(int))

    def __getitem__(self, key):
        """
        Accessor for data values. Now you can do image[key] where k are indices or array slices and it will just work
        Note that JAX does not throw errors for indexing out of bounds
        args:
            key (index): JAX/numpy indexer, i.e. "0", "0,1,3", "4:, 2:3, 0" etc.
        """
        return self.data[key]

    def __setitem__(self, key, thing):
        """
        Jax arrays are immutable, so this reconstructs the data object with copying, and is potentially slow
        """
        val = thing.data if isinstance(thing, Ktensor) else thing
        self.data = self.data.at[key].set(val)
        return self

    def ktensor(self, key):
        """
        Return the Ktensor at the location key. Equivalent to image[key] but returns Ktensor instead of raw data.
        """
        assert len(key) == self.D
        return Ktensor(self[key], self.parity, self.D)

    def shape(self):
        return self.data.shape

    def image_shape(self):
        return self.D*(self.N,)

    def pixel_shape(self):
        return self.k*(self.D,)

    def __str__(self):
        return "<{} object in D={} with N={}, k={}, and parity={}>".format(
            self.__class__, self.D, self.N, self.k, self.parity)

    def keys(self):
        """
        Iterate over the keys of GeometricImage
        """
        return it.product(range(self.N), repeat=self.D)

    def key_array(self):
        # equivalent to the old pixels function
        return jnp.array([key for key in self.keys()], dtype=int)

    def pixels(self, ktensor=True):
        """
        Iterate over the pixels of GeometricImage. If ktensor=True, return the pixels as Ktensor objects
        """
        for key in self.keys():
            yield self.ktensor(key) if ktensor else self[key]

    def items(self, ktensor=True):
        """
        Iterate over the key, pixel pairs of GeometricImage. If ktensor=True, return the pixels as Ktensor objects
        """
        for key in self.keys():
            yield (key, self.ktensor(key)) if ktensor else (key, self[key])

    # Binary Operators, Complicated functions

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
        return self.__class__(self.data + other.data, self.parity, self.D)

    def __mul__(self, other):
        """
        Return the Ktensor product of each pixel as a new GeometricImage
        """
        assert self.D == other.D
        assert self.N == other.N

        image_a_data, image_b_data = GeometricImage.pre_tensor_product_expand(self, other)
        #now that the shapes match, we can do elementwise multiplication
        return self.__class__(image_a_data * image_b_data, self.parity + other.parity, self.D)

    def conv_subimage(self, center_key, filter_image, filter_image_keys=None):
        """
        Get the subimage (on the torus) centered on center_idx that will be convolved with filter_image
        args:
            center_key (index tuple): tuple index of the center of this convolution
            filter_image (GeometricFilter): the GeometricFilter we are convolving with
            filter_image_keys (list): For efficiency, the key offsets of the filter_image. Defaults to None.
        """
        if filter_image_keys is None:
            filter_image_keys = filter_image.key_array(centered=True) #centered key array

        key_list = self.hash(filter_image_keys + jnp.array(center_key)) #key list on the torus
        #values, reshaped to the correct shape, which is the filter_image shape, while still having the ktensor shape
        vals = self[key_list].reshape(filter_image.image_shape() + self.pixel_shape())
        return self.__class__(vals, self.parity, self.D)

    def convolve_with_slow(self, filter_image):
        """
        Apply the convolution filter_image to this geometric image. Keeping this around for testing.
        args:
            filter_image (GeometricFilter-like): convolution that we are applying, can be an image or a filter
        """
        newimage = self.__class__.zeros(self.N, self.k + filter_image.k,
                                         self.parity + filter_image.parity, self.D)

        if (isinstance(filter_image, GeometricImage)):
            filter_image = GeometricFilter.from_image(filter_image) #will break if N is not odd

        filter_image_keys = filter_image.key_array(centered=True)
        for key in self.keys():
            subimage = self.conv_subimage(key, filter_image, filter_image_keys)
            newimage[key] = jnp.sum((subimage * filter_image).data, axis=tuple(range(self.D)))
        return newimage

    @classmethod
    def pre_tensor_product_expand(cls, image_a, image_b):
        """
        Rather than take a tensor product of two tensors, we can first take a tensor product of each with a tensor of
        ones with the shape of the other. Then we have two matching shapes, and we can then do whatever operations.
        This is a class method that takes in two images and returns the expanded data
        args:
            image_a (GeometricImage like): one geometric image whose tensors we will later be doing tensor products on
            image_b (GeometricImage like): other geometric image
        """
        if (len(image_b.pixel_shape())):
            image_a_expanded = jnp.tensordot(
                image_a.data,
                jnp.ones(image_b.pixel_shape()),
                axes=0,
            )
        else:
            image_a_expanded = image_a.data

        if len(image_a.pixel_shape()):
            break1 = image_a.k + image_a.D #after outer product, end of image N^D axes
            #after outer product: [D^ki, N^D, D^kf], convert to [N^D, D^ki, D^kf]
            # we are trying to expand the ones in the middle (D^ki), so we add them on the front, then move to middle
            image_b_expanded = jnp.transpose(
                jnp.tensordot(jnp.ones(image_a.pixel_shape()), image_b.data, axes=0),
                list(
                    tuple(range(image_a.k, break1)) + tuple(range(image_a.k)) + tuple(range(break1, break1 + image_b.k))
                ),
            )
        else:
            image_b_expanded = image_b.data

        return image_a_expanded, image_b_expanded

    def convolve_with(self, filter_image, warnings=True):
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
            filter_image (GeometricFilter): the filter we are performing the convolution with
            warnings (bool): display warnings, defaults to True currently
        """
        if (self.data.dtype != filter_image.data.dtype):
            dtype = 'float32'
            if (warnings):
                print('GeometricImage::convolve_with_fastest: GeometricImage dtype and filter_image dtype mismatch, converting to float32')
        else:
            dtype = self.data.dtype

        output_k = self.k + filter_image.k
        torus_expand_img = self.get_torus_expanded(filter_image)

        img_expanded, filter_expanded = GeometricImage.pre_tensor_product_expand(torus_expand_img, filter_image)
        img_expanded = img_expanded.astype(dtype)
        filter_expanded = filter_expanded.astype(dtype)

        if output_k != 0:
            channel_length = np.multiply.reduce(self.pixel_shape() + filter_image.pixel_shape())
        else:
            channel_length = 1

        # convert the image to NHWC (or NHWDC), treating all the pixel values as channels
        img_formatted = img_expanded.reshape((1,) + torus_expand_img.shape()[:self.D] + (channel_length,))

        # convert filter to HWIO (or HWDIO)
        filter_formatted = filter_expanded.reshape(filter_image.image_shape() + (1,channel_length))

        if self.D == 2:
            dimension_numbers = ('NHWC','HWIO','NHWC')
            window_strides = (1,1)
        elif self.D == 3:
            dimension_numbers = ('NHWDC','HWDIO','NHWDC')
            window_strides = (1,1,1)

        convolved_array = jax.lax.conv_general_dilated(
            img_formatted, #lhs
            filter_formatted, #rhs
            window_strides, #window strides
            'VALID', #padding mode, we can do valid because we already expanded on the torus
            dimension_numbers=dimension_numbers,
            feature_group_count=channel_length, #allows us to have separate filters for separate channels
        ).reshape(self.image_shape() + (self.D,)*output_k) #reshape the pixel to the correct tensor shape
        return self.__class__(convolved_array, self.parity + filter_image.parity, self.D)

    def get_torus_expanded(self, filter_image):
        """
        For a particular filter, expand the image so that we no longer have to do convolutions on the torus, we are
        just doing convolutions on the expanded image and will get the same result. Return a new GeometricImage
        args:
            filter_image (GeometricFilter): filter, how much is expanded depends on filter_image.m
        """
        new_N = self.N + 2 * filter_image.m
        indices = jnp.array([key for key in it.product(range(new_N), repeat=self.D)], dtype=int) - filter_image.m
        return self.__class__(
            self[self.hash(indices)].reshape((new_N,) * self.D + self.pixel_shape()),
            self.parity,
            self.D,
        )

    def times_scalar(self, scalar):
        """
        Scale the data by a scalar, returning a new GeometricImage object
        args:
            scalar (number): number to scale everything by
        """
        return self.__class__(self.data * scalar, self.parity, self.D)

    def normalize(self):
        """
        Normalize so that the max norm of each pixel is 1, and all other ktensors are scaled appropriately
        """
        norm_ord = jnp.inf if self.k == 0 else None #jnp.inf does the infinity norm, or absolute value
        vectorized_pixels = self.data.reshape(((self.N**self.D,) + self.pixel_shape()))
        max_norm = jnp.max(jnp.linalg.norm(vectorized_pixels, axis=0, ord=norm_ord))
        if max_norm > TINY:
            return self.times_scalar(1. / max_norm)
        else:
            return self.times_scalar(1.)

    def contract(self, i, j):
        """
        Use einsum to perform a kronecker contraction on two dimensions
        args:
            i (int): first index of tensor
            j (int): second index of tensor
        """
        assert self.k < 27
        assert self.k >= 2
        assert i != j
        assert i < self.k
        assert j < self.k
        if (j < i): #order of indices does not matter for kronecker contraction
            tmp = i
            i = j
            j = tmp

        i += self.D
        j += self.D
        letters  = "bcdefghijklmnopqrstuvwxyz"
        einstr = letters[:i] + "a" + letters[i:j-1] + "a" + letters[j-1:self.D + self.k-2]
        return self.__class__(np.einsum(einstr, self.data), self.parity, self.D)

    def levi_civita_contract(self, index):
        """
        Perform the Levi-Civita contraction.
        """
        assert self.k >= (self.D - 1) # so we have enough indices to work on since we perform D-1 contractions
        if self.D == 2 and not (isinstance(indices, tuple) or isinstance(indices, list)):
            indices = (indices,)
        assert len(indices) == self.D - 1

        levi_civita = LeviCivitaSymbol.get(self.D)
        outer = jnp.tensordot(self.data, levi_civita, axes=0)

        indices_removed = 0
        while len(indices) > 0:
            idx, *indices = indices
            outer = outer.contract(idx + self.D, self.k - indices_removed + self.D)
            indices = [x if x < idx else x-1 for x in indices]
            #^ decrement indices larger than the one we just contracted, leave smaller ones alone
            indices_removed += 1

        return self.__class__(outer, self.parity + 1, self.D)

        # assert (self.k + 1) >= self.D
        # newimage = self.__class__.zeros(self.N, self.k - self.D + 2,
        #                                  self.parity + 1, self.D)
        # for key, ktensor in self.items():
        #     newimage[key] = ktensor.levi_civita_contract(index)
        # return newimage

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
            # applying the rotation to ktensors is essentially multiplying each index, which we can think of as a
            # vector, by the group action. The image pixels have already been rotated.
            # by the group action
            firstletters  = "abcdefghijklm"
            secondletters = "nopqrstuvwxyz"
            einstr = "".join([firstletters[i] for i in range(self.D)])
            einstr += "".join([firstletters[i + self.D] for i in range(self.k)]) + ","
            einstr += ",".join([secondletters[i] + firstletters[i + self.D] for i in range(self.k)])
            tensor_inputs = (rotated_pixels, ) + self.k * (gg, )
            newdata = np.einsum(einstr, *tensor_inputs) * (parity_flip)

        print(newdata.shape)
        return self.__class__(newdata, self.parity, self.D)

# ------------------------------------------------------------------------------
# PART 3: Define a geometric (k-tensor) filter.

class GeometricFilter(GeometricImage):

    def __init__(self, data, parity, D):
        super(GeometricFilter, self).__init__(data, parity, D)
        self.m = (self.N - 1) // 2
        assert self.N == 2 * self.m + 1, \
        "GeometricFilter: N needs to be odd."

    @classmethod
    def from_image(cls, geometric_image):
        """
        Constructor that copies a GeometricImage and returns a GeometricFilter
        """
        return cls(geometric_image.data, geometric_image.parity, geometric_image.D)

    def __str__(self):
        return "<geometric filter object in D={} with N={} (m={}), k={}, and parity={}>".format(
            self.D, self.N, self.m, self.k, self.parity)

    def bigness(self):
        """
        Gives an idea of size for a filter, sparser filters are smaller while less sparse filters are larger
        """
        numerator, denominator = 0., 0.
        for key, ktensor in self.items():
            numerator += jnp.linalg.norm(jnp.array(key) * ktensor.norm(), ord=2)
            denominator += ktensor.norm()
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

    def items(self, ktensor=True, centered=False):
        """
        Enumerate over all the key, pixel pairs in the geometric filter. Use centered=True when using the keys as
        adjustments
        args:
            ktensor (bool): if true, return the values as Ktensors, otherwise return as raw data. Defaults to true.
            centered (bool): if true, the keys range from -m to m rather than 0 to N. Defaults to false.
        """
        for key in self.keys(): #dont pass centered along because we need the un-centered keys to access the vals
            value = self.ktensor(key) if ktensor else self[key]
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

