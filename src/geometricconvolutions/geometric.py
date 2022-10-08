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

class levi_civita_symbol:

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

class ktensor:

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
        "ktensor: geometry makes no sense if D<2."
        self.parity = parity % 2
        if len(jnp.atleast_1d(data)) == 1:
            self.data = data
            self.k = 0
        else:
            self.data = jnp.array(data)
            self.k = len(data.shape)
            assert jnp.all(jnp.array(data.shape) == self.D), \
            "ktensor: shape must be (D, D, D, ...), but instead it's {}".format(data.shape)

    def __getitem__(self, key):
        return self.data[key]

    def __add__(self, other):
        assert self.k == other.k, \
        "ktensor: can't add objects of different k"
        assert self.parity == other.parity, \
        "ktensor: can't add objects of different parity"
        assert self.D == other.D, \
        "ktensor: can't add objects of different dimension D"
        return ktensor(self.data + other.data, self.parity, self.D)

    def __mul__(self, other):
        if self.k == 0 or other.k == 0:
            return ktensor(self.data * other.data,
                           self.parity + other.parity, self.D)
        return ktensor(np.multiply.outer(self.data, other.data),
                       self.parity + other.parity, self.D)

    def __str__(self):
        return "<k-tensor object in D={} with k={} and parity={}>".format(
            self.D, self.k, self.parity)

    def norm(self):
        if self.k == 0:
            return np.abs(self.data)
        return np.linalg.norm(self.data)

    def times_scalar(self, scalar):
        return ktensor(scalar * self.data, self.parity, self.D)

    def times_group_element(self, gg):
        """
        Multiply ktensor by group element, performing necessary adjustments if its a pseudo-tensor
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
        return ktensor(newdata, self.parity, self.D)

    def contract(self, i, j):
        #this is kinda a dirty hack, but I'm going to leave it for now
        assert self.k < 27
        assert self.k >= 2
        assert i < j
        assert i < self.k
        assert j < self.k
        letters  = "bcdefghijklmnopqrstuvwxyz"
        einstr = letters[:i] + "a" + letters[i:j-1] + "a" + letters[j-1:self.k-2]
        return ktensor(np.einsum(einstr, self.data), self.parity, self.D)

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

        levi_civita = levi_civita_symbol.get(self.D)
        outer = ktensor(jnp.tensordot(self.data, levi_civita, axes=((),())), self.parity + 1, self.D)

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
    tmp = geometric_filter.zeros(M, k, parity, D)
    keys, shape = tmp.keys(), tmp.data.shape
    allfilters = []
    if k == 0:
        for kk in keys:
            thisfilter = geometric_filter.zeros(M, k, parity, D)
            thisfilter[kk] = 1
            allfilters.append(thisfilter)
    else:
        for kk in keys:
            thisfilter = geometric_filter.zeros(M, k, parity, D)
            for indices in it.product(range(D), repeat=k):
                thisfilter[kk] = thisfilter[kk].at[indices].set(1) #is this even right?
                allfilters.append(thisfilter)

    # do the group averaging
    bigshape = (len(allfilters), ) + thisfilter.data.flatten().shape
    filter_matrix = np.zeros(bigshape)
    for i, f1 in enumerate(allfilters):
        ff = geometric_filter.zeros(M, k, parity, D)
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
    filters = [geometric_filter(aa.reshape(shape), parity, D).normalize() for aa in amps]
    norms = [ff.bigness() for ff in filters]
    I = np.argsort(norms)
    filters = [filters[i] for i in I]

    # now do k-dependent rectification:
    # filters = [ff.rectify() for ff in filters]

    return filters

# ------------------------------------------------------------------------------
# PART 5: Define geometric (k-tensor, torus) images.

class geometric_image:

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

    def hash_list(self, indices, extra_indices=None):
        """
        Could probably be combined with hash in some clever fashion
        """
        if extra_indices is not None:
            lst = [self.hash(key1, key2) for key1, key2 in zip(indices, extra_indices)]
        else:
            lst = [self.hash(key) for key in indices]

        coords = self.D*[[]]
        for i in range(self.D):
            coords[i] = [key[i] for key in lst]

        return tuple(coords)

    def hash(self, indices, extra_indices=None):
        """
        Deals with torus by modding (with `np.remainder()`).
        args:
            indices (tuple of ints): indices to apply the remainder to
            extra_indices (tuple of ints): indices to add, defaults to None
        """
        indices = jnp.array(list(indices), dtype=int)
        if extra_indices is not None:
            indices = indices + jnp.array(list(extra_indices), dtype=int)

        return tuple(jnp.remainder(indices, self.N).astype(int))

    def __init__(self, data, parity, D):
        """
        Construct the geometric_image. It will be (N^D x D^k), so if N=100, D=2, k=1, then it's (100 x 100 x 2)
        args:
            data (array-like):
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
        """
        self.D = D
        self.N = len(data)
        self.k = len(data.shape) - D
        assert data.shape[:D] == self.D * (self.N, ), \
        "geometric_image: data must be square."
        assert data.shape[D:] == self.k * (self.D, ), \
        "geometric_image: each pixel must be D cross D, k times"
        self.parity = parity % 2
        self.data = jnp.copy(data)

    def copy(self):
        return self.__class__(self.data, self.parity, self.D)

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
        val = thing.data if isinstance(thing, ktensor) else thing
        self.data = self.data.at[key].set(val)
        return self

    def ktensor(self, key):
        """
        Return the ktensor at the location key. Equivalent to image[key] but returns ktensor instead of raw data.
        """
        assert len(key) == self.D
        return ktensor(self[key], self.parity, self.D)

    def shape(self):
        return self.data.shape

    def image_shape(self):
        return self.D*(self.N,)

    def pixel_shape(self):
        return self.k*(self.D,)

    def __add__(self, other):
        """
        Addition operator for geometric_images. Both must be the same size and parity. Returns a new geometric_image.
        args:
            other (geometric_image): other image to add the the first one
        """
        assert self.D == other.D
        assert self.N == other.N
        assert self.k == other.k
        assert self.parity == other.parity
        return geometric_image(self.data + other.data, self.parity, self.D)

    def __mul__(self, other):
        """
        Return the ktensor product of each pixel as a new geometric_image
        """
        assert self.D == other.D
        assert self.N == other.N
        newimage = geometric_image.zeros(self.N, self.k + other.k,
                                         self.parity + other.parity, self.D)
        for key in self.keys():
            newimage[key] = self.ktensor(key) * other.ktensor(key) # handled by ktensor
        return newimage

    def __str__(self):
        return "<{} object in D={} with N={}, k={}, and parity={}>".format(
            self.__class__, self.D, self.N, self.k, self.parity)

    def keys(self):
        """
        Iterate over the keys of geometric_image
        """
        return it.product(range(self.N), repeat=self.D)

    def key_array(self):
        # equivalent to the old pixels function
        return jnp.array([key for key in self.keys()])

    def pixels(self, ktensor=True):
        """
        Iterate over the pixels of geometric_image. If ktensor=True, return the pixels as ktensor objects
        """
        for key in self.keys():
            yield self.ktensor(key) if ktensor else self[key]

    def items(self, ktensor=True):
        """
        Iterate over the key, pixel pairs of geometric_image. If ktensor=True, return the pixels as ktensor objects
        """
        for key in self.keys():
            yield (key, self.ktensor(key)) if ktensor else (key, self[key])

    def conv_subimage(self, center_key, filter_image, filter_image_keys=None):
        """
        Get the subimage (on the torus) centered on center_idx that will be convolved with filter_image
        args:
            center_key (index tuple): tuple index of the center of this convolution
            filter_image (geometric_filter): the geometric_filter we are convolving with
            filter_image_keys (list): For efficiency, the key offsets of the filter_image. Defaults to None.
        """
        if filter_image_keys is None:
            filter_image_keys = [key for key in filter_image.keys(centered=True)]

        key_list = self.hash_list(len(filter_image_keys)*[center_key], filter_image_keys) #key list on the torus
        #values, reshaped to the correct shape, which is the filter_image shape, while still having the ktensor shape
        vals = self[key_list].reshape(filter_image.image_shape() + self.pixel_shape())
        return self.__class__(vals, self.parity, self.D)

    def convolve_with(self, filter_image):
        """
        Apply the convolution filter_image to this geometric image
        args:
            filter_image (geometric_filter-like): convolution that we are applying, can be an image or a filter
        """
        newimage = self.__class__.zeros(self.N, self.k + filter_image.k,
                                         self.parity + filter_image.parity, self.D)

        if (isinstance(filter_image, geometric_image)):
            filter_image = geometric_filter.from_image(filter_image) #will break if N is not odd

        filter_image_keys = [key for key in filter_image.keys(centered=True)]
        for key in self.keys():
            subimage = self.conv_subimage(key, filter_image, filter_image_keys)
            newimage[key] = jnp.sum((subimage * filter_image).data)
        return newimage

    def times_scalar(self, scalar):
        """
        Scale the data by a scalar, returning a new geometric_image object
        args:
            scalar (number): number to scale everything by
        """
        return self.__class__(self.data * scalar, self.parity, self.D)

    def normalize(self):
        """
        Normalize so that the max norm of each pixel is 1, and all other ktensors are scaled appropriately
        """
        max_norm = jnp.max(jnp.array([ktensor.norm() for ktensor in self.pixels()]))
        if max_norm > TINY:
            return self.times_scalar(1. / max_norm)
        else:
            return self.times_scalar(1.)

    def contract(self, i, j):
        assert self.k >= 2
        newimage = self.__class__.zeros(self.N, self.k - 2, self.parity, self.D)
        for key, ktensor in self.items():
            newimage[key] = ktensor.contract(i, j)
        return newimage

    def levi_civita_contract(self, index):
        assert (self.k + 1) >= self.D
        newimage = self.__class__.zeros(self.N, self.k - self.D + 2,
                                         self.parity + 1, self.D)
        for key, ktensor in self.items():
            newimage[key] = ktensor.levi_civita_contract(index)
        return newimage

    def get_rotated_keys(self, gg):
        """
        Slightly messier than with geometric_filter because self.N-1 / 2 might not be an integer, but should work
        args:
            gg (jnp array-like): group operation
        """
        key_array = jnp.array([np.array(key) - (self.N-1) / 2 for key in self.keys()])
        return jnp.rint((key_array @ gg) + (self.N-1) / 2).astype(int)

    def times_group_element(self, gg, m=None):
        rotated_keys = self.get_rotated_keys(gg)
        ktensor_vals = [ktensor(val, self.parity, self.D) for val in  self[self.hash_list(rotated_keys)]]
        rotated_vals = [ktensor.times_group_element(gg).data for ktensor in ktensor_vals]

        return self.__class__(jnp.array(rotated_vals, dtype=self.data.dtype).reshape(self.shape()), self.parity, self.D)

    # def times_group_element(self, gg):
    #     #untested
    #     newfilter = self.copy()
    #     for key, ktensor in self.items():
    #         print(key)
    #         newfilter[key] = self[self.hash(gg.T @ key)].times_group_element(gg)
    #     return newfilter


# ------------------------------------------------------------------------------
# PART 3: Define a geometric (k-tensor) filter.

class geometric_filter(geometric_image):

    def __init__(self, data, parity, D):
        super(geometric_filter, self).__init__(data, parity, D)
        self.m = (self.N - 1) // 2
        assert self.N == 2 * self.m + 1, \
        "geometric_filter: N needs to be odd."

    @classmethod
    def from_image(cls, geometric_image):
        """
        Constructor that copies a geometric_image and returns a geometric_filter
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
            ktensor (bool): if true, return the values as ktensors, otherwise return as raw data. Defaults to true.
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
        key_array = jnp.array([np.array(key) - self.m for key in self.keys()], dtype=int)
        return (key_array @ gg) + self.m

    # def rectify(self):
    #     if self.k == 0:
    #         if np.sum([self[kk].data for kk in self.keys()]) < 0:
    #             return self.times_scalar(-1)
    #         return self
    #     if self.k == 1:
    #         if self.parity % 2 == 0:
    #             if np.sum([np.dot(pp, self[kk].data) for kk, pp in zip(self.keys(), self.pixels())]) < 0:
    #                 return self.times_scalar(-1)
    #             return self
    #         elif self.D == 2:
    #             if np.sum([np.cross(pp, self[kk].data) for kk, pp in zip(self.keys(), self.pixels())]) < 0:
    #                 return self.times_scalar(-1)
    #             return self
    #     return self

