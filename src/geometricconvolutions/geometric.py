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
- Drop the k-tensor and geometric filter classes; there can be only one class. Or have filter inherit from image with a few extras.
- Move over to jax.
- Make the geometric_filter inherit geometric_image to reduce repeated code.
- Is it possible to make the data a big block (not a dictionary) but have the block addressable with keys()? I bet it is...?
- Create tests for group operations on k-tensor images.
- Fix sizing of multi-filter plots.
- Switch over to jax so this is useful for ML people.
- Switch the structure of the image and filter so they make better use of jax.numpy array objects.
- Need to implement index permutation operation.
- Need to implement Levi-Civita contraction for general dimensions.
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

def make_all_generators(D):

    # Make the flip operator
    foo = np.ones(D).astype(int)
    foo[0] = -1
    gg = np.diag(foo).astype(int)
    generators = [gg, ]

    # Make the 90-degree rotation operators
    for i in range(D):
        for j in range(i + 1, D):
            gg = np.eye(D).astype(int)
            gg[i, i] = 0
            gg[j, j] = 0
            gg[i, j] = -1
            gg[j, i] = 1
            generators.append(gg)

    return np.array(generators)

def make_all_operators(D):
    generators = make_all_generators(D)
    operators = np.array([np.eye(D).astype(int), ])
    foo = 0
    while len(operators) != foo:
        foo = len(operators)
        operators = make_new_operators(operators, generators)
    return(operators)

def make_new_operators(operators, generators):
    for op in operators:
        for gg in generators:
            op2 = (gg @ op).astype(int)
            operators = np.unique(np.append(operators, op2[None, :, :], axis=0), axis=0)
    return operators

def test_group(operators):
    D = len(operators[0])
    # Check that the list of group operators is closed
    for gg in operators:
        for gg2 in operators:
            if ((gg @ gg2).astype(int) not in operators):
                return False
    print("group is closed under multiplication")
    # Check that gg.T is gg.inv for all gg in group?
    for gg in operators:
        if not np.allclose(gg @ gg.T, np.eye(D)):
            return False
    print("group operators are the transposes of their inverses")
    return True

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
        if len(np.atleast_1d(data)) == 1:
            self.data = data
            self.k = 0
        else:
            self.data = np.array(data)
            self.k = len(data.shape)
            assert np.all(np.array(data.shape) == self.D), \
            "ktensor: shape must be (D, D, D, ...), but instead it's {}".format(data.shape)

    def __getitem__(self, key):
        return self.data[key]

    def __add__(self, other):
        assert self.k == other.k, \
        "ktensor: can't add objects of different k"
        assert self.parity == other.parity, \
        "ktensor: can't add objects of different parity"
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
        # Notes / Bugs:
        - THIS IS UNTESTED.
        - This is incomprehensible.
        """
        assert self.k < 14
        assert gg.shape == (self.D, self.D)
        sign, logdet = np.linalg.slogdet(gg)
        assert logdet == 0.
        if self.k == 0:
            newdata = 1. * self.data * sign ** self.parity
        else:
            firstletters  = "abcdefghijklm"
            secondletters = "nopqrstuvwxyz"
            einstr = "".join([firstletters[i] for i in range(self.k)]) +"," + \
            ",".join([secondletters[i] + firstletters[i] for i in range(self.k)])
            foo = (self.data, ) + self.k * (gg, )
            newdata = np.einsum(einstr, *foo) * sign ** self.parity
        return ktensor(newdata, self.parity, self.D)

    def contract(self, i, j):
        assert self.k < 27
        assert self.k >= 2
        assert i < j
        assert i < self.k
        assert j < self.k
        letters  = "bcdefghijklmnopqrstuvwxyz"
        einstr = letters[:i] + "a" + letters[i:j-1] + "a" + letters[j-1:self.k-2]
        return ktensor(np.einsum(einstr, self.data), self.parity, self.D)

    def levi_civita_contract(self, index):
        assert self.D in [2, 3] # BECAUSE WE SUCK
        assert (self.k + 1) >= self.D # so we have enough indices work on
        if self.D == 2:
            otherdata = np.zeros_like(self.data)
            otherdata[..., 0] =  1. * np.take(self.data, 1, axis=index)
            otherdata[..., 1] = -1. * np.take(self.data, 0, axis=index)
            return ktensor(otherdata, self.parity + 1, self.D)
        if self.D == 3:
            assert len(index) == 2
            i, j = index
            assert i < j
            otherdata = np.zeros_like(self.data[..., 0])
            otherdata[..., 0] = np.take(np.take(self.data, 2, axis=j), 1, axis=i) \
                              - np.take(np.take(self.data, 1, axis=j), 2, axis=i)
            otherdata[..., 1] = np.take(np.take(self.data, 0, axis=j), 2, axis=i) \
                              - np.take(np.take(self.data, 2, axis=j), 0, axis=i)
            otherdata[..., 2] = np.take(np.take(self.data, 1, axis=j), 0, axis=i) \
                              - np.take(np.take(self.data, 0, axis=j), 1, axis=i)
            return ktensor(otherdata, self.parity + 1, self.D)
        return

# Now test group actions on k-tensors:
def test_group_actions(operators):
    """
    # Notes:
    - This only does minimal tests!
    """
    D = len(operators[0])

    for parity in [0, 1]:

        # vector dot vector
        v1 = ktensor(np.random.normal(size=D), parity, D)
        v2 = ktensor(np.random.normal(size=D), parity, D)
        dots = [(v1.times_group_element(gg)
                 * v2.times_group_element(gg)).contract(0, 1).data
                for gg in operators]
        dots = np.array(dots)
        if not np.allclose(dots, np.mean(dots)):
            print("failed (parity = {}) vector dot test.".format(parity))
            return False
        print("passed (parity = {}) vector dot test.".format(parity))

        # tensor times tensor
        T3 = ktensor(np.random.normal(size=(D, D)), parity, D)
        T4 = ktensor(np.random.normal(size=(D, D)), parity, D)
        dots = [(T3.times_group_element(gg)
                 * T4.times_group_element(gg)).contract(1, 2).contract(0, 1).data
                for gg in operators]
        dots = np.array(dots)
        if not np.allclose(dots, np.mean(dots)):
            print("failed (parity = {}) tensor times tensor test".format(parity))
            return False
        print("passed (parity = {}) tensor times tensor test".format(parity))

        # vectors dotted through tensor
        v5 = ktensor(np.random.normal(size=D), 0, D)
        dots = [(v5.times_group_element(gg) * T3.times_group_element(gg)
                 * v2.times_group_element(gg)).contract(1, 2).contract(0, 1).data
                for gg in operators]
        dots = np.array(dots)
        if not np.allclose(dots, np.mean(dots)):
            print("failed (parity = {}) v T v test.".format(parity))
            return False
        print("passed (parity = {}) v T v test.".format(parity))
    
    return True


# ------------------------------------------------------------------------------
# PART 4: Use group averaging to find unique invariant filters.

def get_unique_invariant_filters(M, k, parity, D, operators):

    # make the seed filters
    tmp = geometric_filter.zeros(M, k, parity, D)
    M, D, keys, shape = tmp.N, tmp.D, tmp.keys(), tmp.data.shape
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
                thisfilter[kk][indices] = 1
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
    filters = [ff.rectify() for ff in filters]

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
        val = thing.data if type(thing) == ktensor.__class__ else thing
        self.data = self.data.at[key].set(val)
        return self

    def ktensor(self, key):
        """
        Return the ktensor at the location key. Equivalent to image[key] but returns ktensor instead of raw data.
        """
        assert len(key) == self.D
        return ktensor(self[key], self.parity, self.D)

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
        assert self.D == other.D
        assert self.N == other.N
        newimage = geometric_image.zeros(self.N, self.k + other.k,
                                         self.parity + other.parity, self.D)
        for key in self.keys():
            newimage[kk] = (self.ktensor(key) * other.ktensor(key)).data # handled by ktensor
        return newimage

    def __str__(self):
        return "<{} object in D={} with N={}, k={}, and parity={}>".format(
            self.__class__, self.D, self.N, self.k, self.parity)

    def keys(self):
        """
        Iterate over the keys of geometric_image
        """
        return it.product(range(self.N), repeat=self.D)

    def pixels(self, ktensor=True):
        """
        Iterate over the pixels of geometric_image. If ktensor=True, return the pixels as ktensor objects
        """
        for key in self.keys():
            if ktensor:
                yield self.ktensor(key)
            else:
                yield self[key]

    def items(self, ktensor=True):
        """
        Iterate over the key, pixel pairs of geometric_image. If ktensor=True, return the pixels as ktensor objects
        """
        for key in self.keys():
            if ktensor:
                yield (key, self.ktensor(key))
            else:
                yield (key, self[key])

    def convolve_with(self, filter_image):
        """
        Apply the convolution filter_image to this geometric image
        args:
            filter_image (geometric_filter): convolution that we are applying
        """
        newimage = self.__class__.zeros(self.N, self.k + filter_image.k,
                                         self.parity + filter_image.parity, self.D)

        for key in self.keys():
            for filter_key, filter_pixel in filter_image.items(centered=True):
                newimage[key] = newimage[key] + (self.ktensor(self.hash(key, filter_key)) * filter_pixel).data
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

    def times_group_element(self, gg):
        #untested
        newfilter = self.copy()
        for key, ktensor in self.items():
            newfilter[key] = self[self.hash(gg.T @ ktensor)].times_group_element(gg)
        return newfilter


# ------------------------------------------------------------------------------
# PART 3: Define a geometric (k-tensor) filter.

class geometric_filter(geometric_image):

    def __init__(self, data, parity, D):
        super(geometric_filter, self).__init__(data, parity, D)
        self.m = (self.N - 1) // 2
        assert self.N == 2 * self.m + 1, \
        "geometric_filter: N needs to be odd."


    def __str__(self):
        return "<geometric filter object in D={} with N={} (m={}), k={}, and parity={}>".format(
            self.D, self.N, self.m, self.k, self.parity)

    def bigness(self):
        numerator, denominator = 0., 0.
        for key, ktensor in self.items():
            numerator += jnp.linalg.norm(ktensor * ktensor.norm(), ord=2)
            denominator += ktensor.norm()
        return numerator / denominator

    def items(self, ktensor=True, centered=False):
        """
        Enumerate over all the key, pixel pairs in the geometric image. Use centered=True when using the keys as
        adjustments
        args:
            ktensor (bool): if true, return the values as ktensors, otherwise return as raw data. Defaults to true.
            centered (bool): if true, the keys range from -m to m rather than 0 to N. Defaults to false.
        """
        for key in self.keys():
            value = self.ktensor(key) if ktensor else self[key]
            if (centered):
                #subtract m from all the elements of key
                yield (tuple([a+b for a,b in zip(key, len(key) * (-self.m,))]), value)
            else:
                yield (key, value)

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

