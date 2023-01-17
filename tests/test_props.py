import math

import geometricconvolutions.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random
import time
import itertools as it

class TestPropositions:
    # Class to test various propositions, mostly about the GeometricImage

    def testContractionOrderInvariance(self):
        # Test that the order of the two parameters of contraction does not matter
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=(3,3,2,2)), 0, 2)
        assert img1.contract(0,1) == img1.contract(1,0)

        key, subkey = random.split(key)
        img2 = geom.GeometricImage(random.normal(subkey, shape=(3,3,2,2,2)), 0, 2)
        assert img2.contract(0,1) == img2.contract(1,0)
        assert img2.contract(0,2) == img2.contract(2,0)
        assert img2.contract(1,2) == img2.contract(2,1)

    def testSerialContractionInvariance(self):
        # Test that order of contractions performed in series does not matter
        key = random.PRNGKey(0)
        img1 = geom.GeometricImage(random.normal(key, shape=(3,3,2,2,2,2,2)), 0, 2)
        assert jnp.allclose(img1.multicontract(((0,1),(2,3))).data, img1.multicontract(((2,3),(0,1))).data)
        assert jnp.allclose(img1.multicontract(((0,1),(3,4))).data, img1.multicontract(((3,4),(0,1))).data)
        assert jnp.allclose(img1.multicontract(((1,2),(3,4))).data, img1.multicontract(((3,4),(1,2))).data)
        assert jnp.allclose(img1.multicontract(((1,4),(2,3))).data, img1.multicontract(((1,4),(2,3))).data)

    def testConvolutionLinearity(self):
        """
        For scalars alpha, beta, tensor images image1, image2 and filter c1 alpha
        """
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        image1 = geom.GeometricImage(random.uniform(subkey, shape=(3,3,2)), 0, 2)

        key, subkey = random.split(key)
        image2 = geom.GeometricImage(random.uniform(subkey, shape=(3,3,2)), 0, 2)

        key, subkey = random.split(key)
        c1 = geom.GeometricFilter(random.uniform(subkey, shape=(3,3,2)), 0, 2)

        alpha, beta = random.uniform(subkey, shape=(2,))

        B1 = image1.convolve_with(c1).times_scalar(alpha) + image2.convolve_with(c1).times_scalar(beta)
        B2 = (image1.times_scalar(alpha) + image2.times_scalar(beta)).convolve_with(c1)

        assert B1.shape() == B2.shape()
        assert B1.parity == B2.parity
        assert jnp.allclose(B1.data, B2.data)

    def testConvolveConvolveCommutativity(self):
        # Test that performing two contractions is the same (under transposition) no matter the order
        key = random.PRNGKey(time.time_ns())
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=(3,3,2)), 0, 2)

        key, subkey = random.split(key)
        c1 = geom.GeometricFilter(random.normal(subkey, shape=(3,3,2)), 1, 2)

        key, subkey = random.split(key)
        c2 = geom.GeometricFilter(random.normal(subkey, shape=(3,3,2,2)), 0, 2)

        B1 = img1.convolve_with(c1).convolve_with(c2)
        B2 = img1.convolve_with(c2).convolve_with(c1)

        assert B1.D == B2.D
        assert B1.N == B2.N
        assert B1.parity == B2.parity
        assert B1.transpose([0,2,3,1]) == B2

    def testOuterProductCommutativity(self):
        # Test that the tensor product is commutative under transposition, including if there are convolves in there.
        key = random.PRNGKey(time.time_ns())
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=(3,3,2)), 0, 2)

        key, subkey = random.split(key)
        c1 = geom.GeometricFilter(random.normal(subkey, shape=(3,3,2)), 1, 2)

        key, subkey = random.split(key)
        c2 = geom.GeometricFilter(random.normal(subkey, shape=(3,3,2,2)), 0, 2)

        B1 = img1.convolve_with(c1) * img1.convolve_with(c2)
        B2 = img1.convolve_with(c2) * img1.convolve_with(c1)

        assert B1.D == B2.D
        assert B1.N == B2.N
        assert B1.parity == B2.parity
        assert B1.transpose([2,3,4,0,1]) == B2

    def testOuterProductFilterInvariance(self):
        # Test that the outer product of two invariant filters is also invariant
        D = 2
        group_operators = geom.make_all_operators(D)
        all_filters = geom.get_invariant_filters([3], [0,1,2], [0,1], D, group_operators, return_list=True)
        for g in group_operators:
            for c1 in all_filters:
                for c2 in all_filters:
                    assert (c1 * c2).times_group_element(g) == (c1 * c2)

    def testKroneckerAdd(self):
        # Test that multiplying by the kronecker delta symbol, then contracting on those new indices
        # merely scales the original image by D
        N = 3
        D = 2
        k = 3
        kron_delta_k = 2

        key = random.PRNGKey(time.time_ns())
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=((N,)*D + (D,)*k)), 0, D)
        kron_delta_img = geom.KroneckerDeltaSymbol.get_image(N, D, kron_delta_k)

        expanded_img1 = img1 * kron_delta_img
        assert expanded_img1.k == k + kron_delta_k
        assert jnp.allclose(expanded_img1.contract(3,4).data, (img1*D).data)

         #Multiplying by K-D then contracting on exactly one K-D index returns the original, up to a transpose of axes
        assert jnp.allclose(expanded_img1.contract(0,3).transpose((2,0,1)).data, (img1).data)

        D = 3
        key, subkey = random.split(key)
        img2 = geom.GeometricImage(random.normal(subkey, shape=((N,)*D + (D,)*k)), 0, D)
        kron_delta_img = geom.KroneckerDeltaSymbol.get_image(N, D, kron_delta_k)

        expanded_img2 = img2 * kron_delta_img
        assert expanded_img2.k == k + kron_delta_k

        assert expanded_img2.contract(3,4) == (img2*D)

    def testInvariantFilter(self):
        # For every invariant filter of order k, there exists an invariant filter of order k+2 where contracting on
        # any pair of tensor indices results in the invariant filter of order k.
        D = 2
        N = 3
        group_operators = geom.make_all_operators(D)
        max_k = 5

        kron_delta_img = geom.KroneckerDeltaSymbol.get_image(N, D, 2)

        conv_filters_dict = geom.get_invariant_filters([N], range(max_k+1), [0,1], D, group_operators, scale='one')
        for k in range(max_k - 1):
            for parity in [0,1]:
                for conv_filter in conv_filters_dict[(D, N, k, parity)]:
                    found_match = False
                    for upper_conv_filter in conv_filters_dict[(D, N, k+2, parity)]:
                        contracted_filter = upper_conv_filter.contract(0,1)

                        datablock = jnp.stack([conv_filter.data.flatten(), contracted_filter.data.flatten()])
                        u, s, v = jnp.linalg.svd(datablock)

                        if (jnp.sum(s > geom.TINY) == 1): #if one equals...
                            found_match = True
                            for i,j in it.combinations(range(k+2),2):
                                contracted_filter = upper_conv_filter.contract(i,j)
                                datablock = jnp.stack([conv_filter.data.flatten(), contracted_filter.data.flatten()])
                                u, s, v = jnp.linalg.svd(datablock)
                                assert jnp.sum(s > geom.TINY) == 1 # ...they must all equal

                    assert found_match

    def testContractConvolve(self):
        # Test that contracting then convolving is the same as convolving, then contracting
        N = 3
        D = 2
        k = 3

        key = random.PRNGKey(time.time_ns())
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=((N,)*D + (D,)*k)), 0, D)

        key, subkey = random.split(key)
        c1 = geom.GeometricFilter(random.normal(subkey, shape=(3,3,2)), 0, 2)

        key, subkey = random.split(key)
        c2 = geom.GeometricFilter(random.normal(subkey, shape=(3,3,2,2)), 0, 2)

        for c in [c1, c2]:
            for i,j in it.combinations(range(k), 2):
                assert img1.contract(i,j).convolve_with(c) == img1.convolve_with(c).contract(i,j)

