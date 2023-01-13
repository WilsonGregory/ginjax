import math

import geometricconvolutions.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random
import time

TINY = 1.e-5

class TestPropositions:
    # Class to test various propositions, mostly about the GeometricImage

    def testContractionOrderInvariance(self):
        # Test that the order of the two parameters of contraction does not matter
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=(3,3,2,2)), 0, 2)
        assert jnp.allclose(img1.contract(0,1).data, img1.contract(1,0).data)

        key, subkey = random.split(key)
        img2 = geom.GeometricImage(random.normal(subkey, shape=(3,3,2,2,2)), 0, 2)
        assert jnp.allclose(img2.contract(0,1).data, img2.contract(1,0).data)
        assert jnp.allclose(img2.contract(0,2).data, img2.contract(2,0).data)
        assert jnp.allclose(img2.contract(1,2).data, img2.contract(2,1).data)

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
        assert jnp.allclose(B1.transpose([0,2,3,1]).data, B2.data, rtol=geom.TINY, atol=geom.TINY)

    def testOuterProductCommutativity(self):
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
        assert jnp.allclose(B1.transpose([2,3,4,0,1]).data, B2.data, rtol=geom.TINY, atol=geom.TINY)

    def testOuterProductFilterInvariance(self):
        # Test that the outer product of two invariant filters is also invariant
        D = 2
        group_operators = geom.make_all_operators(D)
        all_filters = geom.get_invariant_filters([3], [0,1,2], [0,1], D, group_operators, return_list=True)
        for g in group_operators:
            for c1 in all_filters:
                for c2 in all_filters:
                    assert jnp.allclose(
                        (c1 * c2).times_group_element(g).data,
                        (c1 * c2).data,
                    )

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

        assert jnp.allclose(expanded_img2.contract(3,4).data, (img2*D).data)

    def testInvariantFilter(self):
        # The 3 scalar filters times the kron_delta image are the same as the first 3 k2 filters
        D = 2
        N = 3
        group_operators = geom.make_all_operators(D)
        max_k = 5

        kron_delta_img = geom.KroneckerDeltaSymbol.get_image(N, D, 2)

        k0_filters = geom.get_invariant_filters([N], [0], [0,1], D, group_operators, scale='one', return_list=True)
        k2_filters = geom.get_invariant_filters([N], [2], [0,1], D, group_operators, scale='one', return_list=True)
        for conv_filter in k0_filters:
            found_match = False
            conv_filter_exp = conv_filter * kron_delta_img

            for other_filter in k2_filters:
                if jnp.allclose(conv_filter_exp.data, other_filter.data):
                    found_match = True
                    break

            assert found_match
