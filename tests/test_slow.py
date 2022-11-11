from geometricconvolutions.geometric import (
    GeometricImage,
    GeometricFilter,
    make_all_operators,
    get_unique_invariant_filters,
)
import pytest
import jax.numpy as jnp
from jax import random

TINY = 1.e-5

class TestSlowTests:

    # Test reserved for the slow tests, we only want to test these when we run the full battery

    def testConvolveWithRandoms(self):
        # this test uses convolve_with_slow to test convolve_with, possibly the blind leading the blind
        key = random.PRNGKey(0)
        N=3

        for D in [2,3]:
            for k_img in range(3):
                key, subkey = random.split(key)
                image = GeometricImage(random.uniform(subkey, shape=((N,)*D + (D,)*k_img)), 0, D)

                for k_filter in range(3):
                    key, subkey = random.split(key)
                    geom_filter = GeometricFilter(random.uniform(subkey, shape=((3,)*D + (D,)*k_filter)), 0, D)

                    convolved_image = image.convolve_with(geom_filter)
                    convolved_image_slow = image.convolve_with_slow(geom_filter)

                    assert convolved_image.D == convolved_image_slow.D == image.D
                    assert convolved_image.N == convolved_image_slow.N == image.N
                    assert convolved_image.k == convolved_image_slow.k == image.k + geom_filter.k
                    assert convolved_image.parity == convolved_image_slow.parity == (image.parity + geom_filter.parity) %2
                    assert jnp.allclose(convolved_image.data, convolved_image_slow.data)

    def testUniqueInvariantFilters(self):
        # ensure that all the filters are actually invariant
        key = random.PRNGKey(0)

        for D in [2]: #image dimension
            operators = make_all_operators(D)
            for N in [3]: #filter size
                key, subkey = random.split(key)
                image = GeometricImage(random.uniform(key, shape=(2*N,2*N)), 0, D)
                for k in [0,1,2]: #tensor order of filter
                    for parity in [0,1]:
                        filters = get_unique_invariant_filters(N, k, parity, D, operators)

                        for gg in operators:
                            for geom_filter in filters:

                                # test that the filters are invariant to the group operators
                                assert jnp.allclose(geom_filter.data, geom_filter.times_group_element(gg).data)

                                # test that the convolution with the invariant filters is equivariant to gg
                                # convolutions are currently too slow to test this every time, but should be tested
                                assert jnp.allclose(
                                    image.convolve_with(geom_filter).times_group_element(gg).data,
                                    image.times_group_element(gg).convolve_with(geom_filter).data,
                                )
