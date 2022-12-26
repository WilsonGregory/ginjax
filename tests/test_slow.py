import geometricconvolutions.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random

TINY = 1.e-5

def conv_subimage(image, center_key, filter_image, filter_image_keys=None):
    """
    Get the subimage (on the torus) centered on center_idx that will be convolved with filter_image
    args:
        center_key (index tuple): tuple index of the center of this convolution
        filter_image (GeometricFilter): the GeometricFilter we are convolving with
        filter_image_keys (list): For efficiency, the key offsets of the filter_image. Defaults to None.
    """
    if filter_image_keys is None:
        filter_image_keys = filter_image.key_array(centered=True) #centered key array

    key_list = image.hash(filter_image_keys + jnp.array(center_key)) #key list on the torus
    #values, reshaped to the correct shape, which is the filter_image shape, while still having the tensor shape
    vals = image[key_list].reshape(filter_image.image_shape() + image.pixel_shape())
    return image.__class__(vals, image.parity, image.D)

def convolve_with_slow(image, filter_image):
    """
    Apply the convolution filter_image to this geometric image. Keeping this around for testing.
    args:
        filter_image (GeometricFilter-like): convolution that we are applying, can be an image or a filter
    """
    newimage = image.__class__.zeros(image.N, image.k + filter_image.k,
                                     image.parity + filter_image.parity, image.D)

    if (isinstance(filter_image, geom.GeometricImage)):
        filter_image = geom.GeometricFilter.from_image(filter_image) #will break if N is not odd

    filter_image_keys = filter_image.key_array(centered=True)
    for key in image.keys():
        subimage = conv_subimage(image, key, filter_image, filter_image_keys)
        newimage[key] = jnp.sum((subimage * filter_image).data, axis=tuple(range(image.D)))
    return newimage

class TestSlowTests:

    # Test reserved for the slow tests, we only want to test these when we run the full battery

    def testConvSubimage(self):
        image1 = geom.GeometricImage(jnp.arange(25).reshape((5,5)), 0, 2)
        filter1 = geom.GeometricFilter(jnp.zeros(25).reshape((5,5)), 0, 2)
        subimage1 = conv_subimage(image1, (0,0), filter1)
        assert subimage1.shape() == (5,5)
        assert subimage1.D == image1.D
        assert subimage1.N == filter1.N
        assert subimage1.k == image1.k
        assert subimage1.parity == image1.parity
        assert (subimage1.data == jnp.array(
            [
                [18,19,15,16,17],
                [23,24,20,21,22],
                [3,4,0,1,2],
                [8,9,5,6,7],
                [13,14,10,11,12],
            ],
        dtype=int)).all()

        subimage2 = conv_subimage(image1, (4,4), filter1)
        assert subimage2.shape() == (5,5)
        assert subimage2.D == image1.D
        assert subimage2.N == filter1.N
        assert subimage2.k == image1.k
        assert subimage2.parity == image1.parity
        assert (subimage2.data == jnp.array(
            [
                [12,13,14,10,11],
                [17,18,19,15,16],
                [22,23,24,20,21],
                [2,3,4,0,1],
                [7,8,9,5,6],
            ],
        dtype=int)).all()

        image2 = geom.GeometricImage(jnp.arange(25).reshape((5,5)), 0, 2)*geom.GeometricImage(jnp.ones((5,5,2)), 0, 2)
        subimage3 = conv_subimage(image2, (0,0), filter1)
        assert subimage3.shape() == (5,5,2)
        assert subimage3.D == image2.D
        assert subimage3.N == filter1.N
        assert subimage3.k == image2.k
        assert subimage3.parity == image2.parity
        assert (subimage3.data == jnp.array(
            [
                [x*jnp.array([1,1]) for x in [18,19,15,16,17]],
                [x*jnp.array([1,1]) for x in [23,24,20,21,22]],
                [x*jnp.array([1,1]) for x in [3,4,0,1,2]],
                [x*jnp.array([1,1]) for x in [8,9,5,6,7]],
                [x*jnp.array([1,1]) for x in [13,14,10,11,12]],
            ],
        dtype=int)).all()

    def testConvolveWithRandoms(self):
        # this test uses convolve_with_slow to test convolve_with, possibly the blind leading the blind
        key = random.PRNGKey(0)
        N=3

        for D in [2,3]:
            for k_img in range(3):
                key, subkey = random.split(key)
                image = geom.GeometricImage(random.uniform(subkey, shape=((N,)*D + (D,)*k_img)), 0, D)

                for k_filter in range(3):
                    key, subkey = random.split(key)
                    geom_filter = geom.GeometricFilter(random.uniform(subkey, shape=((3,)*D + (D,)*k_filter)), 0, D)

                    convolved_image = image.convolve_with(geom_filter)
                    convolved_image_slow = convolve_with_slow(image, geom_filter)

                    assert convolved_image.D == convolved_image_slow.D == image.D
                    assert convolved_image.N == convolved_image_slow.N == image.N
                    assert convolved_image.k == convolved_image_slow.k == image.k + geom_filter.k
                    assert convolved_image.parity == convolved_image_slow.parity == (image.parity + geom_filter.parity) %2
                    assert jnp.allclose(convolved_image.data, convolved_image_slow.data)

    def testUniqueInvariantFilters(self):
        # ensure that all the filters are actually invariant
        key = random.PRNGKey(0)

        for D in [2]: #image dimension
            operators = geom.make_all_operators(D)
            for N in [3]: #filter size
                key, subkey = random.split(key)
                image = geom.GeometricImage(random.uniform(key, shape=(2*N,2*N)), 0, D)
                for k in [0,1,2]: #tensor order of filter
                    for parity in [0,1]:
                        filters = geom.get_unique_invariant_filters(N, k, parity, D, operators)

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
