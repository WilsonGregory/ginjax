import ginet.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random
import jax.lax

TINY = 1.0e-5


def conv_subimage(image, center_key, filter_image, filter_image_keys=None):
    """
    Get the subimage (on the torus) centered on center_idx that will be convolved with filter_image
    args:
        center_key (index tuple): tuple index of the center of this convolution
        filter_image (GeometricFilter): the GeometricFilter we are convolving with
        filter_image_keys (list): For efficiency, the key offsets of the filter_image. Defaults to None.
    """
    m = (filter_image.spatial_dims[0] - 1) // 2
    assert (m * 2) + 1 == filter_image.spatial_dims[0]  # for this old function, needs to be odd

    if filter_image_keys is None:
        filter_image_keys = filter_image.key_array() - m  # centered key array

    key_list = image.hash(filter_image_keys + jnp.array(center_key))  # key list on the torus
    # values, reshaped to the correct shape, which is the filter_image shape, while still having the tensor shape
    vals = image[key_list].reshape(filter_image.spatial_dims + (image.D,) * image.k)
    return image.__class__(vals, image.parity, image.D)


def convolve_with_slow(image, filter_image):
    """
    Apply the convolution filter_image to this geometric image. Keeping this around for testing.
    args:
        filter_image (GeometricFilter-like): convolution that we are applying, can be an image or a filter
    """
    m = (filter_image.spatial_dims[0] - 1) // 2
    assert (m * 2) + 1 == filter_image.spatial_dims[0]  # for this old function, needs to be odd

    newimage = image.__class__.zeros(
        image.spatial_dims,
        image.k + filter_image.k,
        image.parity + filter_image.parity,
        image.D,
    )

    if isinstance(filter_image, geom.GeometricImage):
        filter_image = geom.GeometricFilter.from_image(filter_image)  # will break if N is not odd

    filter_image_keys = filter_image.key_array() - m
    for key in image.keys():
        subimage = conv_subimage(image, key, filter_image, filter_image_keys)
        newimage[key] = jnp.sum((subimage * filter_image).data, axis=tuple(range(image.D)))
    return newimage


# Now test group actions on k-tensors:
def do_group_actions(operators):
    """
    # Notes:
    - This only does minimal tests!
    """
    D = len(operators[0])
    key = random.PRNGKey(0)

    for parity in [0, 1]:

        key, subkey = random.split(key)

        # vector dot vector
        v1 = geom.GeometricImage(random.normal(subkey, shape=((1,) * D + (D,))), parity, D)
        key, subkey = random.split(key)
        v2 = geom.GeometricImage(random.normal(subkey, shape=((1,) * D + (D,))), parity, D)
        dots = [
            (v1.times_group_element(gg) * v2.times_group_element(gg)).contract(0, 1).data
            for gg in operators
        ]
        dots = jnp.array(dots)
        if not jnp.allclose(dots, jnp.mean(dots)):
            print("failed (parity = {}) vector dot test.".format(parity))
            return False
        print("passed (parity = {}) vector dot test.".format(parity))

        # tensor times tensor
        key, subkey = random.split(key)
        T3 = geom.GeometricImage(random.normal(subkey, shape=((1,) * D + (D, D))), parity, D)
        key, subkey = random.split(key)
        T4 = geom.GeometricImage(random.normal(subkey, shape=((1,) * D + (D, D))), parity, D)
        dots = [
            (T3.times_group_element(gg) * T4.times_group_element(gg))
            .contract(1, 2)
            .contract(0, 1)
            .data
            for gg in operators
        ]
        dots = jnp.array(dots)
        if not jnp.allclose(dots, jnp.mean(dots)):
            print("failed (parity = {}) tensor times tensor test".format(parity))
            return False
        print("passed (parity = {}) tensor times tensor test".format(parity))

        # vectors dotted through tensor
        key, subkey = random.split(key)
        v5 = geom.GeometricImage(random.normal(subkey, shape=((1,) * D + (D,))), 0, D)
        dots = [
            (v5.times_group_element(gg) * T3.times_group_element(gg) * v2.times_group_element(gg))
            .contract(1, 2)
            .contract(0, 1)
            .data
            for gg in operators
        ]
        dots = jnp.array(dots)
        if not jnp.allclose(dots, jnp.mean(dots)):
            print("failed (parity = {}) v T v test.".format(parity))
            return False
        print("passed (parity = {}) v T v test.".format(parity))

    return True


class TestSlowTests:

    # Test reserved for the slow tests, we only want to test these when we run the full battery

    def testConvSubimage(self):
        image1 = geom.GeometricImage(jnp.arange(25).reshape((5, 5)), 0, 2)
        filter1 = geom.GeometricFilter(jnp.zeros(25).reshape((5, 5)), 0, 2)
        subimage1 = conv_subimage(image1, (0, 0), filter1)
        assert subimage1.shape() == (5, 5)
        assert subimage1.D == image1.D
        assert subimage1.spatial_dims == filter1.spatial_dims
        assert subimage1.k == image1.k
        assert subimage1.parity == image1.parity
        assert (
            subimage1.data
            == jnp.array(
                [
                    [18, 19, 15, 16, 17],
                    [23, 24, 20, 21, 22],
                    [3, 4, 0, 1, 2],
                    [8, 9, 5, 6, 7],
                    [13, 14, 10, 11, 12],
                ],
                dtype=int,
            )
        ).all()

        subimage2 = conv_subimage(image1, (4, 4), filter1)
        assert subimage2.shape() == (5, 5)
        assert subimage2.D == image1.D
        assert subimage2.spatial_dims == filter1.spatial_dims
        assert subimage2.k == image1.k
        assert subimage2.parity == image1.parity
        assert (
            subimage2.data
            == jnp.array(
                [
                    [12, 13, 14, 10, 11],
                    [17, 18, 19, 15, 16],
                    [22, 23, 24, 20, 21],
                    [2, 3, 4, 0, 1],
                    [7, 8, 9, 5, 6],
                ],
                dtype=int,
            )
        ).all()

        image2 = geom.GeometricImage(jnp.arange(25).reshape((5, 5)), 0, 2) * geom.GeometricImage(
            jnp.ones((5, 5, 2)), 0, 2
        )
        subimage3 = conv_subimage(image2, (0, 0), filter1)
        assert subimage3.shape() == (5, 5, 2)
        assert subimage3.D == image2.D
        assert subimage3.spatial_dims == filter1.spatial_dims
        assert subimage3.k == image2.k
        assert subimage3.parity == image2.parity
        assert (
            subimage3.data
            == jnp.array(
                [
                    [x * jnp.array([1, 1]) for x in [18, 19, 15, 16, 17]],
                    [x * jnp.array([1, 1]) for x in [23, 24, 20, 21, 22]],
                    [x * jnp.array([1, 1]) for x in [3, 4, 0, 1, 2]],
                    [x * jnp.array([1, 1]) for x in [8, 9, 5, 6, 7]],
                    [x * jnp.array([1, 1]) for x in [13, 14, 10, 11, 12]],
                ],
                dtype=int,
            )
        ).all()

    def testConvolveWithRandoms(self):
        # this test uses convolve_with_slow to test convolve_with, possibly the blind leading the blind
        key = random.PRNGKey(0)
        N = 3

        for D in [2, 3]:
            for k_img in range(3):
                key, subkey = random.split(key)
                image = geom.GeometricImage(
                    random.uniform(subkey, shape=((N,) * D + (D,) * k_img)), 0, D
                )

                for k_filter in range(3):
                    key, subkey = random.split(key)
                    geom_filter = geom.GeometricFilter(
                        random.uniform(subkey, shape=((3,) * D + (D,) * k_filter)), 0, D
                    )

                    convolved_image = image.convolve_with(geom_filter)
                    convolved_image_slow = convolve_with_slow(image, geom_filter)

                    assert convolved_image.D == convolved_image_slow.D == image.D
                    assert (
                        convolved_image.spatial_dims
                        == convolved_image_slow.spatial_dims
                        == image.spatial_dims
                    )
                    assert convolved_image.k == convolved_image_slow.k == image.k + geom_filter.k
                    assert (
                        convolved_image.parity
                        == convolved_image_slow.parity
                        == (image.parity + geom_filter.parity) % 2
                    )
                    assert jnp.allclose(convolved_image.data, convolved_image_slow.data)

    def testUniqueInvariantFilters(self):
        # ensure that all the filters are actually invariant
        key = random.PRNGKey(0)

        for D in [2]:  # image dimension
            operators = geom.make_all_operators(D)
            for N in [2, 3]:  # filter size
                key, subkey = random.split(key)
                image = geom.GeometricImage(random.uniform(subkey, shape=(2 * N, 2 * N)), 0, D)
                for k in [0, 1, 2]:  # tensor order of filter
                    for parity in [0, 1]:
                        filters = geom.get_unique_invariant_filters(N, k, parity, D, operators)

                        for gg in operators:
                            for geom_filter in filters:

                                # test that the filters are invariant to the group operators
                                assert jnp.allclose(
                                    geom_filter.data,
                                    geom_filter.times_group_element(
                                        gg, precision=jax.lax.Precision.HIGH
                                    ).data,
                                )

                                # test that the convolution with the invariant filters is equivariant to gg
                                # convolutions are currently too slow to test this every time, but should be tested
                                assert jnp.allclose(
                                    image.convolve_with(geom_filter, padding=((1, 1),) * D)
                                    .times_group_element(
                                        gg,
                                        precision=jax.lax.Precision.HIGH,
                                    )
                                    .data,
                                    image.times_group_element(
                                        gg,
                                        precision=jax.lax.Precision.HIGH,
                                    )
                                    .convolve_with(geom_filter, padding=((1, 1),) * D)
                                    .data,
                                )

    def testGroup(self):
        for d in [2, 3]:  # could go longer, but it gets slow to test the closure
            operators = geom.make_all_operators(d)
            D = len(operators[0])
            # Check that the list of group operators is closed, O(d^3)
            for gg in operators:
                for gg2 in operators:
                    product = (gg @ gg2).astype(int)
                    found = False
                    for gg3 in operators:
                        if jnp.allclose(gg3, product):
                            found = True
                            break

                    assert found

            # Check that gg.T is gg.inv for all gg in group
            for gg in operators:
                assert jnp.allclose(gg @ gg.T, jnp.eye(D))

            assert do_group_actions(operators)
