import time

import ginet.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random, vmap

TINY = 1.0e-5


class TestMultiImage:

    def testConstructor(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        multi_image1 = geom.MultiImage({}, D, False)
        assert multi_image1.D == D
        assert multi_image1.is_torus == (False,) * D
        for _, _ in multi_image1.items():
            assert False  # its empty, so this won't ever be called

        k = 0
        multi_image2 = geom.MultiImage(
            {(k, 0): random.normal(key, shape=((1,) + (N,) * D + (D,) * k))}, D, False
        )
        assert multi_image2.D == D
        assert multi_image2.is_torus == (False,) * D
        assert multi_image2[(0, 0)].shape == (1, N, N)

        # multi_images can have multiple k values, and can have different size channels at each k
        multi_image3 = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((10,) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((3,) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )
        assert list(multi_image3.keys()) == [(0, 0), (1, 0)]
        assert multi_image3[(0, 0)].shape == (10, N, N)
        assert multi_image3[(1, 0)].shape == (3, N, N, D)
        assert multi_image3.is_torus == (True,) * D

    def testCopy(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((10,) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((3,) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )

        multi_image2 = multi_image1.copy()
        assert multi_image1 is not multi_image2

        multi_image2[(1, 0)] = jnp.arange(1 * (N**D) * D).reshape((1,) + (N,) * D + (D,) * 1)
        assert multi_image2[(1, 0)].shape == (1, N, N, D)
        assert multi_image1[(1, 0)].shape == (
            3,
            N,
            N,
            D,
        )  # original multi_image we copied from is unchanged

    def testFromImages(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        random_data = random.normal(key, shape=((10,) + (N,) * D + (D,) * 1))
        images = [geom.GeometricImage(data, 0, D) for data in random_data]
        multi_image1 = geom.MultiImage.from_images(images)
        assert multi_image1 is not None
        assert multi_image1.D == D
        assert multi_image1.is_torus == (True,) * D
        assert list(multi_image1.keys()) == [(1, 0)]
        assert multi_image1[(1, 0)].shape == (10, N, N, D)

        # now images has multiple different values of k
        random_data2 = random.normal(key, shape=((33,) + (N,) * D + (D,) * 2))
        images.extend([geom.GeometricImage(data, 0, D) for data in random_data2])
        multi_image2 = geom.MultiImage.from_images(images)
        assert multi_image2 is not None
        assert list(multi_image2.keys()) == [(1, 0), (2, 0)]
        assert multi_image2[(1, 0)].shape == (10, N, N, D)
        assert multi_image2[(2, 0)].shape == (33, N, N, D, D)

    def testEq(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 3

        key, subkey = random.split(key)
        multi_image1 = geom.MultiImage(
            {(0, 0): random.normal(subkey, shape=((10,) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        multi_image1.append(1, 0, random.normal(subkey, shape=((10,) + (N,) * D + (D,) * 1)))

        multi_image2 = multi_image1.copy()
        assert multi_image1 == multi_image2

        # keys do not match
        multi_image3 = geom.MultiImage({(0, 0): jnp.ones((10,) + (N,) * D + (D,) * 0)}, D, True)
        assert multi_image1 != multi_image3

        # values do not match
        multi_image4 = geom.MultiImage(
            {
                (0, 0): jnp.ones((10,) + (N,) * D + (D,) * 0),
                (1, 0): jnp.ones((10,) + (N,) * D + (D,) * 1),
            },
            D,
            True,
        )
        assert multi_image1 != multi_image4

        # is_torus does not match
        multi_image5 = geom.MultiImage(multi_image1.data, D, False)
        assert multi_image1 != multi_image5

    def testGetSpatialDims(self):
        D = 2
        N = 5

        multi_image = geom.MultiImage({}, D)
        assert multi_image.get_spatial_dims() == ()

        multi_image = geom.MultiImage({(0, 0): jnp.ones((1,) + (N,) * D)}, D)
        assert multi_image.get_spatial_dims() == (N,) * D

        multi_image = geom.MultiImage(
            {(1, 0): jnp.ones((1,) + (N,) * D + (D,)), (1, 1): jnp.ones((1,) + (N,) * D + (D,))}, D
        )
        assert multi_image.get_spatial_dims() == (N,) * D

    def testAppend(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((10,) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((3,) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )

        image_block = random.normal(key, shape=((4,) + (N,) * D + (D,) * 1))
        multi_image1.append(1, 0, image_block)
        assert multi_image1[(0, 0)].shape == (10, N, N)  # unchanged
        assert multi_image1[(1, 0)].shape == (7, N, N, D)  # updated 3+4=7

        image_block2 = random.normal(key, shape=((2,) + (N,) * D + (D,) * 2))
        multi_image1.append(2, 0, image_block2)
        assert multi_image1[(0, 0)].shape == (10, N, N)  # unchanged
        assert multi_image1[(1, 0)].shape == (7, N, N, D)  # unchanged
        assert multi_image1[(2, 0)].shape == (2, N, N, D, D)

        # add an image block to the wrong k bucket
        with pytest.raises(AssertionError):
            multi_image1.append(3, 0, image_block2)

        # N is set by append if it is empty
        multi_image2 = multi_image1.empty()
        assert multi_image2.get_spatial_dims() == ()

        multi_image2.append(0, 0, random.normal(key, shape=((10,) + (N,) * D + (D,) * 0)))
        assert multi_image2.get_spatial_dims() == (N,) * D

    def testConcat(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((10,) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((3,) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )
        multi_image2 = geom.MultiImage(
            {
                (1, 0): random.normal(key, shape=((4,) + (N,) * D + (D,) * 1)),
                (2, 0): random.normal(key, shape=((5,) + (N,) * D + (D,) * 2)),
            },
            D,
            True,
        )

        multi_image3 = multi_image1.concat(multi_image2)
        assert list(multi_image3.keys()) == [(0, 0), (1, 0), (2, 0)]
        assert multi_image3[(0, 0)].shape == (10, N, N)
        assert multi_image3[(1, 0)].shape == (7, N, N, D)
        assert multi_image3[(2, 0)].shape == (5, N, N, D, D)

        # mismatched D
        multi_image4 = geom.MultiImage(
            {(0, 0): random.normal(key, shape=((10,) + (N,) * D + (D,) * 0))}, D, True
        )
        multi_image5 = geom.MultiImage(
            {(0, 0): random.normal(key, shape=((10,) + (N,) * 3 + (D,) * 0))}, 3, True
        )
        with pytest.raises(AssertionError):
            multi_image4.concat(multi_image5)

        # mismatched is_torus
        multi_image6 = geom.MultiImage(
            {(0, 0): random.normal(key, shape=((10,) + (N,) * D + (D,) * 0))}, D, True
        )
        multi_image7 = geom.MultiImage(
            {(0, 0): random.normal(key, shape=((10,) + (N,) * D + (D,) * 0))}, D, False
        )
        with pytest.raises(AssertionError):
            multi_image6.concat(multi_image7)

    def testAdd(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        channels = 4

        key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7, subkey8 = random.split(
            key, 9
        )

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=((channels,) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(subkey2, shape=((channels,) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )

        multi_image2 = geom.MultiImage(
            {
                (0, 0): random.normal(subkey3, shape=((channels,) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(subkey4, shape=((channels,) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )

        multi_image3 = multi_image1 + multi_image2
        multi_image4 = geom.MultiImage(
            {
                (0, 0): multi_image1[(0, 0)] + multi_image2[(0, 0)],
                (1, 0): multi_image1[(1, 0)] + multi_image2[(1, 0)],
            },
            D,
            True,
        )

        assert multi_image3 == multi_image4

        # mismatched multi_image types
        multi_image5 = geom.MultiImage(
            {(0, 0): random.normal(subkey5, shape=((channels,) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        multi_image6 = geom.MultiImage(
            {(1, 0): random.normal(subkey6, shape=((channels,) + (N,) * D + (D,) * 1))},
            D,
            True,
        )
        with pytest.raises(AssertionError):
            assert multi_image5 + multi_image6

        # mismatched number of channels
        multi_image7 = geom.MultiImage(
            {(0, 0): random.normal(subkey7, shape=((channels + 1,) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        multi_image8 = geom.MultiImage(
            {(0, 0): random.normal(subkey8, shape=((channels,) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        with pytest.raises(TypeError):
            assert multi_image7 + multi_image8

    def testMul(self):
        key = random.PRNGKey(0)
        channels = 3
        N = 5
        D = 2

        key, subkey1, subkey2 = random.split(key, 3)

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(channels,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(channels,) + (N,) * D + (D,)),
            },
            D,
            True,
        )

        multi_image2 = multi_image1 * 3
        assert jnp.allclose(multi_image2[(0, 0)], multi_image1[(0, 0)] * 3)
        assert jnp.allclose(multi_image2[(1, 0)], multi_image1[(1, 0)] * 3)
        assert multi_image2.D == D
        assert multi_image2.is_torus == (True,) * D

        multi_image3 = multi_image1 * -1
        assert jnp.allclose(multi_image3[(0, 0)], multi_image1[(0, 0)] * -1)
        assert jnp.allclose(multi_image3[(1, 0)], multi_image1[(1, 0)] * -1)
        assert multi_image2.D == D
        assert multi_image2.is_torus == (True,) * D

        # try to multiply two multi_images together
        with pytest.raises(AssertionError):
            assert multi_image1 * multi_image1

    def testDiv(self):
        key = random.PRNGKey(0)
        channels = 3
        N = 5
        D = 2

        key, subkey1, subkey2 = random.split(key, 3)

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(channels,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(channels,) + (N,) * D + (D,)),
            },
            D,
            True,
        )

        multi_image2 = multi_image1 / 3
        assert jnp.allclose(multi_image2[(0, 0)], multi_image1[(0, 0)] / 3)
        assert jnp.allclose(multi_image2[(1, 0)], multi_image1[(1, 0)] / 3)
        assert multi_image2.D == D
        assert multi_image2.is_torus == (True,) * D

        multi_image3 = multi_image1 / -1
        assert jnp.allclose(multi_image3[(0, 0)], multi_image1[(0, 0)] / -1)
        assert jnp.allclose(multi_image3[(1, 0)], multi_image1[(1, 0)] / -1)
        assert multi_image2.D == D
        assert multi_image2.is_torus == (True,) * D

        # try to multiply two multi_images together
        with pytest.raises(AssertionError):
            assert multi_image1 * multi_image1

    def testSize(self):
        D = 2
        N = 5

        # empty multi_image
        multi_image1 = geom.MultiImage({}, D)
        assert multi_image1.size() == 0

        # basic scalar multi_image
        multi_image2 = geom.MultiImage({(0, 0): jnp.ones((1,) + (N,) * D)}, D)
        assert multi_image2.size() == N**D

        # multi_image with channels
        multi_image3 = geom.MultiImage({(0, 0): jnp.ones((4,) + (N,) * D)}, D)
        assert multi_image3.size() == (4 * N**D)

        # more complex multi_image
        multi_image4 = geom.MultiImage(
            {
                (0, 0): jnp.ones((1,) + (N,) * D),
                (1, 0): jnp.ones((4,) + (N,) * D + (D,)),
                (1, 1): jnp.ones((2,) + (N,) * D + (D,)),
                (2, 0): jnp.ones((3,) + (N,) * D + (D, D)),
            },
            D,
        )
        assert multi_image4.size() == (N**D + 4 * N**D * D + 2 * N**D * D + 3 * N**D * D * D)

    def testVector(self):
        # Test the from_vector and to_vector functions
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        multi_image_example = geom.MultiImage(
            {
                (0, 0): jnp.ones((1,) + (N,) * D),
                (1, 0): jnp.ones((1,) + (N,) * D + (D,)),
                (2, 0): jnp.ones((1,) + (N,) * D + (D, D)),
            },
            D,
        )

        key, subkey = random.split(key)
        rand_data = random.normal(subkey, shape=(multi_image_example.size(),))

        rand_multi_image = geom.MultiImage.from_vector(rand_data, multi_image_example)

        assert rand_multi_image.size() == multi_image_example.size()
        assert jnp.allclose(rand_multi_image.to_vector(), rand_data)

    def testToFromScalarMultiImage(self):
        D = 2
        N = 5

        multi_image_example = geom.MultiImage(
            {
                (0, 0): jnp.ones((1,) + (N,) * D),
                (1, 0): jnp.ones((1,) + (N,) * D + (D,)),
                (2, 0): jnp.ones((1,) + (N,) * D + (D, D)),
            },
            D,
        )

        scalar_multi_image = multi_image_example.to_scalar_multi_image()

        assert len(scalar_multi_image.keys()) == 1
        assert next(iter(scalar_multi_image.keys())) == (0, 0)
        assert jnp.allclose(scalar_multi_image[(0, 0)], jnp.ones((1 + D + D * D,) + (N,) * D))

        key = random.PRNGKey(0)
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
        rand_multi_image = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=((3,) + (N,) * D)),
                (1, 0): random.normal(subkey2, shape=((1,) + (N,) * D + (D,))),
                (1, 1): random.normal(subkey3, shape=((2,) + (N,) * D + (D,))),
                (2, 0): random.normal(subkey4, shape=((1,) + (N,) * D + (D, D))),
            },
            D,
        )

        scalar_multi_image2 = rand_multi_image.to_scalar_multi_image()
        assert list(scalar_multi_image2.keys()) == [(0, 0)]
        assert rand_multi_image == rand_multi_image.to_scalar_multi_image().from_scalar_multi_image(
            rand_multi_image.get_signature()
        )

    def testTimesGroupElement(self):
        N = 5
        channels = 3

        vmap_times_gg = vmap(geom.times_group_element, in_axes=(None, 0, None, None))
        key = random.PRNGKey(0)
        for D in [2, 3]:
            multi_image = geom.MultiImage({}, D)

            for parity in [0, 1]:
                for k in [0, 1, 2, 3]:
                    key, subkey = random.split(key)
                    multi_image.append(
                        k,
                        parity,
                        random.normal(subkey, shape=((channels,) + (N,) * D + (D,) * k)),
                    )

            operators = geom.make_all_operators(D)

            for gg in operators:
                rotated_multi_image = multi_image.times_group_element(gg)

                for (k, parity), img_block in multi_image.items():
                    rotated_block = vmap_times_gg(D, img_block, parity, gg)
                    assert jnp.allclose(rotated_multi_image[(k, parity)], rotated_block)

    def testNorm(self):
        N = 5
        D = 2
        channels = 3

        key = random.PRNGKey(0)

        # norm of scalars, pseudo scalars, and vectors
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        multi_image = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(channels,) + (N,) * D),
                (0, 1): random.normal(subkey2, shape=(channels,) + (N,) * D),
                (1, 0): random.normal(subkey3, shape=(channels,) + (N,) * D + (D,)),
            },
            D,
        )

        normed_multi_image = multi_image.norm()
        assert list(normed_multi_image.keys()) == [(0, 0)]  # odd parity is converted to even parity
        assert normed_multi_image[(0, 0)].shape == ((3 * channels,) + (N,) * D)
        assert jnp.allclose(normed_multi_image[(0, 0)][:channels], jnp.abs(multi_image[(0, 0)]))
        assert jnp.allclose(
            normed_multi_image[(0, 0)][channels : 2 * channels], jnp.abs(multi_image[(0, 1)])
        )
        vector_norm = jnp.linalg.norm(
            multi_image[(1, 0)].reshape(multi_image[(1, 0)].shape[: 1 + D] + (-1,)), axis=1 + D
        )
        assert jnp.allclose(normed_multi_image[(0, 0)][2 * channels :], vector_norm)

    def testGetComponent(self):
        N = 5
        D = 2
        channels = 10
        timesteps = 4
        key = random.PRNGKey(0)
        key, subkey1 = random.split(key, 2)
        multi_image = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(channels * timesteps,) + (N,) * D),
            },
            D,
        )
        assert isinstance(multi_image.get_component(0, future_steps=timesteps), geom.MultiImage)
        assert jnp.allclose(
            multi_image.get_component(0, future_steps=timesteps)[(0, 0)],
            multi_image[(0, 0)].reshape((-1, timesteps) + (N,) * D)[0],
        )
        assert jnp.allclose(
            multi_image.get_component(1, future_steps=timesteps)[(0, 0)],
            multi_image[(0, 0)].reshape((-1, timesteps) + (N,) * D)[1],
        )

        # slices work as well
        assert jnp.allclose(
            multi_image.get_component(slice(0, 2), future_steps=timesteps)[(0, 0)],
            multi_image[(0, 0)]
            .reshape((-1, timesteps) + (N,) * D)[:2]
            .reshape((2 * timesteps,) + (N,) * D),
        )

    def testToImages(self):
        N = 5
        D = 2
        channels = 3
        key = random.PRNGKey(0)
        subkey1, subkey2, subkey3, subkey4 = random.split(key, num=4)
        multi_image = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(channels,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(channels,) + (N,) * D + (D,)),
                (1, 1): random.normal(subkey3, shape=(channels,) + (N,) * D + (D,)),
                (2, 0): random.normal(subkey4, shape=(channels,) + (N,) * D + (D, D)),
            },
            D,
        )

        images = multi_image.to_images()
        assert jnp.allclose(images[0].data, multi_image[(0, 0)][0])
        assert images[0].parity == 0
        assert jnp.allclose(images[1].data, multi_image[(0, 0)][1])
        assert images[1].parity == 0
        assert jnp.allclose(images[2].data, multi_image[(0, 0)][2])
        assert images[2].parity == 0
        assert jnp.allclose(images[3].data, multi_image[(1, 0)][0])
        assert images[3].parity == 0
        assert jnp.allclose(images[4].data, multi_image[(1, 0)][1])
        assert images[4].parity == 0
        assert jnp.allclose(images[5].data, multi_image[(1, 0)][2])
        assert images[5].parity == 0
        assert jnp.allclose(images[6].data, multi_image[(1, 1)][0])
        assert images[6].parity == 1
        assert jnp.allclose(images[7].data, multi_image[(1, 1)][1])
        assert images[7].parity == 1
        assert jnp.allclose(images[8].data, multi_image[(1, 1)][2])
        assert images[8].parity == 1
        assert jnp.allclose(images[9].data, multi_image[(2, 0)][0])
        assert images[9].parity == 0
        assert jnp.allclose(images[10].data, multi_image[(2, 0)][1])
        assert images[10].parity == 0
        assert jnp.allclose(images[11].data, multi_image[(2, 0)][2])
        assert images[11].parity == 0

    def testGetSignature(self):
        D = 2
        N = 5

        multi_image = geom.MultiImage({}, D)
        assert multi_image.get_signature() == geom.Signature(())

        multi_image = geom.MultiImage({(0, 0): jnp.ones((1,) + (N,) * D)}, D)
        assert multi_image.get_signature() == geom.Signature((((0, 0), 1),))

        multi_image = geom.MultiImage(
            {(1, 0): jnp.ones((2,) + (N,) * D + (D,)), (1, 1): jnp.ones((5,) + (N,) * D + (D,))}, D
        )
        assert multi_image.get_signature() == geom.Signature((((1, 0), 2), ((1, 1), 5)))

    def testGetNLeading(self):
        D = 2
        N = 5

        multi_image = geom.MultiImage({}, D)
        assert multi_image.get_n_leading() == 0

        multi_image = geom.MultiImage({(0, 0): jnp.ones((N,) * D)}, D)
        assert multi_image.get_n_leading() == 0

        multi_image = geom.MultiImage({(0, 0): jnp.ones((5,) + (N,) * D)}, D)
        assert multi_image.get_n_leading() == 1

        multi_image = geom.MultiImage({(0, 0): jnp.ones((5, 3) + (N,) * D)}, D)
        assert multi_image.get_n_leading() == 2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test TestBatchMultiImage
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TestBatchMultiImage:
    """
    For testing when the first axis of a MultiImage is a batch axis
    """

    def testConstructor(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        multi_image1 = geom.MultiImage({}, D, False)
        assert multi_image1.D == D
        assert multi_image1.is_torus == (False,) * D
        for _, _ in multi_image1.items():
            assert False  # its empty, so this won't ever be called

        k = 0
        multi_image2 = geom.MultiImage(
            {(k, 0): random.normal(key, shape=((10, 1) + (N,) * D + (D,) * k))},
            D,
            False,
        )
        assert multi_image2.D == D
        assert multi_image2.is_torus == (False,) * D
        assert multi_image2[(0, 0)].shape == (10, 1, N, N)

        # multi_images can have multiple k values with different channels,
        # but they should have same batch size, although this is currently unenforced
        multi_image3 = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((5, 10) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((5, 3) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )
        assert list(multi_image3.keys()) == [(0, 0), (1, 0)]
        assert multi_image3[(0, 0)].shape == (5, 10, N, N)
        assert multi_image3[(1, 0)].shape == (5, 3, N, N, D)
        assert multi_image3.is_torus == (True,) * D

    def testGetSpatialDims(self):
        D = 2
        N = 5
        batch = 4
        channels = 3

        multi_image = geom.MultiImage({}, D)
        assert multi_image.get_spatial_dims() == ()

        multi_image = geom.MultiImage({(0, 0): jnp.ones((batch, channels) + (N,) * D)}, D)
        assert multi_image.get_spatial_dims() == (N,) * D

        multi_image = geom.MultiImage(
            {
                (1, 0): jnp.ones((1,) + (N,) * D + (D,)),
                (1, 1): jnp.ones((batch, channels) + (N,) * D + (D,)),
            },
            D,
        )
        assert multi_image.get_spatial_dims() == (N,) * D

    def testGetSubset(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        k = 1

        multi_image1 = geom.MultiImage(
            {(k, 0): random.normal(key, shape=((100, 1) + (N,) * D + (D,) * k))},
            D,
            False,
        )

        multi_image2 = multi_image1.get_subset(jnp.array([3]))
        assert multi_image2.D == multi_image1.D
        assert multi_image2.is_torus == multi_image1.is_torus
        assert multi_image2.get_L() == 1
        assert multi_image2[(k, 0)].shape == (1, 1, N, N, D)
        assert jnp.allclose(multi_image2[(k, 0)][0], multi_image1[(k, 0)][3])

        multi_image3 = multi_image1.get_subset(jnp.array([3, 23, 4, 17]))
        assert multi_image3.get_L() == 4
        assert multi_image3[(k, 0)].shape == (4, 1, N, N, D)
        assert jnp.allclose(multi_image3[(k, 0)], multi_image1[(k, 0)][jnp.array([3, 23, 4, 17])])

        with pytest.raises(AssertionError):
            multi_image1.get_subset(jnp.array(0))

    def testGetOneKeepDimsFalse(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        k = 1

        multi_image1 = geom.MultiImage(
            {(k, 0): random.normal(key, shape=((100, 1) + (N,) * D + (D,) * k))},
            D,
            False,
        )

        multi_image2 = multi_image1.get_one(keepdims=False)
        assert isinstance(multi_image2, geom.MultiImage)
        assert multi_image2[(1, 0)].shape == (1, N, N, D)
        assert jnp.allclose(multi_image1[(1, 0)][0], multi_image2[(1, 0)])

        idx = 12
        multi_image3 = multi_image1.get_one(idx, keepdims=False)
        assert isinstance(multi_image2, geom.MultiImage)
        assert multi_image3[(1, 0)].shape == (1, N, N, D)
        assert jnp.allclose(multi_image1[(1, 0)][idx], multi_image3[(1, 0)])

    def testAppend(self):
        # For MultiImage with batch dimension, append should probably only be used while it is
        # vmapped to a multi_image
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((5, 10) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((5, 3) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )

        def mult(multi_image, param):
            out_multi_image = multi_image.empty()
            for (k, parity), image_block in multi_image.items():
                out_multi_image.append(k, parity, param * jnp.ones(image_block.shape))

            return out_multi_image

        multi_image2 = vmap(mult)(multi_image1, jnp.arange(5))
        assert multi_image2.D == multi_image1.D
        assert multi_image2.is_torus == multi_image1.is_torus
        assert multi_image2.keys() == multi_image1.keys()
        for multi_image2_image, multi_image1_image, num in zip(
            multi_image2[(0, 0)], multi_image1[(0, 0)], jnp.arange(5)
        ):
            assert jnp.allclose(multi_image2_image, num * jnp.ones(multi_image1_image.shape))

        for multi_image2_image, multi_image1_image, num in zip(
            multi_image2[(1, 0)], multi_image1[(1, 0)], jnp.arange(5)
        ):
            assert jnp.allclose(multi_image2_image, num * jnp.ones(multi_image1_image.shape))

    def testConcat(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

        multi_image1 = geom.MultiImage(
            {
                (1, 0): random.normal(subkey1, shape=((5, 10) + (N,) * D + (D,) * 1)),
                (2, 0): random.normal(subkey2, shape=((5, 3) + (N,) * D + (D,) * 2)),
            },
            D,
            True,
        )
        multi_image2 = geom.MultiImage(
            {
                (1, 0): random.normal(subkey3, shape=((7, 10) + (N,) * D + (D,) * 1)),
                (2, 0): random.normal(subkey4, shape=((7, 3) + (N,) * D + (D,) * 2)),
            },
            D,
            True,
        )

        multi_image3 = multi_image1.concat(multi_image2)
        assert multi_image3.D == D
        assert multi_image3.is_torus == (True,) * D
        assert multi_image3[(1, 0)].shape == (12, 10, N, N, D)
        assert multi_image3[(2, 0)].shape == (12, 3, N, N, D, D)
        assert jnp.allclose(
            multi_image3[(1, 0)], jnp.concatenate([multi_image1[(1, 0)], multi_image2[(1, 0)]])
        )

        key, subkey5, subkey6 = random.split(key, 3)
        multi_image4 = geom.MultiImage(
            {(1, 0): random.normal(subkey5, shape=((5, 10) + (N,) * D + (D,) * 1))},
            D,
            True,
        )
        multi_image5 = geom.MultiImage(
            {(1, 0): random.normal(subkey6, shape=((5, 2) + (N,) * D + (D,) * 1))},
            D,
            True,
        )

        multi_image6 = multi_image4.concat(multi_image5, axis=1)
        assert multi_image6.D == D
        assert list(multi_image6.keys()) == [(1, 0)]
        assert multi_image6[(1, 0)].shape == (5, 12, N, N, D)
        assert jnp.allclose(
            multi_image6[(1, 0)],
            jnp.concatenate([multi_image4[(1, 0)], multi_image5[(1, 0)]], axis=1),
        )

    def testSize(self):
        D = 2
        N = 5

        # empty multi_image
        multi_image1 = geom.MultiImage({}, D)
        assert multi_image1.size() == 0

        # basic scalar multi_image
        multi_image2 = geom.MultiImage({(0, 0): jnp.ones((1, 1) + (N,) * D)}, D)
        assert multi_image2.size() == N**D

        # multi_image with channels
        multi_image3 = geom.MultiImage({(0, 0): jnp.ones((2, 4) + (N,) * D)}, D)
        assert multi_image3.size() == (2 * 4 * N**D)

        # more complex multi_image
        multi_image4 = geom.MultiImage(
            {
                (0, 0): jnp.ones((3, 1) + (N,) * D),
                (1, 0): jnp.ones((3, 4) + (N,) * D + (D,)),
                (1, 1): jnp.ones((3, 2) + (N,) * D + (D,)),
                (2, 0): jnp.ones((3, 3) + (N,) * D + (D, D)),
            },
            D,
        )
        assert multi_image4.size() == (3 * (N**D + 4 * N**D * D + 2 * N**D * D + 3 * N**D * D * D))

    def testTimesGroupElement(self):
        N = 5
        batch = 4
        channels = 3

        vmap_times_gg = vmap(
            vmap(geom.times_group_element, in_axes=(None, 0, None, None)),
            in_axes=(None, 0, None, None),
        )
        key = random.PRNGKey(0)
        for D in [2, 3]:
            multi_image = geom.MultiImage({}, D)

            for parity in [0, 1]:
                for k in [0, 1, 2, 3]:
                    key, subkey = random.split(key)
                    multi_image.append(
                        k,
                        parity,
                        random.normal(subkey, shape=((batch, channels) + (N,) * D + (D,) * k)),
                    )

            operators = geom.make_all_operators(D)

            for gg in operators:
                rotated_multi_image = multi_image.times_group_element(gg)

                for (k, parity), img_block in multi_image.items():
                    rotated_block = vmap_times_gg(D, img_block, parity, gg)
                    assert jnp.allclose(rotated_multi_image[(k, parity)], rotated_block)

    def testNorm(self):
        N = 5
        D = 2
        batch = 4
        channels = 3

        key = random.PRNGKey(0)

        # norm of scalars, pseudo scalars, and vectors
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        multi_image = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(batch, channels) + (N,) * D),
                (0, 1): random.normal(subkey2, shape=(batch, channels) + (N,) * D),
                (1, 0): random.normal(subkey3, shape=(batch, channels) + (N,) * D + (D,)),
            },
            D,
        )

        normed_multi_image = multi_image.norm()
        assert list(normed_multi_image.keys()) == [(0, 0)]  # odd parity is converted to even parity
        assert normed_multi_image[(0, 0)].shape == ((batch, 3 * channels) + (N,) * D)
        assert jnp.allclose(normed_multi_image[(0, 0)][:, :channels], jnp.abs(multi_image[(0, 0)]))
        assert jnp.allclose(
            normed_multi_image[(0, 0)][:, channels : 2 * channels], jnp.abs(multi_image[(0, 1)])
        )
        vector_norm = jnp.linalg.norm(
            multi_image[(1, 0)].reshape(multi_image[(1, 0)].shape[: 2 + D] + (-1,)), axis=2 + D
        )
        assert jnp.allclose(normed_multi_image[(0, 0)][:, 2 * channels :], vector_norm)

    def testAdd(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        channels = 4
        batch = 3

        key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7, subkey8 = random.split(
            key, 9
        )

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=((batch, channels) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(subkey2, shape=((batch, channels) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )

        multi_image2 = geom.MultiImage(
            {
                (0, 0): random.normal(subkey3, shape=((batch, channels) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(subkey4, shape=((batch, channels) + (N,) * D + (D,) * 1)),
            },
            D,
            True,
        )

        multi_image3 = multi_image1 + multi_image2
        multi_image4 = geom.MultiImage(
            {
                (0, 0): multi_image1[(0, 0)] + multi_image2[(0, 0)],
                (1, 0): multi_image1[(1, 0)] + multi_image2[(1, 0)],
            },
            D,
            True,
        )

        assert multi_image3 == multi_image4

        # mismatched multi_image types
        multi_image5 = geom.MultiImage(
            {(0, 0): random.normal(subkey5, shape=((batch, channels) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        multi_image6 = geom.MultiImage(
            {(1, 0): random.normal(subkey6, shape=((batch, channels) + (N,) * D + (D,) * 1))},
            D,
            True,
        )
        with pytest.raises(AssertionError):
            assert multi_image5 + multi_image6

        # mismatched number of channels
        multi_image7 = geom.MultiImage(
            {(0, 0): random.normal(subkey7, shape=((batch, channels + 1) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        multi_image8 = geom.MultiImage(
            {(0, 0): random.normal(subkey8, shape=((batch, channels) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        with pytest.raises(TypeError):
            assert multi_image7 + multi_image8

        # mismatched batch size
        key, subkey9, subkey10 = random.split(key, 3)
        multi_image9 = geom.MultiImage(
            {(0, 0): random.normal(subkey9, shape=((batch + 1, channels) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        multi_image10 = geom.MultiImage(
            {(0, 0): random.normal(subkey10, shape=((batch, channels) + (N,) * D + (D,) * 0))},
            D,
            True,
        )
        with pytest.raises(TypeError):
            assert multi_image9 + multi_image10

    def testMul(self):
        key = random.PRNGKey(0)
        batch = 4
        channels = 3
        N = 5
        D = 2

        key, subkey1, subkey2 = random.split(key, 3)

        multi_image1 = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(batch, channels) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(batch, channels) + (N,) * D + (D,)),
            },
            D,
            True,
        )

        multi_image2 = multi_image1 * 3
        assert jnp.allclose(multi_image2[(0, 0)], multi_image1[(0, 0)] * 3)
        assert jnp.allclose(multi_image2[(1, 0)], multi_image1[(1, 0)] * 3)
        assert multi_image2.D == D
        assert multi_image2.is_torus == (True,) * D

        multi_image3 = multi_image1 * -1
        assert jnp.allclose(multi_image3[(0, 0)], multi_image1[(0, 0)] * -1)
        assert jnp.allclose(multi_image3[(1, 0)], multi_image1[(1, 0)] * -1)
        assert multi_image2.D == D
        assert multi_image2.is_torus == (True,) * D

        # try to multiply two multi_images together
        with pytest.raises(AssertionError):
            assert multi_image1 * multi_image1

    def testToFromScalarMultiImage(self):
        D = 2
        batch = 4
        N = 5

        multi_image_example = geom.MultiImage(
            {
                (0, 0): jnp.ones((batch, 4) + (N,) * D),
                (1, 0): jnp.ones((batch, 2) + (N,) * D + (D,)),
                (2, 0): jnp.ones((batch, 1) + (N,) * D + (D, D)),
            },
            D,
        )

        scalar_multi_image = multi_image_example.to_scalar_multi_image()
        print(scalar_multi_image[(0, 0)].shape)

        assert len(scalar_multi_image.keys()) == 1
        assert next(iter(scalar_multi_image.keys())) == (0, 0)
        assert jnp.allclose(
            scalar_multi_image[(0, 0)],
            jnp.ones((batch, 4 + 2 * D + D * D) + (N,) * D),
        )

        key = random.PRNGKey(0)
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
        rand_multi_image = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=((batch, 3) + (N,) * D)),
                (1, 0): random.normal(subkey2, shape=((batch, 1) + (N,) * D + (D,))),
                (1, 1): random.normal(subkey3, shape=((batch, 2) + (N,) * D + (D,))),
                (2, 0): random.normal(subkey4, shape=((batch, 1) + (N,) * D + (D, D))),
            },
            D,
        )

        scalar_multi_image2 = rand_multi_image.to_scalar_multi_image()
        assert list(scalar_multi_image2.keys()) == [(0, 0)]
        assert rand_multi_image == rand_multi_image.to_scalar_multi_image().from_scalar_multi_image(
            rand_multi_image.get_signature()
        )
