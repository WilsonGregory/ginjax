import time

import ginjax.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random
import jax

TINY = 1.0e-5


class TestFunctionalGeometricImage:
    """
    Class to test the functional versions of the geometric image functions.
    """

    def testParseShape(self):
        spatial_dims, k = geom.parse_shape((4, 4, 2), 2)
        assert spatial_dims == (4, 4)
        assert k == 1

        # non-square, D=2
        spatial_dims, k = geom.parse_shape((4, 5, 2), 2)
        assert spatial_dims == (4, 5)
        assert k == 1

        # k = 0
        spatial_dims, k = geom.parse_shape((5, 5), 2)
        assert spatial_dims == (5, 5)
        assert k == 0

        # k = 2
        spatial_dims, k = geom.parse_shape((5, 5, 2, 2), 2)
        assert spatial_dims == (5, 5)
        assert k == 2

        # D = 3
        spatial_dims, k = geom.parse_shape((5, 5, 5, 3, 3), 3)
        assert spatial_dims == (5, 5, 5)
        assert k == 2

        # non-square, D=3
        spatial_dims, k = geom.parse_shape((4, 5, 6, 3), 3)
        assert spatial_dims == (4, 5, 6)
        assert k == 1

        # passing data instead of shape
        with pytest.raises(AssertionError):
            geom.parse_shape(jnp.ones((5, 5, 2)), 2)

        # passing a bungus shape, shape is less than D
        with pytest.raises(AssertionError):
            geom.parse_shape((5, 5), 3)

    def testHash(self):
        D = 2
        indices = jnp.arange(10 * D).reshape((10, D))

        img1 = jnp.ones((4, 4))
        hashed_indices = geom.hash(D, img1, indices)
        assert jnp.allclose(hashed_indices[0], jnp.array([0, 2, 0, 2, 0, 2, 0, 2, 0, 2]))
        assert jnp.allclose(hashed_indices[1], jnp.array([1, 3, 1, 3, 1, 3, 1, 3, 1, 3]))

        img2 = jnp.ones((3, 4))
        hashed_indices = geom.hash(D, img2, indices)
        assert jnp.allclose(hashed_indices[0], jnp.array([0, 2, 1, 0, 2, 1, 0, 2, 1, 0]))
        assert jnp.allclose(hashed_indices[1], jnp.array([1, 3, 1, 3, 1, 3, 1, 3, 1, 3]))

    def testConvolveNonSquare(self):
        D = 2
        in_c = 1
        out_c = 1
        batch = 1

        # Test that non-square convolution works
        img1 = jnp.arange(20).reshape((batch, in_c, 4, 5))
        filter_img = jnp.ones((out_c, in_c, 3, 3))

        # SAME padding
        res = geom.convolve(D, img1, filter_img, False, padding="SAME")
        assert res.shape == (batch, out_c, 4, 5)
        assert jnp.allclose(
            res,
            jnp.array(
                [
                    [
                        [
                            [12, 21, 27, 33, 24],
                            [33, 54, 63, 72, 51],
                            [63, 99, 108, 117, 81],
                            [52, 81, 87, 93, 64],
                        ]
                    ]
                ]
            ),
        )

        # VALID padding
        res = geom.convolve(D, img1, filter_img, False, padding="VALID")
        assert jnp.allclose(
            res,
            jnp.array(
                [
                    [
                        [
                            [54, 63, 72],
                            [99, 108, 117],
                        ]
                    ]
                ]
            ),
        )

        # TORUS padding
        res = geom.convolve(D, img1, filter_img, True, padding="TORUS")
        assert jnp.allclose(
            res,
            jnp.array(
                [
                    [
                        [
                            [75, 69, 78, 87, 81],
                            [60, 54, 63, 72, 66],
                            [105, 99, 108, 117, 111],
                            [90, 84, 93, 102, 96],
                        ]
                    ]
                ]
            ),
        )

        # TORUS padding with dilation
        res = geom.convolve(D, img1, filter_img, True, padding="TORUS", rhs_dilation=(2,) * D)
        assert jnp.allclose(
            res,
            jnp.array(
                [
                    [
                        [
                            [75, 84, 78, 72, 81],
                            [120, 129, 123, 117, 126],
                            [45, 54, 48, 42, 51],
                            [90, 99, 93, 87, 96],
                        ]
                    ]
                ]
            ),
        )

    def testConvolveContract2D(self):
        """
        Test that convolve_contract is the same as convolving, then contracting in 2D
        """
        N = 3
        D = 2
        in_c = 1
        out_c = 1
        batch = 1
        is_torus = (True,) * D
        key = random.PRNGKey(time.time_ns())

        for img_k in range(4):
            for filter_k in range(4):
                key, subkey = random.split(key)
                image = random.normal(subkey, shape=((batch, in_c) + (N,) * D + (D,) * img_k))

                key, subkey = random.split(key)
                conv_filter = random.normal(
                    subkey, shape=((out_c, in_c) + (N,) * D + (D,) * (img_k + filter_k))
                )

                contraction_idxs = tuple((i, i + img_k) for i in range(img_k))
                assert jnp.allclose(
                    geom.convolve_contract(D, image, conv_filter, is_torus),
                    geom.multicontract(
                        geom.convolve(D, image, conv_filter, is_torus),
                        contraction_idxs,
                        idx_shift=D + 2,
                    ),
                    rtol=TINY,
                    atol=TINY,
                )

    def testConvolveContract3D(self):
        """
        Test that convolve_contract is the same as convolving, then contracting in 3D
        """
        N = 3
        D = 3
        in_c = 1
        out_c = 1
        batch = 1
        is_torus = (True,) * D
        key = random.PRNGKey(time.time_ns())

        for img_k in range(3):
            for filter_k in range(2):
                key, subkey = random.split(key)
                image = random.normal(subkey, shape=((batch, in_c) + (N,) * D + (D,) * img_k))

                key, subkey = random.split(key)
                conv_filter = random.normal(
                    subkey, shape=((out_c, in_c) + (N,) * D + (D,) * (img_k + filter_k))
                )

                contraction_idxs = tuple((i, i + img_k) for i in range(img_k))
                assert jnp.allclose(
                    geom.convolve_contract(D, image, conv_filter, is_torus),
                    geom.multicontract(
                        geom.convolve(D, image, conv_filter, is_torus),
                        contraction_idxs,
                        idx_shift=D + 2,
                    ),
                    rtol=TINY,
                    atol=TINY,
                )

    def testConvolveDepth(self):
        # Depth convolve, where input data has 2 channels, so filter needs to have depth 2.
        image_data = jnp.array(
            [
                [
                    [
                        [
                            2,
                            1,
                            0,
                        ],
                        [0, 0, -3],
                        [2, 0, 1],
                    ],
                    [
                        [-7, 4, -4],
                        [3, 1, 3],
                        [-4, -1, 1],
                    ],
                ]
            ]
        )
        assert image_data.shape == (1, 2, 3, 3)

        filter_data = jnp.array(
            [
                [
                    [
                        [1, 0, 1],
                        [0, 0, 0],
                        [1, 0, 1],
                    ],
                    [
                        [0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0],
                    ],
                ]
            ]
        )
        assert filter_data.shape == (1, 2, 3, 3)

        convolved_image = geom.convolve(2, image_data, filter_data, is_torus=(True,) * 2)
        assert convolved_image.shape == (1, 1, 3, 3)
        assert jnp.allclose(
            convolved_image,
            jnp.array(
                [
                    [
                        [
                            [-3, -11, 3],
                            [-5, 14, 6],
                            [-6, 1, -3],
                        ]
                    ]
                ]
            ),
        )

    def testConvolveDepth_IK0_FK1(self):
        # Depth convolve where the filter has k=1
        image_data = jnp.array(
            [
                [
                    [
                        [2, 1, 0],
                        [0, 0, -3],
                        [2, 0, 1],
                    ],
                    [
                        [-7, 4, -4],
                        [3, 1, 3],
                        [-4, -1, 1],
                    ],
                ]
            ]
        )

        # same filter at both depths
        filter_image = jnp.array(
            [
                [
                    [
                        [[0, 0], [0, 1], [0, 0]],
                        [[-1, 0], [0, 0], [1, 0]],
                        [[0, 0], [0, -1], [0, 0]],
                    ],
                    [
                        [[0, 0], [0, 1], [0, 0]],
                        [[-1, 0], [0, 0], [1, 0]],
                        [[0, 0], [0, -1], [0, 0]],
                    ],
                ]
            ]
        )

        convolved_image = geom.convolve(2, image_data, filter_image, is_torus=(True,) * 2)
        assert convolved_image.shape == (1, 1, 3, 3, 2)
        assert jnp.allclose(
            convolved_image,
            jnp.array(
                [
                    [
                        [
                            [[1, 2], [-2, 0], [1, 4]],
                            [[3, 0], [-3, 1], [0, -1]],
                            [[-1, -2], [-1, -1], [2, -3]],
                        ]
                    ]
                ]
            )
            + jnp.array(
                [
                    [
                        [
                            [[8, -7], [3, -2], [-11, -2]],
                            [[-2, -3], [0, 5], [2, -5]],
                            [[-2, 10], [5, -3], [-3, 7]],
                        ]
                    ]
                ]
            ),
        )

    def testConvolveDepthMultiChannelsBatch(self):
        """
        This test makes the assumption that convolve on in_c=1,out_c=1,batch=1 works and uses that to
        test all the other cases. We test this assumption directly with other tests.
        """
        D = 2
        N = 3
        in_c = 5
        out_c = 4
        batch = 7
        key = random.PRNGKey(0)
        vmap_convolve = jax.vmap(
            jax.vmap(
                jax.vmap(
                    lambda img, ff: geom.convolve(D, img, ff, True),
                    in_axes=(
                        0,
                        0,
                    ),  # vmap over the in_channels (then will have to sum over them after)
                ),
                in_axes=(None, 0),  # vmap over the number of out_channels
            ),
            in_axes=(0, None),  # vmap over the batch
        )

        key, subkey = random.split(key)

        for img_k in [0, 1, 2]:

            key, subkey = random.split(key)
            image_data = random.normal(key, shape=((batch, in_c) + (N,) * D + (D,) * img_k))
            for filter_k in [0, 1, 2]:

                key, subkey = random.split(key)
                filter_data = random.normal(
                    subkey, shape=((out_c, in_c) + (N,) * D + (D,) * filter_k)
                )

                convolve_res = geom.convolve(D, image_data, filter_data, True)
                vmap_convolve_res = vmap_convolve(
                    image_data[:, :, None, None],
                    filter_data[:, :, None, None],
                )
                vmap_convolve_res = jnp.sum(vmap_convolve_res, axis=2)[:, :, 0, 0]  # sum over in_c

                assert convolve_res.shape == vmap_convolve_res.shape
                assert jnp.allclose(convolve_res, vmap_convolve_res, rtol=geom.TINY, atol=geom.TINY)
