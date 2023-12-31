import time

import geometricconvolutions.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random

TINY = 1.e-5

class TestFunctionalGeometricImage:
    """
    Class to test the functional versions of the geometric image functions.
    """

    def testParseShape(self):
        spatial_dims, k = geom.parse_shape((4,4,2), 2)
        assert spatial_dims == (4,4)
        assert k == 1

        # non-square, D=2
        spatial_dims, k = geom.parse_shape((4,5,2), 2)
        assert spatial_dims == (4,5)
        assert k == 1

        # k = 0
        spatial_dims, k = geom.parse_shape((5,5), 2)
        assert spatial_dims == (5,5)
        assert k == 0

        # k = 2
        spatial_dims, k = geom.parse_shape((5,5,2,2), 2)
        assert spatial_dims == (5,5)
        assert k == 2

        # D = 3
        spatial_dims, k = geom.parse_shape((5,5,5,3,3), 3)
        assert spatial_dims == (5,5,5)
        assert k == 2

        # non-square, D=3
        spatial_dims, k = geom.parse_shape((4,5,6,3), 3)
        assert spatial_dims == (4,5,6)
        assert k == 1

    def testHash(self):
        D = 2
        indices = jnp.arange(10*D).reshape((10,D))

        img1 = jnp.ones((4,4))
        hashed_indices = geom.hash(D, img1, indices)
        assert jnp.allclose(hashed_indices[0], jnp.array([0,2,0,2,0,2,0,2,0,2]))
        assert jnp.allclose(hashed_indices[1], jnp.array([1,3,1,3,1,3,1,3,1,3]))

        img2 = jnp.ones((3,4))
        hashed_indices = geom.hash(D, img2, indices)
        assert jnp.allclose(hashed_indices[0], jnp.array([0,2,1,0,2,1,0,2,1,0]))
        assert jnp.allclose(hashed_indices[1], jnp.array([1,3,1,3,1,3,1,3,1,3]))

    def testConvolveNonSquare(self):
        D = 2

        # Test that non-square convolution works
        img1 = jnp.arange(20).reshape((4,5))
        filter_img = jnp.ones((3,3))

        # SAME padding
        res = geom.convolve(D, img1, filter_img, False, padding='SAME')
        assert jnp.allclose(
            res,
            jnp.array([
                [12, 21, 27, 33, 24],
                [33, 54, 63, 72, 51],
                [63, 99, 108, 117, 81],
                [52, 81, 87, 93, 64],
            ]),
        )

        # VALID padding
        res = geom.convolve(D, img1, filter_img, False, padding='VALID')
        assert jnp.allclose(
            res,
            jnp.array([
                [54, 63, 72],
                [99, 108, 117],
            ])
        )

        # TORUS padding
        res = geom.convolve(D, img1, filter_img, True, padding='TORUS')
        assert jnp.allclose(
            res,
            jnp.array([
                [75, 69, 78, 87, 81],
                [60, 54, 63, 72, 66],
                [105, 99, 108, 117, 111],
                [90, 84, 93, 102, 96],
            ]),
        )

        # TORUS padding with dilation
        res = geom.convolve(D, img1, filter_img, True, padding='TORUS', rhs_dilation=(2,)*D)
        assert jnp.allclose(
            res,
            jnp.array([
                [75,84,78,72,81],
                [120,129,123,117,126],
                [45,54,48,42,51],
                [90,99,93,87,96],
            ]),
        )

    def testConvolveContract2D(self):
        """
        Test that convolve_contract is the same as convolving, then contracting in 2D
        """
        N = 3
        D = 2
        is_torus = True
        key = random.PRNGKey(time.time_ns())
        
        for img_k in range(4):
            for filter_k in range(4):
                key, subkey = random.split(key)
                image = random.normal(subkey, shape=((N,)*D + (D,)*img_k))

                key, subkey = random.split(key)
                conv_filter = random.normal(subkey, shape=((N,)*D + (D,)*(img_k+filter_k)))

                contraction_idxs = tuple((i,i+img_k) for i in range(img_k))
                assert jnp.allclose(
                    geom.convolve_contract(D, image, conv_filter, is_torus),
                    geom.multicontract(
                        geom.convolve(D, image, conv_filter, is_torus), 
                        contraction_idxs, 
                        idx_shift=D,
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
        is_torus = True
        key = random.PRNGKey(time.time_ns())
        
        for img_k in range(3):
            for filter_k in range(2):
                key, subkey = random.split(key)
                image = random.normal(subkey, shape=((N,)*D + (D,)*img_k))

                key, subkey = random.split(key)
                conv_filter = random.normal(subkey, shape=((N,)*D + (D,)*(img_k+filter_k)))

                contraction_idxs = tuple((i,i+img_k) for i in range(img_k))
                assert jnp.allclose(
                    geom.convolve_contract(D, image, conv_filter, is_torus),
                    geom.multicontract(
                        geom.convolve(D, image, conv_filter, is_torus), 
                        contraction_idxs, 
                        idx_shift=D,
                    ),
                    rtol=TINY,
                    atol=TINY,
                )