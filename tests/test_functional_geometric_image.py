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