import sys
sys.path.insert(0,'src/geometricconvolutions/')

import math

from geometric import geometric_image, ktensor
import pytest
import jax.numpy as jnp
from jax import random

TINY = 1.e-5

class TestGeometricImage:

    def testConstructor(self):
        #note we are not actually relying on randomness in this function, just filling values
        # key = random.PRNGKey(0)

        image1 = ktensor(jnp.array([1,1]), 0, 2)

        assert image1.data.shape == (2,)
        assert image1.D == 2
        assert image1.k == 1

        # image2 = geometric_image(random.uniform(key, shape=(10,10,2)), 0, 2)
        # assert image2.data.shape == (10,10,2)
        # assert image2.D == 2
        # assert image2.k == 1

        # image3 = geometric_image(random.uniform(key, shape=(10,10,2,2,2)), 3, 2)
        # assert image3.data.shape == (10,10,2,2,2)
        # assert image3.k == 3
        # assert image3.parity == 1

        # #D does not match dimensions
        # with pytest.raises(AssertionError):
        #     geometric_image(random.uniform(key, shape=(10,10)), 0, 3)

        # #non square
        # with pytest.raises(AssertionError):
        #     geometric_image(random.uniform(key, shape=(10,20)), 0, 3)

        # #side length of pixel tensors does not match D
        # with pytest.raises(AssertionError):
        #     geometric_image(random.uniform(key, shape=(10,10,3,3)), 0, 2)

    # def testAdd(self):
    #     image1 = geometric_image(jnp.ones((10,10,2), dtype=int), 0, 2)
    #     image2 = geometric_image(5*jnp.ones((10,10,2), dtype=int), 0, 2)
    #     float_image = geometric_image(3.4*jnp.ones((10,10,2)), 0, 2)

    #     result = image1 + image2
    #     assert (result.data == 6).all()
    #     assert result.parity == 0
    #     assert result.D == 2
    #     assert result.k == 1
    #     assert result.N == 10

    #     assert (image1.data == 1).all()
    #     assert (image2.data == 5).all()

    #     result = image1 + float_image
    #     assert (result.data == 4.4).all()

    #     image3 = geometric_image(jnp.ones((10,10,10,3), dtype=int), 0, 3)
    #     with pytest.raises(AssertionError): #D not equal
    #         result = image1 + image3

    #     image4 = geometric_image(jnp.ones((10,10,2), dtype=int), 1, 2)
    #     with pytest.raises(AssertionError): #parity not equal
    #         result = image1 + image4

    #     with pytest.raises(AssertionError):
    #         result = image3 + image4 #D and parity not equal

    #     image5 = geometric_image(jnp.ones((20,20,2), dtype=int), 0, 2)
    #     with pytest.raises(AssertionError): #N not equal
    #         result = image1 + image5

    # def testTimeScalar(self):
    #     image1 = geometric_image(jnp.ones((10,10,2), dtype=int), 0, 2)
    #     assert (image1.data == 1).all()

    #     result = image1.times_scalar(5)
    #     assert (result.data == 5).all()
    #     assert result.parity == image1.parity
    #     assert result.D == image1.D
    #     assert result.k == image1.k
    #     assert result.N == image1.N
    #     assert (image1.data == 1).all() #original is unchanged

    #     result2 = image1.times_scalar(3.4)
    #     assert (result2.data == 3.4).all()
    #     assert (image1.data == 1).all()

    # def testGetItem(self):
    #     #note we are not actually relying on randomness in this function, just filling values
    #     key = random.PRNGKey(0)

    #     random_vals = random.uniform(key, shape=(10,10,2,2,2))
    #     image1 = geometric_image(random_vals, 0, 2)

    #     assert image1[0,5,0,1,1] == random_vals[0,5,0,1,1]
    #     assert image1[4,3,0,0,1] == random_vals[4,3,0,0,1]
    #     assert (image1[0] == random_vals[0]).all()
    #     assert (image1[4:,2:3] == random_vals[4:,2:3]).all()
    #     assert image1[4:, 2:3].shape == random_vals[4:, 2:3].shape

    # def testNormalize(self):
    #     key = random.PRNGKey(0)
    #     image1 = geometric_image(random.uniform(key, shape=(10,10)), 0, 2)

    #     normed_image1 = image1.normalize()
    #     assert math.isclose(jnp.max(jnp.abs(normed_image1.data)), 1.)
    #     assert image1.data.shape == normed_image1.data.shape == (10,10)

    #     image2 = geometric_image(random.uniform(key, shape=(10,10,2)), 0, 2)
    #     normed_image2 = image2.normalize()
    #     assert image2.data.shape == normed_image2.data.shape == (10,10,2)
    #     for row in normed_image2.data:
    #         for pixel in row:
    #             assert jnp.linalg.norm(pixel) < (1 + TINY)

    #     image3 = geometric_image(random.uniform(key, shape=(10,10,2,2)), 0, 2)
    #     normed_image3 = image3.normalize()
    #     assert image3.data.shape == normed_image3.data.shape == (10,10,2,2)
    #     for row in normed_image3.data:
    #         for pixel in row:
    #             assert jnp.linalg.norm(pixel) < (1 + TINY)








