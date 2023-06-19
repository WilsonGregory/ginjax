import time

import geometricconvolutions.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random, vmap

TINY = 1.e-5

class TestLayer:

    def testConstructor(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.Layer({}, D, False)
        assert layer1.D == D
        assert layer1.is_torus == False 
        for _, _ in layer1.items():
            assert False # its empty, so this won't ever be called

        k = 0
        layer2 = geom.Layer({ k: random.normal(key, shape=((1,) + (N,)*D + (D,)*k)) }, D, False)
        assert layer2.D == D
        assert layer2.is_torus == False
        assert layer2[0].shape == (1,N,N)

        #layers can have multiple k values, and can have different size channels at each k
        layer3 = geom.Layer(
            {
                0: random.normal(key, shape=((10,) + (N,)*D + (D,)*0)),
                1: random.normal(key, shape=((3,) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )
        assert list(layer3.keys()) == [0,1]
        assert layer3[0].shape == (10, N, N)
        assert layer3[1].shape == (3, N, N, D)
        assert layer3.is_torus == True

    def testCopy(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.Layer(
            {
                0: random.normal(key, shape=((10,) + (N,)*D + (D,)*0)),
                1: random.normal(key, shape=((3,) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )

        layer2 = layer1.copy()
        assert layer1 != layer2

        layer2[1] = jnp.arange(1*(N**D)*D).reshape((1,) + (N,)*D + (D,)*1)
        assert layer2[1].shape == (1, N, N, D)
        assert layer1[1].shape == (3, N, N, D) # original layer we copied from is unchanged

    def testFromImages(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        random_data = random.normal(key, shape=((10,) + (N,)*D + (D,)*1))
        images = [geom.GeometricImage(data, 0, D) for data in random_data]
        layer1 = geom.Layer.from_images(images)
        assert layer1.D == D 
        assert layer1.is_torus == True
        assert list(layer1.keys()) == [1]
        assert layer1[1].shape == (10, N, N, D)

        # now images has multiple different values of k
        random_data2 = random.normal(key, shape=((33,) + (N,)*D + (D,)*2))
        images.extend([geom.GeometricImage(data, 0, D) for data in random_data2])
        layer2 = geom.Layer.from_images(images)
        assert list(layer2.keys()) == [1,2]
        assert layer2[1].shape == (10, N, N, D)
        assert layer2[2].shape == (33, N, N, D, D)

    def testAppend(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.Layer(
            {
                0: random.normal(key, shape=((10,) + (N,)*D + (D,)*0)),
                1: random.normal(key, shape=((3,) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )

        image_block = random.normal(key, shape=((4,) + (N,)*D + (D,)*1))
        layer1.append(1, image_block)
        assert layer1[0].shape == (10, N, N) #unchanged
        assert layer1[1].shape == (7, N, N, D) #updated 3+4=7

        image_block2 = random.normal(key, shape=((2,) + (N,)*D + (D,)*2))
        layer1.append(2, image_block2)
        assert layer1[0].shape == (10, N, N) #unchanged
        assert layer1[1].shape == (7, N, N, D) #unchanged
        assert layer1[2].shape == (2, N, N, D, D)

        # add an image block to the wrong k bucket
        with pytest.raises(AssertionError):
            layer1.append(3, image_block2)

        # N is set by append if it is empty
        layer2 = layer1.empty()
        assert layer2.N is None

        layer2.append(0, random.normal(key, shape=((10,) + (N,)*D + (D,)*0)))
        assert layer2.N == N

    def testAdd(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.Layer(
            {
                0: random.normal(key, shape=((10,) + (N,)*D + (D,)*0)),
                1: random.normal(key, shape=((3,) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )
        layer2 = geom.Layer(
            {
                1: random.normal(key, shape=((4,) + (N,)*D + (D,)*1)), 
                2: random.normal(key, shape=((5,) + (N,)*D + (D,)*2)),
            },
            D,
            True,
        )

        layer3 = layer1 + layer2
        assert list(layer3.keys()) == [0,1,2]
        assert layer3[0].shape == (10, N, N)
        assert layer3[1].shape == (7, N, N, D)
        assert layer3[2].shape == (5, N, N, D, D)

        # mismatched D
        layer4 = geom.Layer({ 0: random.normal(key, shape=((10,) + (N,)*D + (D,)*0)) }, D, True)
        layer5 = geom.Layer({ 0: random.normal(key, shape=((10,) + (N,)*D + (D,)*0)) }, 3, True)
        with pytest.raises(AssertionError):
            layer4 + layer5

        # mismatched is_torus
        layer6 = geom.Layer({ 0: random.normal(key, shape=((10,) + (N,)*D + (D,)*0)) }, D, True)
        layer7 = geom.Layer({ 0: random.normal(key, shape=((10,) + (N,)*D + (D,)*0)) }, D, False)
        with pytest.raises(AssertionError):
            layer6 + layer7

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test BatchLayer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TestBatchLayer:

    def testConstructor(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.BatchLayer({}, D, False)
        assert layer1.D == D
        assert layer1.is_torus == False 
        for _, _ in layer1.items():
            assert False # its empty, so this won't ever be called

        k = 0
        layer2 = geom.BatchLayer({ k: random.normal(key, shape=((10,1) + (N,)*D + (D,)*k)) }, D, False)
        assert layer2.D == D
        assert layer2.is_torus == False
        assert layer2[0].shape == (10,1,N,N)

        # layers can have multiple k values with different channels, 
        # but they should have same batch size, although this is currently unenforced
        layer3 = geom.Layer(
            {
                0: random.normal(key, shape=((5,10) + (N,)*D + (D,)*0)),
                1: random.normal(key, shape=((5,3) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )
        assert list(layer3.keys()) == [0,1]
        assert layer3[0].shape == (5,10, N, N)
        assert layer3[1].shape == (5,3, N, N, D)
        assert layer3.is_torus == True

    def testGetSubset(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        k = 1

        layer1 = geom.BatchLayer({ k: random.normal(key, shape=((100,1) + (N,)*D + (D,)*k)) }, D, False)

        layer2 = layer1.get_subset(jnp.array([3]))
        assert layer2.D == layer1.D
        assert layer2.is_torus == layer1.is_torus
        assert layer2.L == 1
        assert layer2[k].shape == (1, 1, N, N, D)
        assert jnp.allclose(layer2[k][0], layer1[k][3])

        layer3 = layer1.get_subset(jnp.array([3,23,4,17]))
        assert layer3.L == 4
        assert layer3[k].shape == (4, 1, N, N, D)
        assert jnp.allclose(layer3[k], layer1[k][jnp.array([3,23,4,17])])

        # Indices must be a jax array
        with pytest.raises(AssertionError):
            layer1.get_subset([3])

        with pytest.raises(AssertionError):
            layer1.get_subset((0,2,3))

    def testAppend(self):
        # For BatchLayer, append should probably only be used while it is vmapped to a Layer
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.Layer(
            {
                0: random.normal(key, shape=((5,10) + (N,)*D + (D,)*0)),
                1: random.normal(key, shape=((5,3) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )

        def mult(layer, param):
            out_layer = layer.empty()
            for k, image_block in layer.items():
                out_layer.append(k, param * jnp.ones(image_block.shape))

            return out_layer

        layer2 = vmap(mult)(layer1, jnp.arange(5))
        assert layer2.D == layer1.D
        assert layer2.is_torus == layer1.is_torus
        assert layer2.keys() == layer1.keys()
        for layer2_image, layer1_image, num in zip(layer2[0], layer1[0], jnp.arange(5)):
            assert jnp.allclose(layer2_image, num*jnp.ones(layer1_image.shape))

        for layer2_image, layer1_image, num in zip(layer2[1], layer1[1], jnp.arange(5)):
            assert jnp.allclose(layer2_image, num*jnp.ones(layer1_image.shape))

    def testAdd(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.BatchLayer(
            {
                1: random.normal(key, shape=((5,10) + (N,)*D + (D,)*1)),
                2: random.normal(key, shape=((5,3) + (N,)*D + (D,)*2)), 
            },
            D,
            True,
        )
        layer2 = geom.BatchLayer(
            {
                1: random.normal(key, shape=((7,10) + (N,)*D + (D,)*1)), 
                2: random.normal(key, shape=((7,3) + (N,)*D + (D,)*2)),
            },
            D,
            True,
        )

        layer3 = layer1 + layer2 
        assert layer3.D == D
        assert layer3.is_torus == True 
        assert layer3[1].shape == (12,10,N,N,D)
        assert layer3[2].shape == (12,3,N,N,D,D)

        # test the vmap case, should add the channels. The vmap dimension must match
        layer4 = geom.BatchLayer({ 1: random.normal(key, shape=((5,10) + (N,)*D + (D,)*1)) }, D, True)
        layer5 = geom.BatchLayer({ 1: random.normal(key, shape=((5,2) + (N,)*D + (D,)*1)) }, D, True)

        def adder(layer_a, layer_b):
            return layer_a + layer_b
        
        layer6 = vmap(adder)(layer4, layer5)
        assert layer6.D == D
        assert list(layer6.keys()) == [1]
        assert layer6[1].shape == (5,12,N,N,D)
