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
        assert layer1.is_torus == (False,)*D
        for _, _ in layer1.items():
            assert False # its empty, so this won't ever be called

        k = 0
        layer2 = geom.Layer({ k: random.normal(key, shape=((1,) + (N,)*D + (D,)*k)) }, D, False)
        assert layer2.D == D
        assert layer2.is_torus == (False,)*D
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
        assert layer3.is_torus == (True,)*D

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
        assert layer1 is not layer2

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
        assert layer1.is_torus == (True,)*D
        assert list(layer1.keys()) == [(1,0)]
        assert layer1[(1,0)].shape == (10, N, N, D)

        # now images has multiple different values of k
        random_data2 = random.normal(key, shape=((33,) + (N,)*D + (D,)*2))
        images.extend([geom.GeometricImage(data, 0, D) for data in random_data2])
        layer2 = geom.Layer.from_images(images)
        assert list(layer2.keys()) == [(1,0),(2,0)]
        assert layer2[(1,0)].shape == (10, N, N, D)
        assert layer2[(2,0)].shape == (33, N, N, D, D)

    def testEq(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 3

        key, subkey = random.split(key)
        layer1 = geom.Layer({ (0,0): random.normal(subkey, shape=((10,) + (N,)*D + (D,)*0))}, D, True)
        layer1.append(1, 0, random.normal(subkey, shape=((10,) + (N,)*D + (D,)*1)))

        layer2 = layer1.copy()
        assert layer1 == layer2 

        # keys do not match
        layer3 = geom.Layer({ (0,0): jnp.ones((10,) + (N,)*D + (D,)*0) }, D, True)
        assert layer1 != layer3

        # values do not match
        layer4 = geom.Layer({ (0,): jnp.ones((10,) + (N,)*D + (D,)*0), 1: jnp.ones((10,) + (N,)*D + (D,)*1) }, D, True)
        assert layer1 != layer4 

        # is_torus does not match
        layer5 = geom.Layer(layer1.data, D, False)
        assert layer1 != layer5

    def testAppend(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.Layer(
            {
                (0,0): random.normal(key, shape=((10,) + (N,)*D + (D,)*0)),
                (1,0): random.normal(key, shape=((3,) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )

        image_block = random.normal(key, shape=((4,) + (N,)*D + (D,)*1))
        layer1.append(1, 0, image_block)
        assert layer1[(0,0)].shape == (10, N, N) #unchanged
        assert layer1[(1,0)].shape == (7, N, N, D) #updated 3+4=7

        image_block2 = random.normal(key, shape=((2,) + (N,)*D + (D,)*2))
        layer1.append(2, 0, image_block2)
        assert layer1[(0,0)].shape == (10, N, N) #unchanged
        assert layer1[(1,0)].shape == (7, N, N, D) #unchanged
        assert layer1[(2,0)].shape == (2, N, N, D, D)

        # add an image block to the wrong k bucket
        with pytest.raises(AssertionError):
            layer1.append(3, 0, image_block2)

        # N is set by append if it is empty
        layer2 = layer1.empty()
        assert layer2.get_spatial_dims() is None

        layer2.append(0, 0, random.normal(key, shape=((10,) + (N,)*D + (D,)*0)))
        assert layer2.get_spatial_dims() == (N,)*D

    def testConcat(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.Layer(
            {
                (0,0): random.normal(key, shape=((10,) + (N,)*D + (D,)*0)),
                (1,0): random.normal(key, shape=((3,) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )
        layer2 = geom.Layer(
            {
                (1,0): random.normal(key, shape=((4,) + (N,)*D + (D,)*1)), 
                (2,0): random.normal(key, shape=((5,) + (N,)*D + (D,)*2)),
            },
            D,
            True,
        )

        layer3 = layer1.concat(layer2)
        assert list(layer3.keys()) == [(0,0),(1,0),(2,0)]
        assert layer3[(0,0)].shape == (10, N, N)
        assert layer3[(1,0)].shape == (7, N, N, D)
        assert layer3[(2,0)].shape == (5, N, N, D, D)

        # mismatched D
        layer4 = geom.Layer({ (0,0): random.normal(key, shape=((10,) + (N,)*D + (D,)*0)) }, D, True)
        layer5 = geom.Layer({ (0,0): random.normal(key, shape=((10,) + (N,)*3 + (D,)*0)) }, 3, True)
        with pytest.raises(AssertionError):
            layer4.concat(layer5)

        # mismatched is_torus
        layer6 = geom.Layer({ (0,0): random.normal(key, shape=((10,) + (N,)*D + (D,)*0)) }, D, True)
        layer7 = geom.Layer({ (0,0): random.normal(key, shape=((10,) + (N,)*D + (D,)*0)) }, D, False)
        with pytest.raises(AssertionError):
            layer6.concat(layer7)

    def testAdd(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        channels = 4

        key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7, subkey8 = random.split(key, 9)

        layer1 = geom.Layer(
            {
                (0,0): random.normal(subkey1, shape=((channels,) + (N,)*D + (D,)*0)),
                (1,0): random.normal(subkey2, shape=((channels,) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )

        layer2 = geom.Layer(
            {
                (0,0): random.normal(subkey3, shape=((channels,) + (N,)*D + (D,)*0)), 
                (1,0): random.normal(subkey4, shape=((channels,) + (N,)*D + (D,)*1)),
            },
            D,
            True,
        )

        layer3 = layer1 + layer2
        layer4 = geom.Layer(
            { (0,0): layer1[(0,0)] + layer2[(0,0)], (1,0): layer1[(1,0)] + layer2[(1,0)]},
            D,
            True,
        )

        assert layer3 == layer4

        # mismatched layer types
        layer5 = geom.Layer({ (0,0): random.normal(subkey5, shape=((channels,) + (N,)*D + (D,)*0)) }, D, True)
        layer6 = geom.Layer({ (1,0): random.normal(subkey6, shape=((channels,) + (N,)*D + (D,)*1)) }, D, True)
        with pytest.raises(AssertionError):
            layer5 + layer6

        # mismatched number of channels
        layer7 = geom.Layer({ (0,0): random.normal(subkey7, shape=((channels+1,) + (N,)*D + (D,)*0)) }, D, True)
        layer8 = geom.Layer({ (0,0): random.normal(subkey8, shape=((channels,) + (N,)*D + (D,)*0)) }, D, True)
        with pytest.raises(TypeError):
            layer7 + layer8

    def testMul(self):
        key = random.PRNGKey(0)
        channels = 3
        N = 5
        D = 2

        key, subkey1, subkey2 = random.split(key, 3)

        layer1 = geom.Layer(
            {
                (0,0): random.normal(subkey1, shape=(channels,) + (N,)*D),
                (1,0): random.normal(subkey2, shape=(channels,) + (N,)*D + (D,)),
            },
            D,
            True,
        )

        layer2 = layer1 * 3
        assert jnp.allclose(layer2[(0,0)], layer1[(0,0)]*3)
        assert jnp.allclose(layer2[(1,0)], layer1[(1,0)]*3)
        assert layer2.D == D
        assert layer2.is_torus == (True,)*D

        layer3 = layer1 * -1
        assert jnp.allclose(layer3[(0,0)], layer1[(0,0)]*-1)
        assert jnp.allclose(layer3[(1,0)], layer1[(1,0)]*-1)
        assert layer2.D == D
        assert layer2.is_torus == (True,)*D

        # try to multiply two layers together
        with pytest.raises(AssertionError):
            layer1 * layer1

    def testDiv(self):
        key = random.PRNGKey(0)
        channels = 3
        N = 5
        D = 2

        key, subkey1, subkey2 = random.split(key, 3)

        layer1 = geom.Layer(
            {
                (0,0): random.normal(subkey1, shape=(channels,) + (N,)*D),
                (1,0): random.normal(subkey2, shape=(channels,) + (N,)*D + (D,)),
            },
            D,
            True,
        )

        layer2 = layer1 / 3
        assert jnp.allclose(layer2[(0,0)], layer1[(0,0)]/3)
        assert jnp.allclose(layer2[(1,0)], layer1[(1,0)]/3)
        assert layer2.D == D
        assert layer2.is_torus == (True,)*D

        layer3 = layer1 / -1
        assert jnp.allclose(layer3[(0,0)], layer1[(0,0)]/-1)
        assert jnp.allclose(layer3[(1,0)], layer1[(1,0)]/-1)
        assert layer2.D == D
        assert layer2.is_torus == (True,)*D

        # try to multiply two layers together
        with pytest.raises(AssertionError):
            layer1 * layer1

    def testSize(self):
        D = 2
        N = 5

        # empty layer
        layer1 = geom.Layer({}, D)
        assert layer1.size() == 0

        # basic scalar layer
        layer2 = geom.Layer({ (0,0): jnp.ones((1,) + (N,)*D) }, D)
        assert layer2.size() == N**D

        # layer with channels
        layer3 = geom.Layer({ (0,0): jnp.ones((4,) + (N,)*D) }, D)
        assert layer3.size() == (4 * N**D)

        # more complex layer
        layer4 = geom.Layer(
            {
                (0,0): jnp.ones((1,) + (N,)*D),
                (1,0): jnp.ones((4,) + (N,)*D + (D,)),
                (1,1): jnp.ones((2,) + (N,)*D + (D,)),
                (2,0): jnp.ones((3,) + (N,)*D + (D,D)),
            },
            D,
        )
        assert layer4.size() == (N**D + 4*N**D*D + 2*N**D*D + 3*N**D*D*D)

    def testVector(self):
        # Test the from_vector and to_vector functions
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer_example = geom.Layer(
            {
                (0,0): jnp.ones((1,) + (N,)*D),
                (1,0): jnp.ones((1,) + (N,)*D + (D,)),
                (2,0): jnp.ones((1,) + (N,)*D + (D,D)),
            },
            D,
        )

        key, subkey = random.split(key)
        rand_data = random.normal(subkey, shape=(layer_example.size(),))

        rand_layer = geom.Layer.from_vector(rand_data, layer_example)

        assert rand_layer.size() == layer_example.size()
        assert jnp.allclose(rand_layer.to_vector(), rand_data)

    def testToFromScalarLayer(self):
        D = 2
        N = 5

        layer_example = geom.Layer(
            {
                (0,0): jnp.ones((1,) + (N,)*D),
                (1,0): jnp.ones((1,) + (N,)*D + (D,)),
                (2,0): jnp.ones((1,) + (N,)*D + (D,D)),
            },
            D,
        )

        scalar_layer = layer_example.to_scalar_layer()

        assert len(scalar_layer.keys()) == 1
        assert next(iter(scalar_layer.keys())) == (0,0)
        assert jnp.allclose(scalar_layer[(0,0)], jnp.ones((1+D+D*D,) + (N,)*D))

        key = random.PRNGKey(0)
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
        rand_layer = geom.Layer(
            {
                (0,0): random.normal(subkey1, shape=((3,) + (N,)*D)),
                (1,0): random.normal(subkey2, shape=((1,) + (N,)*D + (D,))),
                (1,1): random.normal(subkey3, shape=((2,) + (N,)*D + (D,))),
                (2,0): random.normal(subkey4, shape=((1,) + (N,)*D + (D,D))),
            },
            D,
        )

        layout = { (0,0): 3, (1,0): 1, (1,1): 2, (2,0): 1 }

        scalar_layer2 = rand_layer.to_scalar_layer()
        assert list(scalar_layer2.keys()) == [(0,0)]
        assert rand_layer == rand_layer.to_scalar_layer().from_scalar_layer(layout)

    def testTimesGroupElement(self):
        N = 5
        channels = 3

        vmap_times_gg = vmap(geom.times_group_element, in_axes=(None,0,None,None))
        key = random.PRNGKey(0)
        for D in [2,3]:
            layer = geom.Layer({}, D)

            for parity in [0,1]:
                for k in [0,1,2,3]:
                    key, subkey = random.split(key)
                    layer.append(k, parity, random.normal(subkey, shape=((channels,) + (N,)*D + (D,)*k)))

            operators = geom.make_all_operators(D)

            for gg in operators:
                rotated_layer = layer.times_group_element(gg)

                for (k,parity), img_block in layer.items():
                    rotated_block = vmap_times_gg(D, img_block, parity, gg)
                    assert jnp.allclose(rotated_layer[(k,parity)], rotated_block)

    def testNorm(self):
        N = 5
        D = 2
        channels = 3

        key = random.PRNGKey(0)

        # norm of scalars, pseudo scalars, and vectors
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        layer = geom.Layer(
            { 
                (0,0): random.normal(subkey1, shape=(channels,) + (N,)*D),
                (0,1): random.normal(subkey2, shape=(channels,) + (N,)*D),
                (1,0): random.normal(subkey3, shape=(channels,) + (N,)*D + (D,)), 
            },
            D,
        )

        normed_layer = layer.norm()
        assert list(normed_layer.keys()) == [(0,0)] # odd parity is converted to even parity
        assert normed_layer[(0,0)].shape == ((3*channels,) + (N,)*D)
        assert jnp.allclose(normed_layer[(0,0)][:channels], jnp.abs(layer[(0,0)]))
        assert jnp.allclose(normed_layer[(0,0)][channels:2*channels], jnp.abs(layer[(0,1)]))
        vector_norm = jnp.linalg.norm(layer[(1,0)].reshape(layer[(1,0)].shape[:1+D] + (-1,)), axis=1+D)
        assert jnp.allclose(normed_layer[(0,0)][2*channels:], vector_norm)

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
        assert layer1.is_torus == (False,)*D
        for _, _ in layer1.items():
            assert False # its empty, so this won't ever be called

        k = 0
        layer2 = geom.BatchLayer({ (k,0): random.normal(key, shape=((10,1) + (N,)*D + (D,)*k)) }, D, False)
        assert layer2.D == D
        assert layer2.is_torus == (False,)*D
        assert layer2[(0,0)].shape == (10,1,N,N)

        # layers can have multiple k values with different channels, 
        # but they should have same batch size, although this is currently unenforced
        layer3 = geom.Layer(
            {
                (0,0): random.normal(key, shape=((5,10) + (N,)*D + (D,)*0)),
                (1,0): random.normal(key, shape=((5,3) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )
        assert list(layer3.keys()) == [(0,0),(1,0)]
        assert layer3[(0,0)].shape == (5,10, N, N)
        assert layer3[(1,0)].shape == (5,3, N, N, D)
        assert layer3.is_torus == (True,)*D

    def testGetSubset(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        k = 1

        layer1 = geom.BatchLayer({ (k,0): random.normal(key, shape=((100,1) + (N,)*D + (D,)*k)) }, D, False)

        layer2 = layer1.get_subset(jnp.array([3]))
        assert layer2.D == layer1.D
        assert layer2.is_torus == layer1.is_torus
        assert layer2.L == 1
        assert layer2[(k,0)].shape == (1, 1, N, N, D)
        assert jnp.allclose(layer2[(k,0)][0], layer1[(k,0)][3])

        layer3 = layer1.get_subset(jnp.array([3,23,4,17]))
        assert layer3.L == 4
        assert layer3[(k,0)].shape == (4, 1, N, N, D)
        assert jnp.allclose(layer3[(k,0)], layer1[(k,0)][jnp.array([3,23,4,17])])

        # Indices must be a jax array
        with pytest.raises(AssertionError):
            layer1.get_subset([3])

        with pytest.raises(AssertionError):
            layer1.get_subset((0,2,3))

        with pytest.raises(AssertionError):
            layer1.get_subset(jnp.array(0))

    def testAppend(self):
        # For BatchLayer, append should probably only be used while it is vmapped to a Layer
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        layer1 = geom.Layer(
            {
                (0,0): random.normal(key, shape=((5,10) + (N,)*D + (D,)*0)),
                (1,0): random.normal(key, shape=((5,3) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )

        def mult(layer, param):
            out_layer = layer.empty()
            for (k,parity), image_block in layer.items():
                out_layer.append(k, parity, param * jnp.ones(image_block.shape))

            return out_layer

        layer2 = vmap(mult)(layer1, jnp.arange(5))
        assert layer2.D == layer1.D
        assert layer2.is_torus == layer1.is_torus
        assert layer2.keys() == layer1.keys()
        for layer2_image, layer1_image, num in zip(layer2[(0,0)], layer1[(0,0)], jnp.arange(5)):
            assert jnp.allclose(layer2_image, num*jnp.ones(layer1_image.shape))

        for layer2_image, layer1_image, num in zip(layer2[(1,0)], layer1[(1,0)], jnp.arange(5)):
            assert jnp.allclose(layer2_image, num*jnp.ones(layer1_image.shape))

    def testConcat(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5

        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

        layer1 = geom.BatchLayer(
            {
                (1,0): random.normal(subkey1, shape=((5,10) + (N,)*D + (D,)*1)),
                (2,0): random.normal(subkey2, shape=((5,3) + (N,)*D + (D,)*2)), 
            },
            D,
            True,
        )
        layer2 = geom.BatchLayer(
            {
                (1,0): random.normal(subkey3, shape=((7,10) + (N,)*D + (D,)*1)), 
                (2,0): random.normal(subkey4, shape=((7,3) + (N,)*D + (D,)*2)),
            },
            D,
            True,
        )

        layer3 = layer1.concat(layer2)
        assert layer3.D == D
        assert layer3.is_torus == (True,)*D
        assert layer3[(1,0)].shape == (12,10,N,N,D)
        assert layer3[(2,0)].shape == (12,3,N,N,D,D)
        assert jnp.allclose(layer3[(1,0)], jnp.concatenate([layer1[(1,0)], layer2[(1,0)]]))

        key, subkey5, subkey6 = random.split(key, 3)
        layer4 = geom.BatchLayer({ (1,0): random.normal(subkey5, shape=((5,10) + (N,)*D + (D,)*1)) }, D, True)
        layer5 = geom.BatchLayer({ (1,0): random.normal(subkey6, shape=((5,2) + (N,)*D + (D,)*1)) }, D, True)
        
        layer6 = layer4.concat(layer5, axis=1)
        assert layer6.D == D
        assert list(layer6.keys()) == [(1,0)]
        assert layer6[(1,0)].shape == (5,12,N,N,D)
        assert jnp.allclose(layer6[(1,0)], jnp.concatenate([layer4[(1,0)], layer5[(1,0)]], axis=1))

    def testSize(self):
        D = 2
        N = 5

        # empty layer
        layer1 = geom.BatchLayer({}, D)
        assert layer1.size() == 0

        # basic scalar layer
        layer2 = geom.BatchLayer({ (0,0): jnp.ones((1,1) + (N,)*D) }, D)
        assert layer2.size() == N**D

        # layer with channels
        layer3 = geom.BatchLayer({ (0,0): jnp.ones((2,4) + (N,)*D) }, D)
        assert layer3.size() == (2*4 * N**D)

        # more complex layer
        layer4 = geom.BatchLayer(
            {
                (0,0): jnp.ones((3,1) + (N,)*D),
                (1,0): jnp.ones((3,4) + (N,)*D + (D,)),
                (1,1): jnp.ones((3,2) + (N,)*D + (D,)),
                (2,0): jnp.ones((3,3) + (N,)*D + (D,D)),
            },
            D,
        )
        assert layer4.size() == (3*(N**D + 4*N**D*D + 2*N**D*D + 3*N**D*D*D))

    def testToFromScalarLayer(self):
        D = 2
        N = 5

        L = 1
        layer_example = geom.BatchLayer(
            {
                (0,0): jnp.ones((L,1) + (N,)*D),
                (1,0): jnp.ones((L,1) + (N,)*D + (D,)),
                (2,0): jnp.ones((L,1) + (N,)*D + (D,D)),
            },
            D,
        )

        scalar_layer = layer_example.to_scalar_layer()

        assert len(scalar_layer.keys()) == 1
        assert next(iter(scalar_layer.keys())) == (0,0)
        assert jnp.allclose(scalar_layer[(0,0)], jnp.ones((1,1+D+D*D) + (N,)*D))

        key = random.PRNGKey(0)
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
        L = 5
        rand_layer = geom.BatchLayer(
            {
                (0,0): random.normal(subkey1, shape=((L,3) + (N,)*D)),
                (1,0): random.normal(subkey2, shape=((L,1) + (N,)*D + (D,))),
                (1,1): random.normal(subkey3, shape=((L,2) + (N,)*D + (D,))),
                (2,0): random.normal(subkey4, shape=((L,1) + (N,)*D + (D,D))),
            },
            D,
        )

        layout = { (0,0): 3, (1,0): 1, (1,1): 2, (2,0): 1 }

        scalar_layer2 = rand_layer.to_scalar_layer()
        assert list(scalar_layer2.keys()) == [(0,0)]
        assert rand_layer == rand_layer.to_scalar_layer().from_scalar_layer(layout)

    def testTimesGroupElement(self):
        N = 5
        batch = 4
        channels = 3

        vmap_times_gg = vmap(vmap(geom.times_group_element, in_axes=(None,0,None,None)), in_axes=(None,0,None,None))
        key = random.PRNGKey(0)
        for D in [2,3]:
            layer = geom.BatchLayer({}, D)

            for parity in [0,1]:
                for k in [0,1,2,3]:
                    key, subkey = random.split(key)
                    layer.append(k, parity, random.normal(subkey, shape=((batch,channels) + (N,)*D + (D,)*k)))

            operators = geom.make_all_operators(D)

            for gg in operators:
                rotated_layer = layer.times_group_element(gg)

                for (k,parity), img_block in layer.items():
                    rotated_block = vmap_times_gg(D, img_block, parity, gg)
                    assert jnp.allclose(rotated_layer[(k,parity)], rotated_block)

    def testNorm(self):
        N = 5
        D = 2
        batch = 4
        channels = 3

        key = random.PRNGKey(0)

        # norm of scalars, pseudo scalars, and vectors
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        layer = geom.BatchLayer(
            { 
                (0,0): random.normal(subkey1, shape=(batch,channels) + (N,)*D),
                (0,1): random.normal(subkey2, shape=(batch,channels) + (N,)*D),
                (1,0): random.normal(subkey3, shape=(batch,channels) + (N,)*D + (D,)), 
            },
            D,
        )

        normed_layer = layer.norm()
        assert list(normed_layer.keys()) == [(0,0)] # odd parity is converted to even parity
        assert normed_layer[(0,0)].shape == ((batch,3*channels) + (N,)*D)
        assert jnp.allclose(normed_layer[(0,0)][:,:channels], jnp.abs(layer[(0,0)]))
        assert jnp.allclose(normed_layer[(0,0)][:,channels:2*channels], jnp.abs(layer[(0,1)]))
        vector_norm = jnp.linalg.norm(layer[(1,0)].reshape(layer[(1,0)].shape[:2+D] + (-1,)), axis=2+D)
        assert jnp.allclose(normed_layer[(0,0)][:,2*channels:], vector_norm)

    def testAdd(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        channels = 4
        batch = 3

        key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7, subkey8 = random.split(key, 9)

        layer1 = geom.BatchLayer(
            {
                (0,0): random.normal(subkey1, shape=((batch,channels) + (N,)*D + (D,)*0)),
                (1,0): random.normal(subkey2, shape=((batch,channels) + (N,)*D + (D,)*1)), 
            },
            D,
            True,
        )

        layer2 = geom.BatchLayer(
            {
                (0,0): random.normal(subkey3, shape=((batch,channels) + (N,)*D + (D,)*0)), 
                (1,0): random.normal(subkey4, shape=((batch,channels) + (N,)*D + (D,)*1)),
            },
            D,
            True,
        )

        layer3 = layer1 + layer2
        layer4 = geom.BatchLayer(
            { (0,0): layer1[(0,0)] + layer2[(0,0)], (1,0): layer1[(1,0)] + layer2[(1,0)]},
            D,
            True,
        )

        assert layer3 == layer4

        # mismatched layer types
        layer5 = geom.BatchLayer(
            { (0,0): random.normal(subkey5, shape=((batch,channels) + (N,)*D + (D,)*0)) }, 
            D, 
            True
        )
        layer6 = geom.BatchLayer(
            { (1,0): random.normal(subkey6, shape=((batch,channels) + (N,)*D + (D,)*1)) },
            D, 
            True,
        )
        with pytest.raises(AssertionError):
            layer5 + layer6

        # mismatched number of channels
        layer7 = geom.Layer(
            { (0,0): random.normal(subkey7, shape=((batch,channels+1) + (N,)*D + (D,)*0)) }, 
            D, 
            True,
        )
        layer8 = geom.Layer(
            { (0,0): random.normal(subkey8, shape=((batch,channels) + (N,)*D + (D,)*0)) },
            D, 
            True,
        )
        with pytest.raises(TypeError):
            layer7 + layer8

        # mismatched batch size
        key, subkey9, subkey10 = random.split(key, 3)
        layer9 = geom.Layer(
            { (0,0): random.normal(subkey9, shape=((batch+1,channels) + (N,)*D + (D,)*0)) }, 
            D, 
            True,
        )
        layer10 = geom.Layer(
            { (0,0): random.normal(subkey10, shape=((batch,channels) + (N,)*D + (D,)*0)) },
            D, 
            True,
        )
        with pytest.raises(TypeError):
            layer9 + layer10

    def testMul(self):
        key = random.PRNGKey(0)
        batch = 4
        channels = 3
        N = 5
        D = 2

        key, subkey1, subkey2 = random.split(key, 3)

        layer1 = geom.BatchLayer(
            {
                (0,0): random.normal(subkey1, shape=(batch,channels) + (N,)*D),
                (1,0): random.normal(subkey2, shape=(batch,channels) + (N,)*D + (D,)),
            },
            D,
            True,
        )

        layer2 = layer1 * 3
        assert jnp.allclose(layer2[(0,0)], layer1[(0,0)]*3)
        assert jnp.allclose(layer2[(1,0)], layer1[(1,0)]*3)
        assert layer2.D == D
        assert layer2.is_torus == (True,)*D

        layer3 = layer1 * -1
        assert jnp.allclose(layer3[(0,0)], layer1[(0,0)]*-1)
        assert jnp.allclose(layer3[(1,0)], layer1[(1,0)]*-1)
        assert layer2.D == D
        assert layer2.is_torus == (True,)*D

        # try to multiply two layers together
        with pytest.raises(AssertionError):
            layer1 * layer1