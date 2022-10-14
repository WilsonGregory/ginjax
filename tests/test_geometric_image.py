import sys
sys.path.insert(0,'src/geometricconvolutions/')

import math

from geometric import GeometricImage, GeometricFilter
import pytest
import jax.numpy as jnp
from jax import random

TINY = 1.e-5

class TestGeometricImage:

    def testZerosConstructor(self):
        image1 = GeometricImage.zeros(20,0,0,2)
        assert image1.data.shape == (20,20)
        assert image1.k == 0

        image2 = GeometricImage.zeros(20,1,0,2)
        assert image2.data.shape == (20,20,2)
        assert image2.k == 1

        image3 = GeometricImage.zeros(20,3,0,2)
        assert image3.data.shape == (20,20,2,2,2)
        assert image3.k == 3

    def testConstructor(self):
        #note we are not actually relying on randomness in this function, just filling values
        key = random.PRNGKey(0)

        image1 = GeometricImage(random.uniform(key, shape=(10,10)), 0, 2)
        assert image1.data.shape == (10,10)
        assert image1.D == 2
        assert image1.k == 0

        image2 = GeometricImage(random.uniform(key, shape=(10,10,2)), 0, 2)
        assert image2.data.shape == (10,10,2)
        assert image2.D == 2
        assert image2.k == 1

        image3 = GeometricImage(random.uniform(key, shape=(10,10,2,2,2)), 3, 2)
        assert image3.data.shape == (10,10,2,2,2)
        assert image3.k == 3
        assert image3.parity == 1

        #D does not match dimensions
        with pytest.raises(AssertionError):
            GeometricImage(random.uniform(key, shape=(10,10)), 0, 3)

        #non square
        with pytest.raises(AssertionError):
            GeometricImage(random.uniform(key, shape=(10,20)), 0, 3)

        #side length of pixel tensors does not match D
        with pytest.raises(AssertionError):
            GeometricImage(random.uniform(key, shape=(10,10,3,3)), 0, 2)

    def testCopy(self):
        image1 = GeometricImage.zeros(20,0,0,2)
        image2 = image1.copy()

        assert type(image1) == type(image2)
        assert (image1.data == image2.data).all()
        assert image1.parity == image2.parity
        assert image1.D == image2.D
        assert image1.k == image2.k
        assert image1.N == image2.N
        assert image1 != image2

    def testAdd(self):
        image1 = GeometricImage(jnp.ones((10,10,2), dtype=int), 0, 2)
        image2 = GeometricImage(5*jnp.ones((10,10,2), dtype=int), 0, 2)
        float_image = GeometricImage(3.4*jnp.ones((10,10,2)), 0, 2)

        result = image1 + image2
        assert (result.data == 6).all()
        assert result.parity == 0
        assert result.D == 2
        assert result.k == 1
        assert result.N == 10

        assert (image1.data == 1).all()
        assert (image2.data == 5).all()

        result = image1 + float_image
        assert (result.data == 4.4).all()

        image3 = GeometricImage(jnp.ones((10,10,10,3), dtype=int), 0, 3)
        with pytest.raises(AssertionError): #D not equal
            result = image1 + image3

        image4 = GeometricImage(jnp.ones((10,10,2), dtype=int), 1, 2)
        with pytest.raises(AssertionError): #parity not equal
            result = image1 + image4

        with pytest.raises(AssertionError):
            result = image3 + image4 #D and parity not equal

        image5 = GeometricImage(jnp.ones((20,20,2), dtype=int), 0, 2)
        with pytest.raises(AssertionError): #N not equal
            result = image1 + image5

    def testMul(self):
        image1 = GeometricImage(2*jnp.ones((3,3), dtype=int), 0, 2)
        image2 = GeometricImage(5*jnp.ones((3,3), dtype=int), 0, 2)

        mult1_2 = image1 * image2
        assert mult1_2.k == 0
        assert mult1_2.parity == 0
        assert mult1_2.D == image1.D == image2.D
        assert mult1_2.N == image1.N == image1.N
        assert (mult1_2.data == 10*jnp.ones((3,3))).all()
        assert (mult1_2.data == (image2 * image1).data).all()

        image3 = GeometricImage(jnp.arange(18).reshape(3,3,2), 0, 2)
        mult1_3 = image1 * image3
        assert mult1_3.k == image1.k + image3.k == 1
        assert mult1_3.parity == (image1.parity + image3.parity) % 2 == 0
        assert mult1_3.D == image1.D == image3.D
        assert mult1_3.N == image1.N == image3.N
        assert (mult1_3.data == jnp.array(
            [
                [[0,2],[4,6],[8,10]],
                [[12,14],[16,18],[20,22]],
                [[24,26],[28,30],[32,34]],
            ],
        dtype=int)).all()

        image4 = GeometricImage(jnp.arange(18).reshape((3,3,2)), 1, 2)
        mult3_4 = image3 * image4
        assert mult3_4.k == image3.k + image3.k == 2
        assert mult3_4.parity == (image3.parity + image4.parity) % 2 == 1
        assert mult3_4.D == image3.D == image4.D
        assert mult3_4.N == image3.N == image4.N
        assert (mult3_4.data == jnp.array(
            [
                [
                    (image3.ktensor((0,0))*image4.ktensor((0,0))).data, #relies on our tests for ktensor multiplication
                    (image3.ktensor((0,1))*image4.ktensor((0,1))).data,
                    (image3.ktensor((0,2))*image4.ktensor((0,2))).data,
                ],
                [
                    (image3.ktensor((1,0))*image4.ktensor((1,0))).data,
                    (image3.ktensor((1,1))*image4.ktensor((1,1))).data,
                    (image3.ktensor((1,2))*image4.ktensor((1,2))).data,
                ],
                [
                    (image3.ktensor((2,0))*image4.ktensor((2,0))).data,
                    (image3.ktensor((2,1))*image4.ktensor((2,1))).data,
                    (image3.ktensor((2,2))*image4.ktensor((2,2))).data,
                ],
            ],
        dtype=int)).all()

        image5 = GeometricImage(jnp.ones((10,10)), 0, 2)
        with pytest.raises(AssertionError): #mismatched N
            image5 * image1

        image6 = GeometricImage(jnp.ones((3,3,3)), 0, 3)
        with pytest.raises(AssertionError): #mismatched D
            image6 * image1

    def testTimeScalar(self):
        image1 = GeometricImage(jnp.ones((10,10,2), dtype=int), 0, 2)
        assert (image1.data == 1).all()

        result = image1.times_scalar(5)
        assert (result.data == 5).all()
        assert result.parity == image1.parity
        assert result.D == image1.D
        assert result.k == image1.k
        assert result.N == image1.N
        assert (image1.data == 1).all() #original is unchanged

        result2 = image1.times_scalar(3.4)
        assert (result2.data == 3.4).all()
        assert (image1.data == 1).all()

    def testGetItem(self):
        #note we are not actually relying on randomness in this function, just filling values
        key = random.PRNGKey(0)

        random_vals = random.uniform(key, shape=(10,10,2,2,2))
        image1 = GeometricImage(random_vals, 0, 2)

        assert image1[0,5,0,1,1] == random_vals[0,5,0,1,1]
        assert image1[4,3,0,0,1] == random_vals[4,3,0,0,1]
        assert (image1[0] == random_vals[0]).all()
        assert (image1[4:,2:3] == random_vals[4:,2:3]).all()
        assert image1[4:, 2:3].shape == random_vals[4:, 2:3].shape

    def testContract(self):
        img1 = GeometricImage(jnp.arange(36).reshape((3,3,2,2)), 0, 2)

        img1_contracted = img1.contract(0,1)
        assert img1_contracted.shape() == (3,3)
        assert (img1_contracted.data == jnp.array([[3,11,19], [27,35,43], [51,59,67]])).all()
        assert (img1.contract(1,0).data == img1_contracted.data).all()

        img2 = GeometricImage(jnp.arange(72).reshape((3,3,2,2,2)), 0, 2)

        img2_contracted_1 = img2.contract(0,1)
        assert img2_contracted_1.shape() == (3,3,2)
        assert (img2_contracted_1.data == jnp.array([
            [[6, 8], [22, 24], [38, 40]],
            [[54, 56], [70, 72], [86, 88]],
            [[102, 104], [118, 120], [134, 136]],
        ])).all()
        assert (img2.contract(1,0).data == img2_contracted_1.data).all()

        img2_contracted_2 = img2.contract(0,2)
        assert img2_contracted_2.shape() == (3,3,2)
        assert (img2_contracted_2.data == jnp.array([
            [[5, 9], [21, 25], [37, 41]],
            [[53, 57], [69, 73], [85, 89]], #nice
            [[101, 105], [117, 121], [133, 137]],
        ])).all()
        assert (img2.contract(2,0).data == img2_contracted_2.data).all()

        img2_contracted_3 = img2.contract(1,2)
        assert img2_contracted_3.shape() == (3,3,2)
        assert (img2_contracted_3.data == jnp.array([
            [[3, 11], [19, 27], [35, 43]],
            [[51, 59], [67, 75], [83, 91]], #nice
            [[99, 107], [115, 123], [131, 139]],
        ])).all()
        assert (img2.contract(2,1).data == img2_contracted_3.data).all()

        with pytest.raises(AssertionError):
            img1.contract(0,0) #same indices

        with pytest.raises(AssertionError):
            img1.contract(2,3) #out of bounds indices

        img3 = GeometricImage(jnp.ones((3,3)), 0, 2)
        with pytest.raises(AssertionError):
            img3.contract(0,1) #k < 2

    def testNormalize(self):
        key = random.PRNGKey(0)
        image1 = GeometricImage(random.uniform(key, shape=(10,10)), 0, 2)

        normed_image1 = image1.normalize()
        assert math.isclose(jnp.max(jnp.abs(normed_image1.data)), 1.)
        assert image1.data.shape == normed_image1.data.shape == (10,10)

        image2 = GeometricImage(random.uniform(key, shape=(10,10,2)), 0, 2)
        normed_image2 = image2.normalize()
        assert image2.data.shape == normed_image2.data.shape == (10,10,2)
        for row in normed_image2.data:
            for pixel in row:
                assert jnp.linalg.norm(pixel) < (1 + TINY)

        image3 = GeometricImage(random.uniform(key, shape=(10,10,2,2)), 0, 2)
        normed_image3 = image3.normalize()
        assert image3.data.shape == normed_image3.data.shape == (10,10,2,2)
        for row in normed_image3.data:
            for pixel in row:
                assert jnp.linalg.norm(pixel) < (1 + TINY)


    def testConvSubimage(self):
        image1 = GeometricImage(jnp.arange(25).reshape((5,5)), 0, 2)
        filter1 = GeometricFilter(jnp.zeros(25).reshape((5,5)), 0, 2)
        subimage1 = image1.conv_subimage((0,0), filter1)
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

        subimage2 = image1.conv_subimage((4,4), filter1)
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

        image2 = GeometricImage(jnp.arange(25).reshape((5,5)), 0, 2)*GeometricImage(jnp.ones((5,5,2)), 0, 2)
        subimage3 = image2.conv_subimage((0,0), filter1)
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

    def testConvolveWithIK0_FK0(self):
        """
        Convolve with where the input is k=0, and the filter is k=0
        """
        #did these out by hand, hopefully, my arithmetic is correct...
        image1 = GeometricImage(jnp.array([[2,1,0], [0,0,-3], [2,0,1]], dtype=int), 0, 2)
        filter_image = GeometricFilter(jnp.array([[1,0,1], [0,0,0], [1,0,1]], dtype=int), 0, 2)

        convolved_image = image1.convolve_with(filter_image)
        assert convolved_image.D == image1.D
        assert convolved_image.N == image1.N
        assert convolved_image.k == image1.k + filter_image.k
        assert convolved_image.parity == (image1.parity + filter_image.parity) % 2
        assert (convolved_image.data == jnp.array([[-2,0,2], [2,5,5], [-2,-1,3]], dtype=int)).all()

        key = random.PRNGKey(0)
        image2 = GeometricImage(jnp.floor(10*random.uniform(key, shape=(5,5))), 0, 2)
        convolved_image2 = image2.convolve_with(filter_image)
        assert convolved_image2.D == image2.D
        assert convolved_image2.N == image2.N
        assert convolved_image2.k == image2.k + filter_image.k
        assert convolved_image2.parity == (image2.parity + filter_image.parity) % 2
        assert (convolved_image2.data == jnp.array(
            [
                [16,9,16,11,10],
                [15,19,15,13,28],
                [17,19,15,16,17],
                [16,12,13,13,18],
                [8,23,11,13,29],
            ],
        dtype=int)).all()

    def testConvolveWithIK0_FK1(self):
        """
        Convolve with where the input is k=0, and the filter is k=1
        """
        image1 = GeometricImage(jnp.array([[2,1,0], [0,0,-3], [2,0,1]], dtype=int), 0, 2)
        filter_image = GeometricFilter(jnp.array([
            [[0,0], [0,1], [0,0]],
            [[-1,0],[0,0], [1,0]],
            [[0,0], [0,-1],[0,0]],
        ], dtype=int), 0, 2) #this is an invariant filter, hopefully not a problem?

        convolved_image = image1.convolve_with(filter_image)
        assert convolved_image.D == image1.D
        assert convolved_image.N == image1.N
        assert convolved_image.k == image1.k + filter_image.k
        assert convolved_image.parity == (image1.parity + filter_image.parity) % 2
        assert (convolved_image.data == jnp.array([
            [[1,2],[-2,0],[1,4]],
            [[3,0],[-3,1],[0,-1]],
            [[-1,-2],[-1,-1],[2,-3]]
        ], dtype=int)).all()

    # def testConvolveCommutativity(self):
    #     image1 = GeometricImage(jnp.array([[2,1,0], [0,0,-3], [2,0,1]], dtype=int), 0, 2)
    #     filter_image = GeometricFilter(jnp.array([[1,0,1], [0,0,0], [1,0,1]], dtype=int), 0, 2)

    #     convolveA = image1.convolve_with(filter_image)
    #     convolveB = filter_image.convolve_with(image1)
    #     assert convolveA.D == convolveB.D
    #     assert convolveA.N == convolveB.N
    #     assert convolveA.k == convolveB.k
    #     assert convolveA.parity == convolveB.parity
    #     assert (convolveA.data == convolveB.data).all()



