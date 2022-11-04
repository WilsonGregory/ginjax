import sys
sys.path.insert(0,'src/geometricconvolutions/')

import math

from geometric import GeometricImage, GeometricFilter
import pytest
import jax.numpy as jnp
from jax import random

TINY = 1.e-5

def levi_civita_contract_old(data, D, k, index):
    assert D in [2, 3] # BECAUSE WE SUCK
    assert k >= D - 1 # so we have enough indices work on
    if D == 2 and not isinstance(index, tuple):
        index = (index,)

    if D == 2:
        index = index[0]
        otherdata = jnp.zeros_like(data)
        otherdata = otherdata.at[..., 0].set(-1. * jnp.take(data, 1, axis=index))
        otherdata = otherdata.at[..., 1].set(1. * jnp.take(data, 0, axis=index)) #i swapped the -1 and 1
        return otherdata
    if D == 3:
        assert len(index) == 2
        i, j = index
        assert i < j
        otherdata = jnp.zeros_like(data[..., 0])
        otherdata = otherdata.at[..., 0].set(jnp.take(jnp.take(data, 2, axis=j), 1, axis=i) \
                          - jnp.take(jnp.take(data, 1, axis=j), 2, axis=i))
        otherdata = otherdata.at[..., 1].set(jnp.take(jnp.take(data, 0, axis=j), 2, axis=i) \
                          - jnp.take(jnp.take(data, 2, axis=j), 0, axis=i))
        otherdata = otherdata.at[..., 2].set(jnp.take(jnp.take(data, 1, axis=j), 0, axis=i) \
                          - jnp.take(jnp.take(data, 0, axis=j), 1, axis=i))
        return otherdata
    return

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
                    jnp.tensordot(image3[0,0], image4[0,0], axes=0),
                    jnp.tensordot(image3[0,1], image4[0,1], axes=0),
                    jnp.tensordot(image3[0,2], image4[0,2], axes=0),
                ],
                [
                    jnp.tensordot(image3[1,0], image4[1,0], axes=0),
                    jnp.tensordot(image3[1,1], image4[1,1], axes=0),
                    jnp.tensordot(image3[1,2], image4[1,2], axes=0),
                ],
                [
                    jnp.tensordot(image3[2,0], image4[2,0], axes=0),
                    jnp.tensordot(image3[2,1], image4[2,1], axes=0),
                    jnp.tensordot(image3[2,2], image4[2,2], axes=0),
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

        key = random.PRNGKey(0)
        img4 = GeometricImage(random.normal(key, shape=(3,3,2,2,2,2,2)), 0, 2)
        assert jnp.allclose(img4.contract(0,1).contract(0,1).data, img4.contract(2,3).contract(0,1).data)
        assert jnp.allclose(img4.contract(0,1).contract(1,2).data, img4.contract(3,4).contract(0,1).data)

    def testLeviCivitaContract(self):
        key = random.PRNGKey(0)
        key, subkey = random.split(key)

        # basic example, parity 0, k=1
        img1 = GeometricImage(random.uniform(subkey, shape=(3,3,2)), 0, 2)
        img1_contracted = img1.levi_civita_contract(0)
        assert img1_contracted.parity == (img1.parity + 1) % 2
        assert img1_contracted.N == img1.N
        assert img1_contracted.D == img1.D
        assert img1_contracted.k == img1.k - img1.D + 2

        lst = []
        for pixel in img1.pixels():
            lst.append(levi_civita_contract_old(pixel, img1.D, img1.k, 0))

        assert (img1_contracted.data == jnp.array(lst).reshape(img1_contracted.shape())).all()

        # parity 1, k=1
        key, subkey = random.split(key)
        img2 = GeometricImage(random.uniform(subkey, shape=(3,3,2)), 1, 2)
        img2_contracted = img2.levi_civita_contract(0)
        assert img2_contracted.parity == (img2.parity + 1) % 2
        assert img2_contracted.N == img2.N
        assert img2_contracted.D == img2.D
        assert img2_contracted.k == img2.k - img2.D + 2

        lst = []
        for pixel in img2.pixels():
            lst.append(levi_civita_contract_old(pixel, img2.D, img2.k, 0))

        assert (img2_contracted.data == jnp.array(lst).reshape(img2_contracted.shape())).all()

        # k=2
        key, subkey = random.split(key)
        img3 = GeometricImage(random.uniform(subkey, shape=(3,3,2,2)), 0, 2)
        for idx in range(img3.k):
            print(idx)
            img3_contracted = img3.levi_civita_contract(idx)
            assert img3_contracted.parity == (img3.parity + 1) % 2
            assert img3_contracted.N == img3.N
            assert img3_contracted.D == img3.D
            assert img3_contracted.k == img3.k - img3.D + 2 #k+D - 2(D-1) = k-D +2

            lst = []
            for pixel in img3.pixels():
                lst.append(levi_civita_contract_old(pixel, img3.D, img3.k, idx))

            assert (img3_contracted.data == jnp.array(lst).reshape(img3_contracted.shape())).all()

        # D=3, k=2
        key, subkey = random.split(key)
        img4 = GeometricImage(random.uniform(subkey, shape=(3,3,3,3,3)), 0, 3)
        img4_contracted = img4.levi_civita_contract((0,1))
        assert img4_contracted.parity == (img4.parity + 1) % 2
        assert img4_contracted.N == img4.N
        assert img4_contracted.D == img4.D
        assert img4_contracted.k == img4.k - img4.D + 2

        lst = []
        for pixel in img4.pixels():
            lst.append(levi_civita_contract_old(pixel, img4.D, img4.k, (0,1)))

        assert (img4_contracted.data == jnp.array(lst).reshape(img4_contracted.shape())).all()
        assert not (img4_contracted.data == img4.levi_civita_contract((1,0)).data).all()

        # D=3, k=3
        key, subkey = random.split(key)
        img5 = GeometricImage(random.uniform(subkey, shape=(3,3,3,3,3,3)), 0, 3)
        for indices in [(0,1),(0,2),(1,2)]:
            img5_contracted = img5.levi_civita_contract(indices)
            assert img5_contracted.parity == (img5.parity + 1) % 2
            assert img5_contracted.N == img5.N
            assert img5_contracted.D == img5.D
            assert img5_contracted.k == img5.k - img5.D + 2

            lst = []
            for pixel in img5.pixels():
                lst.append(levi_civita_contract_old(pixel, img5.D, img5.k, indices))

            assert (img5_contracted.data == jnp.array(lst).reshape(img5_contracted.shape())).all()

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

    def testConvolutionLinearity(self):
        """
        For scalars alpha, beta, tensor images image1, image2 and filter c1 alpha
        """
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        image1 = GeometricImage(random.uniform(subkey, shape=(3,3,2)), 0, 2)

        key, subkey = random.split(key)
        image2 = GeometricImage(random.uniform(subkey, shape=(3,3,2)), 0, 2)

        key, subkey = random.split(key)
        c1 = GeometricFilter(random.uniform(subkey, shape=(3,3,2)), 0, 2)

        alpha, beta = random.uniform(subkey, shape=(2,))

        B1 = image1.convolve_with(c1).times_scalar(alpha) + image2.convolve_with(c1).times_scalar(beta)
        B2 = (image1.times_scalar(alpha) + image2.times_scalar(beta)).convolve_with(c1)

        assert B1.shape() == B2.shape()
        assert B1.parity == B2.parity
        assert jnp.allclose(B1.data, B2.data)

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

    def testTimesGroupElement(self):
        left90 = jnp.array([[0,-1],[1,0]])
        flipX = jnp.array([[-1, 0], [0,1]])

        img1 = GeometricImage(jnp.arange(9).reshape((3,3)), 0, 2)

        #basic rotate
        img1_left90 = img1.times_group_element(left90)
        assert img1_left90.D == img1.D
        assert img1_left90.parity == img1.parity
        assert img1_left90.k == img1.k
        assert (img1_left90.data == jnp.array([[2,5,8], [1,4,7], [0,3,6]])).all()

        #basic flip
        img1_flipX = img1.times_group_element(flipX)
        assert img1_flipX.parity == img1.parity
        assert img1_flipX.k == img1.k
        assert (img1_flipX.data == jnp.array([[6,7,8], [3,4,5], [0,1,2]])).all()

        img2 = GeometricImage(jnp.arange(9).reshape((3,3)), 1, 2)

        #rotate, no sign changes
        img2_left90 = img2.times_group_element(left90)
        assert (img1_left90.data == img2_left90.data).all()
        assert img2_left90.parity == img2.parity

        #rotate and parity 1, sign is flipped from img1
        img2_flipX = img2.times_group_element(flipX)
        assert img2_flipX.parity == img2.parity
        assert (img2_flipX.data == img1_flipX.times_scalar(-1).data).all()

        img3 = GeometricImage(jnp.arange(18).reshape((3,3,2)), 1, 2)

        #k=1 rotate
        img3_left90 = img3.times_group_element(left90)
        assert img3_left90.D == img3.D
        assert img3_left90.parity == img3.parity
        assert img3_left90.k == img3.k
        assert (img3_left90.data == jnp.array([
            [[-5,4], [-11,10], [-17,16]],
            [[-3,2], [-9,8], [-15,14]],
            [[-1,0], [-7,6], [-13,12]],
        ])).all()

        img4 = GeometricImage(jnp.arange(36).reshape((3,3,2,2)), 0, 2)

        #k=2 flip
        img4_flipX = img4.times_group_element(flipX)
        assert img4_flipX.D == img4.D
        assert img4_flipX.parity == img4.parity
        assert img4_flipX.k == img4.k
        print(img4.data)
        print(img4_flipX.data)
        assert (img4_flipX.data == jnp.array([
            [ #first row
                [[24,-25], [-26,27]],
                [[28,-29], [-30,31]],
                [[32,-33], [-34,35]],
            ],
            [ #second row
                [[12,-13], [-14,15]],
                [[16,-17], [-18,19]],
                [[20,-21], [-22,23]],
            ],
            [ #third row
                [[0,-1], [-2,3]],
                [[4,-5], [-6,7]],
                [[8,-9], [-10,11]],
            ],
        ])).all()


