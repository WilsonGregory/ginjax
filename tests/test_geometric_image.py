import math
import time
import numpy as np

import ginjax.geometric as geom
import pytest
import jax.numpy as jnp
from jax import random

TINY = 1.0e-5


def levi_civita_contract_old(data, D, k, index):
    assert D in [2, 3]  # BECAUSE WE SUCK
    assert k >= D - 1  # so we have enough indices work on
    if D == 2 and not isinstance(index, tuple):
        index = (index,)

    if D == 2:
        index = index[0]
        otherdata = jnp.zeros_like(data)
        otherdata = otherdata.at[..., 0].set(-1.0 * jnp.take(data, 1, axis=index))
        otherdata = otherdata.at[..., 1].set(
            1.0 * jnp.take(data, 0, axis=index)
        )  # i swapped the -1 and 1
        return otherdata
    if D == 3:
        assert len(index) == 2
        i, j = index
        assert i < j
        otherdata = jnp.zeros_like(data[..., 0])
        otherdata = otherdata.at[..., 0].set(
            jnp.take(jnp.take(data, 2, axis=j), 1, axis=i)
            - jnp.take(jnp.take(data, 1, axis=j), 2, axis=i)
        )
        otherdata = otherdata.at[..., 1].set(
            jnp.take(jnp.take(data, 0, axis=j), 2, axis=i)
            - jnp.take(jnp.take(data, 2, axis=j), 0, axis=i)
        )
        otherdata = otherdata.at[..., 2].set(
            jnp.take(jnp.take(data, 1, axis=j), 0, axis=i)
            - jnp.take(jnp.take(data, 0, axis=j), 1, axis=i)
        )
        return otherdata
    return


class TestGeometricImage:

    def testZerosConstructor(self):
        image1 = geom.GeometricImage.zeros(20, 0, 0, 2)
        assert image1.data.shape == (20, 20)
        assert image1.k == 0

        image2 = geom.GeometricImage.zeros(20, 1, 0, 2)
        assert image2.data.shape == (20, 20, 2)
        assert image2.k == 1

        image3 = geom.GeometricImage.zeros(20, 3, 0, 2)
        assert image3.data.shape == (20, 20, 2, 2, 2)
        assert image3.k == 3

        # square but spatial_dims in constructor, odd parity, D=3, not torus
        image4 = geom.GeometricImage.zeros((5,) * 3, 1, 1, 3, False)
        assert image4.data.shape == (5, 5, 5, 3)
        assert image4.k == 1
        assert image4.D == 3
        assert image4.parity == 1
        assert image4.is_torus == (False,) * 3

        # non-square, D=3
        image5 = geom.GeometricImage.zeros((4, 5, 6), 0, 0, 3)
        assert image5.data.shape == (4, 5, 6)
        assert image5.D == 3

    def testConstructor(self):
        # note we are not actually relying on randomness in this function, just filling values
        key = random.PRNGKey(0)

        image1 = geom.GeometricImage(random.uniform(key, shape=(10, 10)), 0, 2)
        assert image1.data.shape == (10, 10)
        assert image1.D == 2
        assert image1.k == 0
        assert image1.is_torus == (True,) * 2

        image2 = geom.GeometricImage(random.uniform(key, shape=(10, 10, 2)), 0, 2)
        assert image2.data.shape == (10, 10, 2)
        assert image2.D == 2
        assert image2.k == 1

        image3 = geom.GeometricImage(random.uniform(key, shape=(10, 10, 2, 2, 2)), 3, 2)
        assert image3.data.shape == (10, 10, 2, 2, 2)
        assert image3.k == 3
        assert image3.parity == 1

        image4 = geom.GeometricImage(random.uniform(key, shape=(10, 10, 10)), 0, 3, False)
        assert image4.is_torus == (False,) * 3

        image5 = geom.GeometricImage(random.uniform(key, shape=(4, 5)), 0, 2)
        assert image5.spatial_dims == (4, 5)
        assert image5.k == 0

        image6 = geom.GeometricImage(random.uniform(key, shape=(2, 5, 2)), 0, 2)
        assert image6.spatial_dims == (2, 5)
        assert image6.data.shape == (2, 5, 2)
        assert image6.k == 1

        # D does not match dimensions
        with pytest.raises(AssertionError):
            geom.GeometricImage(random.uniform(key, shape=(10, 10)), 0, 3)

        # side length of pixel tensors does not match D
        with pytest.raises(AssertionError):
            geom.GeometricImage(random.uniform(key, shape=(10, 10, 3, 3)), 0, 2)

    def testEqual(self):
        img1 = geom.GeometricImage(jnp.ones((10, 10, 2)), 0, 2)

        # same
        img2 = geom.GeometricImage(jnp.ones((10, 10, 2)), 0, 2)
        assert img1 == img2

        # different N
        img3 = geom.GeometricImage(jnp.ones((5, 5, 2)), 0, 2)
        assert img1 != img3

        # different k
        img4 = geom.GeometricImage(jnp.ones((10, 10, 2, 2)), 0, 2)
        assert img1 != img4

        # different D
        img5 = geom.GeometricImage(jnp.ones((10, 10, 10)), 0, 3)
        assert img1 != img5
        img6 = geom.GeometricImage(jnp.ones((2, 2, 2)), 0, 3)  # D=3, k=0
        img7 = geom.GeometricImage(jnp.ones((2, 2, 2)), 0, 2)  # D=2, k=1
        assert img6 != img7

        # different parity
        img8 = geom.GeometricImage(jnp.ones((10, 10, 2)), 1, 2)
        assert img1 != img8

        # different is_torus
        img9 = geom.GeometricImage(jnp.ones((10, 10, 2)), 0, 2, False)
        assert img1 != img9

        # different data
        img9 = geom.GeometricImage(2 * jnp.ones((10, 10, 2)), 0, 2)
        assert img1 != img9
        assert img1 != 1.0001 * img1  # outside the error tolerance
        assert img1 == 1.0000001 * img1  # within the error tolerance

        # different N, nonsquare
        img10 = geom.GeometricImage(jnp.ones((9, 10, 2)), 0, 2)
        assert img1 != img10

    def testAdd(self):
        image1 = geom.GeometricImage(jnp.ones((10, 10, 2), dtype=int), 0, 2)
        image2 = geom.GeometricImage(5 * jnp.ones((10, 10, 2), dtype=int), 0, 2)
        float_image = geom.GeometricImage(3.4 * jnp.ones((10, 10, 2)), 0, 2)

        result = image1 + image2
        assert (result.data == 6).all()
        assert result.parity == 0
        assert result.D == 2
        assert result.k == 1
        assert result.spatial_dims == (10, 10)

        assert (image1.data == 1).all()
        assert (image2.data == 5).all()

        result = image1 + float_image
        assert (result.data == 4.4).all()

        image3 = geom.GeometricImage(jnp.ones((10, 10, 10, 3), dtype=int), 0, 3)
        with pytest.raises(AssertionError):  # D not equal
            result = image1 + image3

        image4 = geom.GeometricImage(jnp.ones((10, 10, 2), dtype=int), 1, 2)
        with pytest.raises(AssertionError):  # parity not equal
            result = image1 + image4

        with pytest.raises(AssertionError):
            result = image3 + image4  # D and parity not equal

        image5 = geom.GeometricImage(jnp.ones((20, 20, 2), dtype=int), 0, 2)
        with pytest.raises(AssertionError):  # N not equal
            result = image1 + image5

        image6 = geom.GeometricImage(jnp.ones((10, 10, 2)), 0, 2, False)
        with pytest.raises(AssertionError):  # is_torus not equal
            result = image1 + image6

    def testSub(self):
        image1 = geom.GeometricImage(jnp.ones((10, 10, 2), dtype=int), 0, 2)
        image2 = geom.GeometricImage(5 * jnp.ones((10, 10, 2), dtype=int), 0, 2)
        float_image = geom.GeometricImage(3.4 * jnp.ones((10, 10, 2)), 0, 2)

        result = image1 - image2
        assert (result.data == -4).all()
        assert result.parity == 0
        assert result.D == 2
        assert result.k == 1
        assert result.spatial_dims == (10, 10)

        assert (image1.data == 1).all()
        assert (image2.data == 5).all()

        result = image1 - float_image
        assert (result.data == -2.4).all()

        image3 = geom.GeometricImage(jnp.ones((10, 10, 10, 3), dtype=int), 0, 3)
        with pytest.raises(AssertionError):  # D not equal
            result = image1 - image3

        image4 = geom.GeometricImage(jnp.ones((10, 10, 2), dtype=int), 1, 2)
        with pytest.raises(AssertionError):  # parity not equal
            result = image1 - image4

        with pytest.raises(AssertionError):
            result = image3 - image4  # D and parity not equal

        image5 = geom.GeometricImage(jnp.ones((20, 20, 2), dtype=int), 0, 2)
        with pytest.raises(AssertionError):  # N not equal
            result = image1 - image5

    def testMul(self):
        image1 = geom.GeometricImage(2 * jnp.ones((3, 3), dtype=int), 0, 2)
        image2 = geom.GeometricImage(5 * jnp.ones((3, 3), dtype=int), 0, 2)

        mult1_2 = image1 * image2
        assert mult1_2.k == 0
        assert mult1_2.parity == 0
        assert mult1_2.D == image1.D == image2.D
        assert mult1_2.spatial_dims == image1.spatial_dims == image1.spatial_dims
        assert (mult1_2.data == 10 * jnp.ones((3, 3))).all()
        assert (mult1_2.data == (image2 * image1).data).all()

        image3 = geom.GeometricImage(jnp.arange(18).reshape(3, 3, 2), 0, 2)
        mult1_3 = image1 * image3
        assert mult1_3.k == image1.k + image3.k == 1
        assert mult1_3.parity == (image1.parity + image3.parity) % 2 == 0
        assert mult1_3.D == image1.D == image3.D
        assert mult1_3.spatial_dims == image1.spatial_dims == image3.spatial_dims
        assert (
            mult1_3.data
            == jnp.array(
                [
                    [[0, 2], [4, 6], [8, 10]],
                    [[12, 14], [16, 18], [20, 22]],
                    [[24, 26], [28, 30], [32, 34]],
                ],
                dtype=int,
            )
        ).all()

        image4 = geom.GeometricImage(jnp.arange(18).reshape((3, 3, 2)), 1, 2)
        mult3_4 = image3 * image4
        assert mult3_4.k == image3.k + image3.k == 2
        assert mult3_4.parity == (image3.parity + image4.parity) % 2 == 1
        assert mult3_4.D == image3.D == image4.D
        assert mult3_4.spatial_dims == image3.spatial_dims == image4.spatial_dims
        assert (
            mult3_4.data
            == jnp.array(
                [
                    [
                        jnp.tensordot(image3[0, 0], image4[0, 0], axes=0),
                        jnp.tensordot(image3[0, 1], image4[0, 1], axes=0),
                        jnp.tensordot(image3[0, 2], image4[0, 2], axes=0),
                    ],
                    [
                        jnp.tensordot(image3[1, 0], image4[1, 0], axes=0),
                        jnp.tensordot(image3[1, 1], image4[1, 1], axes=0),
                        jnp.tensordot(image3[1, 2], image4[1, 2], axes=0),
                    ],
                    [
                        jnp.tensordot(image3[2, 0], image4[2, 0], axes=0),
                        jnp.tensordot(image3[2, 1], image4[2, 1], axes=0),
                        jnp.tensordot(image3[2, 2], image4[2, 2], axes=0),
                    ],
                ],
                dtype=int,
            )
        ).all()

        image5 = geom.GeometricImage(jnp.ones((10, 10)), 0, 2)
        with pytest.raises(AssertionError):  # mismatched N
            image5 * image1

        image6 = geom.GeometricImage(jnp.ones((3, 3, 3)), 0, 3)
        with pytest.raises(AssertionError):  # mismatched D
            image6 * image1

        # Test multiplying by a scalar
        result = image1 * 5
        assert (result.data == 10).all()
        assert result.parity == image1.parity
        assert result.D == image1.D
        assert result.k == image1.k
        assert result.spatial_dims == image1.spatial_dims
        assert (image1.data == 2).all()  # original is unchanged

        result2 = image1 * 3.4
        assert (result2.data == 6.8).all()
        assert (image1.data == 2).all()

        # Test multiplying by a scalar right mul
        result = 5 * image1
        assert (result.data == 10).all()
        assert result.parity == image1.parity
        assert result.D == image1.D
        assert result.k == image1.k
        assert result.spatial_dims == image1.spatial_dims
        assert (image1.data == 2).all()  # original is unchanged

        result2 = 3.4 * image1
        assert (result2.data == 6.8).all()
        assert (image1.data == 2).all()

        # check that rmul isn't being used in this case, because it only handles scalar multiplication
        geom_filter = geom.GeometricFilter(jnp.ones(image1.shape()), image1.parity, image1.D)
        res1 = image1 * geom_filter
        res2 = geom_filter * image1
        assert (res1.data == res2.data).all()

    def testTimeScalar(self):
        image1 = geom.GeometricImage(jnp.ones((10, 10, 2), dtype=int), 0, 2)
        assert (image1.data == 1).all()

        result = image1 * 5
        assert (result.data == 5).all()
        assert result.parity == image1.parity
        assert result.D == image1.D
        assert result.k == image1.k
        assert result.spatial_dims == image1.spatial_dims
        assert (image1.data == 1).all()  # original is unchanged

        result2 = image1 * 3.4
        assert (result2.data == 3.4).all()
        assert (image1.data == 1).all()

    def testGetItem(self):
        # note we are not actually relying on randomness in this function, just filling values
        key = random.PRNGKey(0)

        random_vals = random.uniform(key, shape=(10, 10, 2, 2, 2))
        image1 = geom.GeometricImage(random_vals, 0, 2)

        assert image1[0, 5, 0, 1, 1] == random_vals[0, 5, 0, 1, 1]
        assert image1[4, 3, 0, 0, 1] == random_vals[4, 3, 0, 0, 1]
        assert (image1[0] == random_vals[0]).all()
        assert (image1[4:, 2:3] == random_vals[4:, 2:3]).all()
        assert image1[4:, 2:3].shape == random_vals[4:, 2:3].shape

    def testContract(self):
        img1 = geom.GeometricImage(jnp.arange(36).reshape((3, 3, 2, 2)), 0, 2)

        img1_contracted = img1.contract(0, 1)
        assert img1_contracted.shape() == (3, 3)
        assert (img1_contracted.data == jnp.array([[3, 11, 19], [27, 35, 43], [51, 59, 67]])).all()
        assert (img1.contract(1, 0).data == img1_contracted.data).all()

        img2 = geom.GeometricImage(jnp.arange(72).reshape((3, 3, 2, 2, 2)), 0, 2)

        img2_contracted_1 = img2.contract(0, 1)
        assert img2_contracted_1.shape() == (3, 3, 2)
        assert (
            img2_contracted_1.data
            == jnp.array(
                [
                    [[6, 8], [22, 24], [38, 40]],
                    [[54, 56], [70, 72], [86, 88]],
                    [[102, 104], [118, 120], [134, 136]],
                ]
            )
        ).all()
        assert (img2.contract(1, 0).data == img2_contracted_1.data).all()

        img2_contracted_2 = img2.contract(0, 2)
        assert img2_contracted_2.shape() == (3, 3, 2)
        assert (
            img2_contracted_2.data
            == jnp.array(
                [
                    [[5, 9], [21, 25], [37, 41]],
                    [[53, 57], [69, 73], [85, 89]],  # nice
                    [[101, 105], [117, 121], [133, 137]],
                ]
            )
        ).all()
        assert (img2.contract(2, 0).data == img2_contracted_2.data).all()

        img2_contracted_3 = img2.contract(1, 2)
        assert img2_contracted_3.shape() == (3, 3, 2)
        assert (
            img2_contracted_3.data
            == jnp.array(
                [
                    [[3, 11], [19, 27], [35, 43]],
                    [[51, 59], [67, 75], [83, 91]],  # nice
                    [[99, 107], [115, 123], [131, 139]],
                ]
            )
        ).all()
        assert (img2.contract(2, 1).data == img2_contracted_3.data).all()

        img3 = geom.GeometricImage(jnp.ones((3, 3)), 0, 2)
        with pytest.raises(AssertionError):
            img3.contract(0, 1)  # k < 2

    def testMulticontract(self):
        D = 2
        N = 3
        k = 5
        key = random.PRNGKey(time.time_ns())
        img1 = geom.GeometricImage(random.normal(key, shape=((N,) * D + (D,) * k)), 0, D)

        for idxs in geom.get_contraction_indices(k, 1):
            (i1, j1), (i2, j2) = idxs
            i2_shift = int(i2 > i1) + int(i2 > j1)
            j2_shift = int(j2 > i1) + int(j2 > j1)
            assert img1.multicontract(idxs) == img1.contract(i1, j1).contract(
                i2 - i2_shift, j2 - j2_shift
            )

    def testLeviCivitaContract(self):
        key = random.PRNGKey(0)
        key, subkey = random.split(key)

        # basic example, parity 0, k=1
        img1 = geom.GeometricImage(random.uniform(subkey, shape=(3, 3, 2)), 0, 2)
        img1_contracted = img1.levi_civita_contract(0)
        assert img1_contracted.parity == (img1.parity + 1) % 2
        assert img1_contracted.spatial_dims == img1.spatial_dims
        assert img1_contracted.D == img1.D
        assert img1_contracted.k == img1.k - img1.D + 2

        lst = []
        for pixel in img1.pixels():
            lst.append(levi_civita_contract_old(pixel, img1.D, img1.k, 0))

        assert (img1_contracted.data == jnp.array(lst).reshape(img1_contracted.shape())).all()

        # parity 1, k=1
        key, subkey = random.split(key)
        img2 = geom.GeometricImage(random.uniform(subkey, shape=(3, 3, 2)), 1, 2)
        img2_contracted = img2.levi_civita_contract(0)
        assert img2_contracted.parity == (img2.parity + 1) % 2
        assert img2_contracted.spatial_dims == img2.spatial_dims
        assert img2_contracted.D == img2.D
        assert img2_contracted.k == img2.k - img2.D + 2

        lst = []
        for pixel in img2.pixels():
            lst.append(levi_civita_contract_old(pixel, img2.D, img2.k, 0))

        assert (img2_contracted.data == jnp.array(lst).reshape(img2_contracted.shape())).all()

        # k=2
        key, subkey = random.split(key)
        img3 = geom.GeometricImage(random.uniform(subkey, shape=(3, 3, 2, 2)), 0, 2)
        for idx in range(img3.k):
            img3_contracted = img3.levi_civita_contract(idx)
            assert img3_contracted.parity == (img3.parity + 1) % 2
            assert img3_contracted.spatial_dims == img3.spatial_dims
            assert img3_contracted.D == img3.D
            assert img3_contracted.k == img3.k - img3.D + 2  # k+D - 2(D-1) = k-D +2

            lst = []
            for pixel in img3.pixels():
                lst.append(levi_civita_contract_old(pixel, img3.D, img3.k, idx))

            assert (img3_contracted.data == jnp.array(lst).reshape(img3_contracted.shape())).all()

        # D=3, k=2
        key, subkey = random.split(key)
        img4 = geom.GeometricImage(random.uniform(subkey, shape=(3, 3, 3, 3, 3)), 0, 3)
        img4_contracted = img4.levi_civita_contract((0, 1))
        assert img4_contracted.parity == (img4.parity + 1) % 2
        assert img4_contracted.spatial_dims == img4.spatial_dims
        assert img4_contracted.D == img4.D
        assert img4_contracted.k == img4.k - img4.D + 2

        lst = []
        for pixel in img4.pixels():
            lst.append(levi_civita_contract_old(pixel, img4.D, img4.k, (0, 1)))

        assert (img4_contracted.data == jnp.array(lst).reshape(img4_contracted.shape())).all()
        assert not (img4_contracted.data == img4.levi_civita_contract((1, 0)).data).all()

        # D=3, k=3
        key, subkey = random.split(key)
        img5 = geom.GeometricImage(random.uniform(subkey, shape=(3, 3, 3, 3, 3, 3)), 0, 3)
        for indices in [(0, 1), (0, 2), (1, 2)]:
            img5_contracted = img5.levi_civita_contract(indices)
            assert img5_contracted.parity == (img5.parity + 1) % 2
            assert img5_contracted.spatial_dims == img5.spatial_dims
            assert img5_contracted.D == img5.D
            assert img5_contracted.k == img5.k - img5.D + 2

            lst = []
            for pixel in img5.pixels():
                lst.append(levi_civita_contract_old(pixel, img5.D, img5.k, indices))

            assert (img5_contracted.data == jnp.array(lst).reshape(img5_contracted.shape())).all()

    def testNorm(self):
        # 2d image of scalars
        image1 = geom.GeometricImage(jnp.array([[1, 2], [-1, 0]]), 0, 2)
        assert image1.norm() == geom.GeometricImage(jnp.array([[1, 2], [1, 0]]), 0, 2)
        assert image1.norm().parity == 0

        # 2d image of pseudoscalars
        image2 = geom.GeometricImage(jnp.array([[1, 2], [-1, 0]]), 1, 2)
        assert image2.norm() == geom.GeometricImage(jnp.array([[1, 2], [1, 0]]), 0, 2)
        assert image2.norm().parity == 0  # parity of norm is 0

        # 2d image of vectors
        image2 = geom.GeometricImage(jnp.array([[[1, 0], [-1, -1]], [[0, 0], [-4, 3]]]), 0, 2)
        assert image2.norm() == geom.GeometricImage(jnp.array([[1, jnp.sqrt(2)], [0, 5]]), 0, 2)

        # 3d image of scalars
        image3 = geom.GeometricImage(jnp.array([[[4, -3], [0, 1]], [[-2, -3], [1, 2]]]), 0, 3)
        assert image3.norm() == geom.GeometricImage(
            jnp.array([[[4, 3], [0, 1]], [[2, 3], [1, 2]]]), 0, 3
        )

        # 2d image of matrices, N=2
        image4 = geom.GeometricImage(
            jnp.array(
                [
                    [
                        [[1, 0], [-1, -1]],
                        [[0, 0], [-4, 3]],
                    ],
                    [
                        [[1, 2], [-1, 0]],
                        [[4, 0], [2, -4]],
                    ],
                ]
            ),
            0,
            2,
        )
        assert image4.norm() == geom.GeometricImage(
            jnp.array([[jnp.sqrt(3), 5], [jnp.sqrt(6), 6]]), 0, 2
        )

        # 2d images of 3rd order tensors, N=1
        image5 = geom.GeometricImage(jnp.array([[[[[1, 0], [-1, -1]], [[0, 0], [-4, 3]]]]]), 0, 2)
        assert image5.norm() == geom.GeometricImage(jnp.array([[jnp.sqrt(28)]]), 0, 2)

    def testNormalize(self):
        key = random.PRNGKey(0)
        image1 = geom.GeometricImage(random.uniform(key, shape=(10, 10)), 0, 2)

        normed_image1 = image1.normalize()
        assert math.isclose(jnp.max(jnp.abs(normed_image1.data)), 1.0, rel_tol=geom.TINY)
        assert image1.data.shape == normed_image1.data.shape == (10, 10)

        image2 = geom.GeometricImage(random.uniform(key, shape=(10, 10, 2)), 0, 2)
        normed_image2 = image2.normalize()
        assert image2.data.shape == normed_image2.data.shape == (10, 10, 2)
        for row in normed_image2.data:
            for pixel in row:
                assert jnp.linalg.norm(pixel) < (1 + TINY)

        image3 = geom.GeometricImage(random.uniform(key, shape=(10, 10, 2, 2)), 0, 2)
        normed_image3 = image3.normalize()
        assert image3.data.shape == normed_image3.data.shape == (10, 10, 2, 2)
        for row in normed_image3.data:
            for pixel in row:
                assert jnp.linalg.norm(pixel) < (1 + TINY)

    def testConvolveWithIK0_FK0(self):
        """
        Convolve with where the input is k=0, and the filter is k=0
        """
        # did these out by hand, hopefully, my arithmetic is correct...
        image1 = geom.GeometricImage(
            jnp.array([[2, 1, 0], [0, 0, -3], [2, 0, 1]], dtype=float), 0, 2
        )
        filter_image = geom.GeometricFilter(
            jnp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=float), 0, 2
        )

        convolved_image = image1.convolve_with(filter_image)
        assert convolved_image.D == image1.D
        assert convolved_image.spatial_dims == image1.spatial_dims
        assert convolved_image.k == image1.k + filter_image.k
        assert convolved_image.parity == (image1.parity + filter_image.parity) % 2
        assert (
            convolved_image.data == jnp.array([[-2, 0, 2], [2, 5, 5], [-2, -1, 3]], dtype=float)
        ).all()

        image2 = geom.GeometricImage(
            jnp.array(
                [
                    [9, 9, 3, 4, 5],
                    [1, 3, 6, 7, 1],
                    [9, 0, 6, 5, 8],
                    [9, 8, 7, 5, 0],
                    [0, 9, 5, 7, 5],
                ]
            ),
            0,
            2,
        )
        convolved_image2 = image2.convolve_with(filter_image)
        assert convolved_image2.D == image2.D
        assert convolved_image2.spatial_dims == image2.spatial_dims
        assert convolved_image2.k == image2.k + filter_image.k
        assert convolved_image2.parity == (image2.parity + filter_image.parity) % 2
        assert (
            convolved_image2.data
            == jnp.array(
                [
                    [18, 12, 26, 17, 15],
                    [22, 27, 18, 22, 27],
                    [12, 23, 23, 14, 22],
                    [22, 20, 21, 24, 21],
                    [22, 28, 26, 15, 27],
                ],
                dtype=float,
            )
        ).all()

    def testConvolveWithIK0_FK1(self):
        """
        Convolve with where the input is k=0, and the filter is k=1
        """
        image1 = geom.GeometricImage(
            jnp.array([[2, 1, 0], [0, 0, -3], [2, 0, 1]], dtype=float), 0, 2
        )
        filter_image = geom.GeometricFilter(
            jnp.array(
                [
                    [[0, 0], [0, 1], [0, 0]],
                    [[-1, 0], [0, 0], [1, 0]],
                    [[0, 0], [0, -1], [0, 0]],
                ],
                dtype=float,
            ),
            0,
            2,
        )  # this is an invariant filter, hopefully not a problem?

        convolved_image = image1.convolve_with(filter_image)
        assert convolved_image.D == image1.D
        assert convolved_image.spatial_dims == image1.spatial_dims
        assert convolved_image.k == image1.k + filter_image.k
        assert convolved_image.parity == (image1.parity + filter_image.parity) % 2
        assert (
            convolved_image.data
            == jnp.array(
                [
                    [[1, 2], [-2, 0], [1, 4]],
                    [[3, 0], [-3, 1], [0, -1]],
                    [[-1, -2], [-1, -1], [2, -3]],
                ],
                dtype=float,
            )
        ).all()

    def testConvolveNonTorus(self):
        """
        Convolve where the GeometricImage is not a torus.
        """
        image1 = geom.GeometricImage(
            jnp.array([[2, 1, 0], [0, 0, -3], [2, 0, 1]], dtype=float), 0, 2, False
        )
        filter_image = geom.GeometricFilter(
            jnp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=float), 0, 2
        )

        convolved_image = image1.convolve_with(filter_image)
        assert convolved_image.is_torus == (False,) * 2
        assert jnp.allclose(convolved_image.data, jnp.array([[0, -3, 0], [1, 5, 1], [0, -3, 0]]))

    def testConvolveDilation(self):
        """
        Convolve where the Geometric Image is a torus, but we dilate the filter.
        """
        image1 = geom.GeometricImage(
            jnp.array([[2, 1, 0], [0, 0, -3], [2, 0, 1]], dtype=float), 0, 2
        )
        filter_image = geom.GeometricFilter(
            jnp.array([[1, 0, 2], [1, -1, 0], [0, 0, -2]], dtype=float), 0, 2
        )

        convolved_image = image1.convolve_with(filter_image, rhs_dilation=(2, 2))
        assert jnp.allclose(convolved_image.data, jnp.array([[-9, -8, 2], [2, -2, 3], [5, 5, 5]]))

    def testConvolveDilationNonTorus(self):
        """
        Convolve where the Geometric Image is not a torus and we dilate the filter.
        """
        image1 = geom.GeometricImage(
            jnp.array(
                [
                    [2, 1, 0, 1, 0],
                    [0, 0, -3, 2, -2],
                    [2, 0, 1, 0, 1],
                    [1, 0, 1, 2, 2],
                    [-1, -1, 0, 0, 0],
                ],
                dtype=float,
            ),
            0,
            2,
            is_torus=False,
        )
        filter_image = geom.GeometricFilter(
            jnp.array([[1, 0, 2], [1, -1, 0], [0, 0, -2]], dtype=float), 0, 2
        )

        convolved_image = image1.convolve_with(filter_image, rhs_dilation=(2, 2))
        assert jnp.allclose(
            convolved_image.data,
            jnp.array(
                [
                    [-4, -1, 0, 0, 0],
                    [-2, -4, -1, -2, -1],
                    [-2, 2, 3, 1, 0],
                    [-7, 4, -4, -2, -4],
                    [3, 1, 3, -1, 1],
                ]
            ),
        )

    def testConvolvePartialTorus(self):
        # did these out by hand, hopefully, my arithmetic is correct...
        image1 = geom.GeometricImage(
            jnp.array(
                [
                    [2, 1, 0],
                    [0, 0, -3],
                    [2, 0, 1],
                ],
                dtype=float,
            ),
            0,
            2,
            (False, True),
        )
        filter_image = geom.GeometricFilter(
            jnp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=float), 0, 2
        )

        convolved_image = image1.convolve_with(filter_image)
        assert convolved_image.D == 2
        assert convolved_image.spatial_dims == (3, 3)
        assert convolved_image.k == 0
        assert convolved_image.parity == 0
        assert convolved_image.is_torus == (False, True)
        assert jnp.allclose(
            convolved_image.data,
            jnp.array([[-3, -3, 0], [2, 5, 5], [-3, -3, 0]], dtype=float),
        )

        image2 = geom.GeometricImage(
            jnp.array(
                [
                    [2, 1, 0],
                    [0, 0, -3],
                    [2, 0, 1],
                ],
                dtype=float,
            ),
            0,
            2,
            (True, False),
        )
        convolved_image2 = image2.convolve_with(filter_image)
        assert convolved_image2.D == 2
        assert convolved_image2.spatial_dims == (3, 3)
        assert convolved_image2.k == 0
        assert convolved_image2.parity == 0
        assert convolved_image2.is_torus == (True, False)
        assert jnp.allclose(
            convolved_image2.data,
            jnp.array([[0, 0, 0], [1, 5, 1], [1, -1, 1]], dtype=float),
        )

    def testTimesGroupElement(self):
        left90 = np.array([[0, -1], [1, 0]])
        flipX = np.array([[-1, 0], [0, 1]])

        img1 = geom.GeometricImage(jnp.arange(9).reshape((3, 3)), 0, 2)

        # basic rotate
        img1_left90 = img1.times_group_element(left90)
        assert img1_left90.D == img1.D
        assert img1_left90.parity == img1.parity
        assert img1_left90.k == img1.k
        assert (img1_left90.data == jnp.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]])).all()

        # basic flip
        img1_flipX = img1.times_group_element(flipX)
        assert img1_flipX.parity == img1.parity
        assert img1_flipX.k == img1.k
        assert (img1_flipX.data == jnp.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])).all()

        img2 = geom.GeometricImage(jnp.arange(9).reshape((3, 3)), 1, 2)

        # rotate, no sign changes
        img2_left90 = img2.times_group_element(left90)
        assert (img1_left90.data == img2_left90.data).all()
        assert img2_left90.parity == img2.parity

        # rotate and parity 1, sign is flipped from img1
        img2_flipX = img2.times_group_element(flipX)
        assert img2_flipX.parity == img2.parity
        assert (img2_flipX.data == (img1_flipX * -1).data).all()

        img3 = geom.GeometricImage(jnp.arange(18).reshape((3, 3, 2)), 1, 2)

        # k=1 rotate
        img3_left90 = img3.times_group_element(left90)
        assert img3_left90.D == img3.D
        assert img3_left90.parity == img3.parity
        assert img3_left90.k == img3.k
        assert (
            img3_left90.data
            == jnp.array(
                [
                    [[-5, 4], [-11, 10], [-17, 16]],
                    [[-3, 2], [-9, 8], [-15, 14]],
                    [[-1, 0], [-7, 6], [-13, 12]],
                ]
            )
        ).all()

        img4 = geom.GeometricImage(jnp.arange(36).reshape((3, 3, 2, 2)), 0, 2)

        # k=2 flip
        img4_flipX = img4.times_group_element(flipX)
        assert img4_flipX.D == img4.D
        assert img4_flipX.parity == img4.parity
        assert img4_flipX.k == img4.k
        assert (
            img4_flipX.data
            == jnp.array(
                [
                    [  # first row
                        [[24, -25], [-26, 27]],
                        [[28, -29], [-30, 31]],
                        [[32, -33], [-34, 35]],
                    ],
                    [  # second row
                        [[12, -13], [-14, 15]],
                        [[16, -17], [-18, 19]],
                        [[20, -21], [-22, 23]],
                    ],
                    [  # third row
                        [[0, -1], [-2, 3]],
                        [[4, -5], [-6, 7]],
                        [[8, -9], [-10, 11]],
                    ],
                ]
            )
        ).all()

        img5 = geom.GeometricImage(jnp.arange(12).reshape((3, 4)), 0, 2)

        # non-square rotate
        img5_left90 = img5.times_group_element(left90)
        assert img5_left90.D == img5.D
        assert img5_left90.parity == img5.parity
        assert img5_left90.k == img5.k
        assert img5_left90.spatial_dims == (4, 3)
        assert (img5_left90.data == jnp.array([[3, 7, 11], [2, 6, 10], [1, 5, 9], [0, 4, 8]])).all()

        # non-square flip
        img5_flipX = img5.times_group_element(flipX)
        assert img5_flipX.parity == img5.parity
        assert img5_flipX.k == img5.k
        assert img5_flipX.spatial_dims == (3, 4)
        assert (img5_flipX.data == jnp.array([[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]])).all()

    def testTimesGroupElementEven(self):
        left90 = np.array([[0, -1], [1, 0]])
        flipX = np.array([[-1, 0], [0, 1]])

        img1 = geom.GeometricImage(jnp.arange(4).reshape((2, 2)), 0, 2)

        assert jnp.allclose(img1.times_group_element(left90).data, jnp.array([[1, 3], [0, 2]]))
        assert jnp.allclose(img1.times_group_element(flipX).data, jnp.array([[2, 3], [0, 1]]))

        img2 = geom.GeometricImage(jnp.arange(8).reshape((2, 2, 2)), 0, 2)

        assert jnp.allclose(
            img2.times_group_element(left90).data,
            jnp.array(
                [
                    [[-3, 2], [-7, 6]],
                    [[-1, 0], [-5, 4]],
                ]
            ),
        )
        assert jnp.allclose(
            img2.times_group_element(flipX).data,
            jnp.array(
                [
                    [[-4, 5], [-6, 7]],
                    [[0, 1], [-2, 3]],
                ]
            ),
        )

        img3 = geom.GeometricImage(jnp.arange(8).reshape((2, 4)), 0, 2)

        # non-square rotate
        img3_left90 = img3.times_group_element(left90)
        assert img3_left90.spatial_dims == (4, 2)
        assert jnp.allclose(img3_left90.data, jnp.array([[3, 7], [2, 6], [1, 5], [0, 4]]))

        # non-square flip
        img3_flipX = img3.times_group_element(flipX)
        assert img3_flipX.spatial_dims == (2, 4)
        assert jnp.allclose(img3_flipX.data, jnp.array([[4, 5, 6, 7], [0, 1, 2, 3]]))

    def testTimesGroupElement3D(self):
        D = 3
        flipDepth = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        left90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        img1 = geom.GeometricImage(jnp.arange(27).reshape((3, 3, 3)), 0, D)

        # basic flip
        img1_flipDepth = img1.times_group_element(flipDepth)
        assert img1_flipDepth.D == img1.D
        assert img1_flipDepth.parity == img1.parity
        assert img1_flipDepth.k == img1.k
        assert img1_flipDepth.spatial_dims == img1.spatial_dims
        assert jnp.allclose(
            img1_flipDepth.data,
            jnp.array(
                [
                    [
                        [18, 19, 20],
                        [21, 22, 23],
                        [24, 25, 26],
                    ],
                    [
                        [9, 10, 11],
                        [12, 13, 14],
                        [15, 16, 17],
                    ],
                    [
                        [0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8],
                    ],
                ]
            ),
        )

        # basic rotate
        img1_left90 = img1.times_group_element(left90)
        assert img1_left90.parity == img1.parity
        assert img1_left90.k == img1.k
        assert img1_left90.spatial_dims == img1.spatial_dims
        assert jnp.allclose(
            img1_left90.data,
            jnp.array(
                [
                    [
                        [2, 5, 8],
                        [1, 4, 7],
                        [0, 3, 6],
                    ],
                    [
                        [11, 14, 17],
                        [10, 13, 16],
                        [9, 12, 15],
                    ],
                    [
                        [20, 23, 26],
                        [19, 22, 25],
                        [18, 21, 24],
                    ],
                ]
            ),
        )

        img2 = geom.GeometricImage(jnp.arange(24).reshape((2, 3, 4)), 0, D)

        # non-square flip
        img2_flipDepth = img2.times_group_element(flipDepth)
        assert img2_flipDepth.parity == img2.parity
        assert img2_flipDepth.k == img2.k
        assert img2_flipDepth.spatial_dims == (2, 3, 4)
        assert jnp.allclose(
            img2_flipDepth.data,
            jnp.array(
                [
                    [
                        [12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23],
                    ],
                    [
                        [0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                    ],
                ]
            ),
        )

        # non-square rotate
        img2_left90 = img2.times_group_element(left90)
        assert img2_left90.parity == img2.parity
        assert img2_left90.k == img2.k
        assert img2_left90.spatial_dims == (2, 4, 3)
        assert jnp.allclose(
            img2_left90.data,
            jnp.array(
                [
                    [
                        [3, 7, 11],
                        [2, 6, 10],
                        [1, 5, 9],
                        [0, 4, 8],
                    ],
                    [
                        [15, 19, 23],
                        [14, 18, 22],
                        [13, 17, 21],
                        [12, 16, 20],
                    ],
                ]
            ),
        )

    def testMaxPoolUseNorm(self):
        image1 = geom.GeometricImage(
            jnp.array(
                [
                    [4, 1, 0, 1],
                    [0, 0, -3, 2],
                    [1, 0, 1, 0],
                    [1, 0, 2, 1],
                ],
                dtype=float,
            ),
            0,
            2,
        )

        img1_pool2 = image1.max_pool(2)
        assert img1_pool2.spatial_dims == (2, 2)
        assert img1_pool2.parity == 0
        assert img1_pool2.D == 2
        assert img1_pool2.k == 0
        assert img1_pool2.is_torus == (True,) * 2
        assert img1_pool2 == geom.GeometricImage(jnp.array([[4, -3], [1, 2]]), 0, 2)

        img1_pool4 = image1.max_pool(4)
        assert img1_pool4.spatial_dims == (1, 1)
        assert img1_pool4 == geom.GeometricImage(jnp.array([[4]]), 0, 2)

        image2 = geom.GeometricImage(
            jnp.array(
                [
                    [[1, 0], [0, -1], [1, 1], [3, 4]],
                    [[0, 0], [0, 1], [1, 0], [2, 0]],
                    [[-3, 4], [1, 0], [1, 0], [1, 0]],
                    [[1, 0], [1, 0], [1, 0], [-1, 0]],
                ]
            ),
            0,
            2,
        )

        img2_pool2 = image2.max_pool(2)
        assert img2_pool2.spatial_dims == (2, 2)
        assert img2_pool2 == geom.GeometricImage(
            jnp.array(
                [
                    [[1, 0], [3, 4]],
                    [[-3, 4], [1, 0]],
                ]
            ),
            0,
            2,
        )

        image3 = geom.GeometricImage(jnp.arange(4**3).reshape((4, 4, 4)), 0, 3)
        img3_pool2 = image3.max_pool(2)
        assert img3_pool2 == geom.GeometricImage(
            jnp.array(
                [
                    [[21, 23], [29, 31]],
                    [[53, 55], [61, 63]],
                ]
            ),
            0,
            3,
        )

    def testMaxPool(self):
        image1 = geom.GeometricImage(
            jnp.array(
                [
                    [4, 1, 0, 1],
                    [0, 0, -3, 2],
                    [1, 0, 1, 0],
                    [1, 0, 2, 1],
                ],
                dtype=float,
            ),
            0,
            2,
        )

        img1_pool2 = image1.max_pool(2, use_norm=False)
        assert img1_pool2.spatial_dims == (2, 2)
        assert img1_pool2.parity == 0
        assert img1_pool2.D == 2
        assert img1_pool2.k == 0
        assert img1_pool2.is_torus == (True,) * 2
        assert img1_pool2 == geom.GeometricImage(jnp.array([[4, 2], [1, 2]]), 0, 2)

        img1_pool4 = image1.max_pool(4)
        assert img1_pool4.spatial_dims == (1, 1)
        assert img1_pool4 == geom.GeometricImage(jnp.array([[4]]), 0, 2)

        image2 = geom.GeometricImage(
            jnp.array(
                [
                    [[1, 0], [0, -1], [1, 1], [3, 4]],
                    [[0, 0], [0, 1], [1, 0], [2, 0]],
                    [[-3, 4], [1, 0], [1, 0], [1, 0]],
                    [[1, 0], [1, 0], [1, 0], [-1, 0]],
                ]
            ),
            0,
            2,
        )

        with pytest.raises(AssertionError):
            image2.max_pool(2, use_norm=False)

        image3 = geom.GeometricImage(jnp.arange(4**3).reshape((4, 4, 4)), 0, 3)
        img3_pool2 = image3.max_pool(2, use_norm=False)
        assert img3_pool2 == geom.GeometricImage(
            jnp.array(
                [
                    [[21, 23], [29, 31]],
                    [[53, 55], [61, 63]],
                ]
            ),
            0,
            3,
        )

        # TODO: add tests for using comparator image

    def testAveragePool(self):
        image1 = geom.GeometricImage(
            jnp.array(
                [
                    [4, 1, 0, 1],
                    [0, 0, -3, 2],
                    [1, 0, 1, 0],
                    [1, 0, 2, 1],
                ],
                dtype=float,
            ),
            0,
            2,
        )

        img1_pool2 = image1.average_pool(2)
        assert img1_pool2.spatial_dims == (2, 2)
        assert img1_pool2.parity == 0
        assert img1_pool2.D == 2
        assert img1_pool2.k == 0
        assert img1_pool2.is_torus == (True,) * 2
        assert img1_pool2 == geom.GeometricImage(jnp.array([[1.25, 0], [0.5, 1]]), 0, 2)

        img1_pool4 = image1.average_pool(4)
        assert img1_pool4.spatial_dims == (1, 1)
        assert img1_pool4 == geom.GeometricImage(jnp.array([[11 / 16]]), 0, 2)

        image2 = geom.GeometricImage(
            jnp.array(
                [
                    [[1, 0], [0, -1], [1, 1], [3, 4]],
                    [[0, 0], [0, 1], [1, 0], [2, 0]],
                    [[-3, 4], [1, 0], [1, 0], [1, 0]],
                    [[1, 0], [1, 0], [1, 0], [-1, 0]],
                ]
            ),
            0,
            2,
        )

        img2_pool2 = image2.average_pool(2)
        assert img2_pool2.spatial_dims == (2, 2)
        assert img2_pool2 == geom.GeometricImage(
            jnp.array(
                [
                    [[0.25, 0], [1.75, 1.25]],
                    [[0, 1], [0.5, 0]],
                ]
            ),
            0,
            2,
        )

        image3 = geom.GeometricImage(jnp.arange(4**3).reshape((4, 4, 4)), 0, 3)
        img3_pool2 = image3.average_pool(2)
        assert img3_pool2 == geom.GeometricImage(
            jnp.array(
                [
                    [[10.5, 12.5], [18.5, 20.5]],
                    [[42.5, 44.5], [50.5, 52.5]],
                ]
            ),
            0,
            3,
        )

    def testUnpool(self):
        image1 = geom.GeometricImage(
            jnp.array(
                [
                    [4, -3],
                    [1, 2],
                ],
                dtype=float,
            ),
            0,
            2,
        )

        unpooled_image1 = image1.unpool(2)
        assert unpooled_image1.spatial_dims == tuple(2 * N for N in image1.spatial_dims)
        assert unpooled_image1.parity == image1.parity
        assert unpooled_image1.D == image1.D
        assert unpooled_image1.is_torus == image1.is_torus
        assert unpooled_image1.k == image1.k
        assert unpooled_image1 == geom.GeometricImage(
            jnp.array(
                [
                    [4, 4, -3, -3],
                    [4, 4, -3, -3],
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                ],
                dtype=float,
            ),
            0,
            2,
        )

        unpooled_image2 = image1.unpool(3)
        assert unpooled_image2.spatial_dims == tuple(3 * N for N in image1.spatial_dims)
        assert unpooled_image2 == geom.GeometricImage(
            jnp.array(
                [
                    [4, 4, 4, -3, -3, -3],
                    [4, 4, 4, -3, -3, -3],
                    [4, 4, 4, -3, -3, -3],
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                ],
                dtype=float,
            ),
            0,
            2,
        )

        # D=3
        image2 = geom.GeometricImage(
            jnp.array(
                [
                    [[1, 3], [-2, 5]],
                    [[0, 4], [1, 0.3]],
                ]
            ),
            0,
            3,
        )

        unpooled_image3 = image2.unpool(2)
        assert unpooled_image3.spatial_dims == tuple(2 * N for N in image2.spatial_dims)
        assert unpooled_image3.D == image2.D
        assert unpooled_image3 == geom.GeometricImage(
            jnp.array(
                [
                    [[1, 1, 3, 3], [1, 1, 3, 3], [-2, -2, 5, 5], [-2, -2, 5, 5]],
                    [[1, 1, 3, 3], [1, 1, 3, 3], [-2, -2, 5, 5], [-2, -2, 5, 5]],
                    [[0, 0, 4, 4], [0, 0, 4, 4], [1, 1, 0.3, 0.3], [1, 1, 0.3, 0.3]],
                    [[0, 0, 4, 4], [0, 0, 4, 4], [1, 1, 0.3, 0.3], [1, 1, 0.3, 0.3]],
                ]
            ),
            0,
            3,
        )

        # k=1
        image3 = geom.GeometricImage(
            jnp.array(
                [
                    [[1, 0], [3, 4]],
                    [[-3, 4], [1, 0]],
                ]
            ),
            0,
            2,
        )

        unpooled_image4 = image3.unpool(2)
        assert unpooled_image4.spatial_dims == tuple(2 * N for N in image3.spatial_dims)
        assert unpooled_image4.k == image3.k
        assert unpooled_image4 == geom.GeometricImage(
            jnp.array(
                [
                    [[1, 0], [1, 0], [3, 4], [3, 4]],
                    [[1, 0], [1, 0], [3, 4], [3, 4]],
                    [[-3, 4], [-3, 4], [1, 0], [1, 0]],
                    [[-3, 4], [-3, 4], [1, 0], [1, 0]],
                ]
            ),
            0,
            2,
        )
