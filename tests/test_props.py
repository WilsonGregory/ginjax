import time
import itertools as it
import pytest

import ginet.geometric as geom
import ginet.ml as ml
import jax.numpy as jnp
from jax import random, vmap
import jax.lax


class TestPropositions:
    # Class to test various propositions, mostly about the GeometricImage

    def testContractionOrderInvariance(self):
        # Test that the order of the two parameters of contraction does not matter
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=(3, 3, 2, 2)), 0, 2)
        assert img1.contract(0, 1) == img1.contract(1, 0)

        key, subkey = random.split(key)
        img2 = geom.GeometricImage(random.normal(subkey, shape=(3, 3, 2, 2, 2)), 0, 2)
        assert img2.contract(0, 1) == img2.contract(1, 0)
        assert img2.contract(0, 2) == img2.contract(2, 0)
        assert img2.contract(1, 2) == img2.contract(2, 1)

    def testSerialContractionInvariance(self):
        # Test that order of contractions performed in series does not matter
        key = random.PRNGKey(0)
        img1 = geom.GeometricImage(random.normal(key, shape=(3, 3, 2, 2, 2, 2, 2)), 0, 2)
        assert jnp.allclose(
            img1.multicontract(((0, 1), (2, 3))).data,
            img1.multicontract(((2, 3), (0, 1))).data,
        )
        assert jnp.allclose(
            img1.multicontract(((0, 1), (3, 4))).data,
            img1.multicontract(((3, 4), (0, 1))).data,
        )
        assert jnp.allclose(
            img1.multicontract(((1, 2), (3, 4))).data,
            img1.multicontract(((3, 4), (1, 2))).data,
        )
        assert jnp.allclose(
            img1.multicontract(((1, 4), (2, 3))).data,
            img1.multicontract(((1, 4), (2, 3))).data,
        )

    def testContractSwappableIndices(self):
        # Test that convolving with a k=2, parity=0 invariant filter means that we can contract on either
        # filter index and it is the same.
        key = random.PRNGKey(time.time_ns())
        N = 3
        D = 2
        img_k = 3
        operators = geom.make_all_operators(D)
        conv_filters = geom.get_unique_invariant_filters(N, 2, 0, D, operators)

        img1 = geom.GeometricImage(random.normal(key, shape=((N,) * D + (D,) * img_k)), 0, D)
        for conv_filter in conv_filters:
            convolved_img = img1.convolve_with(conv_filter)
            for i in range(img_k):
                assert convolved_img.contract(i, 3) == convolved_img.contract(i, 4)

        for conv_filter1 in conv_filters:
            for conv_filter2 in conv_filters:
                prod_img = img1.convolve_with(conv_filter1) * img1.convolve_with(conv_filter2)
                for i in [0, 1, 2, 5, 6, 7]:
                    assert prod_img.contract(i, 3) == prod_img.contract(i, 4)
                    assert prod_img.contract(i, 8) == prod_img.contract(i, 9)

    def testConvolutionLinearity(self):
        """
        For scalars alpha, beta, tensor images image1, image2 and filter c1 alpha
        """
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        image1 = geom.GeometricImage(random.uniform(subkey, shape=(3, 3, 2)), 0, 2)

        key, subkey = random.split(key)
        image2 = geom.GeometricImage(random.uniform(subkey, shape=(3, 3, 2)), 0, 2)

        key, subkey = random.split(key)
        c1 = geom.GeometricFilter(random.uniform(subkey, shape=(3, 3, 2)), 0, 2)

        alpha, beta = random.uniform(subkey, shape=(2,))

        B1 = image1.convolve_with(c1) * alpha + image2.convolve_with(c1) * beta
        B2 = (image1 * alpha + image2 * beta).convolve_with(c1)

        assert B1.shape() == B2.shape()
        assert B1.parity == B2.parity
        assert jnp.allclose(B1.data, B2.data)

    def testConvolveConvolveCommutativity(self):
        # Test that performing two contractions is the same (under transposition) no matter the order
        key = random.PRNGKey(time.time_ns())
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=(3, 3, 2)), 0, 2)

        key, subkey = random.split(key)
        c1 = geom.GeometricFilter(random.normal(subkey, shape=(3, 3, 2)), 1, 2)

        key, subkey = random.split(key)
        c2 = geom.GeometricFilter(random.normal(subkey, shape=(3, 3, 2, 2)), 0, 2)

        B1 = img1.convolve_with(c1).convolve_with(c2)
        B2 = img1.convolve_with(c2).convolve_with(c1)

        assert B1.D == B2.D
        assert B1.spatial_dims == B2.spatial_dims
        assert B1.parity == B2.parity
        assert B1.transpose([0, 2, 3, 1]) == B2

    def testOuterProductCommutativity(self):
        # Test that the tensor product is commutative under transposition, including if there are convolves in there.
        key = random.PRNGKey(time.time_ns())
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=(3, 3, 2)), 0, 2)

        key, subkey = random.split(key)
        c1 = geom.GeometricFilter(random.normal(subkey, shape=(3, 3, 2)), 1, 2)

        key, subkey = random.split(key)
        c2 = geom.GeometricFilter(random.normal(subkey, shape=(3, 3, 2, 2)), 0, 2)

        B1 = img1.convolve_with(c1) * img1.convolve_with(c2)
        B2 = img1.convolve_with(c2) * img1.convolve_with(c1)

        assert B1.D == B2.D
        assert B1.spatial_dims == B2.spatial_dims
        assert B1.parity == B2.parity
        assert B1.transpose([2, 3, 4, 0, 1]) == B2

    def testOuterProductFilterInvariance(self):
        # Test that the outer product of two invariant filters is also invariant
        D = 2
        group_operators = geom.make_all_operators(D)
        all_filters = geom.get_invariant_filters_list([3], [0, 1, 2], [0, 1], D, group_operators)
        for g in group_operators:
            for c1 in all_filters:
                for c2 in all_filters:
                    assert (c1 * c2).times_group_element(g, precision=jax.lax.Precision.HIGH) == (
                        c1 * c2
                    )

    def testKroneckerAdd(self):
        # Test that multiplying by the kronecker delta symbol, then contracting on those new indices
        # merely scales the original image by D
        N = 3
        D = 2
        k = 3
        kron_delta_k = 2

        key = random.PRNGKey(time.time_ns())
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=((N,) * D + (D,) * k)), 0, D)
        kron_delta_img = geom.get_kronecker_delta_image(N, D, kron_delta_k)

        expanded_img1 = img1 * kron_delta_img
        assert expanded_img1.k == k + kron_delta_k
        assert jnp.allclose(expanded_img1.contract(3, 4).data, (img1 * D).data)

        # Multiplying by K-D then contracting on exactly one K-D index returns the original, up to a transpose of axes
        assert jnp.allclose(expanded_img1.contract(0, 3).transpose((2, 0, 1)).data, (img1).data)

        D = 3
        key, subkey = random.split(key)
        img2 = geom.GeometricImage(random.normal(subkey, shape=((N,) * D + (D,) * k)), 0, D)
        kron_delta_img = geom.get_kronecker_delta_image(N, D, kron_delta_k)

        expanded_img2 = img2 * kron_delta_img
        assert expanded_img2.k == k + kron_delta_k

        assert expanded_img2.contract(3, 4) == (img2 * D)

    def testInvariantFilter(self):
        # For every invariant filter of order k, there exists an invariant filter of order k+2 where contracting on
        # some pair of tensor indices results in the invariant filter of order k.
        D = 2
        N = 3
        group_operators = geom.make_all_operators(D)
        max_k = 5

        conv_filters_dict, _ = geom.get_invariant_filters_dict(
            [N],
            range(max_k + 1),
            [0, 1],
            D,
            group_operators,
            scale="one",
        )
        for k in range(max_k - 1):
            for parity in [0, 1]:
                for conv_filter in conv_filters_dict[(D, N, k, parity)]:
                    found_match = False
                    for upper_conv_filter in conv_filters_dict[(D, N, k + 2, parity)]:
                        for i, j in it.combinations(range(k + 2), 2):
                            contracted_filter = upper_conv_filter.contract(i, j)
                            datablock = jnp.stack(
                                [
                                    conv_filter.data.flatten(),
                                    contracted_filter.data.flatten(),
                                ]
                            )
                            s = jnp.linalg.svd(datablock, compute_uv=False)
                            if jnp.sum(s > geom.TINY) != 1:  # they're the same
                                found_match = True
                                break

                        if found_match:
                            break

                    assert found_match

    def testContractConvolve(self):
        # Test that contracting then convolving is the same as convolving, then contracting
        N = 3
        D = 2
        k = 3

        key = random.PRNGKey(time.time_ns())
        key, subkey = random.split(key)
        img1 = geom.GeometricImage(random.normal(subkey, shape=((N,) * D + (D,) * k)), 0, D)

        key, subkey = random.split(key)
        c1 = geom.GeometricFilter(random.normal(subkey, shape=(3, 3, 2)), 0, 2)

        key, subkey = random.split(key)
        c2 = geom.GeometricFilter(random.normal(subkey, shape=(3, 3, 2, 2)), 0, 2)

        for c in [c1, c2]:
            for i, j in it.combinations(range(k), 2):
                assert img1.contract(i, j).convolve_with(c) == img1.convolve_with(c).contract(i, j)

    def testDiagEquivalence(self):
        # test that the tensor product and contraction is indeed the diag operator
        N = 3
        D = 2
        k = 1

        key = random.PRNGKey(time.time_ns())
        data = random.normal(key, shape=(N,) * D + (D,) * k)
        flattened_data = data.reshape((N**D,) + (D,) * k)

        kd_3 = geom.KroneckerDeltaSymbol.get(D, 3)
        assert jnp.allclose(
            vmap(jnp.diag)(flattened_data),
            vmap(lambda vec: geom.multicontract(jnp.tensordot(vec, kd_3, axes=0), ((0, 1),)))(
                flattened_data
            ),
        )

        D = 3
        data = random.normal(key, shape=(N,) * D + (D,) * k)
        flattened_data = data.reshape((N**D,) + (D,) * k)

        kd_3 = geom.KroneckerDeltaSymbol.get(D, 3)
        assert jnp.allclose(
            vmap(jnp.diag)(flattened_data),
            vmap(lambda vec: geom.multicontract(jnp.tensordot(vec, kd_3, axes=0), ((0, 1),)))(
                flattened_data
            ),
        )

    def testFrobeniusNormEquivariance(self):
        key = random.PRNGKey(0)
        for D in [2, 3]:
            operators = geom.make_all_operators(D)
            for parity in [0, 1]:
                for k in [0, 1, 2, 3]:
                    key, subkey = random.split(key)
                    tensor = random.normal(subkey, shape=((1,) * D + (D,) * k))

                    # assert that norm is equivariant
                    for gg in operators:
                        assert jnp.allclose(
                            jnp.linalg.norm(tensor),
                            jnp.linalg.norm(
                                geom.times_group_element(
                                    D, tensor, parity, gg, jax.lax.Precision.HIGHEST
                                ),
                            ),
                        )

                    # assert that norm is equivalent to prod, followed by the specific contraction
                    idxs = tuple((i, j) for i, j in zip(range(k), range(k, 2 * k)))
                    assert jnp.allclose(
                        jnp.linalg.norm(tensor),
                        jnp.sqrt(
                            geom.multicontract(geom.mul(D, tensor, tensor), idxs, idx_shift=D)[0, 0]
                        ),
                    )

    def testNormEquivariance(self):
        # This follows from the previous test, but just to be sure
        key = random.PRNGKey(0)
        N = 5
        prec = jax.lax.Precision.HIGHEST
        for D in [2, 3]:
            operators = geom.make_all_operators(D)
            for parity in [0, 1]:
                for k in [0, 1, 2, 3]:
                    key, subkey = random.split(key)
                    image = geom.GeometricImage(
                        random.normal(subkey, shape=(N,) * D + (D,) * k), parity, D
                    )

                    # assert that norm is equivariant
                    for gg in operators:
                        first = image.norm().times_group_element(gg, prec)
                        second = image.times_group_element(gg, prec).norm()
                        assert first == second

    def testNormEquivarianceMultiImage(self):
        # Same test but for multi_images
        key = random.PRNGKey(0)
        N = 5
        batch = 2
        channels = 3

        vmap_times_gg = jax.vmap(
            jax.vmap(geom.times_group_element, in_axes=(None, 0, None, None, None)),
            in_axes=(None, 0, None, None, None),
        )

        prec = jax.lax.Precision.HIGHEST
        for D in [2, 3]:
            operators = geom.make_all_operators(D)
            for parity in [0, 1]:
                for k in [0, 1, 2, 3]:
                    key, subkey = random.split(key)
                    multi_image = geom.MultiImage(
                        {
                            (k, parity): random.normal(
                                subkey, shape=(batch, channels) + (N,) * D + (D,) * k
                            )
                        },
                        D,
                    )

                    # assert that norm is equivariant
                    for gg in operators:
                        first = vmap_times_gg(
                            D, geom.norm(D + 2, multi_image[(k, parity)]), 0, gg, prec
                        )
                        second = geom.norm(
                            D + 2, multi_image.times_group_element(gg, prec)[(k, parity)]
                        )
                        assert jnp.allclose(first, second)

    def testMaxPoolEquivariance(self):
        N = 6
        key = random.PRNGKey(0)
        for D in [2, 3]:
            operators = geom.make_all_operators(D)
            for parity in [0, 1]:
                for k in [0, 1, 2, 3]:
                    key, subkey = random.split(key)
                    image = random.normal(subkey, shape=((N,) * D + (D,) * k))

                    # assert that norm is equivariant
                    for gg in operators:
                        first = geom.times_group_element(
                            D,
                            geom.max_pool(D, image, 2),
                            parity,
                            gg,
                            precision=jax.lax.Precision.HIGH,
                        )
                        second = geom.max_pool(
                            D,
                            geom.times_group_element(
                                D, image, parity, gg, precision=jax.lax.Precision.HIGH
                            ),
                            2,
                        )
                        assert jnp.allclose(first, second), f"{jnp.max(jnp.abs(first - second))}"

    def testLayerNormEquivariance(self):
        N = 3
        channels = 2
        prec = jax.lax.Precision.HIGHEST
        key = random.PRNGKey(time.time_ns())
        ks = [0, 1]
        parities = [0, 1]

        for D in [2, 3]:
            key, *subkeys = random.split(key, num=(len(ks) * len(parities)) + 1)
            input_keys = tuple(((k, v), channels) for k, v in it.product(ks, parities))

            data = {
                (k, parity): random.normal(subkey, shape=((channels,) + (N,) * D + (D,) * k))
                for subkey, ((k, parity), _) in zip(subkeys, input_keys)
            }
            multi_image = geom.MultiImage(data, D)
            layer_norm = ml.LayerNorm(multi_image.get_signature(), D, eps=0)

            # assert that layer norm (group_norm with groups=1) is equivariant
            for gg in geom.make_all_operators(D):
                multi_image1 = layer_norm(multi_image).times_group_element(gg, precision=prec)
                multi_image2 = layer_norm(multi_image.times_group_element(gg, precision=prec))
                assert multi_image1.__eq__(multi_image2, rtol=1e-3, atol=1e-2)

    def testLayerNormWhitening(self):
        """
        Show that Layer Norm does center and scale the vectors.
        """
        N = 3
        channels = 2
        key = random.PRNGKey(time.time_ns())
        for D in [2, 3]:
            key, subkey = random.split(key)
            image_block = random.normal(subkey, shape=(channels,) + (N,) * D + (D,))

            whitened_data = ml.layers._group_norm_K1(D, image_block, 1, eps=0)

            # mean centered
            assert jnp.allclose(jnp.mean(whitened_data), 0, atol=geom.TINY, rtol=geom.TINY)

            # identity covariance
            cov = jnp.cov(whitened_data.reshape((-1, D)), rowvar=False, bias=True)
            assert jnp.allclose(cov, jnp.eye(D), atol=1e-2, rtol=1e-2), f"{cov}"

    def testVNNonlinearEquivariance(self):
        N = 5
        in_c = 10
        prec = jax.lax.Precision.HIGHEST
        key = random.PRNGKey(0)

        ks = [0, 1, 2]
        parities = [0, 1]
        for D in [2, 3]:
            key, *subkeys = random.split(key, num=(len(ks) * len(parities)) + 1)

            data = {
                (k, parity): random.normal(subkey, shape=((in_c,) + (N,) * D + (D,) * k))
                for subkey, (k, parity) in zip(subkeys, it.product(ks, parities))
            }
            multi_image = geom.MultiImage(data, D)

            key, subkey = random.split(key)
            vn_nonlinear = ml.VectorNeuronNonlinear(
                multi_image.get_signature(), D, eps=0, key=subkey
            )

            # assert that the vn nonlinearity is equivariant
            for gg in geom.make_all_operators(D):
                multi_image1 = vn_nonlinear(multi_image).times_group_element(gg, precision=prec)
                multi_image2 = vn_nonlinear(multi_image.times_group_element(gg, precision=prec))
                assert multi_image1.__eq__(multi_image2, rtol=1e-3, atol=1e-2)
