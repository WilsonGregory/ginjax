import time
import itertools as it
import numpy as np
from typing_extensions import Self
import pytest

import jax.numpy as jnp
from jax import random
import jax
import equinox as eqx

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models


class TestModels:
    # Class to test the functions in the models.py file

    def testFastConvolve(self):
        D = 2
        M = 3
        N = 5
        in_c = 3
        out_c = 4
        max_k = 2  # k=0,1, or 2
        ks = [(False,) * k for k in list(range(max_k + 1))]  # TODO: test w/ covariant indices
        parities = [0, 1]
        ks_ps_prod = list(it.product(ks, parities))
        key = random.PRNGKey(time.time_ns())

        # Intentionally missing k=3,4 because there are tons of those.
        conv_filters = geom.get_invariant_filters(
            [M], [0, 1, 2], parities, D, geom.make_all_operators(D)
        )

        # power set (excluding empty set) of possible in_k, out_k and parity
        powerset = list(
            it.chain.from_iterable(
                it.combinations(ks_ps_prod, r + 1) for r in range(len(ks_ps_prod))
            )
        )
        for in_ks_ps in powerset:
            for out_ks_ps in powerset:
                input_keys = geom.Signature(tuple((in_key, in_c) for in_key in in_ks_ps))
                target_keys = geom.Signature(tuple((out_key, out_c) for out_key in out_ks_ps))

                key, *subkeys = random.split(key, num=len(input_keys) + 1)
                multi_image = geom.MultiImage(
                    {
                        (k, p): random.normal(subkeys[i], shape=(in_c,) + (N,) * D + (D,) * len(k))
                        for i, ((k, p), _) in enumerate(input_keys)
                    },
                    D,
                )

                key, subkey = random.split(key)
                conv = ml.ConvContract(
                    input_keys, target_keys, conv_filters, use_bias=False, key=subkey
                )
                if conv.missing_filter:
                    continue

                assert conv.fast_convolve(multi_image, conv.weights) == conv.individual_convolve(
                    multi_image, conv.weights
                )

    def testConvContractEquivariance2D(self):
        D = 2
        M = 3
        N = 5
        in_c = 3
        out_c = 4
        max_k = 2  # k=0,1, or 2
        ks = [(False,) * k for k in list(range(max_k + 1))]
        parities = [0, 1]
        ks_ps_prod = list(it.product(ks, parities))
        key = random.PRNGKey(time.time_ns())

        # filters we only want contravariant indices
        operators = geom.make_all_operators(D)
        conv_filters = geom.get_invariant_filters([M], [0, 1, 2], parities, D, operators)

        # power set (excluding empty set) of possible in_k, out_k and parity
        powerset = list(
            it.chain.from_iterable(
                it.combinations(ks_ps_prod, r + 1) for r in range(len(ks_ps_prod))
            )
        )
        for in_ks_ps in powerset:
            for out_ks_ps in powerset:
                input_keys = geom.Signature(tuple((in_key, in_c) for in_key in in_ks_ps))
                target_keys = geom.Signature(tuple((out_key, out_c) for out_key in out_ks_ps))

                key, *subkeys = random.split(key, num=len(input_keys) + 1)
                multi_image = geom.MultiImage(
                    {
                        (k, p): random.normal(subkeys[i], shape=(in_c,) + (N,) * D + (D,) * len(k))
                        for i, ((k, p), _) in enumerate(input_keys)
                    },
                    D,
                    True,
                )

                key, subkey = random.split(key)
                conv = ml.ConvContract(
                    input_keys, target_keys, conv_filters, use_bias=False, key=subkey
                )
                if conv.missing_filter:
                    continue

                for gg in operators:
                    first = conv(multi_image.times_gg_precise(gg))
                    second = conv(multi_image).times_gg_precise(gg)
                    assert first.__eq__(second, 1e-4, 1e-4)

    def testConvContractMetricEquivariance2D(self):
        D = 2
        M = 3
        N = 5
        in_c = 3

        key = random.PRNGKey(time.time_ns())

        key, subkey1, subkey2 = random.split(key, num=3)
        eigvecs = random.orthogonal(subkey1, D, shape=(N,) * D)
        eigvals = jax.vmap(jnp.diag)(random.uniform(subkey2, shape=(N**D, D)) + 0.5)
        metric_tensor_data = jnp.einsum(
            "...ij,...jk,...kl->...il",
            eigvecs,
            eigvals.reshape((N,) * D + (D, D)),
            jnp.moveaxis(eigvecs, -1, -2),
        )
        metric_tensor = geom.GeometricImage(metric_tensor_data, 0, D, covariant_axes=(True, True))
        metric_tensor_inv = geom.get_metric_inverse(metric_tensor)

        # filters we only want contravariant indices
        operators = geom.make_all_operators(D)
        conv_filters = geom.get_invariant_filters([M], [0, 1, 2], [0, 1], D, operators)

        # small test
        key, subkey = random.split(key)
        multi_image1 = geom.MultiImage(
            {((False,), 0): random.normal(subkey, shape=(in_c,) + (N,) * D + (D,))},
            D,
            True,
            metric_tensor,
            metric_tensor_inv,
        )
        key, subkey = random.split(key)
        out_signature = geom.Signature(((((True,), 0), in_c),))
        conv = ml.ConvContract(
            multi_image1.get_signature(), out_signature, conv_filters, use_bias=False, key=subkey
        )

        for gg in operators:
            first = conv(multi_image1.times_gg_precise(gg))
            second = conv(multi_image1).times_gg_precise(gg)
            assert first.get_signature() == second.get_signature() == out_signature
            assert first.__eq__(second, 1e-4, 1e-4)

        # big test
        ks = [(), (True,), (False,), (True, True), (True, False), (False, True), (False, False)]
        parities = [0, 1]
        ks_ps_prod = list(it.product(ks, parities))
        key, *subkeys = random.split(key, num=len(ks_ps_prod) + 1)
        multi_image2 = geom.MultiImage(
            {
                (k, p): random.normal(subkeys[i], shape=(in_c,) + (N,) * D + (D,) * len(k))
                for i, (k, p) in enumerate(ks_ps_prod)
            },
            D,
            True,
            metric_tensor,
            metric_tensor_inv,
        )

        key, subkey = random.split(key)
        conv = ml.ConvContract(
            multi_image2.get_signature(),
            multi_image2.get_signature(),
            conv_filters,
            use_bias=False,
            key=subkey,
        )

        for gg in operators:
            first = conv(multi_image2.times_gg_precise(gg))
            second = conv(multi_image2).times_gg_precise(gg)
            assert first.__eq__(second, 1e-4, 1e-4)

        with pytest.raises(AssertionError):
            key, subkey = random.split(key)
            conv = ml.ConvContract(
                multi_image2.get_signature(),
                multi_image2.get_signature(),
                conv_filters,
                use_bias=False,
                padding=0,  # image will be smaller, this causes a problem with metric tensor
                key=subkey,
            )
            conv(multi_image2)

    def testConvContractContraCovEquivalence(self):
        # This experiments tests whether a convolution conv(A_up) == conv(A_down)
        # conv(A_down) does A_down * C_up where * is convolution, C_up is the filter
        # conv(A_up) does A_up -> A_down -> C_up
        D = 2
        M = 3
        N = 5
        in_c = 3

        key = random.PRNGKey(0)

        key, subkey1, subkey2 = random.split(key, num=3)
        eigvecs = random.orthogonal(subkey1, D, shape=(N,) * D)
        eigvals = jax.vmap(jnp.diag)(random.uniform(subkey2, shape=(N**D, D)) + 0.5)
        metric_tensor_data = jnp.einsum(
            "...ij,...jk,...kl->...il",
            eigvecs,
            eigvals.reshape((N,) * D + (D, D)),
            jnp.moveaxis(eigvecs, -1, -2),
        )
        metric_tensor = geom.GeometricImage(metric_tensor_data, 0, D, covariant_axes=(True, True))
        metric_tensor_inv = geom.get_metric_inverse(metric_tensor)

        # filters we only want contravariant indices
        operators = geom.make_all_operators(D)
        conv_filters = geom.get_invariant_filters([M], [0, 1, 2, 3], [0, 1], D, operators)

        # small test
        key, subkey = random.split(key)
        multi_image1 = geom.MultiImage(
            {((False,), 0): random.normal(subkey, shape=(in_c,) + (N,) * D + (D,))},
            D,
            True,
            metric_tensor,
            metric_tensor_inv,
        )

        multi_image2 = multi_image1.lower_all(jax.lax.Precision.HIGHEST)
        assert multi_image1 != multi_image2  # these two images are different

        key, subkey = random.split(key)
        # define ConvContract lower -> upper
        conv = ml.ConvContract(
            multi_image2.get_signature(),
            multi_image1.get_signature(),
            conv_filters,
            use_bias=False,
            key=subkey,
        )

        # this conv goes upper -> upper
        conv_upper = ml.ConvContract(
            multi_image1.get_signature(),
            multi_image1.get_signature(),
            conv_filters,
            use_bias=False,
            key=subkey,
        )
        # set the weights so they are the same
        conv_upper.weights[((False,), 0)][((False,), 0)] = conv.weights[((True,), 0)][((False,), 0)]
        first = conv_upper(multi_image1)
        second = conv(multi_image2)
        assert first == second

        # more complicated example with multiple axes
        key, subkey = random.split(key)
        multi_image3 = geom.MultiImage(
            {((True, False), 0): random.normal(subkey, shape=(in_c,) + (N,) * D + (D, D))},
            D,
            True,
            metric_tensor,
            metric_tensor_inv,
        )
        multi_image4 = multi_image3.lower_all(jax.lax.Precision.HIGHEST)

        # conv for all lower
        conv = ml.ConvContract(
            multi_image4.get_signature(),
            multi_image2.get_signature(),  # lower vector output
            conv_filters,
            use_bias=False,
            key=subkey,
        )
        conv_mixed = ml.ConvContract(
            multi_image3.get_signature(),
            multi_image2.get_signature(),  # lower vector output
            conv_filters,
            use_bias=False,
            key=subkey,
        )
        conv_mixed.weights[((True, False), 0)][((True,), 0)] = conv.weights[((True, True), 0)][
            ((True,), 0)
        ]
        first = conv_mixed(multi_image3)
        second = conv(multi_image4)
        assert first == second

    def testGroupAverageIsEquivariant(self):
        D = 2
        N = 16
        c = 5
        key = random.PRNGKey(0)
        operators = geom.make_all_operators(D)

        key, subkey1, subkey2 = random.split(key, num=3)
        multi_image_x = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(c,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(c,) + (N,) * D + (D,)),
            },
            D,
        )

        key, subkey1, subkey2 = random.split(key, num=3)
        multi_image_y = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(1,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(1,) + (N,) * D + (D,)),
            },
            D,
        )

        key, subkey = random.split(key)
        always_model = models.GroupAverage(
            models.ResNet(
                D,
                multi_image_x.get_signature(),
                multi_image_y.get_signature(),
                depth=c,
                num_blocks=4,
                equivariant=False,
                kernel_size=3,
                key=subkey,
            ),
            operators,
            always_average=True,
        )

        for gg in operators:
            first, _ = always_model(multi_image_x.times_group_element(gg))
            second = always_model(multi_image_x)[0].times_group_element(gg)
            assert first.__eq__(second, rtol=1e-3, atol=1e-3)

        key, subkey = random.split(key)
        model = models.GroupAverage(
            models.ResNet(
                D,
                multi_image_x.get_signature(),
                multi_image_y.get_signature(),
                depth=c,
                num_blocks=4,
                equivariant=False,
                kernel_size=3,
                key=subkey,
            ),
            operators,
        )
        inference_model = eqx.nn.inference_mode(model)
        assert isinstance(inference_model, models.MultiImageModule)

        for gg in operators:
            first, _ = inference_model(multi_image_x.times_group_element(gg))
            second = inference_model(multi_image_x)[0].times_group_element(gg)
            assert first.__eq__(second, rtol=1e-3, atol=1e-3)

    def testClimate1d(self):
        D = 2
        N = 16
        past_steps = 2
        c = 5
        key = random.PRNGKey(0)

        key, subkey1, subkey2 = random.split(key, num=3)
        x = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(c * past_steps,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(c * past_steps,) + (N,) * D + (D,)),
            },
            D,
            (True, False),
        )
        output_keys = x.get_signature()
        spatial_dims = x.get_spatial_dims()
        output_keys_1d = models.Climate1D.get_1d_signature(output_keys, spatial_dims[1])

        # test that to1d and from1d are inverses
        model = models.Climate1D(
            models.ModelWrapper(1, eqx.nn.Identity(), output_keys_1d, True),
            output_keys,
            past_steps,
            1,
            spatial_dims,
            {},
        )

        out = model.from1d(model.model(model.to1d(x), None)[0])
        assert out == x

        # test that the Climate1D model is equivariant to the equator flip
        key, subkey = random.split(key)
        mlp = eqx.nn.MLP(x.size(), x.size(), 64, 2, key=subkey)

        class MLPReshapeModule(eqx.Module):
            mlp: eqx.Module
            N: int

            def __init__(self: Self, mlp, N: int):
                self.mlp = mlp
                self.N = N

            def __call__(self: Self, x: jax.Array) -> jax.Array:
                assert callable(self.mlp)
                return self.mlp(x.reshape(-1)).reshape((-1, self.N))

        model = models.Climate1D(
            models.ModelWrapper(1, MLPReshapeModule(mlp, N), output_keys_1d, True),
            output_keys,
            past_steps,
            1,
            spatial_dims,
            {},
        )

        equator_flip = np.array([[1, 0], [0, -1]])
        first = model(x.times_group_element(equator_flip))[0]
        second = model(x)[0].times_group_element(equator_flip)
        assert first.__eq__(second, rtol=1e-3, atol=1e-3)

        # test that the conversion to 1d preserves longitude flips
        longitude_flip = np.array([[-1, 0], [0, 1]])
        longitude_flip_1d = np.array([[-1]])
        first = model.to1d(x.times_group_element(longitude_flip))
        second = model.to1d(x).times_group_element(longitude_flip_1d)
        assert first.__eq__(second, rtol=1e-3, atol=1e-3)

        first = model.from1d(first)
        second = model.from1d(second)
        assert first.__eq__(second, rtol=1e-3, atol=1e-3)

    def testClimate1dConstantFields(self):
        D = 2
        N = 16
        past_steps = 2
        c = 5
        key = random.PRNGKey(0)

        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)
        x = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(c * past_steps,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(c * past_steps,) + (N,) * D + (D,)),
            },
            D,
            (True, False),
        )
        const_x = geom.MultiImage(
            {
                (0, 0): random.normal(subkey3, shape=(1,) + (N,) * D),
                (0, 1): random.normal(subkey4, shape=(1,) + (N,) * D),
            },
            D,
            (True, False),
        )
        x = x.concat(const_x)
        output_keys = x.get_signature()
        spatial_dims = x.get_spatial_dims()
        output_keys_1d = models.Climate1D.get_1d_signature(output_keys, spatial_dims[1])

        model = models.Climate1D(
            models.ModelWrapper(1, eqx.nn.Identity(), output_keys_1d, True),
            output_keys,
            past_steps,
            1,
            spatial_dims,
            const_x.get_signature_dict(),
        )

        # test that the conversion to 1d preserves longitude flips
        longitude_flip = np.array([[-1, 0], [0, 1]])
        longitude_flip_1d = np.array([[-1]])
        first = model.to1d(x.times_group_element(longitude_flip))
        second = model.to1d(x).times_group_element(longitude_flip_1d)
        assert first.__eq__(second, rtol=1e-3, atol=1e-3)
