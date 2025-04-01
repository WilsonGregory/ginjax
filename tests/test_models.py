import time
import itertools as it
import numpy as np
from typing_extensions import Self

import jax.numpy as jnp
from jax import random
import jax
import equinox as eqx

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models


class TestModels:
    # Class to test the functions in the models.py file

    def testConvContract2D(self):
        D = 2
        M = 3
        N = 5
        in_c = 3
        out_c = 4
        max_k = 2
        ks = list(range(max_k + 1))
        parities = [0, 1]
        ks_ps_prod = list(it.product(ks, parities))
        key = random.PRNGKey(time.time_ns())

        conv_filters = geom.get_invariant_filters([M], ks, parities, D, geom.make_all_operators(D))
        assert isinstance(conv_filters, geom.MultiImage)

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
                        (k, p): random.normal(subkeys[i], shape=(in_c,) + (N,) * D + (D,) * k)
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
