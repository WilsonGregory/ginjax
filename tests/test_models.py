import time
import itertools as it

import jax.numpy as jnp
from jax import random
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
