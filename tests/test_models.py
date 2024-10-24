import time
import itertools as it

import jax.numpy as jnp
from jax import random

import geometricconvolutions.geometric as geom
import geometricconvolutions.models as models
import geometricconvolutions.ml_eqx as ml_eqx


class TestModels:
    # Class to test the functions in the models.py file

    def testGroupAverage(self):
        N = 1
        D = 2

        def non_equiv_function(params, x, key, train, return_params=None):
            return x + geom.Layer(
                {(1, 0): random.normal(key, shape=(1,) + (N,) * D + (D,) * 1)}, D, False
            )

        def equiv_function(params, x, key, train, return_params=None):
            return geom.Layer({(1, 0): x[(1, 0)][0:1] + x[(1, 0)][1:2]}, D, False)

        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        vec = geom.Layer(
            {(1, 0): random.normal(subkey, shape=(1,) + (N,) * D + (D,) * 1)}, D, False
        )

        group_operators = geom.make_all_operators(D)

        # show that non_equiv_function is not equivariant
        for gg in group_operators:
            key, subkey1, subkey2 = random.split(key, 3)
            first = non_equiv_function(None, vec, subkey1, None).times_group_element(gg)
            second = non_equiv_function(None, vec.times_group_element(gg), subkey2, None)
            assert first != second

        # show that non_equiv_function is equivariant after it undergoes the group averaging
        for gg in group_operators:
            key, subkey1, subkey2 = random.split(key, 3)
            first = models.group_average(
                {}, vec, subkey1, None, non_equiv_function
            ).times_group_element(gg)
            second = models.group_average(
                {}, vec.times_group_element(gg), subkey2, None, non_equiv_function
            )
            assert first == second

        key, subkey = random.split(key)
        vec = geom.Layer(
            {(1, 0): random.normal(subkey, shape=(2,) + (N,) * D + (D,) * 1)}, D, False
        )

        # show that equiv_function is equivariant
        for gg in group_operators:
            key, subkey1, subkey2 = random.split(key, 3)
            first = equiv_function(None, vec, subkey1, None).times_group_element(gg)
            second = equiv_function(None, vec.times_group_element(gg), subkey2, None)
            assert first.__eq__(
                second, 1e-2, 1e-2
            ), f"{jnp.max(jnp.abs(first[(1,0)] - second[(1,0)]))}"

        # show that equiv_function is still equivariant after it undergoes the group averaging
        for gg in group_operators:
            key, subkey1, subkey2 = random.split(key, 3)
            first = models.group_average(
                {}, vec, subkey1, None, equiv_function
            ).times_group_element(gg)
            second = models.group_average(
                {}, vec.times_group_element(gg), subkey2, None, equiv_function
            )
            assert first == second

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

        # power set (excluding empty set) of possible in_k, out_k and parity
        powerset = list(
            it.chain.from_iterable(
                it.combinations(ks_ps_prod, r + 1) for r in range(len(ks_ps_prod))
            )
        )
        for in_ks_ps in powerset:
            for out_ks_ps in powerset:
                input_keys = tuple((in_key, in_c) for in_key in in_ks_ps)
                target_keys = tuple((out_key, out_c) for out_key in out_ks_ps)

                key, *subkeys = random.split(key, num=len(input_keys) + 1)
                layer = geom.Layer(
                    {
                        (k, p): random.normal(subkeys[i], shape=(in_c,) + (N,) * D + (D,) * k)
                        for i, ((k, p), _) in enumerate(input_keys)
                    },
                    D,
                )

                key, subkey = random.split(key)
                conv = ml_eqx.ConvContract(
                    input_keys, target_keys, conv_filters, use_bias=False, key=subkey
                )
                if conv.missing_filter:
                    continue

                assert conv.fast_convolve(layer, conv.weights) == conv.individual_convolve(
                    layer, conv.weights
                )
