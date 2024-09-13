import jax.numpy as jnp
from jax import random

import geometricconvolutions.geometric as geom
import geometricconvolutions.models as models


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
