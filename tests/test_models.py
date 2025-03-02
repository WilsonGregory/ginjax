import time
import itertools as it

from jax import random

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml


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
