import geometricconvolutions.geometric as geom
import geometricconvolutions.data as gc_data
import pytest
import jax.numpy as jnp
import jax.random as random
import jax
import math
import time

import sys

sys.path.append("scripts/phase2vec/")


class TestMisc:

    def testPermutationParity(self):
        assert geom.permutation_parity([0]) == 1
        assert geom.permutation_parity((0, 1)) == 1
        assert geom.permutation_parity((1, 0)) == -1
        assert geom.permutation_parity([1, 0]) == -1
        assert geom.permutation_parity([1, 1]) == 0
        assert geom.permutation_parity([0, 1, 2]) == 1
        assert geom.permutation_parity([0, 2, 1]) == -1
        assert geom.permutation_parity([1, 2, 0]) == 1
        assert geom.permutation_parity([1, 0, 2]) == -1
        assert geom.permutation_parity([2, 1, 0]) == -1
        assert geom.permutation_parity([2, 0, 1]) == 1
        assert geom.permutation_parity([2, 1, 1]) == 0

    def testLeviCivitaSymbol(self):
        with pytest.raises(AssertionError):
            geom.LeviCivitaSymbol.get(1)

        assert (geom.LeviCivitaSymbol.get(2) == jnp.array([[0, 1], [-1, 0]], dtype=int)).all()
        assert (
            geom.LeviCivitaSymbol.get(3)
            == jnp.array(
                [
                    [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                    [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                    [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
                ],
                dtype=int,
            )
        ).all()

        assert geom.LeviCivitaSymbol.get(2) is geom.LeviCivitaSymbol.get(
            2
        )  # test that we aren't remaking them

    def testGroupSize(self):
        for d in range(2, 7):
            operators = geom.make_all_operators(d)

            # test the group size
            assert len(operators) == 2 * (2 ** (d - 1)) * math.factorial(d)

    def testGetOperatorsInversesTranspose(self):
        # test that the transpose of each group operator is its inverse (orthogonal group)
        for D in [2, 3]:
            operators = geom.make_all_operators(D)
            for gg in operators:
                assert jnp.allclose(gg @ gg.T, jnp.eye(D), atol=geom.TINY, rtol=geom.TINY)
                assert jnp.allclose(gg.T @ gg, jnp.eye(D), atol=geom.TINY, rtol=geom.TINY)

    def testGetContractionIndices(self):
        idxs = geom.get_contraction_indices(3, 1)
        known_list = [((0, 1),), ((0, 2),), ((1, 2),)]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(3, 1, ((0, 1),))
        known_list = [((0, 1),), ((0, 2),)]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5, 3)
        known_list = [
            ((0, 1),),
            ((0, 2),),
            ((0, 3),),
            ((0, 4),),
            ((1, 2),),
            ((1, 3),),
            ((1, 4),),
            ((2, 3),),
            ((2, 4),),
            ((3, 4),),
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5, 1)
        known_list = [
            ((0, 1), (2, 3)),
            ((0, 1), (2, 4)),
            ((0, 1), (3, 4)),
            ((0, 2), (1, 3)),
            ((0, 2), (1, 4)),
            ((0, 2), (3, 4)),
            ((0, 3), (1, 2)),
            ((0, 3), (1, 4)),
            ((0, 3), (2, 4)),
            ((0, 4), (1, 2)),
            ((0, 4), (1, 3)),
            ((0, 4), (2, 3)),
            ((1, 2), (3, 4)),
            ((1, 3), (2, 4)),
            ((1, 4), (2, 3)),
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5, 1, ((0, 1),))
        known_list = [
            ((0, 1), (2, 3)),
            ((0, 1), (2, 4)),
            ((0, 1), (3, 4)),
            ((0, 2), (1, 3)),
            ((0, 2), (1, 4)),
            ((0, 2), (3, 4)),
            ((0, 3), (1, 4)),
            ((0, 3), (2, 4)),
            ((0, 4), (2, 3)),
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5, 1, ((0, 1), (2, 3)))
        known_list = [
            ((0, 1), (2, 3)),
            ((0, 1), (2, 4)),
            ((0, 2), (1, 3)),
            ((0, 2), (1, 4)),
            ((0, 2), (3, 4)),
            ((0, 4), (2, 3)),
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

    def testGetInvariantImage(self):
        N = 5

        D = 2
        operators = geom.make_all_operators(D)
        for k in [0, 2, 4]:
            invariant_basis = geom.get_invariant_image(N, D, k, 0, data_only=False)

            for gg in operators:
                assert invariant_basis == invariant_basis.times_group_element(gg)

        D = 3
        operators = geom.make_all_operators(D)
        for k in [0, 2, 4]:
            invariant_basis = geom.get_invariant_image(N, D, k, 0, data_only=False)

            for gg in operators:
                assert invariant_basis == invariant_basis.times_group_element(gg)

    def testGetOperatorsOnLayer(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        operators = jnp.stack(geom.make_all_operators(D))

        layer = geom.Layer(
            {
                (0, 0): jnp.ones((1,) + (N,) * D),
                (1, 0): jnp.ones((1,) + (N,) * D + (D,)),
                (2, 0): jnp.ones((1,) + (N,) * D + (D, D)),
            },
            D,
            False,
        )

        operators_on_layer = geom.get_operators_on_layer(operators, layer)

        # Assert that operators are orthogonal
        for gg_layer in operators_on_layer:
            assert jnp.allclose(gg_layer @ gg_layer.T, jnp.eye(len(gg_layer)))
            assert jnp.allclose(gg_layer.T @ gg_layer, jnp.eye(len(gg_layer)))

        rand_layer = layer.empty()
        for (k, parity), img in layer.items():
            key, subkey = random.split(key)
            rand_layer.append(k, parity, random.normal(subkey, shape=img.shape))

        # Assert that using the operator_on_layer is equivalent to using the times_group_element
        for gg_layer, gg in zip(operators_on_layer, operators):
            assert jnp.allclose(
                rand_layer.times_group_element(gg, precision=jax.lax.Precision.HIGHEST).to_vector(),
                gg_layer @ rand_layer.to_vector(),
            )

    def testTimeSeriesIdxsHardcoded(self):
        past_steps = 2
        future_steps = 1
        num_channels = 5
        delta_t = 1
        input_idxs, output_idxs = gc_data.time_series_idxs(
            past_steps, future_steps, delta_t, num_channels
        )
        assert len(input_idxs) == 3
        assert jnp.allclose(input_idxs[0], jnp.array([0, 1]))
        assert jnp.allclose(input_idxs[1], jnp.array([1, 2]))
        assert jnp.allclose(input_idxs[2], jnp.array([2, 3]))

        assert len(output_idxs) == 3
        assert jnp.allclose(output_idxs[0], jnp.array([2]))
        assert jnp.allclose(output_idxs[1], jnp.array([3]))
        assert jnp.allclose(output_idxs[2], jnp.array([4]))

        num_channels = 10
        delta_t = 2
        input_idxs, output_idxs = gc_data.time_series_idxs(
            past_steps, future_steps, delta_t, num_channels
        )
        assert len(input_idxs) == 6
        assert jnp.allclose(input_idxs[0], jnp.array([0, 2]))
        assert jnp.allclose(input_idxs[1], jnp.array([1, 3]))
        assert jnp.allclose(input_idxs[2], jnp.array([2, 4]))
        assert jnp.allclose(input_idxs[3], jnp.array([3, 5]))
        assert jnp.allclose(input_idxs[4], jnp.array([4, 6]))
        assert jnp.allclose(input_idxs[5], jnp.array([5, 7]))

        assert len(output_idxs) == 6
        assert jnp.allclose(output_idxs[0], jnp.array([4]))
        assert jnp.allclose(output_idxs[1], jnp.array([5]))
        assert jnp.allclose(output_idxs[2], jnp.array([6]))
        assert jnp.allclose(output_idxs[3], jnp.array([7]))
        assert jnp.allclose(output_idxs[4], jnp.array([8]))
        assert jnp.allclose(output_idxs[5], jnp.array([9]))

        future_steps = 2
        input_idxs, output_idxs = gc_data.time_series_idxs(
            past_steps, future_steps, delta_t, num_channels
        )
        assert len(input_idxs) == 4
        assert jnp.allclose(input_idxs[0], jnp.array([0, 2]))
        assert jnp.allclose(input_idxs[1], jnp.array([1, 3]))
        assert jnp.allclose(input_idxs[2], jnp.array([2, 4]))
        assert jnp.allclose(input_idxs[3], jnp.array([3, 5]))

        assert len(output_idxs) == 4
        assert jnp.allclose(output_idxs[0], jnp.array([4, 6]))
        assert jnp.allclose(output_idxs[1], jnp.array([5, 7]))
        assert jnp.allclose(output_idxs[2], jnp.array([6, 8]))
        assert jnp.allclose(output_idxs[3], jnp.array([7, 9]))

    def testTimeSeriesIdxs(self):
        D = 2
        N = 3
        spatial_dims = (N,) * D
        batch = 10
        channels = 11
        key = random.PRNGKey(0)

        for past_steps in [1, 2, 4]:
            for future_steps in [1, 5]:
                for k in [0, 1, 2]:
                    key, subkey = random.split(key)
                    img_data = random.normal(
                        subkey, shape=((batch, channels) + spatial_dims + (D,) * k)
                    )
                    num_windows = channels - future_steps - past_steps + 1

                    input_idxs, output_idxs = gc_data.time_series_idxs(
                        past_steps, future_steps, 1, channels
                    )
                    input_data = img_data[:, input_idxs]
                    assert (
                        input_data.shape
                        == (batch, num_windows, past_steps) + spatial_dims + (D,) * k
                    )

                    output_data = img_data[:, output_idxs]
                    assert (
                        output_data.shape
                        == (batch, num_windows, future_steps) + spatial_dims + (D,) * k
                    )

                    for b in range(batch):
                        for i in range(num_windows):
                            assert jnp.allclose(input_data[b, i], img_data[b, i : i + past_steps])
                            assert jnp.allclose(
                                output_data[b, i],
                                img_data[b, i + past_steps : i + past_steps + future_steps],
                            )

    def testTimeSeriesToLayers(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        batch = 5
        timesteps = 20
        N = 5
        past_steps = 4
        future_steps = 1

        # test basic dynamic fields
        key, subkey1, subkey2 = random.split(key, 3)
        dynamic_fields = {
            (0, 0): random.normal(subkey1, shape=(batch, timesteps) + (N,) * D),
            (1, 0): random.normal(subkey2, shape=(batch, timesteps) + (N,) * D + (D,)),
        }

        X, Y = gc_data.times_series_to_layers(
            D, dynamic_fields, {}, False, past_steps, future_steps
        )
        num_windows = (
            timesteps - past_steps - future_steps + 1
        )  # sliding window per original trajectory
        assert isinstance(X, geom.BatchLayer) and isinstance(Y, geom.BatchLayer)
        assert list(X.keys()) == [(0, 0), (1, 0)]
        assert X[(0, 0)].shape == ((batch * num_windows, past_steps) + (N,) * D)
        assert X[(1, 0)].shape == ((batch * num_windows, past_steps) + (N,) * D + (D,))
        assert jnp.allclose(X[(0, 0)][0], dynamic_fields[(0, 0)][0, :past_steps])
        assert jnp.allclose(X[(1, 0)][0], dynamic_fields[(1, 0)][0, :past_steps])

        assert Y[(0, 0)].shape == ((batch * num_windows, future_steps) + (N,) * D)
        assert Y[(1, 0)].shape == ((batch * num_windows, future_steps) + (N,) * D + (D,))
        assert jnp.allclose(
            Y[(0, 0)][0],
            dynamic_fields[(0, 0)][0, past_steps : past_steps + future_steps],
        )
        assert jnp.allclose(
            Y[(1, 0)][0],
            dynamic_fields[(1, 0)][0, past_steps : past_steps + future_steps],
        )

        # test with a constant fields
        key, subkey3 = random.split(key)
        constant_fields = {(1, 0): random.normal(subkey3, shape=(batch,) + (N,) * D + (D,))}

        X2, Y2 = gc_data.times_series_to_layers(
            D,
            dynamic_fields,
            constant_fields,
            False,
            past_steps,
            future_steps,
        )
        assert list(X.keys()) == [(0, 0), (1, 0)]
        assert X2[(0, 0)].shape == ((batch * num_windows, past_steps) + (N,) * D)
        assert X2[(1, 0)].shape == ((batch * num_windows, past_steps + 1) + (N,) * D + (D,))
        assert jnp.allclose(X2[(0, 0)][0, :past_steps], dynamic_fields[(0, 0)][0, :past_steps])
        assert jnp.allclose(X2[(1, 0)][0, :past_steps], dynamic_fields[(1, 0)][0, :past_steps])
        assert jnp.allclose(X2[(1, 0)][0, past_steps], constant_fields[(1, 0)][0])

        assert Y2[(0, 0)].shape == ((batch * num_windows, future_steps) + (N,) * D)
        assert Y2[(1, 0)].shape == ((batch * num_windows, future_steps) + (N,) * D + (D,))
        assert jnp.allclose(
            Y2[(0, 0)][0],
            dynamic_fields[(0, 0)][0, past_steps : past_steps + future_steps],
        )
        assert jnp.allclose(
            Y2[(1, 0)][0],
            dynamic_fields[(1, 0)][0, past_steps : past_steps + future_steps],
        )
