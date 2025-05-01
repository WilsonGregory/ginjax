import pytest
import math
import time

import jax
import jax.numpy as jnp
import jax.random as random

import ginjax.geometric as geom
import ginjax.data as gc_data


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

    def testTimeSeriesToMultiImages(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        timesteps = 20
        N = 5
        past_steps = 4
        future_steps = 1

        # test basic dynamic fields
        key, subkey1, subkey2 = random.split(key, 3)
        dynamic_fields = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(timesteps,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(timesteps,) + (N,) * D + (D,)),
            },
            D,
        )
        constant_fields = geom.MultiImage({}, D)

        X, Y = gc_data.times_series_to_multi_images(
            dynamic_fields, constant_fields, timesteps, past_steps, future_steps
        )
        num_windows = (
            timesteps - past_steps - future_steps + 1
        )  # sliding window per original trajectory
        assert isinstance(X, geom.MultiImage) and isinstance(Y, geom.MultiImage)
        assert list(X.keys()) == [((), 0), ((False,), 0)]
        assert X[((), 0)].shape == ((num_windows, past_steps) + (N,) * D)
        assert X[((False,), 0)].shape == ((num_windows, past_steps) + (N,) * D + (D,))
        for i in range(num_windows):
            assert jnp.allclose(X[((), 0)][i], dynamic_fields[((), 0)][i : i + past_steps])
            assert jnp.allclose(
                X[((False,), 0)][i], dynamic_fields[((False,), 0)][i : i + past_steps]
            )

        assert Y[((), 0)].shape == ((num_windows, future_steps) + (N,) * D)
        assert Y[((False,), 0)].shape == ((num_windows, future_steps) + (N,) * D + (D,))
        for i in range(num_windows):
            assert jnp.allclose(
                Y[((), 0)][i],
                dynamic_fields[((), 0)][i + past_steps : i + past_steps + future_steps],
            )
            assert jnp.allclose(
                Y[((False,), 0)][i],
                dynamic_fields[((False,), 0)][i + past_steps : i + past_steps + future_steps],
            )

        # test with a constant fields
        key, subkey3, subkey4 = random.split(key, num=3)
        constant_fields = geom.MultiImage(
            {
                (1, 0): random.normal(subkey3, shape=(1,) + (N,) * D + (D,)),
                (2, 0): random.normal(subkey4, shape=(1,) + (N,) * D + (D, D)),
            },
            D,
        )

        X2, Y2 = gc_data.times_series_to_multi_images(
            dynamic_fields, constant_fields, timesteps, past_steps, future_steps
        )
        assert list(X2.keys()) == [((), 0), ((False,), 0), ((False, False), 0)]
        assert X2[((), 0)].shape == ((num_windows, past_steps) + (N,) * D)
        assert X2[((False,), 0)].shape == ((num_windows, past_steps + 1) + (N,) * D + (D,))
        assert X2[((False, False), 0)].shape == ((num_windows, 1) + (N,) * D + (D, D))
        for i in range(num_windows):
            assert jnp.allclose(
                X2[((), 0)][i, :past_steps], dynamic_fields[((), 0)][i : i + past_steps]
            )
            assert jnp.allclose(
                X2[((False,), 0)][i, :past_steps], dynamic_fields[((False,), 0)][i : i + past_steps]
            )
            assert jnp.allclose(X2[((False,), 0)][i, past_steps], constant_fields[((False,), 0)])
            assert jnp.allclose(X2[((False, False), 0)][i], constant_fields[((False, False), 0)])

        assert Y2[((), 0)].shape == ((num_windows, future_steps) + (N,) * D)
        assert Y2[((False,), 0)].shape == ((num_windows, future_steps) + (N,) * D + (D,))
        for i in range(num_windows):
            assert jnp.allclose(
                Y2[((), 0)][i],
                dynamic_fields[((), 0)][i + past_steps : i + past_steps + future_steps],
            )
            assert jnp.allclose(
                Y2[((False,), 0)][i],
                dynamic_fields[((False,), 0)][i + past_steps : i + past_steps + future_steps],
            )

        # test with constant fields with multiple channels
        key, subkey5 = random.split(key)
        constant_fields = geom.MultiImage(
            {(1, 0): random.normal(subkey5, shape=(3,) + (N,) * D + (D,))}, D
        )

        X3, Y3 = gc_data.times_series_to_multi_images(
            dynamic_fields, constant_fields, timesteps, past_steps, future_steps
        )
        assert list(X3.keys()) == [((), 0), ((False,), 0)]
        assert X3[((), 0)].shape == ((num_windows, past_steps) + (N,) * D)
        assert X3[((False,), 0)].shape == ((num_windows, past_steps + 3) + (N,) * D + (D,))
        for i in range(num_windows):
            assert jnp.allclose(
                X3[((), 0)][i, :past_steps], dynamic_fields[((), 0)][i : i + past_steps]
            )
            assert jnp.allclose(
                X3[((False,), 0)][i, :past_steps], dynamic_fields[((False,), 0)][i : i + past_steps]
            )
            assert jnp.allclose(X3[((False,), 0)][i, past_steps:], constant_fields[((False,), 0)])

        assert Y3[((), 0)].shape == ((num_windows, future_steps) + (N,) * D)
        assert Y3[((False,), 0)].shape == ((num_windows, future_steps) + (N,) * D + (D,))

        # test with multiple channels of timestep fields
        key, subkey6, subkey7 = random.split(key, 3)
        data1 = random.normal(subkey6, shape=(timesteps,) + (N,) * D)
        data2 = random.normal(subkey7, shape=(timesteps,) + (N,) * D)
        dynamic_fields = geom.MultiImage(
            {(0, 0): jnp.stack([data1, data2]).reshape((2 * timesteps,) + (N,) * D)}, D
        )
        constant_fields = geom.MultiImage({}, D)

        X4, Y4 = gc_data.times_series_to_multi_images(
            dynamic_fields, constant_fields, timesteps, past_steps, future_steps
        )
        assert list(X4.keys()) == [((), 0)]
        assert X4[((), 0)].shape == ((num_windows, 2 * past_steps) + (N,) * D)
        for i in range(num_windows):
            assert jnp.allclose(
                X4[((), 0)][i].reshape((2, past_steps) + (N,) * D)[0], data1[i : i + past_steps]
            )
            assert jnp.allclose(
                X4[((), 0)][i].reshape((2, past_steps) + (N,) * D)[1], data2[i : i + past_steps]
            )

        assert Y4[((), 0)].shape == ((num_windows, 2 * future_steps) + (N,) * D)
        for i in range(num_windows):
            assert jnp.allclose(
                Y4[((), 0)][i].reshape((2, future_steps) + (N,) * D)[0],
                data1[i + past_steps : i + past_steps + future_steps],
            )
            assert jnp.allclose(
                Y4[((), 0)][i].reshape((2, future_steps) + (N,) * D)[1],
                data2[i + past_steps : i + past_steps + future_steps],
            )

    def testBatchTimeSeries(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        batch = 5
        timesteps = 20
        N = 5
        past_steps = 4
        future_steps = 1

        # test basic dynamic fields
        key, subkey1, subkey2 = random.split(key, 3)
        dynamic_fields = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(batch, timesteps) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(batch, timesteps) + (N,) * D + (D,)),
            },
            D,
        )
        constant_fields = geom.MultiImage({}, D)

        X, Y = gc_data.batch_time_series(
            dynamic_fields, constant_fields, timesteps, past_steps, future_steps
        )
        num_windows = (
            timesteps - past_steps - future_steps + 1
        )  # sliding window per original trajectory
        assert isinstance(X, geom.MultiImage) and isinstance(Y, geom.MultiImage)
        assert list(X.keys()) == [((), 0), ((False,), 0)]
        assert X[((), 0)].shape == ((batch * num_windows, past_steps) + (N,) * D)
        assert X[((False,), 0)].shape == ((batch * num_windows, past_steps) + (N,) * D + (D,))

        X_exp = X.expand(0, num_windows)  # (b,num_windows,past_steps,spatial,tensor)
        for i in range(batch):
            for j in range(num_windows):
                assert jnp.allclose(
                    X_exp[((), 0)][i, j], dynamic_fields[((), 0)][i, j : j + past_steps]
                )
                assert jnp.allclose(
                    X_exp[((False,), 0)][i, j], dynamic_fields[((False,), 0)][i, j : j + past_steps]
                )

        assert Y[((), 0)].shape == ((batch * num_windows, future_steps) + (N,) * D)
        assert Y[((False,), 0)].shape == ((batch * num_windows, future_steps) + (N,) * D + (D,))

        Y_exp = Y.expand(0, num_windows)
        for i in range(batch):
            for j in range(num_windows):
                assert jnp.allclose(
                    Y_exp[((), 0)][i, j],
                    dynamic_fields[((), 0)][i, j + past_steps : j + past_steps + future_steps],
                )
                assert jnp.allclose(
                    Y_exp[((False,), 0)][i, j],
                    dynamic_fields[((False,), 0)][
                        i, j + past_steps : j + past_steps + future_steps
                    ],
                )

        # test with a constant fields
        key, subkey3 = random.split(key)
        constant_fields = geom.MultiImage(
            {(1, 0): random.normal(subkey3, shape=(batch, 1) + (N,) * D + (D,))}, D
        )

        X2, Y2 = gc_data.batch_time_series(
            dynamic_fields, constant_fields, timesteps, past_steps, future_steps
        )
        assert list(X.keys()) == [((), 0), ((False,), 0)]
        assert X2[((), 0)].shape == ((batch * num_windows, past_steps) + (N,) * D)
        assert X2[((False,), 0)].shape == ((batch * num_windows, past_steps + 1) + (N,) * D + (D,))

        X2_dynamic, X2_const = X2.concat_inverse({((False,), 0): 1}, axis=1)
        X2_dynamic = X2_dynamic.expand(0, num_windows)
        for i in range(batch):
            for j in range(num_windows):
                assert jnp.allclose(
                    X2_dynamic[((), 0)][i, j],
                    dynamic_fields[((), 0)][i, j : j + past_steps],
                )
                assert jnp.allclose(
                    X2_dynamic[((False,), 0)][i, j],
                    dynamic_fields[((False,), 0)][i, j : j + past_steps],
                )

        # check that each element of the batch has the correct constant field
        X2_const = X2_const.expand(0, num_windows)
        for i in range(num_windows):
            assert jnp.allclose(X2_const[((False,), 0)][:, i], constant_fields[((False,), 0)])

        assert Y2[((), 0)].shape == ((batch * num_windows, future_steps) + (N,) * D)
        assert Y2[((False,), 0)].shape == ((batch * num_windows, future_steps) + (N,) * D + (D,))

        Y2_exp = Y2.expand(0, num_windows)
        for i in range(batch):
            for j in range(num_windows):
                assert jnp.allclose(
                    Y2_exp[((), 0)][i, j],
                    dynamic_fields[((), 0)][i, j + past_steps : j + past_steps + future_steps],
                )
                assert jnp.allclose(
                    Y2_exp[((False,), 0)][i, j],
                    dynamic_fields[((False,), 0)][
                        i, j + past_steps : j + past_steps + future_steps
                    ],
                )
