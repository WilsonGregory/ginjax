import jax.numpy as jnp
from jax import random
import jax

import ginjax.geometric as geom
import ginjax.ml as ml


class TestMachineLearning:

    def testGetBatches(self):
        num_devices = 1  # since it can only see the cpu
        cpu = [jax.devices("cpu")[0]]
        key = random.PRNGKey(0)
        N = 5
        D = 2
        k = 0

        X = geom.MultiImage({(k, 0): random.normal(key, shape=((10, 1) + (N,) * D + (D,) * k))}, D)
        Y = geom.MultiImage({(k, 0): random.normal(key, shape=((10, 1) + (N,) * D + (D,) * k))}, D)

        batch_size = 2
        X_batches, Y_batches = ml.get_batches(
            (X, Y), batch_size=batch_size, rand_key=key, devices=cpu
        )
        assert len(X_batches) == len(Y_batches) == 5
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert (
                X_batch[(k, 0)].shape
                == Y_batch[(k, 0)].shape
                == (num_devices, batch_size, 1) + (N,) * D + (D,) * k
            )

        X = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 1)),
            },
            D,
        )
        Y = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 1)),
            },
            D,
        )

        # batching when the multi_image has multiple channels at different values of k
        batch_size = 5
        X_batches, Y_batches = ml.get_batches(
            (X, Y), batch_size=batch_size, rand_key=key, devices=cpu
        )
        assert len(X_batches) == len(Y_batches) == 4
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert (
                X_batch[(0, 0)].shape
                == Y_batch[(0, 0)].shape
                == (num_devices, batch_size, 1) + (N,) * D + (D,) * 0
            )
            assert (
                X_batch[(1, 0)].shape
                == Y_batch[(1, 0)].shape
                == (num_devices, batch_size, 1) + (N,) * D + (D,) * 1
            )

        X = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((20, 2) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 1)),
            },
            D,
        )
        Y = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((20, 2) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 1)),
            },
            D,
        )

        # batching when multi_image has multiple channels for one value of k
        batch_size = 5
        X_batches, Y_batches = ml.get_batches(
            (X, Y), batch_size=batch_size, rand_key=key, devices=cpu
        )
        assert len(X_batches) == len(Y_batches) == 4
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert (
                X_batch[(0, 0)].shape
                == Y_batch[(0, 0)].shape
                == (num_devices, batch_size, 2) + (N,) * D + (D,) * 0
            )
            assert (
                X_batch[(1, 0)].shape
                == Y_batch[(1, 0)].shape
                == (num_devices, batch_size, 1) + (N,) * D + (D,) * 1
            )

    def testAutoregressiveStep(self):
        past_steps = 4
        N = 5
        D = 2

        key = random.PRNGKey(0)
        key1, key2, key3, key4, key5, key6, key7 = random.split(key, 7)

        data1 = random.normal(key1, shape=(past_steps,) + (N,) * D)

        input1 = geom.MultiImage({(0, 0): data1}, D)
        one_step1 = geom.MultiImage({(0, 0): random.normal(key2, shape=(1,) + (N,) * D)}, D)

        new_input, output = ml.training.autoregressive_step(
            input1, one_step1, input1.empty(), past_steps
        )
        assert jnp.allclose(
            new_input[(0, 0)],
            jnp.concatenate([input1[(0, 0)][1:], one_step1[(0, 0)]]),
        )
        assert output == one_step1

        data2 = random.normal(key3, shape=(2 * past_steps,) + (N,) * D + (D,))

        input2 = geom.MultiImage({(0, 0): data1, (1, 0): data2}, D)
        one_step2 = geom.MultiImage(
            {
                (0, 0): random.normal(key4, shape=(1,) + (N,) * D),
                (1, 0): random.normal(key5, shape=(2,) + (N,) * D + (D,)),
            },
            D,
        )

        new_input, output = ml.training.autoregressive_step(
            input2, one_step2, input2.empty(), past_steps
        )
        assert jnp.allclose(
            new_input[(0, 0)],
            jnp.concatenate([input2[(0, 0)][1:], one_step2[(0, 0)]]),
        )
        assert output == one_step2
        assert jnp.allclose(new_input[(1, 0)][: past_steps - 1], input2[(1, 0)][1:past_steps])
        assert jnp.allclose(new_input[(1, 0)][past_steps - 1], one_step2[(1, 0)][0])
        assert jnp.allclose(new_input[(1, 0)][past_steps:-1], input2[(1, 0)][past_steps + 1 :])
        assert jnp.allclose(new_input[(1, 0)][-1], one_step2[(1, 0)][1])

        constant_field1 = random.normal(key6, shape=(1,) + (N,) * D)
        constant_field2 = random.normal(key7, shape=(1,) + (N,) * D + (D,))

        input3 = input2.concat(
            geom.MultiImage({(0, 0): constant_field1, (1, 0): constant_field2}, D)
        )
        new_input, output = ml.training.autoregressive_step(
            input3, one_step2, input3.empty(), past_steps, {(0, 0): 1, (1, 0): 1}
        )
        assert jnp.allclose(
            new_input[(0, 0)],
            jnp.concatenate([input3[(0, 0)][1:-1], one_step2[(0, 0)], constant_field1]),
        )
        assert output == one_step2
        assert jnp.allclose(new_input[(1, 0)][: past_steps - 1], input3[(1, 0)][1:past_steps])
        assert jnp.allclose(new_input[(1, 0)][past_steps - 1], one_step2[(1, 0)][0])
        assert jnp.allclose(new_input[(1, 0)][past_steps:-2], input3[(1, 0)][past_steps + 1 : -1])
        assert jnp.allclose(new_input[(1, 0)][-2], one_step2[(1, 0)][1])
        assert jnp.allclose(new_input[(1, 0)][-1:], constant_field2)

        # test when there is a field which is only constant
        input4 = input1.concat(
            geom.MultiImage({(0, 0): constant_field1, (1, 0): constant_field2}, D)
        )
        new_input, output = ml.training.autoregressive_step(
            input4, one_step1, input4.empty(), past_steps, {(0, 0): 1, (1, 0): 1}
        )
        assert jnp.allclose(
            new_input[(0, 0)],
            jnp.concatenate([input4[(0, 0)][1:-1], one_step1[(0, 0)], constant_field1]),
        )
        assert jnp.allclose(new_input[(1, 0)], constant_field2)
        assert output == one_step1
